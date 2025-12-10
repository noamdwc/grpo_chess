import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download
from tqdm import tqdm
import jax
import jax.numpy as jnp
from functools import partial

# --- 1. MEMORY MANAGEMENT ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Prevent PyTorch from eating GPU memory since we only use it for CPU loading
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# --- 2. SETUP ---
repo_id = "dbest-isi/searchless-chess-9M-selfplay"
model_path = snapshot_download(repo_id=repo_id, local_dir="./searchless_chess_model")

sys.path.insert(0, f"{model_path}/searchless_chess_code")
import hf_model
import utils
import tokenizer
import chess

# --- 3. HELPER: REVERSE TOKENIZER ---
REV_CHAR_INDEX = {v: k for k, v in tokenizer._CHARACTERS_INDEX.items()}

def detokenize(tokens):
    """Reconstructs FEN from tokens. (Optimized for standalone use)"""
    chars = [REV_CHAR_INDEX.get(int(t), '?') for t in tokens]
    side = chars[0]
    board_chars = chars[1:65]
    
    rows = []
    for i in range(0, 64, 8):
        row_raw = board_chars[i:i+8]
        row_fen = ""
        empty = 0
        for c in row_raw:
            if c == '.':
                empty += 1
            else:
                if empty: row_fen += str(empty); empty = 0
                row_fen += c
        if empty: row_fen += str(empty)
        rows.append(row_fen)
    
    # Parse tail
    castling = "".join(chars[65:69]).replace('.', '') or "-"
    ep = "".join(chars[69:71]).replace('.', '') or "-"
    # Simple fallback for move counts if parsing fails, as they rarely affect legal moves
    half = "".join(chars[71:74]).replace('.', '') or "0"
    full = "".join(chars[74:77]).replace('.', '') or "1"
    
    return f"{'/'.join(rows)} {side} {castling} {ep} {half} {full}"

# --- 4. DATASET (CPU WORKER) ---
class ChessChildDataset(Dataset):
    def __init__(self, npy_path):
        # Load data in mmap mode (shared memory across workers)
        self.data = np.load(npy_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        This runs in parallel CPU processes.
        Input: Parent Token indices
        Output: All Children Token indices + Action IDs
        """
        parent_tokens = self.data[idx].astype(np.int32)
        
        try:
            fen = detokenize(parent_tokens)
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
        except:
            # Fallback for bad data
            return None 

        if not legal_moves:
            return None

        children_tokens = []
        action_indices = []
        
        for move in legal_moves:
            uci = move.uci()
            a_idx = utils.MOVE_TO_ACTION.get(uci)
            if a_idx is not None:
                board.push(move)
                # Tokenize child (CPU heavy)
                t = tokenizer.tokenize(board.fen())
                children_tokens.append(t)
                action_indices.append(a_idx)
                board.pop()
        
        if not children_tokens:
            return None

        # Convert to numpy for transport
        # We return ragged arrays, collate_fn will stack them
        return {
            "parent_tokens": parent_tokens,
            "children_tokens": np.stack(children_tokens).astype(np.int32),
            "action_indices": np.array(action_indices, dtype=np.int32)
        }

def custom_collate(batch):
    """Groups parallel results into a batch for the GPU."""
    # Filter Nones
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    # Flatten structure for JAX processing
    all_children = []
    metadata = [] # (batch_index_in_output, action_indices)
    
    valid_parents = []
    
    for i, item in enumerate(batch):
        valid_parents.append(item['parent_tokens'])
        
        curr_children = item['children_tokens'] # [n_moves, 77]
        actions = item['action_indices']        # [n_moves]
        
        start_idx = len(all_children)
        all_children.append(curr_children)
        
        # Record where these children belong: (parent_idx, action_id, relative_child_idx)
        # We'll just store the mapping to scatter later
        metadata.append({
            "parent_idx": i,
            "actions": actions,
            "child_offset": start_idx, # Offset in the list of arrays, not flat elements
            "count": len(actions)
        })

    return {
        "parents": np.stack(valid_parents),
        "flat_children": np.concatenate(all_children, axis=0),
        "metadata": metadata
    }

# --- 5. MAIN ---
def main():
    # Config
    DATA_PATH = "../data/numpy_arrays/action_value_test_sequences.npy"
    SAVE_PATH = "../data/numpy_arrays/action_distillation_dataset_test.npz"
    BATCH_SIZE = 256 # Parents per batch
    NUM_WORKERS = 8  # Adjust based on CPU cores (keep high to feed GPU)
    
    # JAX Setup
    print(f"JAX Devices: {jax.devices()}")
    teacher_wrapper = hf_model.SearchlessChessModel.from_pretrained(model_path)
    params = teacher_wrapper.params
    predictor = teacher_wrapper.predictor
    
    if hasattr(teacher_wrapper, "return_buckets_values"):
        bucket_values = jnp.asarray(teacher_wrapper.return_buckets_values)
    else:
        bucket_values = jnp.linspace(0, 1, 128)

    # JIT Functions
    @partial(jax.jit, static_argnames=["is_training"])
    def predict_batch(params, rng, token_batch, is_training=False):
        return predictor.predict(params, rng, token_batch)

    @jax.jit
    def compute_q_scalar(logits, buckets):
        # Extract penultimate token logits -> Prob -> Dot Product
        # Assuming logits shape [B, 79, 128]
        valid_logits = logits[:, -2, :] 
        probs = jax.nn.softmax(valid_logits)
        return jnp.dot(probs, buckets)

    # Dataset & Loader
    print("Initializing Parallel Data Loader...")
    dataset = ChessChildDataset(DATA_PATH)
    
    # PyTorch DataLoader handles the multiprocessing!
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate,
        prefetch_factor=2
    )

    export_parents = []
    export_probs = []
    rng_seq = jax.random.PRNGKey(42)
    
    print("Starting Pipeline...")
    
    # Iterate through loader (CPU workers are pre-fetching next batches now)
    for batch in tqdm(loader):
        if batch is None: continue
        
        # Data is already prepared by CPU workers
        flat_children = batch['flat_children'] # [Total_Children, 77]
        metadata = batch['metadata']
        
        # --- GPU BLOCK ---
        # Process children in chunks if too large for VRAM
        chunk_size = 4096
        all_q_values = []
        
        for i in range(0, len(flat_children), chunk_size):
            chunk = flat_children[i : i+chunk_size]
            chunk_jax = jnp.array(chunk)
            
            rng_seq, rng_now = jax.random.split(rng_seq)
            outputs = predict_batch(params, rng_now, chunk_jax)
            
            # Handle dict vs array output
            logits = outputs['action_value'] if isinstance(outputs, dict) else outputs
            
            q_vals = compute_q_scalar(logits, bucket_values)
            all_q_values.append(np.array(q_vals))
            
        all_q_values = np.concatenate(all_q_values) # [Total_Children]
        
        # Scatter back to parents (CPU, fast)
        # Init batch targets
        n_parents = len(metadata)
        batch_targets = np.full((n_parents, utils.NUM_ACTIONS), -1e9, dtype=np.float32)
        
        cursor = 0
        for meta in metadata:
            count = meta['count']
            actions = meta['actions']
            p_idx = meta['parent_idx']
            
            # Get corresponding Q values from the flat list
            child_qs = all_q_values[cursor : cursor+count]
            cursor += count
            
            # Negate and Assign
            batch_targets[p_idx, actions] = -child_qs
            
        # Softmax
        # Vectorized softmax on rows
        for i in range(n_parents):
            row = batch_targets[i]
            valid = row > -1e8
            if valid.any():
                z = row[valid]
                z = z - z.max()
                e = np.exp(z)
                probs = e / e.sum()
                
                # Zero out old -inf, write probs
                batch_targets[i] = 0
                batch_targets[i, valid] = probs
            else:
                batch_targets[i] = 0

        export_parents.append(batch['parents'])
        export_probs.append(batch_targets)

    # Save
    print("Concatenating and Saving...")
    final_p = np.concatenate(export_parents, axis=0)
    final_t = np.concatenate(export_probs, axis=0)
    
    np.savez_compressed(
        SAVE_PATH,
        tokens=final_p.astype(np.int16),
        action_probs=final_t.astype(np.float16)
    )
    print("Done.")

if __name__ == "__main__":
    main()