import os
import sys
from functools import partial

# --- 1. MEMORY MANAGEMENT ---
# Crucial: Prevent JAX from locking 90% of VRAM so we have RAM left for saving data.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download

# --- 2. PATH SETUP ---
# Download the model if not present
repo_id = "dbest-isi/searchless-chess-9M-selfplay"
model_path = snapshot_download(repo_id=repo_id, local_dir="./searchless_chess_model")
print(f"Model path: {model_path}")

# Add the bundled code to python path to import internal modules
sys.path.insert(0, f"{model_path}/searchless_chess_code")
import hf_model

# --- 3. CONFIGURATION ---
BATCH_SIZE = 4096
# Adjust these paths to match your folder structure
DATA_PATH = os.path.join('..', 'data', 'numpy_arrays', 'state_value_train_sequences.npy')
SAVE_PATH = os.path.join('..', 'data', 'numpy_arrays', 'distillation_dataset.npz')

def main():
    print(f"JAX Devices: {jax.devices()}")

    # --- 4. LOAD TEACHER MODEL ---
    print("Loading JAX Teacher Model...")
    # Load the wrapper class
    teacher_wrapper = hf_model.SearchlessChessModel.from_pretrained(model_path)
    
    # Extract the raw Haiku components required for pure JAX execution
    params = teacher_wrapper.params
    predictor = teacher_wrapper.predictor

    # --- 5. JIT COMPILED INFERENCE STEP ---
    # We compile the pure function 'predictor.predict'.
    # We mark 'is_training' as static because it changes the control flow (dropout).
    @partial(jax.jit, static_argnames=["is_training"])
    def inference_step(params, rng, batch_tokens, is_training=False):
        # Run the forward pass
        outputs = predictor.predict(params, rng, batch_tokens)
        
        # Handle Output:
        # If outputs is a dict (common in Haiku), we extract the main logits/values.
        # If it's a single array, we cast and return.
        if isinstance(outputs, dict):
            # Prefer 'action_value' or 'logits' if they exist
            if 'action_value' in outputs:
                return outputs['action_value'].astype(jnp.float16)
            elif 'logits' in outputs:
                return outputs['logits'].astype(jnp.float16)
            else:
                # Fallback: return the first value found or the dict structure (flattened later)
                return jax.tree_util.tree_map(lambda x: x.astype(jnp.float16), outputs)
        
        return outputs.astype(jnp.float16)

    # --- 6. LOAD DATA ---
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # mmap_mode='r' allows reading chunks from disk without loading the full file to RAM
    all_tokens_mmap = np.load(DATA_PATH, mmap_mode='r')
    total_samples = len(all_tokens_mmap)
    print(f"Total samples found: {total_samples}")

    # --- 7. BATCH INFERENCE LOOP ---
    all_logits = []
    
    # Create a PRNG key for Haiku initialization
    rng_seq = jax.random.PRNGKey(42)
    
    num_batches = int(np.ceil(total_samples / BATCH_SIZE))
    
    print("Starting Distillation Inference...")
    for i in tqdm(range(num_batches), desc="Generating Logits"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, total_samples)
        
        # 1. Load Batch (Disk -> RAM)
        # Ensure int32 for JAX indices
        batch_tokens_np = all_tokens_mmap[start_idx:end_idx].astype(np.int32)
        
        # 2. Move to GPU
        batch_tokens_jax = jnp.array(batch_tokens_np)
        
        # 3. Split RNG for this batch
        rng_seq, batch_rng = jax.random.split(rng_seq)
        
        # 4. Run JIT Inference
        batch_logits_jax = inference_step(params, batch_rng, batch_tokens_jax, is_training=False)
        
        # 5. Move back to CPU
        batch_logits_np = np.array(batch_logits_jax)
        all_logits.append(batch_logits_np)

    # --- 8. SAVE RESULTS ---
    print("Concatenating logits...")
    final_logits = np.concatenate(all_logits, axis=0)
    
    print(f"Saving dataset to {SAVE_PATH}...")
    np.savez_compressed(
        SAVE_PATH,
        tokens=all_tokens_mmap, # Save reference to inputs
        logits=final_logits     # Save the Teacher's soft targets
    )
    print("Export Complete.")

if __name__ == "__main__":
    main()