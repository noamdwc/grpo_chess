import os
import sys
import chess
import jax
import numpy as np
import jax.numpy as jnp

from typing import Dict
from huggingface_hub import snapshot_download

def download_model(model_path: str = './searchless_chess_model') -> str:
    ''''Download the model from https://huggingface.co/dbest-isi/searchless-chess-9M-selfplay'''
    # The defult path where the model will be stored
    if os.path.exists(model_path):
        print(f"Model already exists at: {model_path}")
        return model_path
    model_path = snapshot_download(
        repo_id="dbest-isi/searchless-chess-9M-selfplay",
        local_dir=model_path)

    print(f"Model downloaded to: {model_path}")
    return model_path




def _create_base_predict_func(model, utils, tokenizer):
    """
    Create a predict function that returns Q-values and action probabilities over the 1968 action space.
    - Batches all legal next positions into a single model call.
    - Corrects for perspective flip by negating next-position values.
    - Robust to short token sequences.
    """
    self = model
    NUM_ACTIONS = utils.NUM_ACTIONS
    ACTION_TO_MOVE = utils.ACTION_TO_MOVE
    MOVE_TO_ACTION = utils.MOVE_TO_ACTION
    bucket_values = jnp.asarray(self.return_buckets_values)  # [128], jnp for math
    if not hasattr(model, "_predict_one"):
        model._predict_one = jax.jit(
            lambda params, targets:
                model.predictor.predict(params=params, targets=targets, rng=None)
        )
    else:
        print("Using cached _predict_one jitted function.")
    def _value_from_bucket_log_probs(bucket_log_probs_2d):
        """
        bucket_log_probs_2d: jnp[seq_len, num_buckets] (log-probs)
        We want the penultimate position’s bucket distribution if present; otherwise last.
        """
        seq_len = bucket_log_probs_2d.shape[0]
        idx = seq_len - 2 if seq_len >= 2 else seq_len - 1
        # use softmax for extra safety even if they are log-probs
        probs = jax.nn.softmax(bucket_log_probs_2d[idx])  # [num_buckets]
        return jnp.vdot(probs, bucket_values)  # scalar

    def _base_predict(fen: str) -> Dict:
        if self.params is None:
            raise ValueError("Model parameters not loaded. Call load_params() first.")

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        # Handle terminal positions cleanly
        if not legal_moves:
            # If no legal moves, define a state value directly from current FEN
            tokens = tokenizer.tokenize(fen)[None, :]
            bucket_log_probs = self.predictor.predict(params=self.params, targets=tokens, rng=None)[0]
            q_state = float(_value_from_bucket_log_probs(bucket_log_probs))
            # No actions available
            action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            return {
                "q_value": q_state,
                "action_probs": action_probs,
                "best_action": -1,
                "best_move": None,
            }

        # Map legal moves to action indices and build next-FEN batch
        legal_action_indices = []
        next_fens = []
        for mv in legal_moves:
            uci = mv.uci()
            a_idx = MOVE_TO_ACTION.get(uci)
            if a_idx is None:
                continue
            legal_action_indices.append(a_idx)
            board.push(mv)
            next_fens.append(board.fen())
            board.pop()

        if not legal_action_indices:  # mapping incomplete
            # fallback: treat as no-legal for safety
            action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            return {
                "q_value": float("nan"),
                "action_probs": action_probs,
                "best_action": -1,
                "best_move": None,
            }

        # Tokenize batch of next positions: shape [B, T]
        token_batch = [tokenizer.tokenize(f) for f in next_fens]
        # Pad to common length if tokenizer returns ragged sequences
        max_len = max(len(t) for t in token_batch)
        token_batch = np.stack([np.pad(t, (0, max_len - len(t))) for t in token_batch], axis=0)
        token_batch = jnp.asarray(token_batch)  # instead of np.stack(...), use jnp


        # One batched forward pass
        bucket_log_probs_batch = model._predict_one(model.params, token_batch) # [B, seq_len, num_buckets]

        # Compute q-values for each next position and flip perspective
        q_next = []
        for b in range(bucket_log_probs_batch.shape[0]):
            q_b = _value_from_bucket_log_probs(jnp.asarray(bucket_log_probs_batch[b]))
            q_next.append(float(q_b))  # negate: opponent-to-move -> current player perspective
        q_next = np.array(q_next, dtype=np.float32)

        # Fill full action vector (masked elsewhere as -inf)
        action_values = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
        action_values[np.array(legal_action_indices, dtype=np.int32)] = q_next

        # Softmax over legal actions (better numerics)
        if temperature <= 1e-8:
            # Pure argmax
            best_idx_in_legal = int(np.argmax(q_next))
            best_action = int(legal_action_indices[best_idx_in_legal])
            action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_probs[best_action] = 1.0
            q_value = float(action_values[best_action])
        else:
            logits = q_next / temperature
            logits -= logits.max()  # stability
            exp_ = np.exp(logits, dtype=np.float64)
            probs_legal = (exp_ / exp_.sum()).astype(np.float32)
            action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_probs[np.array(legal_action_indices, dtype=np.int32)] = probs_legal
            best_idx_in_legal = int(np.argmax(probs_legal))
            best_action = int(legal_action_indices[best_idx_in_legal])
            q_value = float(action_values[best_action])

        best_move = ACTION_TO_MOVE.get(best_action, "unknown")
        return {
            "q_value": q_value,
            "action_probs": action_probs,
            "best_action": best_action,
            "best_move": best_move,
        }

    return _base_predict



def create_predict_func(model, utils, tokenizer):
    return _create_base_predict_func(model, utils, tokenizer)


def load_model(model_path: str, add_to_path: bool = True):
    """Load the model and attach a batched, perspective-correct predict()."""
    if add_to_path:
        sys.path.insert(0, f"{model_path}/searchless_chess_code")    
    import utils
    import tokenizer
    from hf_model import SearchlessChessModel 

    model = SearchlessChessModel.from_pretrained(model_path)
    model.predict = create_predict_func(model, utils, tokenizer)
    return model