import os
import sys
import chess
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


def create_predict_func(model, utils, tokenizer):
    '''Create a predict function to adjuts model to return Q-values for all possiable chess moves.'''
    self = model
    def predict(fen: str, temperature: float = 1.0) -> Dict:
            """
            Predict move from FEN position by evaluating all legal actions.
            Returns:
            - q_value: expected return of the chosen move
            - action_probs: np.ndarray, shape (1968,)
            - best_action: int
            - best_move: UCI string
            """
            if self.params is None:
                raise ValueError("Model parameters not loaded. Call load_params() first.")

            board = chess.Board(fen)
            action_values = np.full(utils.NUM_ACTIONS, -1e9, dtype=np.float32)  # mask illegal

            for move in board.legal_moves:
                uci = move.uci()  # e.g., "e7e5" or "e7e8q"
                action_idx = utils.MOVE_TO_ACTION.get(uci)
                if action_idx is None:
                    continue

                # Evaluate the *resulting* position after the move.
                board.push(move)
                next_fen = board.fen()
                board.pop()

                # Tokenize and run model once (no RNG)
                tokens = tokenizer.tokenize(next_fen)[None, :]  # [1, T]
                bucket_log_probs = self.predictor.predict(params=self.params, targets=tokens, rng=None)

                # Use the penultimate position’s bucket distribution as the state value
                # (matches the current wrapper’s convention)
                action_bucket_log_probs = bucket_log_probs[0, -2]  # [num_return_buckets=128]
                action_bucket_probs = jnp.exp(action_bucket_log_probs)
                q_val = float(jnp.dot(action_bucket_probs, self.return_buckets_values))
                action_values[action_idx] = q_val

            # Temperature + softmax over the 1968 actions
            logits = action_values / max(1e-8, temperature)
            # Keep masked (-1e9) moves out of the softmax safely
            logits = logits - logits.max()  # numerical stability
            exp = np.exp(logits, dtype=np.float64)
            exp[action_values < -1e8] = 0.0  # ensure masked are zeroed
            denom = exp.sum()
            action_probs = (exp / denom) if denom > 0 else exp  # handle stalemate/checkmate

            best_action = int(np.argmax(action_probs))
            best_move = utils.ACTION_TO_MOVE.get(best_action, "unknown")
            q_value = float(action_values[best_action])

            return {
                "q_value": q_value,
                "action_probs": action_probs.astype(np.float32),
                "best_action": best_action,
                "best_move": best_move,
            }
    return predict

    

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