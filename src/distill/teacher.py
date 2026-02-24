"""DeepMind searchless chess teacher model: loading, data prep, and inference helpers."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from src.searchless_chess_imports import MOVE_TO_ACTION, tokenize
from src.chess.chess_logic import ChessPlayer

# Path to searchless_chess submodule
_SC_ROOT = Path(__file__).resolve().parent.parent.parent / "searchless_chess"
_SC_SRC = _SC_ROOT / "src"

# Action space size
ACTION_SPACE_SIZE = max(MOVE_TO_ACTION.values()) + 1

# Model configs matching searchless_chess constants.py
MODEL_CONFIGS = {
    "9M": {"num_layers": 8, "embedding_dim": 256, "num_heads": 8},
    "136M": {"num_layers": 8, "embedding_dim": 1024, "num_heads": 8},
    "270M": {"num_layers": 16, "embedding_dim": 1024, "num_heads": 8},
    "local": {"num_layers": 4, "embedding_dim": 64, "num_heads": 4},
}


# ---------------------------------------------------------------------------
# Searchless chess compatibility layer
# ---------------------------------------------------------------------------

def _patch_searchless_chess_compat():
    """Patch missing dependencies so searchless_chess modules can load.

    searchless_chess was built against a specific JAX/Beam environment.
    We only use it for inference, so we stub out the parts we don't need.
    See scripts/DISTILL_DEPS.md for full details.
    """
    # apache_beam stub — constants.py imports coders at module level,
    # but they're only used for data pipelines, never inference.
    try:
        from apache_beam import coders  # noqa: F401
    except Exception:
        import types

        def _make_stub(name):
            mod = types.ModuleType(name)
            mod.__path__ = []
            return mod

        beam = _make_stub("apache_beam")
        coders_mod = _make_stub("apache_beam.coders")

        class _DummyCoder:
            def __init__(self, *args, **kwargs):
                pass

        for coder_name in (
            "StrUtf8Coder", "BigIntegerCoder", "FloatCoder", "TupleCoder",
        ):
            setattr(coders_mod, coder_name, _DummyCoder)

        beam.coders = coders_mod
        sys.modules.setdefault("apache_beam", beam)
        sys.modules.setdefault("apache_beam.coders", coders_mod)

    # jax.sharding.PositionalSharding — training_utils.py uses it as a type
    # annotation on replicate(), which we never call.
    import jax
    if not hasattr(jax.sharding, "PositionalSharding"):
        jax.sharding.PositionalSharding = type("PositionalSharding", (), {})


def _load_sc_module(name: str):
    """Load a module from searchless_chess/src/ via importlib."""
    _patch_searchless_chess_compat()
    spec = importlib.util.spec_from_file_location(name, _SC_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_sc_engine_module(name: str):
    """Load a module from searchless_chess/src/engines/ via importlib."""
    _patch_searchless_chess_compat()
    spec = importlib.util.spec_from_file_location(
        name, _SC_SRC / "engines" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Teacher engine construction
# ---------------------------------------------------------------------------

def build_teacher_engine(
    model_name: str,
    checkpoint_dir: str,
    checkpoint_step: int,
    batch_size: int,
    use_half: bool = True,
):
    """Build the DeepMind action-value engine and return (engine, bucket_values).

    Replicates _build_neural_engine from searchless_chess constants.py
    with configurable checkpoint_dir.

    Args:
        use_half: Cast params to bfloat16 for ~4x faster inference on GPUs
                  with Tensor Core support (L4, A100, etc.). Safe for
                  distillation labels where exact float32 precision isn't needed.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    import jax
    import jax.numpy as jnp
    from jax import random as jrandom

    devices = jax.devices()
    print(f"JAX backend: {jax.default_backend()}, devices: {devices}")

    sc_transformer = _load_sc_module("transformer")
    sc_training_utils = _load_sc_module("training_utils")
    sc_utils = _load_sc_module("utils")
    sc_tokenizer = _load_sc_module("tokenizer")
    sc_neural_engines = _load_sc_engine_module("neural_engines")

    cfg = MODEL_CONFIGS[model_name]
    num_return_buckets = 128

    predictor = sc_transformer.build_transformer_predictor(
        config=sc_transformer.TransformerConfig(
            vocab_size=sc_utils.NUM_ACTIONS,
            output_size=num_return_buckets,
            pos_encodings=sc_transformer.PositionalEncodings.LEARNED,
            max_sequence_length=sc_tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            embedding_dim=cfg["embedding_dim"],
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
    )

    ckpt_dir = os.path.abspath(os.path.join(checkpoint_dir, model_name))
    print(f"Loading checkpoint from {ckpt_dir} at step {checkpoint_step}...")

    params = sc_training_utils.load_parameters(
        checkpoint_dir=ckpt_dir,
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=checkpoint_step,
    )

    if use_half:
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        )
        print("Params cast to bfloat16 for faster inference")

    _, return_buckets_values = sc_utils.get_uniform_buckets_edges_values(num_return_buckets)  # [num_return_buckets]

    engine = sc_neural_engines.ActionValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=sc_neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=params,
            batch_size=batch_size,
        ),
    )

    return engine, return_buckets_values


# ---------------------------------------------------------------------------
# DataLoader components (CPU-bound work runs in workers)
# ---------------------------------------------------------------------------

class FENPrepDataset(Dataset):
    """Prepares teacher-model input sequences from FEN strings.

    CPU-bound work (board parsing, legal moves, tokenization) runs in
    DataLoader workers.  The main process then runs JAX inference on the
    combined sequences and does lightweight post-processing.
    """

    def __init__(self, fens: list[str]):
        self.fens = fens

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen = self.fens[idx]
        try:
            board = chess.Board(fen)
        except ValueError:
            return None

        if len(list(board.legal_moves)) < 2:
            return None

        sorted_legal = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION[x.uci()])

        # Build input sequences for the teacher model (one per legal move)
        sequences = _build_teacher_sequences(board, sorted_legal)

        # Pre-compute student-model outputs (board tokens, legal mask, action map)
        try:
            board_tokens = torch.tensor(list(tokenize(fen)), dtype=torch.long)
        except Exception:
            return None

        legal_mask = _build_legal_mask(board)

        action_indices = torch.tensor(
            [MOVE_TO_ACTION[m.uci()] for m in sorted_legal], dtype=torch.long
        )

        return {
            "sequences": torch.from_numpy(sequences),  # [num_legal, seq_len]
            "board_tokens": board_tokens,               # [SEQUENCE_LENGTH]
            "legal_mask": legal_mask,                    # [ACTION_SPACE_SIZE]
            "action_indices": action_indices,            # [num_legal]
        }


def _build_teacher_sequences(board: chess.Board, sorted_legal: list) -> np.ndarray:
    """Build tokenized input sequences for the teacher model.

    Returns: np.ndarray [num_legal, SEQUENCE_LENGTH + 2]
        +2 for the appended action token and dummy return bucket.
    """
    legal_actions = np.array(
        [MOVE_TO_ACTION[x.uci()] for x in sorted_legal], dtype=np.int32
    )
    legal_actions = np.expand_dims(legal_actions, axis=-1)  # [num_legal, 1]
    dummy_return_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)  # [num_legal, 1]
    tokenized_fen = tokenize(board.fen()).astype(np.int32)  # [SEQUENCE_LENGTH]
    sequences = np.stack([tokenized_fen] * len(legal_actions))  # [num_legal, SEQUENCE_LENGTH]
    return np.concatenate([sequences, legal_actions, dummy_return_buckets], axis=1)  # [num_legal, SEQUENCE_LENGTH + 2]


def _build_legal_mask(board: chess.Board) -> torch.Tensor:
    """Build a boolean mask over the action space for legal moves.

    Returns: torch.Tensor [ACTION_SPACE_SIZE] (bool)
    """
    legal_mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.bool)
    for move in board.legal_moves:
        move_idx = MOVE_TO_ACTION.get(move.uci())
        if move_idx is not None:
            legal_mask[move_idx] = True
    return legal_mask


def collate_positions(batch):
    """Collate variable-length position data, filtering invalid positions."""
    valid = [x for x in batch if x is not None]
    num_failed = len(batch) - len(valid)
    if not valid:
        return {"num_failed": num_failed, "valid": False}
    return {
        "valid": True,
        "num_failed": num_failed,
        "sequences": torch.cat([x["sequences"] for x in valid], dim=0),  # [total_legal, SEQUENCE_LENGTH + 2]
        "seq_counts": [x["sequences"].shape[0] for x in valid],          # list[int], len = num_valid
        "board_tokens": [x["board_tokens"] for x in valid],              # list of [SEQUENCE_LENGTH]
        "legal_masks": [x["legal_mask"] for x in valid],                 # list of [ACTION_SPACE_SIZE]
        "action_indices": [x["action_indices"] for x in valid],          # list of [num_legal_i]
    }


# ---------------------------------------------------------------------------
# Post-processing (runs in main process after JAX inference)
# ---------------------------------------------------------------------------

def postprocess_teacher_output(
    log_probs: np.ndarray,
    bucket_values: np.ndarray,
    action_indices: torch.Tensor,
    top_k: int,
    temperature: float,
    score_norm_mode: str = "plain",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw teacher log-probs to top-k action indices and soft probs.

    Args:
        log_probs: [num_legal, num_buckets] from predict_fn for one position.
        bucket_values: Return bucket center values.
        action_indices: [num_legal] action indices for sorted legal moves.
        top_k: Number of top moves to keep.
        temperature: Softmax temperature for teacher probabilities.
        score_norm_mode: Score normalization mode before softmax. One of:
            - "plain": no normalization (backwards-compatible behavior)
            - "zscore": per-position z-score normalization over top-k EVs

    Returns:
        (teacher_action_indices [top_k], teacher_probs [top_k])
    """
    return_buckets_probs = np.exp(log_probs)         # [num_legal, num_buckets]
    win_probs = np.inner(return_buckets_probs, bucket_values)  # [num_legal]

    k = min(top_k, len(win_probs))
    top_indices = np.argsort(win_probs)[-k:][::-1].copy()  # [k], copy for positive stride

    # Optional score normalization to sharpen weak/flat EV differences.
    top_win_probs = win_probs[top_indices]           # [k]
    if score_norm_mode == "zscore":
        top_win_probs = (top_win_probs - top_win_probs.mean()) / (top_win_probs.std() + 1e-6)
    elif score_norm_mode != "plain":
        raise ValueError(f"Unsupported score_norm_mode={score_norm_mode!r}; expected 'plain' or 'zscore'.")

    # Stable softmax over top-k scores
    if temperature != 1.0:
        top_win_probs = top_win_probs / temperature
    top_win_probs = top_win_probs - top_win_probs.max()
    exp_probs = np.exp(top_win_probs)
    soft_probs = exp_probs / exp_probs.sum()

    teacher_action_indices = action_indices[top_indices]
    teacher_probs = torch.tensor(soft_probs, dtype=torch.float32)
    return teacher_action_indices, teacher_probs


def postprocess_teacher_batch(
    all_log_probs: np.ndarray,
    bucket_values: np.ndarray,
    batch: dict,
    top_k: int,
    temperature: float,
    score_norm_mode: str = "plain",
    return_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict[str, float]]:
    """Vectorized post-processing for an entire batch of positions.

    Args:
        all_log_probs: [total_seqs, num_buckets] from predict_fn.
        bucket_values: [num_buckets] return bucket center values.
        batch: Collated batch dict from collate_positions.
        top_k: Number of top moves to keep per position.
        temperature: Softmax temperature for teacher probabilities.
        score_norm_mode: Score normalization mode before softmax. One of:
            - "plain": no normalization (backwards-compatible behavior)
            - "zscore": per-position z-score normalization over top-k EVs

    Returns:
        List of sample dicts (one per valid position in the batch), or
        (samples, diagnostics) if return_diagnostics=True.
    """
    seq_counts = batch["seq_counts"]
    N = len(seq_counts)
    max_legal = max(seq_counts)
    k = min(top_k, max_legal)

    # Vectorized: win probs for all sequences at once
    all_win_probs = np.exp(all_log_probs) @ bucket_values  # [total_seqs]

    # Pad into [N, max_legal] for vectorized argsort/softmax (-inf for padding)
    offsets = np.cumsum([0] + list(seq_counts))
    padded = np.full((N, max_legal), -np.inf)
    for i in range(N):
        padded[i, : seq_counts[i]] = all_win_probs[offsets[i] : offsets[i + 1]]

    # Vectorized top-k: argsort ascending, take last k (largest), reverse to descending
    sorted_idx = np.argsort(padded, axis=1)          # [N, max_legal]
    top_idx = sorted_idx[:, -k:][:, ::-1].copy()      # [N, k], copy for positive stride
    top_win = np.take_along_axis(padded, top_idx, axis=1)  # [N, k]
    raw_top_win = top_win.copy()

    valid_top_mask = np.isfinite(raw_top_win)
    safe_raw_top = np.where(valid_top_mask, raw_top_win, np.nan)
    raw_spread = np.nanstd(safe_raw_top, axis=1)
    raw_spread = np.nan_to_num(raw_spread, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional score normalization before softmax.
    softmax_inputs = top_win
    if score_norm_mode == "zscore":
        safe_scores = np.where(valid_top_mask, softmax_inputs, np.nan)
        means = np.nanmean(safe_scores, axis=1, keepdims=True)
        stds = np.nanstd(safe_scores, axis=1, keepdims=True)
        means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
        stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
        softmax_inputs = (softmax_inputs - means) / (stds + 1e-6)
        softmax_inputs = np.where(valid_top_mask, softmax_inputs, -np.inf)
    elif score_norm_mode != "plain":
        raise ValueError(f"Unsupported score_norm_mode={score_norm_mode!r}; expected 'plain' or 'zscore'.")

    # Vectorized stable softmax (padding entries are -inf → exp=0, safe)
    if temperature != 1.0:
        softmax_inputs = softmax_inputs / temperature
    softmax_inputs = softmax_inputs - softmax_inputs.max(axis=1, keepdims=True)
    exp_p = np.exp(softmax_inputs)                   # [N, k]
    soft_probs = exp_p / exp_p.sum(axis=1, keepdims=True)  # [N, k]

    if k >= 2:
        top1_minus_top2_raw = raw_top_win[:, 0] - raw_top_win[:, 1]
        top1_minus_top2_scaled = softmax_inputs[:, 0] - softmax_inputs[:, 1]
        top1_minus_top2_prob = soft_probs[:, 0] - soft_probs[:, 1]
    else:
        top1_minus_top2_raw = raw_top_win[:, 0]
        top1_minus_top2_scaled = softmax_inputs[:, 0]
        top1_minus_top2_prob = soft_probs[:, 0]

    eps = 1e-12
    entropy = -(soft_probs * np.log(np.clip(soft_probs, eps, 1.0))).sum(axis=1)
    effective_k = np.exp(entropy)
    top1_prob = soft_probs[:, 0]

    def _summary_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        values = values.astype(np.float64, copy=False)
        return {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_p50": float(np.percentile(values, 50)),
            f"{prefix}_p90": float(np.percentile(values, 90)),
        }

    diagnostics = {
        "n_positions": float(N),
        "top_k_effective": float(k),
        **_summary_stats(top1_minus_top2_raw, "teacher_top1_minus_top2_raw"),
        **_summary_stats(top1_minus_top2_scaled, "teacher_top1_minus_top2_scaled"),
        **_summary_stats(raw_spread, "teacher_topk_raw_score_std"),
        **_summary_stats(top1_prob, "teacher_top1_prob"),
        **_summary_stats(top1_minus_top2_prob, "teacher_top1_minus_top2_prob"),
        **_summary_stats(entropy, "teacher_entropy"),
        **_summary_stats(effective_k, "teacher_effective_k"),
    }

    # Build sample dicts (cheap indexing only)
    samples = []
    for i in range(N):
        ki = min(top_k, seq_counts[i])
        idx = top_idx[i, :ki]
        samples.append({
            "board_tokens": batch["board_tokens"][i],              # [SEQUENCE_LENGTH]
            "legal_mask": batch["legal_masks"][i],                 # [ACTION_SPACE_SIZE]
            "teacher_action_indices": batch["action_indices"][i][idx],  # [ki]
            "teacher_probs": torch.from_numpy(soft_probs[i, :ki].copy()).float(),  # [ki]
        })
    if return_diagnostics:
        return samples, diagnostics
    return samples


# ---------------------------------------------------------------------------
# Single-position convenience (for testing / one-off use via engine.analyse)
# ---------------------------------------------------------------------------

def process_position(
    engine,
    bucket_values: np.ndarray,
    fen: str,
    top_k: int,
    temperature: float,
    score_norm_mode: str = "plain",
) -> Optional[dict]:
    """Run teacher inference on a single position via engine.analyse().

    Returns dict with board_tokens, legal_mask, teacher_action_indices,
    teacher_probs — or None if position is invalid.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    if len(list(board.legal_moves)) < 2:
        return None

    try:
        result = engine.analyse(board)
    except Exception:
        return None

    sorted_legal = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION[x.uci()])
    action_indices = torch.tensor(
        [MOVE_TO_ACTION[m.uci()] for m in sorted_legal], dtype=torch.long
    )

    teacher_action_indices, teacher_probs = postprocess_teacher_output(
        result["log_probs"], bucket_values, action_indices, top_k, temperature, score_norm_mode,
    )

    try:
        board_tokens = torch.tensor(list(tokenize(fen)), dtype=torch.long)
    except Exception:
        return None

    return {
        "board_tokens": board_tokens,               # [SEQUENCE_LENGTH]
        "legal_mask": _build_legal_mask(board),      # [ACTION_SPACE_SIZE]
        "teacher_action_indices": teacher_action_indices,  # [top_k]
        "teacher_probs": teacher_probs,              # [top_k]
    }


# ---------------------------------------------------------------------------
# ChessPlayer wrapper for evaluation
# ---------------------------------------------------------------------------

class DeepMindPlayer(ChessPlayer):
    """Wraps the JAX ActionValueEngine as a ChessPlayer for evaluation."""

    def __init__(self, engine, bucket_values: np.ndarray):
        self._engine = engine
        self._bucket_values = bucket_values

    def act(self, board: chess.Board) -> chess.Move:
        sorted_legal = sorted(
            [m for m in board.legal_moves if m.uci() in MOVE_TO_ACTION],
            key=lambda m: MOVE_TO_ACTION[m.uci()],
        )
        if not sorted_legal:
            import random
            return random.choice(list(board.legal_moves))

        result = self._engine.analyse(board)       # {"log_probs": [num_legal, 128]}
        log_probs = result["log_probs"]            # np.ndarray [num_legal, 128]
        win_probs = np.exp(log_probs) @ self._bucket_values  # [num_legal]
        best_idx = int(np.argmax(win_probs))
        return sorted_legal[best_idx]
