"""Generate distillation dataset from DeepMind's searchless chess teacher model.

Requires JAX environment. Run as:
    python -m src.distill.generate_dataset --config distill.yaml
"""

import argparse
import os
import sys
import random
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.searchless_chess_imports import MOVE_TO_ACTION, tokenize
from src.pretrain.pretrain_dataset import get_positions_from_game
from src.configs.config_loader import load_yaml_file, dict_to_dataclass

# Path to searchless_chess submodule
_SC_ROOT = Path(__file__).resolve().parent.parent.parent / "searchless_chess"
_SC_SRC = _SC_ROOT / "src"

# Action space size
_ACTION_SPACE_SIZE = max(MOVE_TO_ACTION.values()) + 1


@dataclass
class GenerateConfig:
    teacher_model: str = "270M"
    checkpoint_dir: str = "searchless_chess/checkpoints"
    checkpoint_step: int = 6_400_000
    teacher_batch_size: int = 64
    top_k: int = 8
    teacher_temperature: float = 1.0
    output_dir: str = "data/distill"
    shard_size: int = 50_000
    min_elo: int = 1800
    max_samples: int = 2_000_000
    skip_first_n_moves: int = 5
    skip_last_n_moves: int = 5
    sample_positions_per_game: int = 3


def _patch_searchless_chess_compat():
    """Patch missing dependencies so searchless_chess modules can load.

    searchless_chess was built against a specific JAX/Beam environment.
    We only use it for inference, so we stub out the parts we don't need.
    See scripts/DISTILL_DEPS.md for full details.
    """
    # 1) apache_beam stub — constants.py imports coders at module level,
    #    but they're only used for data pipelines, never inference.
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

    # 2) jax.sharding.PositionalSharding — training_utils.py uses it as a type
    #    annotation on replicate(), which we never call. It was removed/moved
    #    in newer JAX versions.
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


def build_teacher_engine(
    model_name: str,
    checkpoint_dir: str,
    checkpoint_step: int,
    batch_size: int,
):
    """Build the DeepMind action-value engine and return (engine, bucket_values).

    Replicates _build_neural_engine from searchless_chess constants.py
    with configurable checkpoint_dir.
    """
    from jax import random as jrandom

    # Load searchless_chess modules
    sc_transformer = _load_sc_module("transformer")
    sc_training_utils = _load_sc_module("training_utils")
    sc_utils = _load_sc_module("utils")
    sc_tokenizer = _load_sc_module("tokenizer")
    sc_neural_engines = _load_sc_engine_module("neural_engines")

    # Model configs matching constants.py
    model_configs = {
        "9M": {"num_layers": 8, "embedding_dim": 256, "num_heads": 8},
        "136M": {"num_layers": 8, "embedding_dim": 1024, "num_heads": 8},
        "270M": {"num_layers": 16, "embedding_dim": 1024, "num_heads": 8},
        "local": {"num_layers": 4, "embedding_dim": 64, "num_heads": 4},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")

    cfg = model_configs[model_name]
    num_return_buckets = 128

    predictor_config = sc_transformer.TransformerConfig(
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

    predictor = sc_transformer.build_transformer_predictor(config=predictor_config)

    # Resolve checkpoint directory (orbax requires absolute paths)
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

    _, return_buckets_values = sc_utils.get_uniform_buckets_edges_values(num_return_buckets)

    engine = sc_neural_engines.ActionValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=sc_neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=params,
            batch_size=batch_size,
        ),
    )

    return engine, return_buckets_values


def process_position(
    engine,
    bucket_values: np.ndarray,
    fen: str,
    top_k: int,
    temperature: float,
) -> Optional[dict]:
    """Run teacher inference on a single position.

    Returns dict with board_tokens, legal_mask, teacher_action_indices, teacher_probs
    or None if position is invalid.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    legal_moves = list(board.legal_moves)
    if len(legal_moves) < 2:
        return None

    # Teacher inference
    try:
        result = engine.analyse(board)
    except Exception:
        return None

    log_probs = result["log_probs"]  # [num_legal, 128]
    return_buckets_probs = np.exp(log_probs)
    win_probs = np.inner(return_buckets_probs, bucket_values)  # [num_legal]

    # Get sorted legal moves (same order as engine)
    sorted_legal_moves = sorted(board.legal_moves, key=lambda x: MOVE_TO_ACTION[x.uci()])

    # Top-k by win probability
    k = min(top_k, len(win_probs))
    top_indices = np.argsort(win_probs)[-k:][::-1]

    # Soft teacher probs
    top_win_probs = win_probs[top_indices]
    if temperature != 1.0:
        top_win_probs = top_win_probs / temperature
    # Stable softmax
    top_win_probs = top_win_probs - top_win_probs.max()
    exp_probs = np.exp(top_win_probs)
    soft_probs = exp_probs / exp_probs.sum()

    # Convert to action indices in our action space
    teacher_action_indices = []
    for idx in top_indices:
        move = sorted_legal_moves[idx]
        action_idx = MOVE_TO_ACTION.get(move.uci())
        if action_idx is None:
            return None
        teacher_action_indices.append(action_idx)

    # Tokenize board
    try:
        board_tokens = list(tokenize(fen))
    except Exception:
        return None

    # Legal mask
    legal_mask = [False] * _ACTION_SPACE_SIZE
    for move in board.legal_moves:
        move_idx = MOVE_TO_ACTION.get(move.uci())
        if move_idx is not None:
            legal_mask[move_idx] = True

    return {
        "board_tokens": torch.tensor(board_tokens, dtype=torch.long),
        "legal_mask": torch.tensor(legal_mask, dtype=torch.bool),
        "teacher_action_indices": torch.tensor(teacher_action_indices, dtype=torch.long),
        "teacher_probs": torch.tensor(soft_probs, dtype=torch.float32),
    }


def save_shard(samples: list[dict], shard_path: str):
    """Save a list of samples as a .pt shard file."""
    shard = {
        "board_tokens": torch.stack([s["board_tokens"] for s in samples]),
        "legal_mask": torch.stack([s["legal_mask"] for s in samples]),
        "teacher_action_indices": [s["teacher_action_indices"] for s in samples],
        "teacher_probs": [s["teacher_probs"] for s in samples],
    }
    torch.save(shard, shard_path)


def load_positions(config: GenerateConfig) -> list[str]:
    """Load chess positions from HuggingFace dataset."""
    print("Downloading angeluriot/chess_games...")
    dataset = load_dataset("angeluriot/chess_games", split="train")
    print(f"Loaded {len(dataset):,} games")

    # Filter by ELO
    min_elo = config.min_elo

    def batch_filter(batch):
        keep = []
        for i in range(len(batch["white_elo"])):
            white_elo = batch["white_elo"][i]
            black_elo = batch["black_elo"][i]
            if white_elo is None or black_elo is None:
                keep.append(False)
                continue
            if white_elo < min_elo or black_elo < min_elo:
                keep.append(False)
                continue
            if len(batch["moves_uci"][i]) < 10:
                keep.append(False)
                continue
            keep.append(True)
        return keep

    dataset = dataset.filter(batch_filter, batched=True, batch_size=10000, desc="Filtering")
    print(f"After filtering: {len(dataset):,} games")

    # Extract positions
    positions = []
    max_games = config.max_samples // config.sample_positions_per_game + 1000
    game_indices = list(range(min(len(dataset), max_games)))
    random.shuffle(game_indices)

    for idx in tqdm(game_indices, desc="Extracting positions"):
        game = dataset[idx]
        moves = game["moves_uci"]
        if not moves:
            continue
        game_positions = get_positions_from_game(
            moves,
            skip_first_n=config.skip_first_n_moves,
            skip_last_n=config.skip_last_n_moves,
            sample_n=config.sample_positions_per_game,
        )
        for fen, _, _ in game_positions:
            positions.append(fen)
            if len(positions) >= config.max_samples:
                break
        if len(positions) >= config.max_samples:
            break

    print(f"Extracted {len(positions):,} positions")
    return positions


def generate(config: GenerateConfig):
    """Main generation loop: load teacher, run inference, save shards."""
    print("=" * 60)
    print("Distillation Dataset Generation")
    print("=" * 60)
    print(f"  Teacher model:    {config.teacher_model}")
    print(f"  Checkpoint dir:   {config.checkpoint_dir}")
    print(f"  Checkpoint step:  {config.checkpoint_step:,}")
    print(f"  Top-k:            {config.top_k}")
    print(f"  Temperature:      {config.teacher_temperature}")
    print(f"  Max samples:      {config.max_samples:,}")
    print(f"  Shard size:       {config.shard_size:,}")
    print(f"  Min ELO:          {config.min_elo}")
    print(f"  Output dir:       {config.output_dir}")
    print(f"  Batch size:       {config.teacher_batch_size}")
    print("=" * 60)

    os.makedirs(config.output_dir, exist_ok=True)

    # Count existing shards to support resuming
    existing_shards = sorted(Path(config.output_dir).glob("shard_*.pt"))
    existing_samples = 0
    if existing_shards:
        for sp in existing_shards:
            shard_data = torch.load(sp, weights_only=False)
            existing_samples += len(shard_data["board_tokens"])
        print(f"Found {len(existing_shards)} existing shards with {existing_samples:,} samples")
    shard_idx = len(existing_shards)

    if existing_samples >= config.max_samples:
        print("Already have enough samples, skipping generation.")
        return

    # Build teacher first (fast-fail if JAX/Haiku not available)
    print("Building teacher engine...")
    engine, bucket_values = build_teacher_engine(
        model_name=config.teacher_model,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_step=config.checkpoint_step,
        batch_size=config.teacher_batch_size,
    )
    print("Teacher engine ready.")

    # Load positions (slow — downloads 7.3GB dataset)
    positions = load_positions(config)

    # Skip already-processed positions (approximate)
    if existing_samples > 0:
        positions = positions[existing_samples:]
        print(f"Resuming from position {existing_samples:,}, {len(positions):,} remaining")

    # Process positions
    current_shard = []
    total_processed = existing_samples
    failed = 0

    for fen in tqdm(positions, desc="Processing positions"):
        if total_processed >= config.max_samples:
            break

        sample = process_position(
            engine, bucket_values, fen, config.top_k, config.teacher_temperature
        )
        if sample is None:
            failed += 1
            continue

        current_shard.append(sample)
        total_processed += 1

        # Save shard when full
        if len(current_shard) >= config.shard_size:
            shard_path = os.path.join(config.output_dir, f"shard_{shard_idx:04d}.pt")
            save_shard(current_shard, shard_path)
            print(f"Saved {shard_path} ({len(current_shard):,} samples, total: {total_processed:,})")
            current_shard = []
            shard_idx += 1

    # Save remaining samples
    if current_shard:
        shard_path = os.path.join(config.output_dir, f"shard_{shard_idx:04d}.pt")
        save_shard(current_shard, shard_path)
        print(f"Saved {shard_path} ({len(current_shard):,} samples)")

    print(f"\nDone! Total: {total_processed:,} samples, Failed: {failed:,}")


def main():
    parser = argparse.ArgumentParser(description="Generate distillation dataset")
    parser.add_argument("--config", type=str, default="distill.yaml", help="Path to config file")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    args = parser.parse_args()

    data = load_yaml_file(args.config)
    config = dict_to_dataclass(GenerateConfig, data.get("generate", {}))

    if args.max_samples:
        config.max_samples = args.max_samples

    generate(config)


if __name__ == "__main__":
    main()
