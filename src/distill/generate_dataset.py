"""Generate distillation dataset from DeepMind's searchless chess teacher model.

Requires JAX environment. Run as:
    python -m src.distill.generate_dataset --config distill.yaml
"""

import argparse
import json
import os
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from src.pretrain.pretrain_dataset import get_positions_from_game
from src.configs.config_loader import load_yaml_file, dict_to_dataclass
from src.distill.teacher import (
    build_teacher_engine,
    FENPrepDataset,
    collate_positions,
    postprocess_teacher_batch,
    ACTION_SPACE_SIZE,
)


@dataclass
class GenerateConfig:
    teacher_model: str = "270M"
    checkpoint_dir: str = "searchless_chess/checkpoints"
    checkpoint_step: int = 6_400_000
    teacher_batch_size: int = 64
    process_batch_size: int = 256
    num_workers: int = 0
    top_k: int = 8
    teacher_temperature: float = 1.0
    teacher_target_score_norm: str = "plain"
    hf_cache_dir: Optional[str] = None  # e.g. "/content/drive/MyDrive/hf_cache"
    output_dir: str = "data/distill"
    shard_size: int = 50_000
    min_elo: int = 1800
    max_samples: int = 2_000_000
    skip_first_n_moves: int = 5
    skip_last_n_moves: int = 5
    sample_positions_per_game: int = 3


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def save_shard(samples: list[dict], shard_path: str):
    """Save a list of samples as a .pt shard file."""
    shard = {
        "board_tokens": torch.stack([s["board_tokens"] for s in samples]),
        "legal_mask": torch.stack([s["legal_mask"] for s in samples]),
        "teacher_action_indices": [s["teacher_action_indices"] for s in samples],
        "teacher_probs": [s["teacher_probs"] for s in samples],
    }
    torch.save(shard, shard_path)


def count_existing_shards(output_dir: str) -> tuple[int, int]:
    """Count existing shards for resume support.

    Returns (shard_idx, existing_samples).
    """
    existing_shards = sorted(Path(output_dir).glob("shard_*.pt"))
    existing_samples = 0
    for sp in existing_shards:
        shard_data = torch.load(sp, weights_only=False)
        existing_samples += len(shard_data["board_tokens"])
    if existing_shards:
        print(f"Found {len(existing_shards)} existing shards with {existing_samples:,} samples")
    return len(existing_shards), existing_samples


# ---------------------------------------------------------------------------
# Position loading from HuggingFace
# ---------------------------------------------------------------------------

def _positions_cache_path(config: GenerateConfig) -> str:
    """Build a cache file path that encodes the extraction parameters."""
    return os.path.join(
        config.output_dir,
        f"positions_cache_elo{config.min_elo}_n{config.max_samples}"
        f"_skip{config.skip_first_n_moves}-{config.skip_last_n_moves}"
        f"_samp{config.sample_positions_per_game}.parquet",
    )


def load_positions(config: GenerateConfig) -> list[str]:
    """Load chess positions, using a parquet cache if available."""
    cache_path = _positions_cache_path(config)

    if os.path.exists(cache_path):
        print(f"Loading cached positions from {cache_path}...")
        positions = pd.read_parquet(cache_path)["fen"].tolist()
        print(f"Loaded {len(positions):,} cached positions")
        return positions

    print("Downloading angeluriot/chess_games...")
    dataset = load_dataset("angeluriot/chess_games", split="train", cache_dir=config.hf_cache_dir)
    print(f"Loaded {len(dataset):,} games")

    dataset = _filter_by_elo(dataset, config.min_elo)
    positions = _extract_positions(dataset, config)

    # Save cache for next run
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    pd.DataFrame({"fen": positions}).to_parquet(cache_path, index=False)
    print(f"Saved positions cache to {cache_path}")

    return positions


def _filter_by_elo(dataset, min_elo: int):
    """Filter games by minimum ELO and move count."""
    def batch_filter(batch):
        keep = []
        for i in range(len(batch["white_elo"])):
            white_elo = batch["white_elo"][i]
            black_elo = batch["black_elo"][i]
            if white_elo is None or black_elo is None:
                keep.append(False)
            elif white_elo < min_elo or black_elo < min_elo:
                keep.append(False)
            elif len(batch["moves_uci"][i]) < 10:
                keep.append(False)
            else:
                keep.append(True)
        return keep

    dataset = dataset.filter(batch_filter, batched=True, batch_size=10000, desc="Filtering")
    print(f"After filtering: {len(dataset):,} games")
    return dataset


def _extract_positions(dataset, config: GenerateConfig) -> list[str]:
    """Sample positions from filtered games."""
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


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _print_config(config: GenerateConfig):
    """Print generation configuration."""
    print("=" * 60)
    print("Distillation Dataset Generation")
    print("=" * 60)
    print(f"  Teacher model:    {config.teacher_model}")
    print(f"  Checkpoint dir:   {config.checkpoint_dir}")
    print(f"  Checkpoint step:  {config.checkpoint_step:,}")
    print(f"  Top-k:            {config.top_k}")
    print(f"  Temperature:      {config.teacher_temperature}")
    print(f"  Score norm mode:  {config.teacher_target_score_norm}")
    print(f"  Max samples:      {config.max_samples:,}")
    print(f"  Shard size:       {config.shard_size:,}")
    print(f"  Min ELO:          {config.min_elo}")
    print(f"  Output dir:       {config.output_dir}")
    print(f"  Teacher batch:    {config.teacher_batch_size}")
    print(f"  Process batch:    {config.process_batch_size}")
    print(f"  Num workers:      {config.num_workers}")
    print("=" * 60)


def generate(config: GenerateConfig):
    """Main generation loop: load teacher, run inference, save shards."""
    _print_config(config)
    os.makedirs(config.output_dir, exist_ok=True)

    shard_idx, existing_samples = count_existing_shards(config.output_dir)
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

    # Process positions via DataLoader (workers prepare sequences in parallel)
    loader = DataLoader(
        FENPrepDataset(positions),
        batch_size=config.process_batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_positions,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    current_shard = []
    total_processed = existing_samples
    failed = 0
    diag_batches = 0
    diag_positions = 0.0
    diag_weighted_sums: dict[str, float] = {}
    diag_running_sums: dict[str, float] = {}

    t_prev = time.time()

    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches"), start=1):
        t_data = time.time()
        if total_processed >= config.max_samples:
            break

        failed += batch["num_failed"]
        if not batch["valid"]:
            t_prev = time.time()
            continue

        # Batched JAX inference + vectorized post-processing
        seqs = batch["sequences"].numpy()
        t_to_numpy = time.time()
        all_log_probs = engine.predict_fn(seqs)[:, -1]
        t_infer = time.time()
        samples, batch_diag = postprocess_teacher_batch(
            all_log_probs, bucket_values, batch,
            config.top_k, config.teacher_temperature,
            score_norm_mode=config.teacher_target_score_norm,
            return_diagnostics=True,
        )
        t_post = time.time()

        diag_batches += 1
        batch_positions = float(batch_diag.get("n_positions", 0.0))
        diag_positions += batch_positions
        for key, value in batch_diag.items():
            if key in {"n_positions"}:
                continue
            diag_running_sums[key] = diag_running_sums.get(key, 0.0) + float(value)
            if key.endswith("_mean") and batch_positions > 0.0:
                diag_weighted_sums[key] = diag_weighted_sums.get(key, 0.0) + float(value) * batch_positions

        n_seqs = seqs.shape[0]
        batch_compute = max(t_post - t_data, 1e-6)
        infer_time = max(t_infer - t_to_numpy, 1e-6)
        if batch_idx % 5 == 0:
            pos_per_s = len(samples) / batch_compute
            seq_per_s = n_seqs / infer_time
            print(f"  [timing] data_wait={t_data - t_prev:.1f}s  "
                  f"to_numpy={t_to_numpy - t_data:.2f}s  "
                  f"inference={t_infer - t_to_numpy:.1f}s ({n_seqs} seqs, {seq_per_s:.1f} seq/s)  "
                  f"postprocess={t_post - t_infer:.2f}s  "
                  f"batch={batch_compute:.1f}s ({pos_per_s:.2f} pos/s)")
        if batch_idx % 20 == 0:
            print(
                "  [teacher_diagnostics] "
                f"top1_prob_mean={batch_diag.get('teacher_top1_prob_mean', float('nan')):.4f}, "
                f"top1_margin_prob_mean={batch_diag.get('teacher_top1_minus_top2_prob_mean', float('nan')):.4f}, "
                f"entropy_mean={batch_diag.get('teacher_entropy_mean', float('nan')):.4f}, "
                f"effective_k_mean={batch_diag.get('teacher_effective_k_mean', float('nan')):.4f}, "
                f"top1_margin_raw_mean={batch_diag.get('teacher_top1_minus_top2_raw_mean', float('nan')):.4f}, "
                f"top1_margin_scaled_mean={batch_diag.get('teacher_top1_minus_top2_scaled_mean', float('nan')):.4f}, "
                f"topk_raw_std_mean={batch_diag.get('teacher_topk_raw_score_std_mean', float('nan')):.4f}"
            )

        current_shard.extend(samples)
        total_processed += len(samples)
        t_prev = time.time()

        # Flush full shards
        while len(current_shard) >= config.shard_size:
            to_save = current_shard[: config.shard_size]
            current_shard = current_shard[config.shard_size :]
            shard_path = os.path.join(config.output_dir, f"shard_{shard_idx:04d}.pt")
            save_shard(to_save, shard_path)
            print(f"Saved {shard_path} ({len(to_save):,} samples, total: {total_processed:,})")
            shard_idx += 1

    # Trim to max_samples and save remaining
    excess = total_processed - config.max_samples
    if excess > 0:
        current_shard = current_shard[: len(current_shard) - excess]
        total_processed = config.max_samples

    if current_shard:
        shard_path = os.path.join(config.output_dir, f"shard_{shard_idx:04d}.pt")
        save_shard(current_shard, shard_path)
        print(f"Saved {shard_path} ({len(current_shard):,} samples)")

    if diag_batches > 0:
        diagnostics_payload = {
            "n_batches": diag_batches,
            "n_positions": int(diag_positions),
            "score_norm_mode": config.teacher_target_score_norm,
            "batch_avg": {
                key: value / diag_batches for key, value in diag_running_sums.items()
            },
            "weighted_by_positions_mean_metrics": {
                key: (value / diag_positions) if diag_positions > 0 else float("nan")
                for key, value in diag_weighted_sums.items()
            },
        }
        diagnostics_path = os.path.join(config.output_dir, "teacher_diagnostics.json")
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics_payload, f, indent=2, sort_keys=True)
        print(f"Wrote teacher diagnostics: {diagnostics_path}")

    print(f"\nDone! Total: {total_processed:,} samples, Failed: {failed:,}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate distillation dataset")
    parser.add_argument("--config", type=str, default="distill.yaml", help="Path to config file")
    parser.add_argument("--max_samples", type=int, help="Override max_samples")
    parser.add_argument("--teacher_model", type=str, help="Override teacher model (e.g. 136M, 270M)")
    parser.add_argument("--batch_size", type=int, help="Override process_batch_size (positions per batch)")
    parser.add_argument("--teacher_batch_size", type=int, help="Override teacher_batch_size (sequences per GPU call)")
    parser.add_argument("--num_workers", type=int, help="Override num_workers for DataLoader")
    parser.add_argument("--output_dir", type=str, help="Override output directory for shards and cache")
    parser.add_argument("--hf_cache_dir", type=str, help="HuggingFace dataset cache directory")
    parser.add_argument(
        "--teacher_target_score_norm",
        type=str,
        choices=["plain", "zscore"],
        help="Score normalization mode for top-k EV scores before softmax",
    )
    args = parser.parse_args()

    data = load_yaml_file(args.config)
    config = dict_to_dataclass(GenerateConfig, data.get("generate", {}))

    if args.max_samples:
        config.max_samples = args.max_samples
    if args.teacher_model:
        config.teacher_model = args.teacher_model
    if args.batch_size:
        config.process_batch_size = args.batch_size
    if args.teacher_batch_size:
        config.teacher_batch_size = args.teacher_batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.hf_cache_dir:
        config.hf_cache_dir = args.hf_cache_dir
    if args.teacher_target_score_norm:
        config.teacher_target_score_norm = args.teacher_target_score_norm

    generate(config)


if __name__ == "__main__":
    main()
