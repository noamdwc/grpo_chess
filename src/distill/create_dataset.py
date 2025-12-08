#!/usr/bin/env python
"""
Create a distillation dataset from a Lichess HF dataset using DeepMind's
Searchless Chess action_value model as a teacher.

For each FEN position, we store the top-K moves and their probabilities.

Usage:
    python create_searchless_distill_dataset.py \
        --output-dir /path/to/output \
        --k 8
"""

import argparse
import json
import os
from typing import Dict, List
from grpo_chess.src.distill.position_sampler import iter_fens_from_lichess
from grpo_chess.src.distill.teacher import SearchlessActionValueTeacher, TeacherEngine
from grpo_chess.src.distill.constants import (
    MAX_GAMES,
    MAX_POSITIONS,
    RANDOM_SUBSAMPLE_FRACTION,
    SHARD_SIZE,
)


# ----------------------------------------------------------------------
# Writing JSONL shards
# ----------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_jsonl_shard(
    output_dir: str,
    shard_idx: int,
    records: List[Dict],
) -> None:
    if not records:
        return
    ensure_dir(output_dir)
    filename = os.path.join(output_dir, f"distill_shard_{shard_idx:05d}.jsonl")
    with open(filename, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote {len(records)} examples to {filename}")


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def create_distillation_dataset(
    output_dir: str,
    k: int,
    teacher: TeacherEngine,
) -> None:
    """
    Main loop:
      - iterate FENs
      - query teacher
      - save to JSONL shards
    """
    shard_idx = 0
    buffer: List[Dict] = []

    for i, sample in enumerate(
        iter_fens_from_lichess(
            max_games=MAX_GAMES,
            max_positions=MAX_POSITIONS,
            subsample_fraction=RANDOM_SUBSAMPLE_FRACTION,
        )
    ):
        fen = sample.fen

        try:
            moves, probs = teacher.topk(fen, k)
        except Exception as e:
            # If the teacher fails on some weird FEN, skip and continue.
            print(f"[WARN] Teacher failed on FEN #{i}: {fen} ({e})")
            continue

        if not moves:
            continue

        record = {
            "fen": fen,
            "moves": moves,
            "probs": probs,
        }
        buffer.append(record)

        if len(buffer) >= SHARD_SIZE:
            shard_idx += 1
            write_jsonl_shard(output_dir, shard_idx, buffer)
            buffer = []

        if (i + 1) % 10_000 == 0:
            print(f"[INFO] Processed {i+1} positions so far...")

    # Flush last shard
    if buffer:
        shard_idx += 1
        write_jsonl_shard(output_dir, shard_idx, buffer)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a distillation dataset for DeepMind Searchless Chess action_value model."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write JSONL shards.",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Top-K moves to store per FEN.",
    )
    # Optional override of defaults if you want from CLI:
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Override MAX_GAMES (number of games to process). None = use config constant.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Override MAX_POSITIONS (number of positions to process).",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=None,
        help="Override RANDOM_SUBSAMPLE_FRACTION (0..1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MAX_GAMES, MAX_POSITIONS, RANDOM_SUBSAMPLE_FRACTION

    if args.max_games is not None:
        MAX_GAMES = args.max_games
    if args.max_positions is not None:
        MAX_POSITIONS = args.max_positions
    if args.subsample_fraction is not None:
        RANDOM_SUBSAMPLE_FRACTION = args.subsample_fraction

    # TODO: once you've implemented SearchlessActionValueTeacher, construct it here.
    # Example:
    #
    # teacher = SearchlessActionValueTeacher(agent="270M", device="gpu")
    #
    # For now, this will raise NotImplementedError to remind you to plug it in.
    teacher = SearchlessActionValueTeacher()

    create_distillation_dataset(
        output_dir=args.output_dir,
        k=args.k,
        teacher=teacher,
    )


if __name__ == "__main__":
    main()
