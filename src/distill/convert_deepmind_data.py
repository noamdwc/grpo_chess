"""Convert DeepMind .bag shards to .pt shard format for distillation.

Downloads a subset of pre-computed action_value shards from
storage.googleapis.com/searchless_chess/data/ and converts them
to the .pt format consumed by DistillDataset.

Each .bag record is a (fen, move, win_prob) tuple. A single position
has many records (one per legal move). Records are grouped by FEN and
win_probs are converted to a soft probability distribution via softmax,
keeping only the top-k moves — matching the format produced by
generate_dataset.py.

Run as:
    python -m src.distill.convert_deepmind_data --num_shards 2
"""

import argparse
import os
import struct
import urllib.request
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np
import torch
from tqdm import tqdm

from src.searchless_chess_imports import tokenize, MOVE_TO_ACTION
from src.distill.generate_dataset import save_shard
from src.configs.config_loader import load_yaml_file, dict_to_dataclass


BAG_URL_TEMPLATE = (
    "https://storage.googleapis.com/searchless_chess/data/train/"
    "action_value-{shard_idx:05d}-of-02148_data.bag"
)
TOTAL_SHARDS = 2148


@dataclass
class ConvertConfig:
    num_shards: int = 50
    top_k: int = 8
    temperature: float = 1.0
    min_win_prob: float = 0.55
    output_dir: str = "data/distill"
    shard_size: int = 50_000
    max_samples: Optional[int] = None


# ---------------------------------------------------------------------------
# Minimal .bag reader (avoids zstandard/etils deps from bagz.py)
# ---------------------------------------------------------------------------

def read_bag_records(path: str):
    """Iterate over all records in an uncompressed .bag file."""
    file_size = os.path.getsize(path)
    if file_size < 8:
        return

    with open(path, "rb") as f:
        data = f.read()

    # Last 8 bytes: uint64 pointing to index start
    index_start = struct.unpack_from("<Q", data, file_size - 8)[0]

    # Index: array of int64 cumulative end-offsets
    index_data = data[index_start : file_size - 8]
    num_records = len(index_data) // 8

    prev_end = 0
    for i in range(num_records):
        end = struct.unpack_from("<q", index_data, i * 8)[0]
        yield data[prev_end:end]
        prev_end = end


def count_bag_records(path: str) -> int:
    """Count records in a .bag file without reading them all."""
    file_size = os.path.getsize(path)
    if file_size < 8:
        return 0
    with open(path, "rb") as f:
        f.seek(-8, 2)
        index_start = struct.unpack("<Q", f.read(8))[0]
    index_size = file_size - 8 - index_start
    return index_size // 8


# ---------------------------------------------------------------------------
# Apache Beam coder decoding (TupleCoder of StrUtf8, StrUtf8, Float)
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a protobuf-style varint. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, offset


def decode_action_value(data: bytes) -> tuple[str, str, float]:
    """Decode (fen, move, win_prob) from Apache Beam TupleCoder format.

    Layout: varint(len) + fen_utf8 | varint(len) + move_utf8 | float64_be
    """
    offset = 0
    fen_len, offset = _decode_varint(data, offset)
    fen = data[offset : offset + fen_len].decode("utf-8")
    offset += fen_len
    move_len, offset = _decode_varint(data, offset)
    move = data[offset : offset + move_len].decode("utf-8")
    offset += move_len
    win_prob = struct.unpack(">d", data[offset : offset + 8])[0]
    return fen, move, win_prob


# ---------------------------------------------------------------------------
# Grouped-record processing
# ---------------------------------------------------------------------------

def make_grouped_sample(
    fen: str,
    moves_and_probs: list[tuple[str, float]],
    top_k: int,
    temperature: float,
) -> dict | None:
    """Create a distillation sample from grouped (move, win_prob) records for one FEN.

    Converts win_probs to a soft distribution via softmax with temperature,
    keeps top-k moves, and outputs the same format as generate_dataset.py.
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    legal_uci = {m.uci() for m in board.legal_moves}

    # Filter to legal moves that also exist in our action space.
    valid = []
    for move_str, win_prob in moves_and_probs:
        if move_str not in legal_uci:
            continue
        action_idx = MOVE_TO_ACTION.get(move_str)
        if action_idx is not None:
            valid.append((action_idx, win_prob))

    if not valid:
        return None

    # Sort by win_prob descending and keep top-k
    valid.sort(key=lambda x: x[1], reverse=True)
    valid = valid[:top_k]

    action_indices = [v[0] for v in valid]
    win_probs = np.array([v[1] for v in valid], dtype=np.float64)

    # Softmax over win_probs with temperature
    logits = win_probs / temperature
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()

    board_tokens = torch.tensor(tokenize(fen), dtype=torch.long)

    legal_mask = torch.zeros(1968, dtype=torch.bool)
    for legal_move in board.legal_moves:
        idx = MOVE_TO_ACTION.get(legal_move.uci())
        if idx is not None:
            legal_mask[idx] = True

    return {
        "board_tokens": board_tokens,
        "legal_mask": legal_mask,
        "teacher_action_indices": torch.tensor(action_indices, dtype=torch.long),
        "teacher_probs": torch.tensor(probs, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_shard(shard_idx: int, download_dir: str) -> str:
    """Download a .bag shard from GCS. Returns local path."""
    url = BAG_URL_TEMPLATE.format(shard_idx=shard_idx)
    filename = f"action_value-{shard_idx:05d}-of-02148_data.bag"
    local_path = os.path.join(download_dir, filename)

    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  Already downloaded: {filename} ({size_mb:.0f} MB)")
        return local_path

    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.0f} MB")
    return local_path


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def _flush_grouped_positions(
    fen_groups: dict[str, list[tuple[str, float]]],
    buffer: list[dict],
    config: ConvertConfig,
) -> tuple[int, int]:
    """Convert grouped FEN records into samples and append to buffer.

    Returns (positions_kept, positions_skipped).
    """
    kept = 0
    skipped = 0
    for fen, moves_and_probs in fen_groups.items():
        sample = make_grouped_sample(fen, moves_and_probs, config.top_k, config.temperature)
        if sample is None:
            skipped += 1
        else:
            buffer.append(sample)
            kept += 1
    return kept, skipped


def convert(config: ConvertConfig):
    """Download and convert DeepMind .bag shards to .pt format."""
    os.makedirs(config.output_dir, exist_ok=True)
    download_dir = os.path.join(config.output_dir, "raw_bags")
    os.makedirs(download_dir, exist_ok=True)

    # Evenly-spaced shard indices for position diversity
    shard_indices = np.linspace(0, TOTAL_SHARDS - 1, config.num_shards, dtype=int).tolist()

    print(f"Converting {config.num_shards} shards (of {TOTAL_SHARDS})")
    print(f"  top_k: {config.top_k}, temperature: {config.temperature}")
    print(f"  min_win_prob: {config.min_win_prob}")
    print(f"  Output: {config.output_dir}")
    print(f"  Shard size: {config.shard_size:,}")
    if config.max_samples is not None:
        print(f"  Max samples: {config.max_samples:,}")

    buffer: list[dict] = []
    out_shard_idx = 0
    total_saved = 0

    for i, bag_idx in enumerate(shard_indices):
        print(f"\nShard {i + 1}/{config.num_shards} (bag index {bag_idx}):")

        bag_path = download_shard(bag_idx, download_dir)
        num_records = count_bag_records(bag_path)

        # Group all records by FEN within this .bag shard
        fen_groups: dict[str, list[tuple[str, float]]] = {}
        filtered_records = 0

        for raw in tqdm(
            read_bag_records(bag_path),
            total=num_records,
            desc=f"  Processing",
            unit=" records",
        ):
            fen, move, win_prob = decode_action_value(raw)

            if win_prob < config.min_win_prob:
                filtered_records += 1
                continue

            if fen not in fen_groups:
                fen_groups[fen] = []
            fen_groups[fen].append((move, win_prob))

        # Convert grouped positions to samples
        kept, skipped = _flush_grouped_positions(fen_groups, buffer, config)

        print(
            f"  {num_records:,} records → {len(fen_groups):,} positions "
            f"({kept:,} kept, {skipped} bad, {filtered_records:,} below min_win_prob)"
        )

        # Flush full output shards
        while len(buffer) >= config.shard_size:
            if config.max_samples is not None and total_saved >= config.max_samples:
                break
            remaining = (
                config.shard_size
                if config.max_samples is None
                else min(config.shard_size, config.max_samples - total_saved)
            )
            if remaining <= 0:
                break
            to_save = buffer[:remaining]
            buffer = buffer[remaining:]
            shard_path = os.path.join(config.output_dir, f"shard_{out_shard_idx:04d}.pt")
            save_shard(to_save, shard_path)
            total_saved += len(to_save)
            print(f"  Saved {shard_path} ({len(to_save):,} samples, total: {total_saved:,})")
            out_shard_idx += 1

        if config.max_samples is not None and total_saved >= config.max_samples:
            print("Reached max_samples limit; stopping conversion early.")
            break

    # Save remaining buffer
    if buffer and (config.max_samples is None or total_saved < config.max_samples):
        if config.max_samples is None:
            to_save = buffer
        else:
            remaining = config.max_samples - total_saved
            to_save = buffer[:remaining]
        if to_save:
            shard_path = os.path.join(config.output_dir, f"shard_{out_shard_idx:04d}.pt")
            save_shard(to_save, shard_path)
            total_saved += len(to_save)
            print(f"  Saved {shard_path} ({len(to_save):,} samples)")
            out_shard_idx += 1

    print(f"\nDone! {total_saved:,} samples in {out_shard_idx} shards → {config.output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepMind .bag shards to .pt distillation format"
    )
    parser.add_argument("--config", type=str, default="distill.yaml")
    parser.add_argument("--num_shards", type=int, help="Number of .bag shards to download")
    parser.add_argument("--top_k", type=int, help="Keep top-k moves per position")
    parser.add_argument("--temperature", type=float, help="Softmax temperature")
    parser.add_argument("--min_win_prob", type=float, help="Minimum win_prob to keep a record")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--shard_size", type=int, help="Samples per output shard")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to save")
    args = parser.parse_args()

    data = load_yaml_file(args.config)
    config = dict_to_dataclass(ConvertConfig, data.get("deepmind_data", {}))

    for field in [
        "num_shards",
        "top_k",
        "temperature",
        "min_win_prob",
        "output_dir",
        "shard_size",
        "max_samples",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    convert(config)


if __name__ == "__main__":
    main()
