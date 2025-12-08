from typing import Optional

# ----------------------------------------------------------------------
# CONFIG: which Lichess dataset + how much to process
# ----------------------------------------------------------------------

# Default HuggingFace dataset: condensed Lichess DB with FEN lists. :contentReference[oaicite:2]{index=2}
HF_DATASET = "mauricett/lichess_sf"
HF_SPLIT = "train"
HF_TRUST_REMOTE_CODE = True  # needed for mauricett/lichess_sf

# Process only part of the dataset (set to None for "no limit")
MAX_GAMES: Optional[int] = 10_000      # None -> all games
MAX_POSITIONS: Optional[int] = None    # None -> no limit on positions

# Optional random subsampling of positions (1.0 = use all)
RANDOM_SUBSAMPLE_FRACTION: float = 1.0

# Sharding: how many (FEN, top-K) examples per JSONL file
SHARD_SIZE: int = 100_000

# Random seed for deterministic subsampling
RANDOM_SEED: int = 42

