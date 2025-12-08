from dataclasses import dataclass
from typing import Iterable, Optional
import random
from datasets import load_dataset
from grpo_chess.src.distill.constants import (
    MAX_GAMES,
    MAX_POSITIONS,
    RANDOM_SUBSAMPLE_FRACTION,
    HF_DATASET,
    HF_SPLIT,
    HF_TRUST_REMOTE_CODE,
    RANDOM_SEED,
)
# ----------------------------------------------------------------------
# Lichess dataset iterator
# ----------------------------------------------------------------------


@dataclass
class PositionSample:
    fen: str
    # You can add metadata if you want (elo, etc) later.


def iter_fens_from_lichess(
    max_games: Optional[int] = MAX_GAMES,
    max_positions: Optional[int] = MAX_POSITIONS,
    subsample_fraction: float = RANDOM_SUBSAMPLE_FRACTION,
) -> Iterable[PositionSample]:
    """
    Stream FEN positions from the HF dataset.

    For mauricett/lichess_sf, each sample is a full game:
        example["fens"] is a list of FENs (stripped format: no halfmove/fullmove). :contentReference[oaicite:4]{index=4}
    """
    random.seed(RANDOM_SEED)

    dataset = load_dataset(
        path=HF_DATASET,
        split=HF_SPLIT,
        streaming=True,
        trust_remote_code=HF_TRUST_REMOTE_CODE,
    )

    num_games = 0
    num_positions = 0

    for game in dataset:
        if max_games is not None and num_games >= max_games:
            break
        num_games += 1

        fens = game["fens"]

        for fen in fens:
            if subsample_fraction < 1.0 and random.random() > subsample_fraction:
                continue

            yield PositionSample(fen=fen)
            num_positions += 1

            if max_positions is not None and num_positions >= max_positions:
                return

