"""Vocabulary layout for the reasoning model.

The reasoning model reuses DM's action vocabulary (1968 move indices) and
adds three structural tokens. The base vocab size must match the DM 9M
checkpoint exactly — if DM changes, the parity test catches it.
"""
from typing import Sequence

import numpy as np

# Must match searchless_chess/src/utils.py::NUM_ACTIONS exactly.
# We do not import that module here to keep reasoning free of JAX imports.
BASE_VOCAB_SIZE = 1968

THINK_ID = BASE_VOCAB_SIZE
END_THINK_ID = BASE_VOCAB_SIZE + 1
MOVE_ID = BASE_VOCAB_SIZE + 2

TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + 3

# Length of the DM FEN tokenization (see searchless_chess/src/tokenizer.py).
FEN_LEN = 77


def build_input_sequence(
    fen_tokens: np.ndarray,
    think_moves: Sequence[int],
    final_move: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a full reasoning sequence and return position markers.

    Layout:
        [fen(77) | <think> | m1 ... m_T | <end_think> | <move> | m_final]

    The position markers are absolute indices into the returned sequence; the
    loss and the sampler both rely on them.
    """
    if fen_tokens.shape != (FEN_LEN,):
        raise ValueError(f"fen_tokens must be length {FEN_LEN}")
    think_arr = np.asarray(list(think_moves), dtype=np.int64)
    parts = [
        fen_tokens.astype(np.int64),
        np.array([THINK_ID], dtype=np.int64),
        think_arr,
        np.array([END_THINK_ID, MOVE_ID, final_move], dtype=np.int64),
    ]
    seq = np.concatenate(parts)
    think_start = FEN_LEN
    think_end = FEN_LEN + 1 + len(think_arr)
    move_token = think_end + 1
    final_move_pos = move_token + 1
    return seq, {
        "think_start": think_start,
        "think_end": think_end,
        "move_token": move_token,
        "final_move": final_move_pos,
    }
