"""Imports from the searchless_chess submodule (google-deepmind/searchless_chess)."""
import importlib.util
from pathlib import Path

import numpy as np

_SC_SRC = Path(__file__).resolve().parent.parent / "searchless_chess" / "src"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SC_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils = _load("utils")

ACTION_TO_MOVE = _utils.ACTION_TO_MOVE
MOVE_TO_ACTION = _utils.MOVE_TO_ACTION

# Local tokenizer mirror to avoid hard runtime dependency on JAX via jaxtyping.
_CHARACTERS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "p",
    "n",
    "r",
    "k",
    "q",
    "P",
    "B",
    "N",
    "R",
    "Q",
    "K",
    "w",
    ".",
]
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({"1", "2", "3", "4", "5", "6", "7", "8"})
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> np.ndarray:
    """Tokenize FEN into fixed-length sequence matching searchless_chess tokenizer."""
    board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(" ")
    board = board.replace("/", "")
    board = side + board

    indices: list[int] = []

    for char in board:
        if char in _SPACES_CHARACTERS:
            indices.extend(int(char) * [_CHARACTERS_INDEX["."]])
        else:
            indices.append(_CHARACTERS_INDEX[char])

    if castling == "-":
        indices.extend(4 * [_CHARACTERS_INDEX["."]])
    else:
        for char in castling:
            indices.append(_CHARACTERS_INDEX[char])
        if len(castling) < 4:
            indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX["."]])

    if en_passant == "-":
        indices.extend(2 * [_CHARACTERS_INDEX["."]])
    else:
        for char in en_passant:
            indices.append(_CHARACTERS_INDEX[char])

    halfmoves_last += "." * (3 - len(halfmoves_last))
    indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

    fullmoves += "." * (3 - len(fullmoves))
    indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

    if len(indices) != SEQUENCE_LENGTH:
        raise ValueError(f"Unexpected token length: {len(indices)} (expected {SEQUENCE_LENGTH})")

    return np.asarray(indices, dtype=np.uint8)
