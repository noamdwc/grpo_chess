import numpy as np

from src.searchless_chess_imports import SEQUENCE_LENGTH, tokenize


def test_tokenize_returns_fixed_length_without_jax() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tokens = tokenize(fen)
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.uint8
    assert tokens.shape == (SEQUENCE_LENGTH,)
