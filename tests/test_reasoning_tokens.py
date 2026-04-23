"""Tests for the reasoning vocabulary layout."""
import numpy as np
import pytest

from src.reasoning.tokens import (
    BASE_VOCAB_SIZE,
    FEN_LEN,
    TOTAL_VOCAB_SIZE,
    THINK_ID,
    END_THINK_ID,
    MOVE_ID,
    build_input_sequence,
)


def test_special_ids_are_past_base_vocab():
    assert THINK_ID == BASE_VOCAB_SIZE
    assert END_THINK_ID == BASE_VOCAB_SIZE + 1
    assert MOVE_ID == BASE_VOCAB_SIZE + 2
    assert TOTAL_VOCAB_SIZE == BASE_VOCAB_SIZE + 3


def test_build_input_sequence_layout():
    fen_tokens = np.arange(FEN_LEN, dtype=np.int64)
    think_moves = [100, 200, 300]
    seq, positions = build_input_sequence(fen_tokens, think_moves, final_move=400)
    # Expected: [fen(77) | <think> | m1 m2 m3 | <end_think> | <move> | m_final]
    assert seq.shape == (FEN_LEN + 1 + 3 + 1 + 1 + 1,)
    assert seq[FEN_LEN] == THINK_ID
    assert list(seq[78:81]) == [100, 200, 300]
    assert seq[81] == END_THINK_ID
    assert seq[82] == MOVE_ID
    assert seq[83] == 400
    # Position markers
    assert positions["think_start"] == FEN_LEN
    assert positions["think_end"] == 81
    assert positions["move_token"] == 82
    assert positions["final_move"] == 83


def test_build_input_sequence_rejects_wrong_fen_length():
    fen_tokens = np.arange(FEN_LEN - 1, dtype=np.int64)

    with pytest.raises(ValueError, match=f"fen_tokens must be length {FEN_LEN}"):
        build_input_sequence(fen_tokens, [], final_move=400)
