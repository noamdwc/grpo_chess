"""Tests for the prefix-LM + padding attention mask."""
import torch
import pytest

from src.reasoning.attention_mask import build_prefix_lm_mask
from src.reasoning.tokens import FEN_LEN


def test_fen_positions_attend_bidirectionally_to_fen():
    seq_len = FEN_LEN + 3
    mask = build_prefix_lm_mask(
        seq_lens=torch.tensor([seq_len]),
        max_len=seq_len,
        fen_len=FEN_LEN,
    )  # [1, 1, T, T]
    assert mask.shape == (1, 1, seq_len, seq_len)
    # Within the FEN block, every pair should attend.
    fen_block = mask[0, 0, :FEN_LEN, :FEN_LEN]
    assert fen_block.all()


def test_think_positions_are_causal_relative_to_each_other():
    seq_len = FEN_LEN + 3
    tail_prev = FEN_LEN + 1
    tail_next = FEN_LEN + 2
    mask = build_prefix_lm_mask(
        seq_lens=torch.tensor([seq_len]),
        max_len=seq_len,
        fen_len=FEN_LEN,
    )
    # Thinking token at position tail_prev should NOT attend to tail_next.
    assert not mask[0, 0, tail_prev, tail_next]
    # But thinking token at tail_next should attend to tail_prev.
    assert mask[0, 0, tail_next, tail_prev]
    # And every thinking token should attend to all FEN tokens.
    for t in range(FEN_LEN, seq_len):
        assert mask[0, 0, t, :FEN_LEN].all()


def test_padding_positions_are_fully_masked():
    # Batch of two sequences of different lengths.
    short_len = FEN_LEN + 3
    long_len = FEN_LEN + 8
    max_len = FEN_LEN + 13
    mask = build_prefix_lm_mask(
        seq_lens=torch.tensor([short_len, long_len]),
        max_len=max_len,
        fen_len=FEN_LEN,
    )
    # For the shorter sample, positions short_len..max_len-1 should be padding:
    # no real token should attend TO a padding position, and a padding
    # position should not attend TO anything.
    for t in range(short_len, max_len):
        assert not mask[0, 0, :, t].any(), f"pos {t} is padding, nothing should key it"
        assert not mask[0, 0, t, :].any(), f"pos {t} is padding, it should not query anything"


def test_invalid_lengths_raise_value_error():
    with pytest.raises(ValueError, match="fen_len must be between 0 and max_len"):
        build_prefix_lm_mask(seq_lens=torch.tensor([5]), max_len=4, fen_len=5)

    with pytest.raises(ValueError, match="seq_lens must be non-negative"):
        build_prefix_lm_mask(seq_lens=torch.tensor([-1]), max_len=4, fen_len=2)

    with pytest.raises(ValueError, match="seq_lens cannot exceed max_len"):
        build_prefix_lm_mask(seq_lens=torch.tensor([5]), max_len=4, fen_len=2)
