"""Prefix-LM attention mask for the reasoning model.

FEN positions (0..fen_len-1) attend bidirectionally within the FEN block.
Thinking/move positions (fen_len..) attend bidirectionally to the FEN block
and causally to themselves. Padding positions are fully masked.

Combining the causal constraint with the padding mask is a single bitwise AND
— adding the causal bit on top of a mask we already have to build is free.

The causal constraint is required for the PPO recompute in
grpo_logic/reasoning_loss.py: without it, log_prob at position t would be
computed conditional on tokens at positions > t, which breaks the ratio.
See failure mode FM 7.6 (Clip-fraction blowup).
"""
import torch


def build_prefix_lm_mask(
    seq_lens: torch.Tensor,
    max_len: int,
    fen_len: int,
) -> torch.Tensor:
    """Returns a [B, 1, max_len, max_len] bool mask (True = attend).

    Args:
        seq_lens: int tensor [B] — real (unpadded) length of each sequence.
        max_len:  int            — padded length all sequences are right-padded to.
        fen_len:  int            — length of the FEN prefix, typically 77.

    The mask is applied inside DMMultiHeadAttention as
        scores.masked_fill(~mask, neg_inf)
    """
    if fen_len < 0 or fen_len > max_len:
        raise ValueError("fen_len must be between 0 and max_len")
    if torch.any(seq_lens < 0):
        raise ValueError("seq_lens must be non-negative")
    if torch.any(seq_lens > max_len):
        raise ValueError("seq_lens cannot exceed max_len")

    device = seq_lens.device
    idx = torch.arange(max_len, device=device)

    # Causal mask: position t can attend to positions <= t. Shape [max_len, max_len].
    causal = idx[None, :] <= idx[:, None]

    # Bidirectional within the FEN block: rows 0..fen-1 attend to columns 0..fen-1.
    is_fen_row = idx[:, None] < fen_len
    is_fen_col = idx[None, :] < fen_len

    # A (row, col) cell is valid if:
    #   - both row and col are in the FEN block (bidirectional FEN prefix), OR
    #   - the row is in the thinking tail and the col is in the FEN block,  OR
    #   - the row is in the thinking tail and the tail is causal.
    is_tail_row = idx[:, None] >= fen_len
    mask_rc = (is_fen_row & is_fen_col) | (is_tail_row & is_fen_col) | (causal & is_tail_row)

    # Padding mask: key positions past seq_lens are invalid, and query positions
    # past seq_lens are also invalid (don't query from pad).
    token_valid = idx[None, :] < seq_lens[:, None]     # [B, max_len]
    # Broadcast: [B, 1, max_len, max_len]
    mask = mask_rc[None, :, :] & token_valid[:, None, :] & token_valid[:, :, None]
    return mask.unsqueeze(1)  # [B, 1, max_len, max_len]
