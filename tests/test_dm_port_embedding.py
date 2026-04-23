"""Tests for the DM-port transformer's embedding, shift_right, and PE shapes."""
import torch

from src.dm_port.transformer import DMTransformerConfig, DMTransformer, shift_right


def test_shift_right_prepends_zero_and_drops_last():
    x = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    y = shift_right(x)
    assert y.shape == x.shape
    assert y[0, 0].item() == 0
    assert y[0, 1].item() == 5
    assert y[0, 2].item() == 6
    assert y[0, 3].item() == 7


def test_embed_sequences_shapes():
    cfg = DMTransformerConfig(vocab_size=20, embedding_dim=16, num_layers=1, num_heads=4, max_sequence_length=8)
    model = DMTransformer(cfg)
    x = torch.zeros((2, 8), dtype=torch.long)
    h = model.embed_inputs(x)
    assert h.shape == (2, 8, 16)
