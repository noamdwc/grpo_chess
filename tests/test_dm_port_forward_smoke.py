"""Shape + finiteness smoke tests for the DM-port forward pass.

Numerical parity with JAX is tested in tests/test_dm_port_parity.py.
"""
import torch

from src.dm_port.transformer import DMTransformer, DMTransformerConfig


def test_forward_shapes():
    cfg = DMTransformerConfig(
        vocab_size=32,
        output_size=32,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        max_sequence_length=12,
    )
    model = DMTransformer(cfg)
    x = torch.zeros((3, 12), dtype=torch.long)
    log_probs = model(x)
    assert log_probs.shape == (3, 12, 32)
    assert torch.isfinite(log_probs).all()
