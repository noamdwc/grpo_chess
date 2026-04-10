"""PyTorch reimplementation of the DeepMind searchless-chess transformer body.

Mirrors searchless_chess/src/transformer.py layer-for-layer. The only reason
this exists is so we can fine-tune the weights with PyTorch/Lightning after
porting them out of JAX/Haiku. The Phase-0 parity test in
tests/test_dm_port_parity.py is the contract: if its thresholds fail, fix this
file, do not relax the thresholds.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DMTransformerConfig:
    vocab_size: int = 1968
    output_size: int = 1968        # we replace DM's 128-bucket head; parity uses 128 via override
    embedding_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    widening_factor: int = 4
    max_sequence_length: int = 79  # DM default; we extend this in the reasoning model
    apply_post_ln: bool = True
    apply_qk_layernorm: bool = False
    emb_init_scale: float = 0.02
    eps: float = 1e-5               # LayerNorm epsilon; JAX default


DM_9M_CONFIG = DMTransformerConfig(
    vocab_size=1968,
    output_size=128,
    embedding_dim=256,
    num_layers=8,
    num_heads=8,
)


def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Right-shift by prepending a zero column and dropping the last column.

    Matches searchless_chess/src/transformer.py::shift_right exactly. The DM
    model conditions on the previous token at every position; shifting here
    means position t's input is token[t-1] (with 0 at position 0).
    """
    bos = torch.zeros_like(sequences[:, :1])
    return torch.cat([bos, sequences[:, :-1]], dim=1)


class DMTransformer(nn.Module):
    """PyTorch port of the DM searchless-chess transformer body.

    Mirrors the JAX model layer-for-layer. See tests/test_dm_port_parity.py
    for the numerical contract against the JAX reference.
    """

    def __init__(self, config: DMTransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Token and position embeddings (both hk.Embed in JAX).
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.trunc_normal_(self.token_embedding.weight, std=config.emb_init_scale)
        self.pos_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=config.emb_init_scale)

        # Placeholder for body layers and heads — filled in in Task 3.
        self.layers = nn.ModuleList()
        self.post_ln: Optional[nn.LayerNorm] = nn.LayerNorm(config.embedding_dim, eps=config.eps) if config.apply_post_ln else None
        self.output_linear = nn.Linear(config.embedding_dim, config.output_size, bias=True)

    def embed_inputs(self, targets: torch.Tensor) -> torch.Tensor:
        """Right-shift targets, embed, scale by sqrt(D), add learned PE."""
        inputs = shift_right(targets)
        embeddings = self.token_embedding(inputs) * (self.config.embedding_dim ** 0.5)
        positions = torch.arange(inputs.size(1), device=inputs.device)
        pos_enc = self.pos_embedding(positions)[None, :, :]
        return embeddings + pos_enc
