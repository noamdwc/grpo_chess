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
