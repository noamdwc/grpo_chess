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


class DMMultiHeadAttention(nn.Module):
    """Bidirectional multi-head attention, matching the DM JAX implementation.

    Matches hk.Linear(with_bias=False) semantics for Q, K, V, and output proj.
    No dropout (DM's inference transformer has none).
    """

    def __init__(self, embedding_dim: int, num_heads: int, apply_qk_layernorm: bool, eps: float) -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        total = self.num_heads * self.head_dim
        # JAX has separate hk.Linear for q, k, v — keep them separate so the
        # converter can map each one 1:1 without fighting a fused qkv layout.
        self.q_proj = nn.Linear(embedding_dim, total, bias=False)
        self.k_proj = nn.Linear(embedding_dim, total, bias=False)
        self.v_proj = nn.Linear(embedding_dim, total, bias=False)
        self.out_proj = nn.Linear(total, embedding_dim, bias=False)
        self.apply_qk_layernorm = apply_qk_layernorm
        if apply_qk_layernorm:
            self.q_ln = nn.LayerNorm(total, eps=eps)
            self.k_ln = nn.LayerNorm(total, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        if self.apply_qk_layernorm:
            q = self.q_ln(q)
            k = self.k_ln(k)
        v = self.v_proj(x)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        # bthd,bThd -> bhtT  (matches JAX einsum)
        scores = torch.einsum("bthd,bThd->bhtT", q, k) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            # attn_mask is [B, 1, T, T] bool, True = keep
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        # bhtT,bThd -> bthd
        out = torch.einsum("bhtT,bThd->bthd", weights, v).contiguous()
        out = out.view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out)


class DMSwiGluMLP(nn.Module):
    """SwiGLU MLP block: (silu(W1 x) * W2 x) W3 — matches JAX _mlp_block."""

    def __init__(self, embedding_dim: int, widening_factor: int) -> None:
        super().__init__()
        ffn_dim = embedding_dim * widening_factor
        self.w1 = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(embedding_dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DMTransformerBlock(nn.Module):
    """Pre-LN attention block, pre-LN MLP block, both residual. Matches JAX."""

    def __init__(self, config: DMTransformerConfig) -> None:
        super().__init__()
        self.attn_ln = nn.LayerNorm(config.embedding_dim, eps=config.eps)
        self.attn = DMMultiHeadAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            apply_qk_layernorm=config.apply_qk_layernorm,
            eps=config.eps,
        )
        self.mlp_ln = nn.LayerNorm(config.embedding_dim, eps=config.eps)
        self.mlp = DMSwiGluMLP(config.embedding_dim, config.widening_factor)

    def forward(self, h: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = h + self.attn(self.attn_ln(h), attn_mask=attn_mask)
        h = h + self.mlp(self.mlp_ln(h))
        return h


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

        self.layers = nn.ModuleList([DMTransformerBlock(config) for _ in range(config.num_layers)])
        self.post_ln: Optional[nn.LayerNorm] = nn.LayerNorm(config.embedding_dim, eps=config.eps) if config.apply_post_ln else None
        self.output_linear = nn.Linear(config.embedding_dim, config.output_size, bias=True)

    def embed_inputs(self, targets: torch.Tensor) -> torch.Tensor:
        """Right-shift targets, embed, scale by sqrt(D), add learned PE."""
        inputs = shift_right(targets)
        embeddings = self.token_embedding(inputs) * (self.config.embedding_dim ** 0.5)
        positions = torch.arange(inputs.size(1), device=inputs.device)
        pos_enc = self.pos_embedding(positions)[None, :, :]
        return embeddings + pos_enc

    def forward(
        self,
        targets: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns log_softmax logits of shape [B, T, output_size].

        `attn_mask` is an optional bool mask of shape [B, 1, T, T] (True = keep).
        The DM 9M checkpoint was trained with `use_causal_mask=False`, so the
        default here is bidirectional (no mask). The reasoning model passes a
        prefix-LM mask in later phases.
        """
        h = self.embed_inputs(targets)
        for block in self.layers:
            h = block(h, attn_mask=attn_mask)
        if self.post_ln is not None:
            h = self.post_ln(h)
        logits = self.output_linear(h)
        return F.log_softmax(logits, dim=-1)
