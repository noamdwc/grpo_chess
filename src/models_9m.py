"""
Exact PyTorch mirror of the DeepMind searchless_chess 9M model.

Architecture matches the JAX TransformerConfig used in teacher.py:
  vocab_size=1968, embed_dim=256, num_layers=8, num_heads=8,
  ffn_dim=1024 (widening_factor=4), max_seq_len=79 (SEQUENCE_LENGTH+2),
  output_size=128, pos_encodings=LEARNED, apply_post_ln=True,
  use_causal_mask=False (bidirectional attention, matching inference config).

Weights are 1:1 transferable from the Orbax checkpoint via
  src/distill/convert_jax_checkpoint.py.

Input contract: The JAX model applies shift_right (prepend 0, drop last)
  internally in transformer_decoder. For exact output parity, callers
  should pass the already-shifted sequence (length 79) with a leading 0
  token prepended and the return-bucket token dropped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Searchless9MConfig:
    vocab_size: int = 1968        # NUM_ACTIONS — move indices (same as JAX 9M)
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 1024           # widening_factor=4 × embed_dim
    max_seq_len: int = 79         # SEQUENCE_LENGTH(77) + 2
    output_size: int = 128        # return buckets


class SwiGLU(nn.Module):
    """Gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x)). No bias."""

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Searchless9MLayer(nn.Module):
    """Pre-norm transformer layer (LLaMa-style) matching the JAX 9M architecture."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # bias=False: JAX attention projections have no bias
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = SwiGLU(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class Searchless9MTransformer(nn.Module):
    """
    Exact PyTorch mirror of the DeepMind searchless_chess 9M model.

    Input:  integer token sequences [B, T]
    Output: log-probabilities over 128 return buckets [B, T, 128]
              (last position [:, -1, :] is the action-value prediction)

    Attention is bidirectional (use_causal_mask=False), matching the
    inference configuration in teacher.py for the action-value task.

    All weights are transferable 1:1 from the JAX Orbax checkpoint via
    src/distill/convert_jax_checkpoint.py.
    """

    def __init__(self, config: Searchless9MConfig = None):
        super().__init__()
        if config is None:
            config = Searchless9MConfig()
        self.config = config
        self.embedding    = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.layers       = nn.ModuleList([
            Searchless9MLayer(config.embed_dim, config.num_heads, config.ffn_dim)
            for _ in range(config.num_layers)
        ])
        self.final_norm   = nn.LayerNorm(config.embed_dim)
        # hk.Linear(output_size) in JAX has bias; replicate that here
        self.policy_head  = nn.Linear(config.embed_dim, config.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] integer token sequences (0 ≤ tokens < vocab_size)
        Returns:
            [B, T, 128] log-softmax over return buckets
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.embedding(x) * (self.config.embed_dim ** 0.5)  # JAX scales by sqrt(d)
        h = h + self.pos_encoding(positions)
        # No causal mask — matches use_causal_mask=False in teacher.py
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        logits = self.policy_head(h)           # [B, T, 128]
        return F.log_softmax(logits, dim=-1)   # matches JAX jnn.log_softmax


class Searchless9MGRPOPolicy(nn.Module):
    """9M transformer body repurposed as a GRPO chess policy.

    Uses FEN-tokenized board states ([B, 77], tokens 0-30) as input.
    Outputs [B, 1968] action logits, compatible with the ChessTransformer interface
    expected by sampling.py and grpo_logic/model.py.

    Body weights (embedding, transformer layers, final_norm) are loaded from the
    converted 9M checkpoint; the action head is randomly initialized.
    """

    action_size: int = 1968

    def __init__(self, checkpoint_path: str = None, freeze_body: bool = False):
        super().__init__()
        cfg = Searchless9MConfig()
        self._embed_dim = cfg.embed_dim
        # 9M body (same submodule names as Searchless9MTransformer for weight compat)
        self.embedding    = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_encoding = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.layers       = nn.ModuleList([
            Searchless9MLayer(cfg.embed_dim, cfg.num_heads, cfg.ffn_dim)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm   = nn.LayerNorm(cfg.embed_dim)
        # New action head: 256 → 1968 (replaces 9M policy_head 256 → 128)
        self.action_head  = nn.Linear(cfg.embed_dim, 1968)

        if checkpoint_path:
            self._load_9m_body(checkpoint_path)

        if freeze_body:
            for name, p in self.named_parameters():
                if not name.startswith("action_head"):
                    p.requires_grad = False

    def _load_9m_body(self, checkpoint_path: str) -> None:
        import torch as _torch
        ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        body_missing = [k for k in missing if not k.startswith("action_head")]
        if body_missing:
            print(f"Warning: Missing 9M body keys: {body_missing}")
        print(f"Loaded 9M body weights from {checkpoint_path}.")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """FEN token ids [B, T] → mean-pooled representation [B, embed_dim]."""
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.embedding(x) * (self._embed_dim ** 0.5)
        h = h + self.pos_encoding(positions)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return h.mean(dim=1)  # [B, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → [B, 1968] action logits."""
        return self.action_head(self._encode(x))

    def get_legal_moves_logits(self, tensor_state: torch.Tensor,
                               legal_moves_mask: torch.Tensor,
                               temperature: float = 1.0) -> torch.Tensor:
        logits = self(tensor_state) / temperature
        return logits.masked_fill(~legal_moves_mask, -float('inf'))

    def get_legal_moves_probs(self, tensor_state: torch.Tensor,
                              legal_moves_mask: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(
            self.get_legal_moves_logits(tensor_state, legal_moves_mask, temperature), dim=-1
        )

    def get_group_log_probs(self, trajectories_states: torch.Tensor,
                            action_idx: torch.Tensor,
                            legal_moves_mask: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
        B, G, T, L = trajectories_states.shape
        x_flat = trajectories_states.view(B * G * T, L)
        lm_flat = legal_moves_mask.view(B * G * T, -1)
        masked_logits = self.get_legal_moves_logits(x_flat, lm_flat, temperature)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        action_flat = action_idx.view(B * G * T, 1)
        return log_probs_all.gather(1, action_flat).squeeze(-1).view(B, G, T)
