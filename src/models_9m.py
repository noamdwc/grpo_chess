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

import numpy as np
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


class Searchless9MActionValuePolicy(nn.Module):
    """GRPO policy that preserves exact 9M action-value checkpoint behavior.

    For each legal action a, the model scores Q(s, a) by:
    1. Building the 9M action-value target sequence [fen_tokens, action, return_dummy]
    2. Applying shift-right (matching JAX transformer_decoder input convention)
    3. Predicting return-bucket log-probs at the last position
    4. Taking expected value over bucket centers.
    """

    action_size: int = 1968
    sequence_length: int = 77

    def __init__(self, checkpoint_path: str | None = None, chunk_size: int = 8192):
        super().__init__()
        self.value_model = Searchless9MTransformer()
        self.chunk_size = int(chunk_size)
        bucket_values = self._make_bucket_values(128)
        self.register_buffer("bucket_values", bucket_values, persistent=False)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    @staticmethod
    def _make_bucket_values(num_buckets: int) -> torch.Tensor:
        full = np.linspace(0.0, 1.0, num_buckets + 1, dtype=np.float32)
        values = (full[:-1] + full[1:]) / 2.0
        return torch.from_numpy(values)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = self.value_model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Failed strict load for Searchless9MActionValuePolicy. "
                f"missing={missing}, unexpected={unexpected}"
            )
        print(f"Loaded exact 9M action-value weights from {checkpoint_path}.")

    @staticmethod
    def _shift_right(targets: torch.Tensor) -> torch.Tensor:
        # targets: [N, 79] where 79 = 77(fen)+1(action)+1(dummy return bucket)
        bos = torch.zeros((targets.shape[0], 1), dtype=targets.dtype, device=targets.device)
        padded = torch.cat([bos, targets], dim=1)
        return padded[:, :-1]

    def _build_inputs(self, states: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        states = states.to(dtype=torch.long)
        action_ids = action_ids.to(dtype=torch.long).unsqueeze(1)
        dummy_return_bucket = torch.zeros_like(action_ids)
        targets = torch.cat([states, action_ids, dummy_return_bucket], dim=1)  # [N, 79]
        return self._shift_right(targets)  # [N, 79]

    def _score_action_pairs(self, states: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        """Compute Q(s,a) expected values for paired states/actions."""
        if states.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=states.device)

        outputs: list[torch.Tensor] = []
        bucket_values = self.bucket_values.to(device=states.device)
        n = states.shape[0]
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            inputs = self._build_inputs(states[start:end], action_ids[start:end])  # [m, 79]
            log_probs = self.value_model(inputs)[:, -1, :]  # [m, 128]
            scores = torch.matmul(log_probs.exp(), bucket_values.to(dtype=log_probs.dtype))  # [m]
            outputs.append(scores)
        return torch.cat(outputs, dim=0)

    def _compute_masked_logits(
        self,
        states: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if legal_moves_mask.dtype != torch.bool:
            legal_moves_mask = legal_moves_mask.to(dtype=torch.bool)
        if legal_moves_mask.ndim != 2:
            raise ValueError(f"legal_moves_mask must be rank-2 [B,A], got {tuple(legal_moves_mask.shape)}")
        if states.ndim != 2:
            raise ValueError(f"states must be rank-2 [B,T], got {tuple(states.shape)}")
        if states.shape[0] != legal_moves_mask.shape[0]:
            raise ValueError(
                f"Batch mismatch between states and legal mask: {states.shape[0]} vs {legal_moves_mask.shape[0]}"
            )
        if legal_moves_mask.shape[1] != self.action_size:
            raise ValueError(
                f"Action dimension mismatch: legal mask has {legal_moves_mask.shape[1]}, expected {self.action_size}"
            )

        # Ensure at least one legal action per row to avoid NaNs in softmax/log_softmax.
        row_has_legal = legal_moves_mask.any(dim=1)
        if not row_has_legal.all():
            legal_moves_mask = legal_moves_mask.clone()
            legal_moves_mask[~row_has_legal, 0] = True

        nz = legal_moves_mask.nonzero(as_tuple=False)  # [N_legal, 2]
        row_idx = nz[:, 0]
        action_idx = nz[:, 1]
        states_rep = states[row_idx]  # [N_legal, 77]
        scores = self._score_action_pairs(states_rep, action_idx)

        logits = torch.full(
            (states.shape[0], self.action_size),
            -float("inf"),
            dtype=scores.dtype,
            device=states.device,
        )
        temp = max(1e-6, float(temperature))
        logits[row_idx, action_idx] = scores / temp
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense action logits [B, 1968] by scoring all actions (slow; avoid in hot paths)."""
        all_legal = torch.ones((x.shape[0], self.action_size), dtype=torch.bool, device=x.device)
        return self._compute_masked_logits(x, all_legal, temperature=1.0)

    def get_legal_moves_logits(
        self,
        tensor_state: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return self._compute_masked_logits(tensor_state, legal_moves_mask, temperature=temperature)

    def get_legal_moves_probs(
        self,
        tensor_state: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        return F.softmax(self.get_legal_moves_logits(tensor_state, legal_moves_mask, temperature), dim=-1)

    def get_group_log_probs(
        self,
        trajectories_states: torch.Tensor,
        action_idx: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if legal_moves_mask is None:
            raise ValueError("legal_moves_mask is required for Searchless9MActionValuePolicy.")
        bsz, groups, depth, seq_len = trajectories_states.shape
        x_flat = trajectories_states.view(bsz * groups * depth, seq_len)
        lm_flat = legal_moves_mask.view(bsz * groups * depth, -1)
        masked_logits = self.get_legal_moves_logits(x_flat, lm_flat, temperature)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)
        action_flat = action_idx.view(bsz * groups * depth, 1)
        return log_probs_all.gather(1, action_flat).squeeze(-1).view(bsz, groups, depth)


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
