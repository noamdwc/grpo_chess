"""ReasoningModel: DM 9M port + minimal extensions for autoregressive reasoning.

Three extensions to the ported DM transformer:
  1) Token embedding extended from 1968 -> 1971 (new rows initialized from scratch).
  2) Positional embedding extended from 79 -> `max_seq_len` (new rows initialized
     by COPYING the last DM-trained row (position 78), not random. This avoids
     the cold-start blowup documented in FM 7.9.
  3) A new policy head (Linear -> TOTAL_VOCAB_SIZE logits) replacing DM's 128-bucket head.
Plus a 2-layer MLP value head read off at the <end_think> hidden state (for
the auxiliary MSE regression against the frozen oracle; see FM 7.5).

The attention mask used during forward is prefix-LM + padding (see
src/reasoning/attention_mask.py). It is constructed once per forward call
and passed to every transformer block.
"""
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from src.dm_port.transformer import DMTransformer, DMTransformerConfig
from src.reasoning.attention_mask import build_prefix_lm_mask
from src.reasoning.tokens import FEN_LEN, TOTAL_VOCAB_SIZE


@dataclass
class ReasoningModelConfig:
    dm_checkpoint: str
    max_seq_len: int = 120
    value_head_hidden: int = 256
    freeze_body: bool = False


class ReasoningModelOutput(NamedTuple):
    logits: torch.Tensor
    hidden: torch.Tensor


class ReasoningModel(nn.Module):
    def __init__(self, config: ReasoningModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.max_seq_len < FEN_LEN:
            raise ValueError(f"max_seq_len must be at least {FEN_LEN}")

        ckpt = torch.load(config.dm_checkpoint, weights_only=False)
        dm_cfg_dict = dict(ckpt["config"])
        dm_cfg_dict["vocab_size"] = TOTAL_VOCAB_SIZE
        dm_cfg_dict["output_size"] = TOTAL_VOCAB_SIZE
        dm_cfg_dict["max_sequence_length"] = config.max_seq_len
        dm_cfg = DMTransformerConfig(**dm_cfg_dict)
        self.dm = DMTransformer(dm_cfg)

        self._load_and_extend_dm_weights(ckpt["state_dict"])

        nn.init.trunc_normal_(self.dm.output_linear.weight, std=0.02)
        nn.init.zeros_(self.dm.output_linear.bias)

        embedding_dim = dm_cfg.embedding_dim
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, config.value_head_hidden),
            nn.SiLU(),
            nn.Linear(config.value_head_hidden, 1),
        )
        nn.init.trunc_normal_(self.value_head[0].weight, std=0.02)
        nn.init.zeros_(self.value_head[0].bias)
        nn.init.trunc_normal_(self.value_head[2].weight, std=0.02)
        nn.init.zeros_(self.value_head[2].bias)
        if config.freeze_body:
            for block in self.dm.layers:
                for param in block.parameters():
                    param.requires_grad_(False)

    def _load_and_extend_dm_weights(self, dm_state_dict: dict) -> None:
        """Load DM body weights, extending token embedding and PE to new sizes."""
        own_state = self.dm.state_dict()
        loaded_keys: set[str] = set()
        for key, value in dm_state_dict.items():
            if key == "token_embedding.weight":
                own_state[key][:value.shape[0]] = value
                loaded_keys.add(key)
            elif key == "pos_embedding.weight":
                own_state[key][:value.shape[0]] = value
                last = value[-1]
                for i in range(value.shape[0], own_state[key].shape[0]):
                    own_state[key][i] = last
                loaded_keys.add(key)
            elif key in {"output_linear.weight", "output_linear.bias"}:
                continue
            else:
                if key not in own_state:
                    raise RuntimeError(f"Unexpected checkpoint key: {key}")
                if own_state[key].shape == value.shape:
                    own_state[key] = value
                    loaded_keys.add(key)
                else:
                    raise RuntimeError(
                        f"Shape mismatch loading {key}: own={own_state[key].shape} ckpt={value.shape}"
                    )
        expected_keys = set(own_state) - {"output_linear.weight", "output_linear.bias"}
        missing_keys = sorted(expected_keys - loaded_keys)
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            raise RuntimeError(f"Missing checkpoint keys: {preview}")
        self.dm.load_state_dict(own_state)

    @property
    def policy_head(self) -> nn.Linear:
        """Authoritative policy head alias without duplicate state_dict keys."""
        return self.dm.output_linear

    def forward(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> ReasoningModelOutput:
        """Forward pass with prefix-LM + padding mask."""
        _, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        mask = build_prefix_lm_mask(
            seq_lens=seq_lens,
            max_len=seq_len,
            fen_len=FEN_LEN,
        )

        h = self.dm.embed_inputs(tokens)
        for block in self.dm.layers:
            h = block(h, attn_mask=mask)
        if self.dm.post_ln is not None:
            h = self.dm.post_ln(h)
        logits = self.policy_head(h)
        return ReasoningModelOutput(logits=logits, hidden=h)

    def value_at_end_think(
        self,
        hidden: torch.Tensor,
        end_think_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Read the value head at each sample's <end_think> position. Returns [B]."""
        gather_idx = end_think_positions[:, None, None].expand(-1, 1, hidden.shape[-1])
        h_end = torch.gather(hidden, 1, gather_idx).squeeze(1)
        return self.value_head(h_end).squeeze(-1)

    def logprob_sequence(
        self,
        tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return log-prob of each token conditional on its prefix. Shape [B, T]."""
        out = self.forward(tokens, seq_lens)
        scaled = out.logits / temperature
        log_probs_all = torch.log_softmax(scaled, dim=-1)
        gathered = torch.gather(log_probs_all, 2, tokens.unsqueeze(-1)).squeeze(-1)
        return gathered
