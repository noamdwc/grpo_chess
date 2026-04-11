import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from typing import Optional

from dataclasses import dataclass
from src.searchless_chess_imports import ACTION_TO_MOVE
from src.chess.chess_logic import board_to_tensor, get_legal_moves_indices


@dataclass
class ChessTransformerConfig:
    """Configuration for the Chess Transformer model."""

    vocab_size: int = 300
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    action_dim: int = 1968
    pad_idx: int = 0
    readout: str = "mean"
    cls_token_id: Optional[int] = None
    ffn_mult: Optional[int] = None
    head_mult: int = 1
    activation: str = "relu"
    dropout: float = 0.1


torch.serialization.add_safe_globals([ChessTransformerConfig])


class ChessTransformer(nn.Module):
    """Transformer-based chess policy network retained for distill compatibility."""

    def __init__(self, transformer_config: ChessTransformerConfig):
        super().__init__()
        vocab_size = transformer_config.vocab_size
        embed_dim = transformer_config.embed_dim
        num_layers = transformer_config.num_layers
        num_heads = transformer_config.num_heads
        action_dim = transformer_config.action_dim
        pad_idx = transformer_config.pad_idx
        readout = transformer_config.readout.lower()
        activation = transformer_config.activation.lower()
        dropout = transformer_config.dropout
        head_mult = max(1, int(transformer_config.head_mult))

        if readout not in {"mean", "last", "cls"}:
            raise ValueError(f"Unsupported readout mode: {readout!r}.")
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {activation!r}.")

        self.pad_idx = int(pad_idx)
        self.readout = readout
        self.cls_token_id: Optional[int] = None

        embedding_vocab_size = vocab_size
        if readout == "cls":
            cls_token_id = transformer_config.cls_token_id
            if cls_token_id is None:
                cls_token_id = vocab_size
            cls_token_id = int(cls_token_id)
            if cls_token_id == self.pad_idx:
                raise ValueError("cls_token_id cannot be equal to pad_idx.")
            embedding_vocab_size = max(vocab_size, cls_token_id + 1)
            self.cls_token_id = cls_token_id

        self.embedding = nn.Embedding(embedding_vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, embed_dim))

        if transformer_config.ffn_mult is None:
            dim_feedforward = 2048
        else:
            dim_feedforward = max(1, int(transformer_config.ffn_mult)) * embed_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        head_hidden_dim = head_mult * embed_dim
        head_activation: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            head_activation,
            nn.Linear(head_hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = x.shape
        token_ids = x

        if self.readout == "cls":
            assert self.cls_token_id is not None
            cls_tokens = torch.full(
                (batch, 1),
                self.cls_token_id,
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            token_ids = torch.cat([cls_tokens, token_ids], dim=1)

        seq = token_ids.shape[1]
        if seq > self.pos_encoding.shape[1]:
            raise ValueError(
                f"Input sequence length ({seq}) exceeds positional encoding limit ({self.pos_encoding.shape[1]})."
            )

        src_key_padding_mask = token_ids == self.pad_idx
        x = self.embedding(token_ids) + self.pos_encoding[:, :seq, :]
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if self.readout == "mean":
            mask = ~src_key_padding_mask
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        elif self.readout == "last":
            nonpad_mask = ~src_key_padding_mask
            lengths = nonpad_mask.sum(dim=1)
            last_idx = (lengths - 1).clamp_min(0)
            pooled = out[torch.arange(batch, device=out.device), last_idx]
        else:
            pooled = out[:, 0, :]

        return self.policy_head(pooled)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def action_size(self) -> int:
        return self.policy_head[-1].out_features

    def get_legal_moves_logits(
        self,
        tensor_state: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        assert legal_moves_mask is not None, "legal_moves_mask cannot be None"
        logits = self(tensor_state) / temperature
        return logits.masked_fill(~legal_moves_mask, -float("inf"))

    def get_legal_moves_probs(
        self,
        tensor_state: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        mask_logits = self.get_legal_moves_logits(tensor_state, legal_moves_mask, temperature)
        return F.softmax(mask_logits, dim=-1)

    def get_group_log_probs(
        self,
        trajectories_states: torch.Tensor,
        action_idx: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        assert legal_moves_mask is not None, "legal_moves_mask cannot be None"
        assert legal_moves_mask.dtype == torch.bool, "legal_moves_mask must be bool dtype"
        x = trajectories_states
        batch, groups, steps, seq_len = x.shape
        x_flat = x.view(batch * groups * steps, seq_len)
        legal_moves_mask = legal_moves_mask.view(batch * groups * steps, -1)
        masked_logits = self.get_legal_moves_logits(x_flat, legal_moves_mask, temperature)
        log_probs_all = F.log_softmax(masked_logits, dim=-1)

        action_idx_flat = action_idx.view(batch * groups * steps, 1)
        log_probs_flat = log_probs_all.gather(1, action_idx_flat).squeeze(-1)
        return log_probs_flat.view(batch, groups, steps)

    def _get_action_logits(self, board: chess.Board, temperature: float = 1.0) -> Optional[torch.Tensor]:
        legal_moves = list(board.legal_moves)
        legal_indices = get_legal_moves_indices(board)
        if not legal_moves:
            return None

        state = board_to_tensor(board, self.device)
        logits = self(state) / temperature
        action_logits = torch.full((self.action_size,), -float("inf"), device=self.device, dtype=logits.dtype)
        indices = torch.tensor(legal_indices, device=self.device, dtype=torch.long)
        action_logits[indices] = logits[0, indices]
        return action_logits

    @torch.no_grad()
    def act(self, board: chess.Board, temperature: float = 1.0) -> Optional[chess.Move]:
        action_logits = self._get_action_logits(board, temperature)
        if action_logits is None:
            return None
        action_idx = int(torch.argmax(action_logits).item())
        return chess.Move.from_uci(ACTION_TO_MOVE[action_idx])
