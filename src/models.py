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
    """Configuration for the Chess Transformer model.

    Attributes:
        vocab_size: Size of the vocabulary (token dictionary)
        embed_dim: Embedding dimension for transformer
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        action_dim: Dimension of action space (number of possible moves)
        pad_idx: Padding token id
        readout: Sequence readout mode ("mean", "last", or "cls")
        cls_token_id: Optional CLS token id (used when readout="cls")
        ffn_mult: Optional transformer FFN width multiplier (dim_feedforward = ffn_mult * embed_dim)
        head_mult: Policy head hidden width multiplier (hidden = head_mult * embed_dim)
        activation: Transformer/head activation ("relu" or "gelu")
        dropout: Transformer dropout
    """
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


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([ChessTransformerConfig])


class ChessTransformer(nn.Module):
    """Transformer-based chess policy network.
    
    Takes FEN-encoded board states as input and outputs action logits.
    Uses a transformer encoder with learnable positional encodings.
    """
    def __init__(self, transformer_config: ChessTransformerConfig):
        """
        Initialize Chess Transformer.
        
        Args:
            transformer_config: Configuration for the transformer model
        """
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
            raise ValueError(f"Unsupported readout mode: {readout!r}. Expected one of: 'mean', 'last', 'cls'.")
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"Unsupported activation: {activation!r}. Expected 'relu' or 'gelu'.")

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

        # DeepMind uses absolute or relative pos encoding.
        # For simplicity, we use learnable absolute encoding for FEN length (~80 chars)
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

        # Head outputs 1968 logits (one for each possible unique move type)
        head_hidden_dim = head_mult * embed_dim
        head_activation: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            head_activation,
            nn.Linear(head_hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input tensor of token IDs [batch, seq_len]
            
        Returns:
            Action logits [batch, action_dim]
        """
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

        # Create padding mask: True indicates a masked position.
        src_key_padding_mask = token_ids == self.pad_idx
        x = self.embedding(token_ids) + self.pos_encoding[:, :seq, :]

        # Pass the padding mask to the transformer
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        if self.readout == "mean":
            # Mean pool over non-padding tokens.
            mask = ~src_key_padding_mask
            mask_expanded = mask.unsqueeze(-1).float()  # [B, SEQ, 1]
            pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
        elif self.readout == "last":
            # Readout from final non-padding token.
            nonpad_mask = ~src_key_padding_mask
            lengths = nonpad_mask.sum(dim=1)  # [B]
            last_idx = (lengths - 1).clamp_min(0)
            pooled = out[torch.arange(batch, device=out.device), last_idx]
        else:  # self.readout == "cls"
            pooled = out[:, 0, :]

        return self.policy_head(pooled)
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    @property
    def action_size(self) -> int:
        """Get the size of the action space."""
        return self.policy_head[-1].out_features

    def get_legal_moves_logits(self, tensor_state: torch.Tensor, 
                               legal_moves_mask: torch.Tensor, 
                               temperature: float = 1.0) -> torch.Tensor:
        """Get logits for legal moves only, masking illegal moves.
        
        Args:
            tensor_state: Board state tensor [B, SEQ]
            legal_moves_mask: Boolean mask for legal moves [B, A]
            temperature: Temperature for scaling logits
            
        Returns:
            Masked logits [B, A] with illegal moves set to -inf
        """
        assert legal_moves_mask is not None, "legal_moves_mask cannot be None"
        logits = self(tensor_state) / temperature
        return logits.masked_fill(~legal_moves_mask, -float('inf'))
    
    def get_legal_moves_probs(self, tensor_state: torch.Tensor,
                              legal_moves_mask: torch.Tensor,
                              temperature: float = 1.0) -> torch.Tensor:
        """Get probability distribution over legal moves.
        
        Args:
            tensor_state: Board state tensor [B, SEQ]
            legal_moves_mask: Boolean mask for legal moves [B, A]
            temperature: Temperature for scaling logits
            
        Returns:
            Probability distribution [B, A] over legal moves
        """
        mask_logits = self.get_legal_moves_logits(tensor_state, legal_moves_mask, temperature)
        return F.softmax(mask_logits, dim=-1)    

    def get_group_log_probs(self,
                            trajectories_states: torch.Tensor,
                            action_idx: torch.Tensor,
                            legal_moves_mask: torch.Tensor,
                            temperature: float = 1.0) -> torch.Tensor:
        """Get log probabilities for actions in batched trajectories.
        
        Args:
            trajectories_states: State tensors [B, G, T, SEQ]
            action_idx: Action indices [B, G, T]
            legal_moves_mask: Legal moves mask [B, G, T, A]
            temperature: Temperature for scaling logits
            
        Returns:
            Log probabilities [B, G, T] for the selected actions
        """
        assert legal_moves_mask is not None, "legal_moves_mask cannot be None"
        assert legal_moves_mask.dtype == torch.bool, "legal_moves_mask must be bool dtype"
        x = trajectories_states  # [B, G, T, SEQ]
        B, G, T, L = x.shape
        x_flat = x.view(B * G * T, L)  # [B*G*T, SEQ]
        if legal_moves_mask is not None:
            legal_moves_mask = legal_moves_mask.view(B * G * T, -1)  # [B*G*T, O]
        masked_logits = self.get_legal_moves_logits(x_flat, legal_moves_mask, temperature)  # [B*G*T, O]
        log_probs_all = F.log_softmax(masked_logits, dim=-1)  # [B*G*T, O]

        action_idx_flat = action_idx.view(B * G * T, 1)  # [B*G*T, 1]
        log_probs_flat = log_probs_all.gather(1, action_idx_flat).squeeze(-1)  # [B*G*T]
        log_probs = log_probs_flat.view(B, G, T)  # [B, G, T]
        return log_probs

    def _get_action_logits(self, board: chess.Board, temperature: float = 1.0) -> Optional[torch.Tensor]:
        """Get action logits for a single board position.
        
        Args:
            board: Chess board position
            temperature: Temperature for scaling logits
            
        Returns:
            Logits tensor [1, action_dim] or None if no legal moves
        """
        legal_moves = list(board.legal_moves)
        legal_indices = get_legal_moves_indices(board)

        if not legal_moves:
            return None

        # Run model
        state = board_to_tensor(board, device=self.device)
        logits = self(state)  # [1, O]

        output = torch.full_like(logits, -float('inf'))
        output[0, legal_indices] = logits[0, legal_indices] / temperature
        return output

    def select_action(self, board: chess.Board, temperature: float = 1.0) -> tuple[Optional[chess.Move], Optional[torch.Tensor], Optional[int]]:
        """Sample an action from the policy for a given board position.
        
        Args:
            board: Chess board position
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Tuple of (move, log_prob, action_idx) or (None, None, None) if no legal moves
        """
        logits = self._get_action_logits(board, temperature)
        if logits is None:
            return None, None, None
        logits = logits.squeeze(0)  # Remove batch dimension
        probs = F.softmax(logits, dim=0)

        # Sample
        action_idx = int(torch.multinomial(probs, 1).item())
        chosen_move = ACTION_TO_MOVE[action_idx]
        log_prob = torch.log(probs[action_idx] + 1e-12)  # Avoid log(0)

        return chess.Move.from_uci(chosen_move), log_prob, action_idx


def select_action_greedy(model: ChessTransformer, board: chess.Board, temperature: float = 1.0) -> Optional[chess.Move]:
    """Select the best action greedily (no sampling).
    
    Args:
        model: Chess transformer model
        board: Chess board position
        temperature: Temperature for scaling logits (unused in greedy selection)
        
    Returns:
        Best move or None if no legal moves
    """
    logits = model._get_action_logits(board, temperature)
    if logits is None:
        return None
    logits = logits.squeeze(0)  # Remove batch dimension
    probs = F.softmax(logits, dim=0)
    action_idx = int(torch.argmax(probs).item())
    chosen_move = ACTION_TO_MOVE[action_idx]
    return chess.Move.from_uci(chosen_move)
