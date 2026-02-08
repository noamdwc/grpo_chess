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
    """
    vocab_size: int = 300
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    action_dim: int = 1968


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

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # DeepMind uses absolute or relative pos encoding.
        # For simplicity, we use learnable absolute encoding for FEN length (~80 chars)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head outputs 1968 logits (one for each possible unique move type)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input tensor of token IDs [batch, seq_len]
            
        Returns:
            Action logits [batch, action_dim]
        """
        batch, seq = x.shape

        # Create padding mask: True indicates a masked position (padding token 0)
        src_key_padding_mask = (x == 0)
        x = self.embedding(x) + self.pos_encoding[:, :seq, :]

        # Pass the padding mask to the transformer
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: Mean of the non-masked tokens
        mask = ~src_key_padding_mask
        mask_expanded = mask.unsqueeze(-1).float()  # [B, SEQ, 1]
        pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1)

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
