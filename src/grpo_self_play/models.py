import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from dataclasses import dataclass
from chess_logic import board_to_tensor, get_legal_moves_indices
from src.grpo_self_play.searchless_chess_imports import ACTION_TO_MOVE

@dataclass
class ChessTransformerConfig:
    vocab_size: int = 300
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    action_dim: int = 1968


class ChessTransformer(nn.Module):
    def __init__(self, transformer_confg: ChessTransformerConfig):
        super().__init__()
        vocab_size = transformer_confg.vocab_size
        embed_dim = transformer_confg.embed_dim
        num_layers = transformer_confg.num_layers
        num_heads = transformer_confg.num_heads
        action_dim = transformer_confg.action_dim

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

    def forward(self, x):
        # x: [batch, seq_len] - This `x` contains token_ids
        batch, seq = x.shape

        # Create padding mask: True indicates a masked position (padding token 0)
        src_key_padding_mask = (x == 0)
        x = self.embedding(x) + self.pos_encoding[:, :seq, :]

        # Pass the padding mask to the transformer
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pool: usually [CLS] token or Mean.
        # If DeepMind tokenizer puts [CLS] at index 0:
        cls_token = out[:, 0, :]

        return self.policy_head(cls_token)
    
    @property
    def device(self):
      return next(self.parameters()).device

    def get_group_log_probs(self,
                            trajectories_states,
                            action_idx, 
                            temperature=1.0):
        '''
        trajectories_states = [B, G, T, SEQ]
        action_idx = [B, G, T]
        '''
        x = trajectories_states  # [B, G, T, SEQ]
        B, G, T, L = x.shape
        x_flat = x.view(B * G * T, L) # [B*G*T, SEQ]
        logits  = self(x_flat) / temperature # [B*G*T, O]
        log_probs_all = F.log_softmax(logits, dim=-1)
        action_idx_flat = action_idx.view(B * G * T, 1) # [B*G*T, 1]
        log_probs_flat = log_probs_all.gather(1, action_idx_flat).squeeze(-1) # [B*G*T, 1]
        log_probs = log_probs_flat.view(B, G, T) # [B, G, T]
        return log_probs

    def _get_action_logits(self, board, temperature=1.0):
        legal_moves = list(board.legal_moves)
        legal_indices = get_legal_moves_indices(board)

        if not legal_moves:
            return None

        # Run model
        state = board_to_tensor(board, device=self.device)
        logits = self(state) # [1, O]

        output = torch.full_like(logits,  -float('inf'))
        output[0, legal_indices] = logits[0, legal_indices] / temperature
        return output

    def select_action(self, board, temperature=1.0):
        logits = self._get_action_logits(board,
                                         temperature)
        if logits is None: return None, None, None
        logits = logits.squeeze(0) # We don't have a batch when selection actions
        probs = F.softmax(logits, dim=0)

        # Sample
        action_idx = int(torch.multinomial(probs, 1).item())
        chosen_move = ACTION_TO_MOVE[action_idx]
        log_prob = torch.log(probs[action_idx])

        return chess.Move.from_uci(chosen_move), log_prob, action_idx


def select_action_greedy(model, board, temperature=1.0):
    '''Funcation to play with the bot'''
    logits = model._get_action_logits(board, temperature)
    if logits is None: return None
    logits = logits.squeeze(0) # We don't have a batch when selection actions
    probs = F.softmax(logits, dim=0)
    action_idx = int(torch.argmax(probs).item())
    chosen_move = ACTION_TO_MOVE[action_idx]
    return chess.Move.from_uci(chosen_move)
