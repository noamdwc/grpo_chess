"""An adapter for the DM action-value model to work as policy head and value head after porting into pytorch."""

from this import d
from jax._src.prng import targets
import torch
import torch.nn as nn

from src.dm_port.transformer import DMTransformer, DMTransformerConfig
from src.models import ChessTransformer
from src.searchless_chess_imports import get_uniform_buckets_edges_values, ACTION_DIM   

NUM_BUCKETS = 128

class PPOActorHead(nn.Module):
    def __init__(self, dm_model: DMTransformer):
        super().__init__()
        self.score_head = nn.Linear(ACTION_DIM, ACTION_DIM)


    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, A]
        # return: [B, A]
        # take the scores output from the dm_model and pass it through the score_head to get the logits
        logits = self.score_head(scores)
        return logits


class AVAdapterPolicy(nn.Module):
    def __init__(self, dm_model: DMTransformer, actor_head: PPOActorHead):
        super().__init__()
        self._bucket_values = get_uniform_buckets_edges_values(NUM_BUCKETS)[1]
        self.dm_model = dm_model # the model [board, action] -> score for action
        self.actor_head = actor_head

    def _dm_forward(self, board_tokens: torch.Tensor, legal_actions: torch.Tensor) -> torch.Tensor:
        """Score legal actions for each board with the ported DeepMind AV model.

           Args:
               board_tokens: Tokenized board/FEN tensor of shape [B, FEN_LEN].
               legal_masks: Boolean legal-action mask of shape [B, A].

           Returns:
               Dense action-score tensor of shape [B, A], with illegal actions set to -inf.

           The original searchless_chess AV engine scores each (board, action) pair by
           passing [FEN tokens..., action_token, dummy_return_bucket] through the shared
           transformer and reading the final-position 128-bucket output. The dummy token is
           required because transformer_decoder() applies shift_right() to targets, so the
           final position only sees action_token if there is one extra token after it.

           See:
               searchless_chess/src/engines/neural_engines.py
                   ActionValueEngine.analyse()
               searchless_chess/src/transformer.py
                   transformer_decoder()
        """

        dummy = torch.zeros((board_tokens.shape[0], 1), device=board_tokens.device) # [B, 1]
        targets = torch.cat([board_tokens, legal_actions, dummy], dim=1) # [B, FEN_LEN + 2] ( +2 because of the action + dummy)
        return self.dm_model(targets)


    def _score_actions(self, board_tokens: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        # board_tokens: [B, FEN_LEN]
        # legal_masks: [B, A]
        # return: [B, A]
        B, fen_len = board_tokens.shape
        A = legal_masks.shape[1]
        device = board_tokens.device
        # board_idx -> legal_actions
        board_idx, action_idx = legal_masks.nonzero(as_tuple=True) # [B, A]
        
        # N_legal = board_idx/action_idx
        legal_board_tokens = board_tokens[board_idx] # [N_legal, FEN_LEN]
        legal_actions = action_idx.unsqueeze(1) # [N_legal, 1]
        bucket_log_scores = self._dm_forward(legal_board_tokens, legal_actions) # [N_legal, 128]
        legal_actions_scores = bucket_log_scores.exp() @ self._bucket_values # [N_legal]
        full_scores = torch.full((B, A), -float('inf'), dtype=legal_actions_scores.dtype, device=device) # [B, A] B=batch_size, A=action_size
        full_scores[board_idx, action_idx] = legal_actions_scores
        return full_scores


    def forward(self, board_tokens: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        action_scores = self._score_actions(board_tokens, legal_masks) # [B, A]
        return self.actor_head(action_scores)