"""An adapter for the DM action-value model to work as policy head and value head after porting into pytorch."""
import torch
import torch.nn as nn

from src.dm_port.transformer import DMTransformer
from src.searchless_chess_imports import get_uniform_buckets_edges_values, ACTION_DIM   

NUM_BUCKETS = 128


class AVAdapter(nn.Module):
    def __init__(self, dm_model: DMTransformer):
        super().__init__()
        self.register_buffer("bucket_values",
                             torch.tensor(get_uniform_buckets_edges_values(NUM_BUCKETS)[1], 
                                          dtype=torch.float32)) # [NUM_BUCKETS]
        self.dm_model = dm_model # the model [board, action] -> score for action

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

        dummy = torch.zeros((board_tokens.shape[0], 1), device=board_tokens.device, dtype=board_tokens.dtype) # [B, 1]
        targets = torch.cat([board_tokens, legal_actions, dummy], dim=1) # [B, FEN_LEN + 2] ( +2 because of the action + dummy)
        return self.dm_model(targets)


    def _score_actions(self, board_tokens: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        # board_tokens: [B, FEN_LEN]
        # legal_masks: [B, A]
        # return: [B, A]
        B, _ = board_tokens.shape
        A = legal_masks.shape[1]
        device = board_tokens.device
        # board_idx -> legal_actions
        board_idx, action_idx = legal_masks.nonzero(as_tuple=True) # [B, A]
        
        # N_legal = board_idx/action_idx
        legal_board_tokens = board_tokens[board_idx] # [N_legal, FEN_LEN]
        legal_actions = action_idx.unsqueeze(1) # [N_legal, 1]
        bucket_log_scores = self._dm_forward(legal_board_tokens, legal_actions)[:, -1, :] # [N_legal, 128], the dm output incluts T for the sequence
        bucket_values = self.bucket_values.to(device=bucket_log_scores.device, dtype=bucket_log_scores.dtype)
        legal_actions_scores = bucket_log_scores.exp() @ bucket_values # [N_legal]
        full_scores = torch.full((B, A), -float('inf'), dtype=legal_actions_scores.dtype, device=device) # [B, A] B=batch_size, A=action_size
        full_scores[board_idx, action_idx] = legal_actions_scores
        return full_scores


    def forward(self, board_tokens: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        return self._score_actions(board_tokens, legal_masks)



class PPOActorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_head = nn.Linear(ACTION_DIM, ACTION_DIM)


    def forward(self, action_scores: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        # scores: [B, A]
        # return: [B, A]
        scores = self.actor_head(action_scores) # [B, A]
        scores = scores.masked_fill(~legal_masks, -float('inf')) # mask out the illegal actions
        return scores # [B, A]


class PPOCriticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Linear(ACTION_DIM, 1)

    def forward(self, action_scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, A]
        # return: [B, 1]
        return self.value_head(action_scores) # [B, 1]


class PPOActorCritic(nn.Module):
    def __init__(self, av_adapter: AVAdapter):
        super().__init__()
        self.av_adapter = av_adapter
        self.actor_head = PPOActorHead()
        self.critic_head = PPOCriticHead()

    def forward(self, board_tokens: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
        action_scores = self.av_adapter(board_tokens, legal_masks) # [B, A]
        action_scores = action_scores.masked_fill(~legal_masks, -1) # Dont feed -inf to the heads
        
        actor_logits = self.actor_head(action_scores, legal_masks) # [B, A]
        value = self.critic_head(action_scores).squeeze(-1) # [B]
        return {"actor_logits": actor_logits, "value": value}