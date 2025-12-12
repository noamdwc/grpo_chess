
import random
from shutil import move
import torch
import torch.nn.functional as F
from src.grpo_self_play.chess.chess_logic import board_to_tensor, get_legal_moves_indices, action_to_move

from dataclasses import dataclass

@dataclass
class PolicyConfig:
    temperature: float = 1.0
    greedy: bool = False  # if True, pick argmax among legal moves
    branching_factor: int = 4  # for search; 0 = no limit
    search_depth: int = 2  # for search; 0 = no search


class PolicyPlayer:
    def __init__(self, model, device=None, cfg=PolicyConfig()):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.cfg = cfg
        self.stats = {"no_legal_idxs": 0, "mapping_failed": 0, "random_fallback": 0}


    @torch.no_grad()
    def act(self, board):
        legal_moves_indices = get_legal_moves_indices(board)
        if not legal_moves_indices:
            self.stats["no_legal_idxs"] += 1
            self.stats["random_fallback"] += 1
            return random.choice(list(board.legal_moves))
        return self.sample_move(board, legal_moves_indices)

    @torch.no_grad()
    def sample_move(self, board, legal_moves_indices=None):
        if legal_moves_indices is None:
            legal_moves_indices = get_legal_moves_indices(board)
        if not legal_moves_indices:
            self.stats["no_legal_idxs"] += 1
            self.stats["random_fallback"] += 1
            return random.choice(list(board.legal_moves))
        board_tensor = board_to_tensor(board, self.device)
        logits = self.model(board_tensor) # [1, A]

        A = logits.size(-1)
        masked = torch.full(
            (A,),
            -float("inf"),
            device=self.device,
            dtype=logits.dtype,
        )
        li = torch.tensor(legal_moves_indices, device=self.device, dtype=torch.long)
        masked[li] = logits[0, li]

        if self.cfg.greedy:
            action_idx = int(torch.argmax(masked).item())
        else:
            temp = max(1e-6, self.cfg.temperature)
            probs = F.softmax(masked / temp, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())
        move = action_to_move(board, action_idx)
        if move is None:
            self.stats["mapping_failed"] += 1
            self.stats["random_fallback"] += 1
            return random.choice(list(board.legal_moves))
        return move

    @torch.no_grad()
    def eval_board(self, board, root_color):
        board_tensor = board_to_tensor(board, self.device)
        legal_moves_indices = get_legal_moves_indices(board)
        if not legal_moves_indices:
             # no moves -> treat via game result if available
             outcome = board.outcome()
             if outcome is not None:
                 if outcome.winner is None:
                     return 0.0
                 return 1.0 if outcome.winner == root_color else -1.0
        
        logits = self.model(board_tensor) # [1, A]
        A = logits.size(-1)
        masked = torch.full(
            (A,),
            -float("inf"),
            device=self.device,
            dtype=logits.dtype,
        )
        li = torch.tensor(legal_moves_indices, device=self.device, dtype=torch.long)
        masked[li] = logits[-1, li]
        best_logit = float(torch.max(F.sigmoid(masked)).item())
        return best_logit if board.turn == root_color else -best_logit
