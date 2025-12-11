
import random
import torch
import torch.nn.functional as F
from src.grpo_self_play.chess.chess_logic import board_to_tensor, get_legal_moves_indices, action_to_move

from dataclasses import dataclass

@dataclass
class PolicyConfig:
    temperature: float = 1.0
    greedy: bool = False  # if True, pick argmax among legal moves


class PolicyPlayer:
    def __init__(self, model, device=None, cfg=PolicyConfig()):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.cfg = cfg
        self.stats = {"no_legal_idxs": 0, "mapping_failed": 0, "random_fallback": 0}

    @torch.no_grad()
    def choose_move(self, board):
        legal_idxs = get_legal_moves_indices(board)
        if not legal_idxs:
            self.stats["no_legal_idxs"] += 1
            self.stats["random_fallback"] += 1
            return random.choice(list(board.legal_moves))

        x = board_to_tensor(board, self.device)
        logits = self.model(x)

        A = logits.size(-1)
        masked = torch.full((A,), -float("inf"), device=self.device, dtype=logits.dtype)
        li = torch.tensor(legal_idxs, device=self.device, dtype=torch.long)
        masked[li] = logits[0, li]

        if self.cfg.greedy:
            action_idx = int(torch.argmax(masked).item())
        else:
            temp = max(self.cfg.temperature, 1e-6)
            probs = F.softmax(masked / temp, dim=-1)
            action_idx = int(torch.multinomial(probs, 1).item())

        mv = action_to_move(board, action_idx)
        if mv is None:
            self.stats["mapping_failed"] += 1
            # try best legal by score
            sorted_legal = sorted(legal_idxs, key=lambda i: float(masked[i].item()), reverse=True)
            for i in sorted_legal:
                mv2 = action_to_move(board, i)
                if mv2 is not None:
                    return mv2
            self.stats["random_fallback"] += 1
            return random.choice(list(board.legal_moves))

        return mv