'''
Implement search method to choose moves based on a policy network.
'''

import chess
import torch

from typing import Optional
from dataclasses import dataclass
from src.chess.chess_logic import ChessPlayer
from src.chess.policy_player import PolicyPlayer

@dataclass
class SearchConfig:
    n_trajectories: int = 1   # G: number of sampled trajectories
    trajectory_depth: int = 1 # T: max plies per trajectory


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([SearchConfig])


class TrajectorySearcher(ChessPlayer):
    """
    Searcher that uses a PolicyPlayer to:   
      - sample trajectories using the policy
      - evaluate their final states using the policy
    and picks the first move of the best-scoring trajectory.
    """

    def __init__(self, policy: PolicyPlayer, cfg: SearchConfig = SearchConfig()):
        self.policy = policy
        self.cfg = cfg


    @torch.no_grad()
    def act(self, board: chess.Board) -> Optional[chess.Move]:
        """
        If n_trajectories or trajectory_depth <= 1:
          Just use the policy's one-step act() (no search).

        Otherwise:
          Sample G trajectories, score each by final state,
          pick first move of best trajectory.
        """
        if self.cfg.n_trajectories <= 1 or self.cfg.trajectory_depth <= 1:
            return self.policy.act(board)

        root_color = board.turn
        best_score = -float("inf")
        best_first_move = None

        for g in range(self.cfg.n_trajectories):
            rollout_board = board.copy()

            first_move = None
            for step in range(self.cfg.trajectory_depth):
                if rollout_board.is_game_over():
                    break

                mv = self.policy.sample_move(rollout_board)
                if mv is None:
                    # no move available -> end trajectory
                    break

                if first_move is None:
                    first_move = mv

                rollout_board.push(mv)

            if first_move is None:
                # This trajectory failed to get any move (should be rare)
                continue

            score = self.policy.eval_board(rollout_board, root_color)

            if score > best_score:
                best_score = score
                best_first_move = first_move

        if best_first_move is None:
            # Fallback to simple 1-step policy
            return self.policy.act(board)

        return best_first_move


    @property
    def stats(self) -> dict:
        return self.policy.stats