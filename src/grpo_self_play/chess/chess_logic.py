import chess
import random
import torch
from collections import deque

from typing import Optional, Dict
from torch.utils.data import IterableDataset
from src.grpo_self_play.searchless_chess_imports import (MOVE_TO_ACTION, 
                                                         ACTION_TO_MOVE,
                                                         tokenize as deepmind_tokenize)
from src.grpo_self_play.chess.rewards import evaluate_fen

MAX_ACTION = max(ACTION_TO_MOVE.keys())


def board_to_tensor(board, device: str | torch.device ='cpu') -> torch.Tensor:
  fen = board.fen()
  token_ids = deepmind_tokenize(fen) # Returns list of ints
  input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
  return input_tensor


def get_legal_moves_indices(board):
  legal_moves = list(board.legal_moves)
  legal_indices = []
  for move in legal_moves:
    # move.uci() returns "e2e4" or "a7a8q" which matches your dict keys
    uci_str = move.uci()
    if uci_str in MOVE_TO_ACTION:
        legal_indices.append(MOVE_TO_ACTION[uci_str])
    else:
        # Fallback: unlikely if MOVE_TO_ACTION is complete
        raise ValueError(f"Invalid move: {uci_str}")
  return legal_indices


def get_legal_moves_mask(board, device: str | torch.device ='cpu') -> torch.Tensor:
    legal_moves = list(board.legal_moves)
    mask = torch.zeros(MAX_ACTION + 1, dtype=torch.bool)
    for move in legal_moves:
        uci_str = move.uci()
        action_idx = MOVE_TO_ACTION.get(uci_str)
        if action_idx is not None:
            mask[action_idx] = True
    return mask.to(device)


def action_to_move(board: chess.Board, action_idx: int):
    uci = ACTION_TO_MOVE.get(action_idx)
    if uci is None:
        return None
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return mv if mv in board.legal_moves else None


class ChessPlayer:
    """
    An abstract chess player interface.
    """
    def act(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Given a chess.Board, return a chess.Move or None to resign.
        """
        raise NotImplementedError()
