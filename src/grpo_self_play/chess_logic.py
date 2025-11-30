import chess
import random
import torch
from torch.utils.data import IterableDataset
from src.grpo_self_play.searchless_chess_imports import MOVE_TO_ACTION, tokenize as deepmind_tokenize


def generate_random_board(step_num=30):
    board = chess.Board()
    random_steps = random.randint(0, step_num)
    for _ in range(random_steps):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    return board


def reward_board(env, board_start):
  score = 0
  piece_vals = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
  for sq, piece in env.piece_map().items():
      val = piece_vals.get(piece.piece_type, 0)
      if piece.color == board_start.turn: # Friendly piece
          score += val
      else:
          score -= val

  # Add outcome bonus
  result = env.result()
  if result == "1-0" and board_start.turn == chess.WHITE: score += 100
  elif result == "0-1" and board_start.turn == chess.BLACK: score += 100
  return score


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


class ChessStartStatesDataset(IterableDataset):
  """
  Infinite dataset that yields random mid-game FEN strings.
  """
  def __init__(self, max_steps=10000, random_walk_gen_steps=30):
      self.max_steps = max_steps
      self.random_walk_gen_steps = random_walk_gen_steps

  def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()
      if worker_info is not None:
          # Seed workers differently to get diverse games
          random.seed(worker_info.id + random.randint(0, 10000))

      for _ in range(self.max_steps):
          board = generate_random_board(self.random_walk_gen_steps)
          if not board.is_game_over():
              yield board.fen()