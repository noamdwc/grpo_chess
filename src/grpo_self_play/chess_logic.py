import chess
import random
import torch
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