"""Real-play rival: push m_final, have the rival respond, return post FEN."""
import chess
import numpy as np
import torch

from src.reasoning.model import ReasoningModel
from src.reasoning.tokens import BASE_VOCAB_SIZE, END_THINK_ID, FEN_LEN, MOVE_ID, THINK_ID
from src.searchless_chess_imports import ACTION_TO_MOVE, MOVE_TO_ACTION, tokenize


def _rival_argmax_move(model: ReasoningModel, board: chess.Board) -> chess.Move:
    """Run the model on [FEN | <think> | <end_think> | <move>] and argmax over legal moves."""
    device = next(model.parameters()).device
    fen_tokens = torch.from_numpy(np.asarray(tokenize(board.fen()), dtype=np.int64)).to(device)
    seq = torch.zeros((1, FEN_LEN + 4), dtype=torch.long, device=device)
    seq[0, :FEN_LEN] = fen_tokens
    seq[0, FEN_LEN] = THINK_ID
    seq[0, FEN_LEN + 1] = END_THINK_ID
    seq[0, FEN_LEN + 2] = MOVE_ID
    seq_lens = torch.tensor([FEN_LEN + 3], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(seq[:, : FEN_LEN + 3], seq_lens)
    logits = out.logits[0, FEN_LEN + 2, :BASE_VOCAB_SIZE]

    legal_indices: list[int] = []
    for move in board.legal_moves:
        idx = MOVE_TO_ACTION.get(move.uci())
        if idx is not None:
            legal_indices.append(idx)
    if not legal_indices:
        raise RuntimeError(f"No legal moves available for rival on {board.fen()}")

    legal_logits = logits[legal_indices]
    best_idx = legal_indices[int(torch.argmax(legal_logits).item())]
    return chess.Move.from_uci(ACTION_TO_MOVE[best_idx])


def play_final_and_respond(
    *,
    model: ReasoningModel,
    root_fen: str,
    m_final_token: int,
) -> tuple[str, bool]:
    """Push m_final, then run the rival. Return (post_fen, is_terminal)."""
    board = chess.Board(root_fen)
    board.push(chess.Move.from_uci(ACTION_TO_MOVE[m_final_token]))
    if board.is_game_over(claim_draw=True):
        return board.fen(), True
    rival_move = _rival_argmax_move(model, board)
    board.push(rival_move)
    return board.fen(), board.is_game_over(claim_draw=True)
