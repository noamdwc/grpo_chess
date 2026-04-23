"""Deterministic parity checks for bare BC inference vs wrapped zero-think inference."""
from __future__ import annotations

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from src.dm_port.convert_behavioral_cloning import convert as convert_bc_checkpoint
from src.dm_port.stockfish_eval import load_dm_port_and_reasoning_models
from src.reasoning.tokens import BASE_VOCAB_SIZE, END_THINK_ID, FEN_LEN, MOVE_ID, THINK_ID
from src.searchless_chess_imports import ACTION_TO_MOVE, MOVE_TO_ACTION, tokenize
from tests.debug_zero_think_parity import build_fixed_fen_corpus


CONVERTED_BC_CKPT = Path("checkpoints/dm_port/9M_behavioral_cloning.pt")
UPSTREAM_BC_CKPT = Path("searchless_chess/checkpoints/9M_behavioral_cloning")


def _mapped_legal_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(
        [move for move in board.legal_moves if move.uci() in MOVE_TO_ACTION],
        key=lambda move: MOVE_TO_ACTION[move.uci()],
    )


@torch.no_grad()
def _bare_bc_choice(model, board: chess.Board) -> tuple[int, str]:
    device = next(model.parameters()).device
    legal_moves = _mapped_legal_moves(board)
    if not legal_moves:
        raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

    fen_tokens = torch.tensor(np.asarray(tokenize(board.fen()), dtype=np.int64), dtype=torch.long, device=device)
    seq = torch.zeros((1, FEN_LEN + 1), dtype=torch.long, device=device)
    seq[0, :FEN_LEN] = fen_tokens
    logits = model(seq)[0, FEN_LEN, :BASE_VOCAB_SIZE]

    legal_action_ids = [MOVE_TO_ACTION[move.uci()] for move in legal_moves]
    legal_logits = logits[legal_action_ids]
    chosen_idx = int(torch.argmax(legal_logits).item())
    chosen_action_id = legal_action_ids[chosen_idx]
    return chosen_action_id, ACTION_TO_MOVE[chosen_action_id]


@torch.no_grad()
def _wrapped_zero_think_choice(model, board: chess.Board) -> tuple[int, str]:
    device = next(model.parameters()).device
    legal_moves = _mapped_legal_moves(board)
    if not legal_moves:
        raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

    fen_tokens = torch.tensor(np.asarray(tokenize(board.fen()), dtype=np.int64), dtype=torch.long, device=device)
    seq = torch.zeros((1, FEN_LEN + 4), dtype=torch.long, device=device)
    seq[0, :FEN_LEN] = fen_tokens
    seq[0, FEN_LEN] = THINK_ID
    seq[0, FEN_LEN + 1] = END_THINK_ID
    seq[0, FEN_LEN + 2] = MOVE_ID
    seq_lens = torch.tensor([FEN_LEN + 3], dtype=torch.long, device=device)
    logits = model(seq[:, : FEN_LEN + 3], seq_lens).logits[0, FEN_LEN + 2, :BASE_VOCAB_SIZE]

    legal_action_ids = [MOVE_TO_ACTION[move.uci()] for move in legal_moves]
    legal_logits = logits[legal_action_ids]
    chosen_idx = int(torch.argmax(legal_logits).item())
    chosen_action_id = legal_action_ids[chosen_idx]
    return chosen_action_id, ACTION_TO_MOVE[chosen_action_id]


@pytest.fixture(scope="session")
def bc_checkpoint_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if CONVERTED_BC_CKPT.exists():
        return CONVERTED_BC_CKPT
    if not UPSTREAM_BC_CKPT.exists():
        pytest.skip("Behavioral-cloning checkpoint missing")

    out = tmp_path_factory.mktemp("bc_ckpt") / "9M_behavioral_cloning.pt"
    convert_bc_checkpoint(str(UPSTREAM_BC_CKPT.resolve()), str(out))
    return out


def test_single_fen_comparison_exposes_bc_wrapper_divergence(bc_checkpoint_path: Path):
    torch.manual_seed(0)
    bc_model, reasoning_model = load_dm_port_and_reasoning_models(bc_checkpoint_path)
    board = chess.Board()

    bare_action_id, bare_move = _bare_bc_choice(bc_model, board)
    wrapped_action_id, wrapped_move = _wrapped_zero_think_choice(reasoning_model, board)

    legal_moves_uci = [move.uci() for move in _mapped_legal_moves(board)]
    assert bare_move in legal_moves_uci
    assert wrapped_move in legal_moves_uci
    assert bare_move == "e2e4"
    assert wrapped_move == "d2d3"
    assert bare_action_id != wrapped_action_id


def test_fen_sweep_shows_bc_wrapper_has_no_position_parity(bc_checkpoint_path: Path):
    torch.manual_seed(0)
    bc_model, reasoning_model = load_dm_port_and_reasoning_models(bc_checkpoint_path)

    mismatches = 0
    for _, fen in build_fixed_fen_corpus():
        board = chess.Board(fen)
        bare_action_id, bare_move = _bare_bc_choice(bc_model, board)
        wrapped_action_id, wrapped_move = _wrapped_zero_think_choice(reasoning_model, board)
        assert chess.Move.from_uci(bare_move) in board.legal_moves
        assert chess.Move.from_uci(wrapped_move) in board.legal_moves
        if bare_action_id != wrapped_action_id:
            mismatches += 1

    assert mismatches == len(build_fixed_fen_corpus())


def test_fixed_game_trace_diverges_on_first_ply_for_bc_wrapper(bc_checkpoint_path: Path):
    torch.manual_seed(0)
    bc_model, reasoning_model = load_dm_port_and_reasoning_models(bc_checkpoint_path)
    bare_board = chess.Board()
    wrapped_board = chess.Board()

    bare_action_id, bare_move = _bare_bc_choice(bc_model, bare_board)
    wrapped_action_id, wrapped_move = _wrapped_zero_think_choice(reasoning_model, wrapped_board)

    assert bare_board.fen() == wrapped_board.fen()
    assert bare_move == "e2e4"
    assert wrapped_move == "d2d3"
    assert bare_action_id != wrapped_action_id

    bare_after = bare_board.copy(stack=False)
    wrapped_after = wrapped_board.copy(stack=False)
    bare_after.push(chess.Move.from_uci(bare_move))
    wrapped_after.push(chess.Move.from_uci(wrapped_move))
    assert bare_after.fen() != wrapped_after.fen()
