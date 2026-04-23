"""Deterministic harnesses for bare-DM vs wrapped zero-think parity debugging."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from src.dm_port.stockfish_eval import DEFAULT_DM_PORT_CHECKPOINT, load_dm_port_and_reasoning_models
from src.reasoning.tokens import BASE_VOCAB_SIZE, END_THINK_ID, FEN_LEN, MOVE_ID, THINK_ID
from src.searchless_chess_imports import ACTION_TO_MOVE, MOVE_TO_ACTION, tokenize


@dataclass(frozen=True)
class ActionSummary:
    action_id: int
    move_uci: str
    score: float
    probability: float


@dataclass(frozen=True)
class PathAnalysis:
    path_name: str
    fen: str
    side_to_move: str
    fen_token_ids: list[int]
    input_token_ids: list[int]
    legal_moves_uci: list[str]
    legal_action_ids: list[int]
    legal_mask_active_ids: list[int]
    roundtrip_violations: list[str]
    top_actions: list[ActionSummary]
    chosen_action_id: int
    chosen_move_uci: str
    decoded_move_is_legal: bool
    selection_semantics: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["top_actions"] = [asdict(entry) for entry in self.top_actions]
        return data


@dataclass(frozen=True)
class PositionComparison:
    label: str
    bare: PathAnalysis
    wrapped: PathAnalysis
    wrapped_dm_semantics: PathAnalysis

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bare": self.bare.to_dict(),
            "wrapped": self.wrapped.to_dict(),
            "wrapped_dm_semantics": self.wrapped_dm_semantics.to_dict(),
        }


@dataclass(frozen=True)
class TraceStep:
    ply_index: int
    fen_before: str
    bare_move_uci: str
    wrapped_move_uci: str
    bare_action_id: int
    wrapped_action_id: int
    bare_move_is_legal: bool
    wrapped_move_is_legal: bool
    bare_fen_after: str
    wrapped_fen_after: str
    diverged: bool


@dataclass(frozen=True)
class TraceResult:
    steps: list[TraceStep]
    first_divergence: TraceStep | None


def _bucket_values(num_return_buckets: int, *, device: torch.device) -> torch.Tensor:
    full = np.linspace(0.0, 1.0, num_return_buckets + 1, dtype=np.float32)
    return torch.from_numpy((full[:-1] + full[1:]) / 2.0).to(device)


def _load_dm_value_head(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    weight = state["output_linear.weight"].to(device=device)
    bias = state["output_linear.bias"].to(device=device)
    return weight, bias


def _mapped_legal_moves(board: chess.Board) -> list[chess.Move]:
    return sorted(
        [move for move in board.legal_moves if move.uci() in MOVE_TO_ACTION],
        key=lambda move: MOVE_TO_ACTION[move.uci()],
    )


def _roundtrip_violations(legal_moves: list[chess.Move]) -> list[str]:
    violations: list[str] = []
    for move in legal_moves:
        action_id = MOVE_TO_ACTION[move.uci()]
        decoded = ACTION_TO_MOVE[action_id]
        if decoded != move.uci():
            violations.append(f"{move.uci()}->{action_id}->{decoded}")
    return violations


def _top_actions(
    action_ids: list[int],
    scores: torch.Tensor,
    *,
    k: int = 5,
) -> list[ActionSummary]:
    if not action_ids:
        return []
    probs = torch.softmax(scores, dim=0)
    ranking = torch.argsort(scores, descending=True)[: min(k, len(action_ids))]
    top: list[ActionSummary] = []
    for idx in ranking.tolist():
        action_id = action_ids[idx]
        top.append(
            ActionSummary(
                action_id=action_id,
                move_uci=ACTION_TO_MOVE[action_id],
                score=float(scores[idx].item()),
                probability=float(probs[idx].item()),
            )
        )
    return top


def _base_position_info(board: chess.Board) -> tuple[list[int], list[str], list[int], list[str]]:
    fen_tokens = np.asarray(tokenize(board.fen()), dtype=np.int64).tolist()
    legal_moves = _mapped_legal_moves(board)
    legal_moves_uci = [move.uci() for move in legal_moves]
    legal_action_ids = [MOVE_TO_ACTION[move] for move in legal_moves_uci]
    return fen_tokens, legal_moves_uci, legal_action_ids, _roundtrip_violations(legal_moves)


@torch.no_grad()
def analyze_bare_dm_position(dm_model, board: chess.Board) -> PathAnalysis:
    device = next(dm_model.parameters()).device
    fen_tokens, legal_moves_uci, legal_action_ids, roundtrip_violations = _base_position_info(board)
    if not legal_action_ids:
        raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

    fen_tensor = torch.tensor(fen_tokens, dtype=torch.long, device=device)
    action_tokens = torch.tensor(legal_action_ids, dtype=torch.long, device=device)
    targets = fen_tensor.unsqueeze(0).repeat(len(legal_action_ids), 1)
    targets = torch.cat(
        [
            targets,
            action_tokens.unsqueeze(1),
            torch.zeros((len(legal_action_ids), 1), dtype=torch.long, device=device),
        ],
        dim=1,
    )
    log_probs = dm_model(targets)[:, -1, :]
    win_probs = torch.exp(log_probs) @ _bucket_values(dm_model.config.output_size, device=device)
    chosen_idx = int(torch.argmax(win_probs).item())
    chosen_action_id = legal_action_ids[chosen_idx]
    chosen_move = ACTION_TO_MOVE[chosen_action_id]
    return PathAnalysis(
        path_name="bare_dm",
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        fen_token_ids=fen_tokens,
        input_token_ids=fen_tokens + [chosen_action_id, 0],
        legal_moves_uci=legal_moves_uci,
        legal_action_ids=legal_action_ids,
        legal_mask_active_ids=legal_action_ids.copy(),
        roundtrip_violations=roundtrip_violations,
        top_actions=_top_actions(legal_action_ids, win_probs),
        chosen_action_id=chosen_action_id,
        chosen_move_uci=chosen_move,
        decoded_move_is_legal=chess.Move.from_uci(chosen_move) in board.legal_moves,
        selection_semantics="append each legal move token, read DM return head, argmax expected win probability",
    )


@torch.no_grad()
def analyze_wrapped_zero_think_position(reasoning_model, board: chess.Board) -> PathAnalysis:
    device = next(reasoning_model.parameters()).device
    fen_tokens, legal_moves_uci, legal_action_ids, roundtrip_violations = _base_position_info(board)
    if not legal_action_ids:
        raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

    seq = torch.zeros((1, FEN_LEN + 4), dtype=torch.long, device=device)
    seq[0, :FEN_LEN] = torch.tensor(fen_tokens, dtype=torch.long, device=device)
    seq[0, FEN_LEN] = THINK_ID
    seq[0, FEN_LEN + 1] = END_THINK_ID
    seq[0, FEN_LEN + 2] = MOVE_ID
    seq_lens = torch.tensor([FEN_LEN + 3], dtype=torch.long, device=device)
    out = reasoning_model(seq[:, : FEN_LEN + 3], seq_lens)
    logits = out.logits[0, FEN_LEN + 2, :BASE_VOCAB_SIZE]
    legal_logits = logits[legal_action_ids]
    chosen_idx = int(torch.argmax(legal_logits).item())
    chosen_action_id = legal_action_ids[chosen_idx]
    chosen_move = ACTION_TO_MOVE[chosen_action_id]
    return PathAnalysis(
        path_name="wrapped_zero_think",
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        fen_token_ids=fen_tokens,
        input_token_ids=fen_tokens + [THINK_ID, END_THINK_ID, MOVE_ID],
        legal_moves_uci=legal_moves_uci,
        legal_action_ids=legal_action_ids,
        legal_mask_active_ids=legal_action_ids.copy(),
        roundtrip_violations=roundtrip_violations,
        top_actions=_top_actions(legal_action_ids, legal_logits),
        chosen_action_id=chosen_action_id,
        chosen_move_uci=chosen_move,
        decoded_move_is_legal=chess.Move.from_uci(chosen_move) in board.legal_moves,
        selection_semantics="run [FEN|<think>|<end_think>|<move>], read policy logits at <move>, argmax legal action token",
    )


@torch.no_grad()
def analyze_wrapped_dm_semantics_position(
    reasoning_model,
    board: chess.Board,
    *,
    checkpoint_path: str | Path = DEFAULT_DM_PORT_CHECKPOINT,
) -> PathAnalysis:
    device = next(reasoning_model.parameters()).device
    dm_head_weight, dm_head_bias = _load_dm_value_head(checkpoint_path, device=device)
    fen_tokens, legal_moves_uci, legal_action_ids, roundtrip_violations = _base_position_info(board)
    if not legal_action_ids:
        raise RuntimeError(f"No mapped legal moves available for {board.fen()}")

    targets = torch.tensor(fen_tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(len(legal_action_ids), 1)
    action_tokens = torch.tensor(legal_action_ids, dtype=torch.long, device=device)
    targets = torch.cat(
        [
            targets,
            action_tokens.unsqueeze(1),
            torch.zeros((len(legal_action_ids), 1), dtype=torch.long, device=device),
        ],
        dim=1,
    )
    hidden = reasoning_model.dm.embed_inputs(targets)
    for block in reasoning_model.dm.layers:
        hidden = block(hidden, attn_mask=None)
    if reasoning_model.dm.post_ln is not None:
        hidden = reasoning_model.dm.post_ln(hidden)
    logits = F.linear(hidden, dm_head_weight, dm_head_bias)
    log_probs = F.log_softmax(logits, dim=-1)[:, -1, :]
    win_probs = torch.exp(log_probs) @ _bucket_values(logits.shape[-1], device=device)
    chosen_idx = int(torch.argmax(win_probs).item())
    chosen_action_id = legal_action_ids[chosen_idx]
    chosen_move = ACTION_TO_MOVE[chosen_action_id]
    return PathAnalysis(
        path_name="wrapped_dm_semantics",
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        fen_token_ids=fen_tokens,
        input_token_ids=fen_tokens + [chosen_action_id, 0],
        legal_moves_uci=legal_moves_uci,
        legal_action_ids=legal_action_ids,
        legal_mask_active_ids=legal_action_ids.copy(),
        roundtrip_violations=roundtrip_violations,
        top_actions=_top_actions(legal_action_ids, win_probs),
        chosen_action_id=chosen_action_id,
        chosen_move_uci=chosen_move,
        decoded_move_is_legal=chess.Move.from_uci(chosen_move) in board.legal_moves,
        selection_semantics="reuse wrapped transformer body with the frozen DM return head on [FEN|move|0]",
    )


def compare_position(
    dm_model,
    reasoning_model,
    board: chess.Board,
    *,
    label: str,
    checkpoint_path: str | Path = DEFAULT_DM_PORT_CHECKPOINT,
) -> PositionComparison:
    return PositionComparison(
        label=label,
        bare=analyze_bare_dm_position(dm_model, board),
        wrapped=analyze_wrapped_zero_think_position(reasoning_model, board),
        wrapped_dm_semantics=analyze_wrapped_dm_semantics_position(
            reasoning_model,
            board,
            checkpoint_path=checkpoint_path,
        ),
    )


def _board_after(*moves: str) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board


def build_fixed_fen_corpus() -> list[tuple[str, str]]:
    positions = [
        ("start", chess.STARTING_FEN),
        ("after_e4", _board_after("e2e4").fen()),
        ("open_game", _board_after("e2e4", "e7e5").fen()),
        ("italian_shape", _board_after("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5").fen()),
        ("queen_gambit_declined", _board_after("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6").fen()),
        ("sicilian_open", _board_after("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4").fen()),
        ("caro_kann", _board_after("e2e4", "c7c6", "d2d4", "d7d5", "b1c3").fen()),
        ("french_advance", _board_after("e2e4", "e7e6", "d2d4", "d7d5", "e4e5").fen()),
        ("kings_indian_shape", _board_after("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7").fen()),
        ("rook_castles_both", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        ("white_castles_only", "4k2r/8/8/8/8/8/8/R3K2R w KQ - 0 1"),
        ("black_castles_only", "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"),
        ("en_passant_white", "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"),
        ("en_passant_black", "rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 2"),
        ("white_promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
        ("black_promotion", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
        ("white_in_check", "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1"),
        ("black_in_check", "4k3/4R3/8/8/8/8/8/4K3 b - - 0 1"),
        ("tactical_midgame", "r1bq1rk1/pp2bppp/2n2n2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 9"),
        ("queenside_pressure", "2rq1rk1/pp2bppp/2n1pn2/2bp4/3P4/2N1PN2/PPQ1BPPP/2RR2K1 w - - 0 12"),
        ("minor_piece_endgame", "8/2k5/3p4/3P4/2K5/8/8/8 w - - 0 1"),
        ("rook_endgame", "8/8/3k4/8/3K4/8/4R3/8 w - - 0 1"),
        ("mate_net_defense", "6k1/5ppp/8/8/8/5Q2/5PPP/6K1 b - - 0 1"),
        ("imbalanced_center", "rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 0 7"),
    ]
    validated: list[tuple[str, str]] = []
    for label, fen in positions:
        chess.Board(fen)
        validated.append((label, fen))
    return validated


def run_fen_sweep(
    dm_model,
    reasoning_model,
    *,
    checkpoint_path: str | Path = DEFAULT_DM_PORT_CHECKPOINT,
) -> list[PositionComparison]:
    return [
        compare_position(
            dm_model,
            reasoning_model,
            chess.Board(fen),
            label=label,
            checkpoint_path=checkpoint_path,
        )
        for label, fen in build_fixed_fen_corpus()
    ]


def trace_first_divergence(
    dm_model,
    reasoning_model,
    *,
    checkpoint_path: str | Path = DEFAULT_DM_PORT_CHECKPOINT,
    start_fen: str = chess.STARTING_FEN,
    max_plies: int = 20,
) -> TraceResult:
    bare_board = chess.Board(start_fen)
    wrapped_board = chess.Board(start_fen)
    steps: list[TraceStep] = []

    for ply_index in range(max_plies):
        if bare_board.is_game_over(claim_draw=True) or wrapped_board.is_game_over(claim_draw=True):
            break
        if bare_board.fen() != wrapped_board.fen():
            raise AssertionError("Boards drifted before the first divergence step was recorded")
        comparison = compare_position(
            dm_model,
            reasoning_model,
            bare_board,
            label=f"ply_{ply_index}",
            checkpoint_path=checkpoint_path,
        )
        bare_move = chess.Move.from_uci(comparison.bare.chosen_move_uci)
        wrapped_move = chess.Move.from_uci(comparison.wrapped.chosen_move_uci)

        bare_after = bare_board.copy(stack=False)
        bare_after.push(bare_move)
        wrapped_after = wrapped_board.copy(stack=False)
        wrapped_after.push(wrapped_move)

        step = TraceStep(
            ply_index=ply_index,
            fen_before=bare_board.fen(),
            bare_move_uci=comparison.bare.chosen_move_uci,
            wrapped_move_uci=comparison.wrapped.chosen_move_uci,
            bare_action_id=comparison.bare.chosen_action_id,
            wrapped_action_id=comparison.wrapped.chosen_action_id,
            bare_move_is_legal=bare_move in bare_board.legal_moves,
            wrapped_move_is_legal=wrapped_move in wrapped_board.legal_moves,
            bare_fen_after=bare_after.fen(),
            wrapped_fen_after=wrapped_after.fen(),
            diverged=comparison.bare.chosen_action_id != comparison.wrapped.chosen_action_id,
        )
        steps.append(step)
        if step.diverged:
            return TraceResult(steps=steps, first_divergence=step)

        bare_board = bare_after
        wrapped_board = wrapped_after

    return TraceResult(steps=steps, first_divergence=None)


def load_debug_models(
    *,
    checkpoint_path: str | Path = DEFAULT_DM_PORT_CHECKPOINT,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(seed)
    return load_dm_port_and_reasoning_models(checkpoint_path, device=device)
