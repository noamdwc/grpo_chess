"""Utilities for evaluating chess policies against Stockfish."""
import io
import math
import chess
import chess.pgn
import chess.engine
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.chess.chess_logic import MOVE_TO_ACTION
from src.chess.policy_player import PolicyPlayer, PolicyConfig
from src.chess.searcher import TrajectorySearcher, SearchConfig
from src.chess.stockfish import StockfishPlayer, StockfishConfig, StockfishManager, DEFAULT_STOCKFISH_PATH as STOCKFISH_PATH


@dataclass
class EvalConfig:
    games: int = 50
    seed: int = 0
    max_plies: int = 400  # safety to avoid extremely long games
    randomize_opening: bool = False
    opening_plies: int = 6  # random legal moves to diversify early positions
    num_workers: int = 1  # number of parallel workers for game play


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([EvalConfig])


def debug_legal_coverage(board: chess.Board) -> tuple[int, int, list[str]]:
    """Debug function to check coverage of legal moves in action space.
    
    Args:
        board: Chess board position
        
    Returns:
        Tuple of (covered_count, total_legal_moves, list_of_missing_moves)
    """
    legals = list(board.legal_moves)
    covered = 0
    missing = []
    for mv in legals:
        u = mv.uci()
        if u in MOVE_TO_ACTION:
            covered += 1
        else:
            missing.append(u)
    return covered, len(legals), missing[:10]




     
def play_one_game(
    policy: PolicyPlayer | TrajectorySearcher,
    stockfish: StockfishPlayer,
    policy_is_white: bool,
    cfg: EvalConfig,
    game_number: int = 0,
    rng: random.Random | None = None,
) -> Tuple[str, str, str]:
    """Play a single game between policy and Stockfish.

    Args:
        policy: Policy player to evaluate
        stockfish: Stockfish player
        policy_is_white: Whether policy plays as white
        cfg: Evaluation configuration
        game_number: Game number for PGN metadata
        rng: Optional Random instance for thread-safe randomness.
             If None, uses the global random module.

    Returns:
        Tuple of (result_str, termination_reason, pgn_str)
        result_str in {"1-0", "0-1", "1/2-1/2"}
    """
    _choice = rng.choice if rng is not None else random.choice

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Policy vs Stockfish Evaluation"
    game.headers["White"] = "Policy" if policy_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if policy_is_white else "Policy"
    game.headers["Round"] = str(game_number + 1)
    node = game

    # Optional random opening to reduce overfitting to a single line
    if cfg.randomize_opening and cfg.opening_plies > 0:
        for _ in range(cfg.opening_plies):
            if board.is_game_over():
                break
            move = _choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)

    for ply in range(cfg.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        is_white_to_move = board.turn
        policy_turn = (is_white_to_move and policy_is_white) or ((not is_white_to_move) and (not policy_is_white))

        if policy_turn:
            move = policy.act(board)
        else:
            move = stockfish.act(board)
        if move is None:
            break  # no legal moves

        board.push(move)
        node = node.add_variation(move)

    # Determine result
    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        reason = "game_over"
    else:
        # Reached max plies: treat as draw
        res = "1/2-1/2"
        reason = "max_plies"

    game.headers["Result"] = res

    # Generate PGN string
    pgn_output = io.StringIO()
    exporter = chess.pgn.FileExporter(pgn_output)
    game.accept(exporter)
    pgn_str = pgn_output.getvalue()

    return res, reason, pgn_str


def estimate_elo_diff(score: float) -> float:
    """Estimate Elo difference from match score.
    
    Uses logistic model: S = 1/(1+10^(-d/400)) => d = -400*log10(1/S - 1)
    Clamped for numeric stability.
    
    Args:
        score: Win rate score in [0, 1]
        
    Returns:
        Estimated Elo difference
    """
    eps = 1e-6
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def _aggregate_results(
    game_results: List[Tuple[int, str, str, str]],
    eval_cfg: EvalConfig,
    policy: PolicyPlayer | TrajectorySearcher,
) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher, List[str]]:
    """Aggregate individual game results into summary statistics.

    Args:
        game_results: List of (game_number, result_str, reason, pgn_str)
        eval_cfg: Evaluation configuration
        policy: Policy player (returned as-is)

    Returns:
        Same format as evaluate_policy_vs_stockfish
    """
    wins = draws = losses = 0
    term_reasons: Dict[str, int] = {}
    pgns: List[str] = []

    # Sort by game number to maintain deterministic PGN ordering
    game_results.sort(key=lambda x: x[0])

    for g, res, reason, pgn in game_results:
        policy_is_white = (g % 2 == 0)
        term_reasons[reason] = term_reasons.get(reason, 0) + 1
        pgns.append(pgn)

        if res == "1-0":
            if policy_is_white:
                wins += 1
            else:
                losses += 1
        elif res == "0-1":
            if policy_is_white:
                losses += 1
            else:
                wins += 1
        else:
            draws += 1

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total else 0.0
    elo_diff = estimate_elo_diff(score) if total else 0.0

    return {
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff_vs_stockfish_approx": elo_diff,
        "termination_reasons": term_reasons,
        "eval_cfg": eval_cfg,
    }, policy, pgns


def evaluate_policy_vs_stockfish(
    policy: PolicyPlayer | TrajectorySearcher,
    sf: StockfishPlayer,
    eval_cfg: EvalConfig,
) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher, List[str]]:
    """Evaluate a policy by playing multiple games against Stockfish.

    Args:
        policy: Policy player to evaluate
        sf: Stockfish player
        eval_cfg: Evaluation configuration

    Returns:
        Tuple of (results_dict, policy_player, pgns)
        results_dict contains: games, wins, draws, losses, score, elo_diff, etc.
        pgns is a list of PGN strings for all games played
    """
    if eval_cfg.num_workers > 1:
        return _evaluate_parallel(policy, sf, eval_cfg)
    return _evaluate_sequential(policy, sf, eval_cfg)


def _evaluate_sequential(
    policy: PolicyPlayer | TrajectorySearcher,
    sf: StockfishPlayer,
    eval_cfg: EvalConfig,
) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher, List[str]]:
    """Sequential evaluation (original behavior)."""
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)

    game_results = []
    try:
        for g in range(eval_cfg.games):
            policy_is_white = (g % 2 == 0)
            res, reason, pgn = play_one_game(policy, sf, policy_is_white, eval_cfg, game_number=g)
            game_results.append((g, res, reason, pgn))
    finally:
        sf.close()

    return _aggregate_results(game_results, eval_cfg, policy)


def _evaluate_parallel(
    policy: PolicyPlayer | TrajectorySearcher,
    sf: StockfishPlayer,
    eval_cfg: EvalConfig,
) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher, List[str]]:
    """Parallel evaluation using ThreadPoolExecutor.

    Each worker thread gets its own Stockfish engine. The policy player
    is shared across threads (PyTorch inference is thread-safe).
    """
    sf_cfg = sf.cfg
    sf.close()  # Close original; workers create their own

    engine_names_lock = threading.Lock()
    engine_names: set = set()

    def play_game(g: int) -> Tuple[int, str, str, str]:
        # Per-thread Stockfish engine (reused across games on same thread)
        ename = f"eval_parallel_{threading.current_thread().ident}"
        with engine_names_lock:
            engine_names.add(ename)
        sf_player = StockfishPlayer(sf_cfg, engine_name=ename)

        # Per-game deterministic seed for opening randomization
        rng = random.Random(eval_cfg.seed + g)
        policy_is_white = (g % 2 == 0)
        res, reason, pgn = play_one_game(
            policy, sf_player, policy_is_white, eval_cfg, game_number=g, rng=rng,
        )
        return g, res, reason, pgn

    try:
        with ThreadPoolExecutor(max_workers=eval_cfg.num_workers) as executor:
            game_results = list(executor.map(play_game, range(eval_cfg.games)))
    finally:
        for ename in engine_names:
            StockfishManager.close(ename)

    return _aggregate_results(game_results, eval_cfg, policy)
    