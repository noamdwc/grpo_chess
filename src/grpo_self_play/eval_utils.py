'''
This code is for evluating a chess bot using stockfish
'''
import math
from turtle import st
import chess
import chess.engine
import random

import torch

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.grpo_self_play.chess.chess_logic import MOVE_TO_ACTION
from src.grpo_self_play.chess.policy_player import PolicyPlayer, PolicyConfig
from src.grpo_self_play.chess.searcher import TrajectorySearcher, SearchConfig
from src.grpo_self_play.chess.stockfish import StockfishPlayer, StockfishConfig, DEFAULT_STOCKFISH_PATH as STOCKFISH_PATH


@dataclass
class EvalConfig:
    games: int = 50
    seed: int = 0
    max_plies: int = 400  # safety to avoid extremely long games
    randomize_opening: bool = False
    opening_plies: int = 6  # random legal moves to diversify early positions


def debug_legal_coverage(board):
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
    cfg: EvalConfig
) -> Tuple[str, str]:
    """
    Returns (result_str, termination_reason)
    result_str in {"1-0","0-1","1/2-1/2"}
    """

    board = chess.Board()

    # Optional random opening to reduce overfitting to a single line
    if cfg.randomize_opening and cfg.opening_plies > 0:
        for _ in range(cfg.opening_plies):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

    for ply in range(cfg.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        is_white_to_move = board.turn
        policy_turn = (is_white_to_move and policy_is_white) or ((not is_white_to_move) and (not policy_is_white))

        if policy_turn:
            move = policy.act(board)
        else:
            move = stockfish.act(board)
        if move is None: break  # no legal moves
        
        board.push(move)

    # Determine result
    if board.is_game_over(claim_draw=True):
        res = board.result(claim_draw=True)
        return res, "game_over"
    else:
        # Reached max plies: treat as draw
        return "1/2-1/2", "max_plies"


def estimate_elo_diff(score: float) -> float:
    """
    Rough Elo difference estimate from match score S in [0,1].
    Uses logistic model: S = 1/(1+10^(-d/400))  => d = -400*log10(1/S - 1)
    Clamped for numeric stability.
    """
    eps = 1e-6
    s = min(max(score, eps), 1 - eps)
    return -400.0 * math.log10(1.0 / s - 1.0)


def evaluate_policy_vs_stockfish(
    policy: PolicyPlayer | TrajectorySearcher,
    sf: StockfishPlayer,
    eval_cfg: EvalConfig,
) -> Tuple[Dict, PolicyPlayer | TrajectorySearcher]:
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)

    wins = draws = losses = 0
    term_reasons = {}

    try:
        for g in range(eval_cfg.games):
            policy_is_white = (g % 2 == 0)
            res, reason = play_one_game(policy, sf, policy_is_white, eval_cfg)
            term_reasons[reason] = term_reasons.get(reason, 0) + 1

            # From policy perspective
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

    finally:
        sf.close()

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
    }, policy


def eval_ladder(model, device):
    results = {}
    for skill in [1, 3, 5, 8, 10]:
        stockfish_cfg = StockfishConfig(
            path=STOCKFISH_PATH,
            skill_level=skill,
            movetime_ms=20,
        )
        eval_cfg = EvalConfig(
            games=50,
            seed=0,
            randomize_opening=False
        )
        policy = PolicyPlayer(model=model, 
                              device=device,
                              cfg=PolicyConfig(temperature=0.3, greedy=False))
        searcher_policy = TrajectorySearcher(policy=policy,
                                             cfg=SearchConfig(n_trajectories=8,
                                                              trajectory_depth=2))
        stockfish_player = StockfishPlayer(StockfishConfig(path=STOCKFISH_PATH,
                                                           skill_level=skill,
                                                           movetime_ms=20))

        r, policy = evaluate_policy_vs_stockfish(searcher_policy,
                                                 stockfish_player,
                                                 eval_cfg)
        results[skill] = r["score"]
        print("skill", skill, r)
        print('policy stats', policy.stats)
    return results
