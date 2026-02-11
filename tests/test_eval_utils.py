"""Tests for eval_utils: play_one_game, aggregation, sequential & parallel evaluation."""
import os
import shutil
import math
import random
from unittest.mock import MagicMock, patch

import pytest
import chess

from src.eval_utils import (
    EvalConfig,
    play_one_game,
    evaluate_policy_vs_stockfish,
    estimate_elo_diff,
    _aggregate_results,
)
from src.chess.stockfish import StockfishPlayer, StockfishConfig, StockfishManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_stockfish_path() -> str | None:
    paths_to_try = [
        shutil.which("stockfish"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ]
    for path in paths_to_try:
        if path and os.path.isfile(path):
            return path
    return None


def _get_test_sf_config() -> StockfishConfig:
    path = get_stockfish_path()
    if path is None:
        raise RuntimeError("Stockfish not found")
    return StockfishConfig(path=path, skill_level=0, movetime_ms=10)


class RandomPlayer:
    """Minimal player that picks random legal moves (no torch needed)."""
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.stats = {"random_fallback": 0}

    def act(self, board: chess.Board) -> chess.Move | None:
        legals = list(board.legal_moves)
        if not legals:
            return None
        return self._rng.choice(legals)


@pytest.fixture(scope="module", autouse=True)
def require_stockfish():
    if get_stockfish_path() is None:
        pytest.skip("Stockfish not found on system")


@pytest.fixture(autouse=True)
def cleanup_engines():
    yield
    StockfishManager.close_all()


# ---------------------------------------------------------------------------
# estimate_elo_diff
# ---------------------------------------------------------------------------

class TestEstimateEloDiff:
    def test_even_score(self):
        assert estimate_elo_diff(0.5) == pytest.approx(0.0, abs=0.1)

    def test_high_score_positive(self):
        assert estimate_elo_diff(0.75) > 0

    def test_low_score_negative(self):
        assert estimate_elo_diff(0.25) < 0

    def test_symmetry(self):
        assert estimate_elo_diff(0.75) == pytest.approx(-estimate_elo_diff(0.25), abs=0.1)

    def test_extreme_clamp(self):
        # Should not raise for 0.0 or 1.0
        assert math.isfinite(estimate_elo_diff(0.0))
        assert math.isfinite(estimate_elo_diff(1.0))


# ---------------------------------------------------------------------------
# _aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def _make_policy(self):
        return MagicMock()

    def test_all_wins_as_white(self):
        # Even game numbers => policy is white. "1-0" => white wins => policy win
        results = [(g, "1-0", "game_over", "pgn") for g in range(0, 6, 2)]
        cfg = EvalConfig(games=3)
        out, _, pgns = _aggregate_results(results, cfg, self._make_policy())
        assert out["wins"] == 3
        assert out["draws"] == 0
        assert out["losses"] == 0
        assert out["score"] == 1.0
        assert len(pgns) == 3

    def test_all_losses_as_black(self):
        # Odd game numbers => policy is black. "1-0" => white wins => policy loses
        results = [(g, "1-0", "game_over", "pgn") for g in range(1, 6, 2)]
        cfg = EvalConfig(games=3)
        out, _, _ = _aggregate_results(results, cfg, self._make_policy())
        assert out["wins"] == 0
        assert out["losses"] == 3

    def test_all_draws(self):
        results = [(g, "1/2-1/2", "game_over", "pgn") for g in range(4)]
        cfg = EvalConfig(games=4)
        out, _, _ = _aggregate_results(results, cfg, self._make_policy())
        assert out["draws"] == 4
        assert out["score"] == 0.5

    def test_mixed_results(self):
        results = [
            (0, "1-0", "game_over", "pgn"),   # white win => policy win
            (1, "0-1", "game_over", "pgn"),    # black wins => policy win (policy is black)
            (2, "0-1", "game_over", "pgn"),    # black wins => policy loss (policy is white)
            (3, "1/2-1/2", "max_plies", "pgn"),
        ]
        cfg = EvalConfig(games=4)
        out, _, _ = _aggregate_results(results, cfg, self._make_policy())
        assert out["wins"] == 2
        assert out["losses"] == 1
        assert out["draws"] == 1
        assert out["score"] == pytest.approx(2.5 / 4)

    def test_ordering_by_game_number(self):
        # Results arrive out of order (as in parallel); PGNs should be sorted
        results = [
            (2, "1-0", "game_over", "pgn_2"),
            (0, "0-1", "game_over", "pgn_0"),
            (1, "1/2-1/2", "game_over", "pgn_1"),
        ]
        cfg = EvalConfig(games=3)
        _, _, pgns = _aggregate_results(results, cfg, self._make_policy())
        assert pgns == ["pgn_0", "pgn_1", "pgn_2"]

    def test_termination_reasons_counted(self):
        results = [
            (0, "1/2-1/2", "max_plies", ""),
            (1, "1-0", "game_over", ""),
            (2, "1/2-1/2", "max_plies", ""),
        ]
        cfg = EvalConfig(games=3)
        out, _, _ = _aggregate_results(results, cfg, self._make_policy())
        assert out["termination_reasons"] == {"max_plies": 2, "game_over": 1}

    def test_empty_results(self):
        cfg = EvalConfig(games=0)
        out, _, pgns = _aggregate_results([], cfg, self._make_policy())
        assert out["games"] == 0
        assert out["score"] == 0.0
        assert pgns == []


# ---------------------------------------------------------------------------
# play_one_game
# ---------------------------------------------------------------------------

class TestPlayOneGame:
    def test_game_completes(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_play_one")
        policy = RandomPlayer()
        cfg = EvalConfig(games=1, max_plies=100)

        res, reason, pgn = play_one_game(policy, sf, True, cfg, game_number=0)
        assert res in {"1-0", "0-1", "1/2-1/2"}
        assert reason in {"game_over", "max_plies"}
        assert len(pgn) > 0

    def test_max_plies_draw(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_max_plies")
        policy = RandomPlayer()
        cfg = EvalConfig(games=1, max_plies=4)  # Very short => likely max_plies

        res, reason, pgn = play_one_game(policy, sf, True, cfg, game_number=0)
        # With 4 plies, game almost certainly doesn't end naturally
        if reason == "max_plies":
            assert res == "1/2-1/2"

    def test_rng_param_used_for_opening(self):
        """Passing rng should produce deterministic opening moves."""
        sf_cfg = _get_test_sf_config()
        # Only play the opening plies (no further moves) to isolate rng effect
        cfg = EvalConfig(games=1, max_plies=0, randomize_opening=True, opening_plies=4)

        results = []
        for _ in range(2):
            sf = StockfishPlayer(sf_cfg, engine_name="test_rng_opening")
            rng = random.Random(999)
            policy = RandomPlayer(seed=42)
            _, _, pgn = play_one_game(policy, sf, True, cfg, game_number=0, rng=rng)
            results.append(pgn)
            StockfishManager.close("test_rng_opening")

        # Same rng seed => identical opening moves (max_plies=0 so no game moves after)
        moves_0 = [l for l in results[0].strip().split("\n") if l and not l.startswith("[")]
        moves_1 = [l for l in results[1].strip().split("\n") if l and not l.startswith("[")]
        assert moves_0 == moves_1

    def test_pgn_headers(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_pgn_headers")
        policy = RandomPlayer()
        cfg = EvalConfig(games=1, max_plies=10)

        _, _, pgn = play_one_game(policy, sf, True, cfg, game_number=5)
        assert '[White "Policy"]' in pgn
        assert '[Black "Stockfish"]' in pgn
        assert '[Round "6"]' in pgn

        StockfishManager.close("test_pgn_headers")
        sf2 = StockfishPlayer(sf_cfg, engine_name="test_pgn_headers_b")
        _, _, pgn2 = play_one_game(policy, sf2, False, cfg, game_number=3)
        assert '[White "Stockfish"]' in pgn2
        assert '[Black "Policy"]' in pgn2


# ---------------------------------------------------------------------------
# evaluate_policy_vs_stockfish — sequential (num_workers=1)
# ---------------------------------------------------------------------------

class TestEvaluateSequential:
    def test_basic_evaluation(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_seq_basic")
        policy = RandomPlayer()
        cfg = EvalConfig(games=4, max_plies=60, num_workers=1)

        results, returned_policy, pgns = evaluate_policy_vs_stockfish(policy, sf, cfg)
        assert results["games"] == 4
        assert results["wins"] + results["draws"] + results["losses"] == 4
        assert 0.0 <= results["score"] <= 1.0
        assert len(pgns) == 4
        assert returned_policy is policy

    def test_alternating_colors(self):
        """Games alternate: even=white, odd=black."""
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_seq_colors")
        policy = RandomPlayer()
        cfg = EvalConfig(games=4, max_plies=10, num_workers=1)

        _, _, pgns = evaluate_policy_vs_stockfish(policy, sf, cfg)
        for i, pgn in enumerate(pgns):
            if i % 2 == 0:
                assert '[White "Policy"]' in pgn
            else:
                assert '[White "Stockfish"]' in pgn


# ---------------------------------------------------------------------------
# evaluate_policy_vs_stockfish — parallel (num_workers>1)
# ---------------------------------------------------------------------------

class TestEvaluateParallel:
    def test_basic_parallel(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_par_basic")
        policy = RandomPlayer()
        cfg = EvalConfig(games=6, max_plies=60, num_workers=2)

        results, returned_policy, pgns = evaluate_policy_vs_stockfish(policy, sf, cfg)
        assert results["games"] == 6
        assert results["wins"] + results["draws"] + results["losses"] == 6
        assert len(pgns) == 6
        assert returned_policy is policy

    def test_pgns_ordered_by_game_number(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_par_order")
        policy = RandomPlayer()
        cfg = EvalConfig(games=6, max_plies=10, num_workers=3)

        _, _, pgns = evaluate_policy_vs_stockfish(policy, sf, cfg)
        # Verify alternating colors in PGN order
        for i, pgn in enumerate(pgns):
            if i % 2 == 0:
                assert '[White "Policy"]' in pgn, f"Game {i} should have Policy as white"
            else:
                assert '[White "Stockfish"]' in pgn, f"Game {i} should have Stockfish as white"

    def test_worker_engines_cleaned_up(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_par_cleanup")
        policy = RandomPlayer()
        cfg = EvalConfig(games=4, max_plies=10, num_workers=2)

        evaluate_policy_vs_stockfish(policy, sf, cfg)

        # No eval_parallel engines should remain
        for name in list(StockfishManager._engines.keys()):
            assert not name.startswith("eval_parallel_"), f"Engine {name} was not cleaned up"

    def test_more_workers_than_games(self):
        sf_cfg = _get_test_sf_config()
        sf = StockfishPlayer(sf_cfg, engine_name="test_par_excess")
        policy = RandomPlayer()
        cfg = EvalConfig(games=2, max_plies=30, num_workers=8)

        results, _, pgns = evaluate_policy_vs_stockfish(policy, sf, cfg)
        assert results["games"] == 2
        assert len(pgns) == 2

    def test_result_structure_matches_sequential(self):
        """Parallel and sequential should return the same dict keys."""
        sf_cfg = _get_test_sf_config()
        policy = RandomPlayer()

        sf1 = StockfishPlayer(sf_cfg, engine_name="test_struct_seq")
        cfg1 = EvalConfig(games=2, max_plies=30, num_workers=1)
        seq_results, _, _ = evaluate_policy_vs_stockfish(policy, sf1, cfg1)

        sf2 = StockfishPlayer(sf_cfg, engine_name="test_struct_par")
        cfg2 = EvalConfig(games=2, max_plies=30, num_workers=2)
        par_results, _, _ = evaluate_policy_vs_stockfish(policy, sf2, cfg2)

        assert set(seq_results.keys()) == set(par_results.keys())
