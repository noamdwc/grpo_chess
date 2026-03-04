"""Test that tqdm usage in eval_utils doesn't cause RecursionError.

The bug: tqdm's Rich-based rendering in Jupyter notebooks can exceed
Python's recursion limit when called from deep within the Lightning
training loop + IPython stack. This manifests as RecursionError in
evaluate_policy_vs_stockfish → tqdm.__init__ → refresh → Rich segment rendering.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestTqdmRecursionSafety(unittest.TestCase):
    """Verify that eval_utils avoids tqdm-triggered RecursionError."""

    def test_safe_range_falls_back_on_recursion_error(self):
        """_safe_range must return plain range when tqdm raises RecursionError."""
        from src.eval_utils import _safe_range

        def exploding_tqdm(*args, **kwargs):
            raise RecursionError("simulated deep stack")

        with patch("src.eval_utils._tqdm_auto", exploding_tqdm):
            result = list(_safe_range(3))
        self.assertEqual(result, [0, 1, 2])

    def test_safe_range_uses_tqdm_when_available(self):
        """_safe_range uses tqdm when it works normally."""
        from src.eval_utils import _safe_range

        # With default _tqdm_auto (real tqdm), should not raise
        result = list(_safe_range(3))
        self.assertEqual(result, [0, 1, 2])

    def test_safe_range_works_without_tqdm(self):
        """_safe_range falls back to plain range when tqdm is absent."""
        from src.eval_utils import _safe_range

        with patch("src.eval_utils._tqdm_auto", None):
            result = list(_safe_range(3))
        self.assertEqual(result, [0, 1, 2])

    def test_evaluate_no_recursion_when_tqdm_explodes(self):
        """Full evaluate_policy_vs_stockfish must survive tqdm RecursionError."""
        from src.eval_utils import evaluate_policy_vs_stockfish, EvalConfig
        from src.chess.stockfish import StockfishConfig, StockfishPlayer
        from src.chess.policy_player import PolicyPlayer, PolicyConfig
        from src.models import ChessTransformer, ChessTransformerConfig

        model = ChessTransformer(ChessTransformerConfig())
        policy = PolicyPlayer(model, cfg=PolicyConfig(greedy=True))
        sf = StockfishPlayer(StockfishConfig(), engine_name="test_recursion")
        eval_cfg = EvalConfig(games=2, max_plies=10)

        def exploding_tqdm(*args, **kwargs):
            raise RecursionError("simulated deep stack")

        with patch("src.eval_utils._tqdm_auto", exploding_tqdm):
            results, _, pgns = evaluate_policy_vs_stockfish(policy, sf, eval_cfg)

        self.assertEqual(results["games"], 2)
        self.assertEqual(len(pgns), 2)


if __name__ == "__main__":
    unittest.main()
