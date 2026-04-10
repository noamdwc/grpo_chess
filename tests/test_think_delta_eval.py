"""Evaluator produces think_delta = score_with_think - score_no_think."""
from pathlib import Path

import pytest

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_evaluator_logs_think_delta_keys():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.evaluator import evaluate_with_and_without_thinking

    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    metrics = evaluate_with_and_without_thinking(
        model=model,
        n_games=2,
        stockfish_level=0,
        seed=0,
    )
    assert "score_with_think" in metrics
    assert "score_no_think" in metrics
    assert "think_delta" in metrics
    assert metrics["think_delta"] == pytest_approx_eq(
        metrics["score_with_think"] - metrics["score_no_think"]
    )


def pytest_approx_eq(x):
    import pytest

    return pytest.approx(x, abs=1e-6)
