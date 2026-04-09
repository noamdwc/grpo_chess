"""Tests for Stockfish callback scheduling/path helpers."""

import pytest

from src.chess.stockfish import resolve_stockfish_path
from src.evaluator import Evaluator, StockfishEvalCallback


def test_resolve_stockfish_path_falls_back_to_path_lookup(monkeypatch):
    """If configured path is invalid, resolver should use stockfish from PATH."""
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("src.chess.stockfish.os.path.isfile", lambda _: False)
    monkeypatch.setattr("src.chess.stockfish.os.access", lambda *_: False)
    monkeypatch.setattr(
        "src.chess.stockfish.shutil.which",
        lambda cmd: "/opt/homebrew/bin/stockfish" if cmd in {"stockfish", "sf"} else None,
    )

    path = resolve_stockfish_path("sf")
    assert path == "/opt/homebrew/bin/stockfish"


def test_resolve_stockfish_path_raises_when_missing(monkeypatch):
    """Resolver should fail loudly when no candidate exists."""
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr("src.chess.stockfish.os.path.isfile", lambda _: False)
    monkeypatch.setattr("src.chess.stockfish.os.access", lambda *_: False)
    monkeypatch.setattr("src.chess.stockfish.shutil.which", lambda _: None)

    with pytest.raises(FileNotFoundError):
        resolve_stockfish_path("/missing/stockfish")


def test_stockfish_eval_callback_validates_frequency():
    """Callback frequency must be positive to avoid modulo-by-zero schedules."""
    with pytest.raises(ValueError):
        StockfishEvalCallback(Evaluator(), every_n_epochs=0)


def test_evaluator_handles_recursion_error_without_crashing(monkeypatch):
    """Recursion errors in eval should not crash the training loop callback path."""
    evaluator = Evaluator()

    class DummyLightningModule:
        def __init__(self):
            self.training = True
            self.logger = None

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def log(self, *args, **kwargs):
            return None

    # Force eval failure path.
    monkeypatch.setattr(
        evaluator,
        "single_evaluation",
        lambda model: (_ for _ in ()).throw(RecursionError("maximum recursion depth exceeded")),
    )
    # Simulate traceback formatter itself failing recursively.
    monkeypatch.setattr(
        "src.evaluator.traceback.format_exc",
        lambda: (_ for _ in ()).throw(RecursionError("traceback recursion")),
    )

    close_calls = []
    monkeypatch.setattr(
        "src.evaluator.StockfishManager.close",
        lambda name: close_calls.append(name),
    )

    result = evaluator.evaluate_and_log(DummyLightningModule(), model=object(), metric_prefix="eval_stockfish")

    assert result is None
    assert close_calls, "Expected eval engine reset to be attempted on failure"
