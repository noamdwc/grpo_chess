from types import SimpleNamespace

import chess
import pytest
import torch
from pytorch_lightning.callbacks import Callback
from src.chess.stockfish import StockfishConfig


class DummyCallback(Callback):
    pass


def test_get_trainer_appends_extra_callbacks():
    from src.trainer import get_trainer

    extra = DummyCallback()
    trainer = get_trainer(use_wandb=False, callbacks=[extra])

    assert extra in trainer.callbacks
    monitors = {getattr(cb, "monitor", None) for cb in trainer.callbacks}
    assert "train_total_loss" in monitors


def test_build_stockfish_eval_callback_uses_baseline_and_periodic_metrics(monkeypatch):
    from src.train_self_play import build_stockfish_eval_callback

    resolve_calls = []
    evaluator_kwargs = []

    monkeypatch.setattr(
        "src.train_self_play.resolve_stockfish_path",
        lambda path: resolve_calls.append(path) or "/usr/bin/stockfish",
    )
    monkeypatch.setattr(
        "src.train_self_play.ReasoningEvaluator",
        lambda **kwargs: evaluator_kwargs.append(kwargs) or SimpleNamespace(**kwargs),
    )

    cfg = SimpleNamespace(
        grpo=SimpleNamespace(eval_every_n_epochs=7),
        reasoning=SimpleNamespace(max_think_tokens=9),
        eval=SimpleNamespace(games=64),
        stockfish=StockfishConfig(path="stockfish"),
    )

    callback = build_stockfish_eval_callback(cfg)

    assert resolve_calls == ["stockfish"]
    assert callback.every_n_epochs == 7
    assert callback.metric_prefix == "eval_stockfish"
    assert callback.run_on_fit_start is True
    assert callback.init_metric_prefix == "eval_stockfish_init"
    # Baseline evaluator gets think_tokens=0; periodic gets max_think_tokens.
    assert [kw["think_tokens"] for kw in evaluator_kwargs] == [0, 9]
    assert callback.evaluator.think_tokens == 9
    assert callback._baseline_evaluator.think_tokens == 0


def test_split_think_callback_uses_baseline_evaluator_only_on_fit_start():
    from src.train_self_play import _SplitThinkStockfishEvalCallback

    class RecordingEvaluator:
        def __init__(self, name):
            self.name = name
            self.single_evaluation_calls = 0
            self.evaluate_and_log_calls = 0

        def single_evaluation(self, model):
            self.single_evaluation_calls += 1
            return {
                "score": 0.5, "elo_diff_vs_stockfish_approx": 0.0, "games": 1,
                "wins": 0, "draws": 1, "losses": 0, "termination_reasons": {},
            }, None, []

        def evaluate_and_log(self, pl_module, model, metric_prefix):
            del pl_module, model, metric_prefix
            self.evaluate_and_log_calls += 1
            return {"score": 0.6}

    baseline = RecordingEvaluator("baseline")
    periodic = RecordingEvaluator("periodic")
    callback = _SplitThinkStockfishEvalCallback(
        baseline_evaluator=baseline,
        periodic_evaluator=periodic,
        every_n_epochs=1,
        metric_prefix="eval_stockfish",
        run_on_fit_start=True,
        init_metric_prefix="eval_stockfish_init",
    )

    trainer = SimpleNamespace(logger=None, global_step=0, current_epoch=0)
    pl_module = SimpleNamespace(model=object(), log=lambda *a, **kw: None)

    callback.on_fit_start(trainer, pl_module)
    assert baseline.single_evaluation_calls == 1
    assert periodic.single_evaluation_calls == 0
    # Evaluator is restored to the periodic one after fit-start.
    assert callback.evaluator is periodic

    callback.on_train_epoch_end(trainer, pl_module)
    assert periodic.evaluate_and_log_calls == 1
    assert baseline.evaluate_and_log_calls == 0


def test_reasoning_eval_policy_selects_best_legal_move():
    from src.train_self_play import ReasoningEvalPolicy

    board = chess.Board()
    calls = []

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *args, **kwargs):
            raise AssertionError("select_reasoning_move should be stubbed in this test")

    dummy_model = DummyModel()

    def fake_select_reasoning_move(model, board_arg, *, think_tokens):
        calls.append((model, board_arg.fen(), think_tokens))
        return chess.Move.from_uci("e2e4")

    from src import train_self_play as tsp

    original = tsp.select_reasoning_move
    tsp.select_reasoning_move = fake_select_reasoning_move
    try:
        move = ReasoningEvalPolicy(dummy_model, think_tokens=3).act(board)
    finally:
        tsp.select_reasoning_move = original

    assert move == chess.Move.from_uci("e2e4")
    assert calls == [(dummy_model, board.fen(), 3)]


def test_build_stockfish_eval_callback_raises_when_stockfish_missing(monkeypatch):
    from src.train_self_play import build_stockfish_eval_callback

    monkeypatch.setattr(
        "src.train_self_play.resolve_stockfish_path",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )

    cfg = SimpleNamespace(
        grpo=SimpleNamespace(eval_every_n_epochs=10),
        eval=SimpleNamespace(games=64),
        stockfish=SimpleNamespace(path="/missing/stockfish"),
    )

    with pytest.raises(RuntimeError, match="Stockfish evaluation is required"):
        build_stockfish_eval_callback(cfg)


def test_train_passes_eval_callback_to_trainer(monkeypatch):
    from src.train_self_play import train

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(),
        training=SimpleNamespace(
            batch_size=2,
            num_epochs=3,
            checkpoint_dir="checkpoints",
            checkpoint_every_n_epochs=5,
            keep_n_checkpoints=2,
            use_wandb=False,
            wandb_project="Chess-GRPO-Bot",
        ),
        stockfish=SimpleNamespace(path="stockfish"),
        eval=SimpleNamespace(games=64),
        reasoning=SimpleNamespace(max_think_tokens=4),
        grpo=SimpleNamespace(eval_every_n_epochs=1),
    )
    callback = DummyCallback()
    trainer_calls = {}

    monkeypatch.setattr("src.train_self_play.load_experiment_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr("src.train_self_play.ChessStartStatesDataset", lambda dataset_cfg: ["fen-a", "fen-b"])
    monkeypatch.setattr("src.train_self_play.ReasoningGRPOLightningModule", lambda loaded_cfg: object())
    monkeypatch.setattr("src.train_self_play.build_stockfish_eval_callback", lambda loaded_cfg: callback)

    class DummyTrainer:
        def fit(self, module, dataloader, ckpt_path=None):
            trainer_calls["fit"] = {
                "module": module,
                "dataloader_batch_size": dataloader.batch_size,
                "ckpt_path": ckpt_path,
            }

    def fake_get_trainer(**kwargs):
        trainer_calls["kwargs"] = kwargs
        return DummyTrainer()

    monkeypatch.setattr("src.train_self_play.get_trainer", fake_get_trainer)

    train(config_path="default.yaml", resume_from_checkpoint="resume.ckpt")

    assert trainer_calls["kwargs"]["callbacks"] == [callback]
    assert trainer_calls["fit"]["dataloader_batch_size"] == 2
    assert trainer_calls["fit"]["ckpt_path"] == "resume.ckpt"


def test_train_omits_eval_callback_when_eval_config_is_unavailable(monkeypatch):
    from src.train_self_play import train

    cfg = SimpleNamespace(
        dataset=SimpleNamespace(),
        training=SimpleNamespace(
            batch_size=2,
            num_epochs=1,
            checkpoint_dir="checkpoints/test",
            checkpoint_every_n_epochs=1,
            keep_n_checkpoints=1,
            use_wandb=False,
            wandb_project="Chess-GRPO-Bot",
        ),
    )
    trainer_calls = {}

    monkeypatch.setattr("src.train_self_play.load_experiment_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr("src.train_self_play.ChessStartStatesDataset", lambda dataset_cfg: ["fen-a", "fen-b"])
    monkeypatch.setattr("src.train_self_play.ReasoningGRPOLightningModule", lambda loaded_cfg: object())

    def fail_build_stockfish_eval_callback(_cfg):
        raise AssertionError("build_stockfish_eval_callback should not be called")

    monkeypatch.setattr("src.train_self_play.build_stockfish_eval_callback", fail_build_stockfish_eval_callback)

    class DummyTrainer:
        def fit(self, module, dataloader, ckpt_path=None):
            trainer_calls["fit"] = {
                "module": module,
                "dataloader_batch_size": dataloader.batch_size,
                "ckpt_path": ckpt_path,
            }

    def fake_get_trainer(**kwargs):
        trainer_calls["kwargs"] = kwargs
        return DummyTrainer()

    monkeypatch.setattr("src.train_self_play.get_trainer", fake_get_trainer)

    train(config_path="default.yaml", resume_from_checkpoint="resume.ckpt")

    assert trainer_calls["kwargs"]["callbacks"] == []
    assert trainer_calls["fit"]["dataloader_batch_size"] == 2
    assert trainer_calls["fit"]["ckpt_path"] == "resume.ckpt"
