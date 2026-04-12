from types import SimpleNamespace

import chess
import pytest
import torch
from pytorch_lightning.callbacks import Callback


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

    monkeypatch.setattr(
        "src.train_self_play.resolve_stockfish_path",
        lambda path: resolve_calls.append(path) or "/usr/bin/stockfish",
    )
    monkeypatch.setattr(
        "src.train_self_play.ReasoningEvaluator",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    cfg = SimpleNamespace(
        grpo=SimpleNamespace(eval_every_n_epochs=7),
        eval=SimpleNamespace(games=64),
        stockfish=SimpleNamespace(path="stockfish"),
    )

    callback = build_stockfish_eval_callback(cfg)

    assert resolve_calls == ["stockfish"]
    assert callback.every_n_epochs == 7
    assert callback.metric_prefix == "eval_stockfish"
    assert callback.run_on_fit_start is True
    assert callback.init_metric_prefix == "eval_stockfish_init"


def test_reasoning_eval_policy_selects_best_legal_move():
    from src.reasoning.tokens import BASE_VOCAB_SIZE
    from src.searchless_chess_imports import MOVE_TO_ACTION
    from src.train_self_play import ReasoningEvalPolicy

    board = chess.Board()
    legal_best = MOVE_TO_ACTION["e2e4"]
    legal_other = MOVE_TO_ACTION["d2d4"]
    illegal_higher = MOVE_TO_ACTION["a1a8"]

    class DummyModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            yield self.param

        def eval(self):
            return self

        def __call__(self, tokens, seq_lens):
            del seq_lens
            vocab_size = max(BASE_VOCAB_SIZE, illegal_higher, legal_other, legal_best) + 1
            logits = torch.full((1, tokens.shape[1], vocab_size), -1000.0)
            logits[0, -1, legal_other] = 1.0
            logits[0, -1, legal_best] = 5.0
            logits[0, -1, illegal_higher] = 99.0
            hidden = torch.zeros((1, tokens.shape[1], 1), dtype=torch.float32)
            return SimpleNamespace(logits=logits, hidden=hidden)

    move = ReasoningEvalPolicy(DummyModel()).act(board)
    assert move == chess.Move.from_uci("e2e4")


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
