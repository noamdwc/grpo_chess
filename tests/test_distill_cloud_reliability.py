import os
from types import SimpleNamespace

import pytest
import torch

from src.chess.policy_player import PolicyConfig
from src.chess.stockfish import StockfishConfig
from src.distill.distill import (
    DistillChessTransformer,
    DistillConfig,
    DistillQualityGateCallback,
    build_stockfish_eval_callback,
    prepare_wandb_config,
)
from src.eval_utils import EvalConfig
from src.models import ChessTransformerConfig


def _make_tiny_batch(action_dim: int = 16, batch_size: int = 2):
    board_tokens = torch.randint(0, 300, (batch_size, 77), dtype=torch.long)
    legal_masks = torch.zeros((batch_size, action_dim), dtype=torch.bool)
    legal_masks[:, 0] = True
    teacher_indices = torch.zeros((batch_size, 1), dtype=torch.long)
    teacher_probs = torch.ones((batch_size, 1), dtype=torch.float32)
    k_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    return board_tokens, legal_masks, teacher_indices, teacher_probs, k_mask


def _tiny_model(distill_cfg: DistillConfig) -> DistillChessTransformer:
    model_cfg = ChessTransformerConfig(
        vocab_size=300,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        action_dim=16,
    )
    model = DistillChessTransformer(model_cfg, distill_cfg)
    model.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return model


def test_prepare_wandb_config_auto_disables_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_KEY", raising=False)

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=True)
    updated = prepare_wandb_config(cfg)

    assert updated.use_wandb is False


def test_prepare_wandb_config_requires_key_when_auto_disable_off(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_KEY", raising=False)

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY not found"):
        prepare_wandb_config(cfg)


def test_prepare_wandb_config_maps_wandb_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_KEY", "abc123")

    cfg = DistillConfig(use_wandb=True, auto_disable_wandb_if_missing_key=True)
    updated = prepare_wandb_config(cfg)

    assert updated.use_wandb is True
    assert os.environ["WANDB_API_KEY"] == "abc123"


def test_training_step_skips_nonfinite_loss_when_allowed():
    model = _tiny_model(DistillConfig(fail_on_nonfinite=False, max_nonfinite_batches=3))

    def bad_compute_loss(*_args, **_kwargs):
        metrics = {
            "top1_match": torch.tensor(0.0),
            "top5_match": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "kl_divergence": torch.tensor(0.0),
            "teacher_legal_fraction": torch.tensor(0.0),
            "teacher_valid_sample_fraction": torch.tensor(0.0),
        }
        return torch.tensor(float("inf"), device=next(model.model.parameters()).device), metrics

    model._compute_loss = bad_compute_loss  # type: ignore[method-assign]
    loss = model.training_step(_make_tiny_batch(), batch_idx=0)

    assert torch.isfinite(loss)
    assert model._train_nonfinite_batches == 1


def test_training_step_raises_on_nonfinite_when_configured():
    model = _tiny_model(DistillConfig(fail_on_nonfinite=True, max_nonfinite_batches=3))

    def bad_compute_loss(*_args, **_kwargs):
        metrics = {
            "top1_match": torch.tensor(0.0),
            "top5_match": torch.tensor(0.0),
            "entropy": torch.tensor(0.0),
            "kl_divergence": torch.tensor(0.0),
            "teacher_legal_fraction": torch.tensor(0.0),
            "teacher_valid_sample_fraction": torch.tensor(0.0),
        }
        return torch.tensor(float("inf"), device=next(model.model.parameters()).device), metrics

    model._compute_loss = bad_compute_loss  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="fail_on_nonfinite=true"):
        model.training_step(_make_tiny_batch(), batch_idx=0)


def test_build_stockfish_eval_callback_skips_when_missing_and_optional(monkeypatch):
    monkeypatch.setattr(
        "src.distill.distill.resolve_stockfish_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing stockfish")),
    )

    callback = build_stockfish_eval_callback(
        distill_config=DistillConfig(require_stockfish_eval=False),
        eval_cfg=EvalConfig(),
        stockfish_cfg=StockfishConfig(),
        policy_cfg=PolicyConfig(),
    )

    assert callback is None


def test_build_stockfish_eval_callback_raises_when_missing_and_required(monkeypatch):
    monkeypatch.setattr(
        "src.distill.distill.resolve_stockfish_path",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("missing stockfish")),
    )

    with pytest.raises(RuntimeError, match="Stockfish evaluation is required"):
        build_stockfish_eval_callback(
            distill_config=DistillConfig(require_stockfish_eval=True),
            eval_cfg=EvalConfig(),
            stockfish_cfg=StockfishConfig(),
            policy_cfg=PolicyConfig(),
        )


def test_distill_quality_gate_skips_before_gate_epoch():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=1,  # Completed epoch 2
        callback_metrics={"val/top1_match": torch.tensor(0.01)},
    )

    callback.on_validation_epoch_end(trainer, pl_module=object())


def test_distill_quality_gate_raises_when_metric_below_threshold():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=2,  # Completed epoch 3
        callback_metrics={"val/top1_match": torch.tensor(0.03)},
    )

    with pytest.raises(RuntimeError, match="Distill quality gate failed"):
        callback.on_validation_epoch_end(trainer, pl_module=object())


def test_distill_quality_gate_passes_when_metric_meets_threshold():
    callback = DistillQualityGateCallback(min_val_top1=0.08, gate_epoch=3)
    trainer = SimpleNamespace(
        current_epoch=2,  # Completed epoch 3
        callback_metrics={"val/top1_match": torch.tensor(0.08)},
    )

    callback.on_validation_epoch_end(trainer, pl_module=object())
