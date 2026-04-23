"""One full training step at small scale — finite loss, gradients flow."""
from pathlib import Path

import pytest
import torch

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_one_training_step_is_sane():
    from src.configs.config_loader import load_experiment_config
    from src.grpo_logic.model import ReasoningGRPOLightningModule

    cfg = load_experiment_config(
        "default.yaml",
        overrides={
            "training": {"batch_size": 2, "steps_per_epoch": 1, "num_epochs": 1},
            "grpo": {"k_samples_per_root": 2},
            "reasoning": {"min_think_tokens": 1, "max_think_tokens": 3},
            "leaf_evaluator": {"mode": "stockfish"},
        },
    )
    module = ReasoningGRPOLightningModule(cfg)
    module.train()

    batch = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 2
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)

    loss.backward()
    body_grads = [param.grad for param in module.policy_model.dm.layers.parameters() if param.grad is not None]
    assert len(body_grads) > 0
    assert module.policy_model.dm.token_embedding.weight.grad is not None
    assert module.policy_model.dm.pos_embedding.weight.grad is not None
    assert module.policy_model.policy_head.weight.grad is not None
    for param in module.policy_model.value_head.parameters():
        assert param.grad is not None
