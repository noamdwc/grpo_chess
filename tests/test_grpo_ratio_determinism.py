"""Regression tests for deterministic GRPO ratio/log-prob computation."""

import random

import chess
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.grpo_logic.loss import GRPOLossInfo, importance_ratio
from src.grpo_logic.mode_utils import temporary_eval
from src.grpo_logic.model import GRPOChessTransformer, GRPOConfig
from src.grpo_logic.sampling import batched_policy_step
from src.models import ChessTransformerConfig


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model(action_dim: int = 64) -> GRPOChessTransformer:
    transformer_cfg = ChessTransformerConfig(
        vocab_size=64,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        action_dim=action_dim,
    )
    grpo_cfg = GRPOConfig(
        num_trajectories=2,
        trajectory_depth=3,
        rollout_temperature=1.1,
    )
    model = GRPOChessTransformer(transformer_cfg, grpo_cfg)
    model.train()
    return model


def _make_trajectory_batch(
    *,
    seed: int,
    action_dim: int,
    vocab_size: int = 64,
    batch_size: int = 2,
    num_trajectories: int = 3,
    depth: int = 4,
    seq_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    states = torch.randint(
        1,
        vocab_size,
        (batch_size, num_trajectories, depth, seq_len),
        generator=g,
        dtype=torch.long,
    )
    actions = torch.randint(
        0,
        action_dim,
        (batch_size, num_trajectories, depth),
        generator=g,
        dtype=torch.long,
    )
    legal_masks = torch.rand(
        (batch_size, num_trajectories, depth, action_dim),
        generator=g,
    ) > 0.7
    legal_masks.scatter_(-1, actions.unsqueeze(-1), True)
    return states, actions, legal_masks


class _RecordingSamplingPolicy(nn.Module):
    """Sampling stub that records train/eval mode seen at probability computation."""

    def __init__(self, action_size: int = 1968):
        super().__init__()
        self._param = nn.Parameter(torch.zeros(1))
        self._action_size = action_size
        self.seen_training_flags: list[bool] = []

    @property
    def action_size(self) -> int:
        return self._action_size

    def get_legal_moves_probs(
        self,
        states_tensor: torch.Tensor,
        legal_moves_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.seen_training_flags.append(self.training)
        probs = legal_moves_mask.float()
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0)


def test_same_policy_ratio_is_deterministic_across_repeated_calls():
    _seed_all(0)
    model = _build_model(action_dim=64)
    states, actions, legal_masks = _make_trajectory_batch(seed=1, action_dim=64)

    old_log_probs = model._compute_old_log_probs(states, actions, legal_masks)

    ratios = []
    for _ in range(30):
        with torch.no_grad():
            new_log_probs = model._compute_new_log_probs(states, actions, legal_masks)
            ratios.append(importance_ratio(new_log_probs, old_log_probs))

    ones = torch.ones_like(ratios[0])
    for ratio in ratios:
        assert torch.allclose(ratio, ones, atol=1e-6, rtol=1e-6)

    ratio_stack = torch.stack(ratios, dim=0)
    assert float(ratio_stack.var(unbiased=False)) <= 1e-12


def test_same_policy_ratio_vector_is_all_ones():
    _seed_all(2)
    model = _build_model(action_dim=96)
    states, actions, legal_masks = _make_trajectory_batch(
        seed=3,
        action_dim=96,
        batch_size=3,
        num_trajectories=2,
        depth=5,
    )

    old_log_probs = model._compute_old_log_probs(states, actions, legal_masks)
    with torch.no_grad():
        new_log_probs = model._compute_new_log_probs(states, actions, legal_masks)
    ratio = importance_ratio(new_log_probs, old_log_probs)

    assert ratio.shape == old_log_probs.shape
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6)


def test_temporary_eval_restores_train_mode_and_handles_exceptions():
    module = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    module.train()

    with pytest.raises(RuntimeError):
        with temporary_eval(module):
            assert module.training is False
            raise RuntimeError("boom")

    assert module.training is True


def test_temporary_eval_restores_eval_mode():
    module = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
    module.eval()

    with temporary_eval(module):
        assert module.training is False

    assert module.training is False


def test_new_logprob_path_preserves_gradients_with_dropout_disabled():
    _seed_all(4)
    model = _build_model(action_dim=64)
    states, actions, legal_masks = _make_trajectory_batch(seed=5, action_dim=64)

    model.policy_model.zero_grad(set_to_none=True)

    new_log_probs = model._compute_new_log_probs(states, actions, legal_masks)
    assert new_log_probs.requires_grad

    loss = -new_log_probs.mean()
    loss.backward()

    grads = [param.grad for param in model.policy_model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in grads)


def test_batched_policy_step_uses_eval_mode_and_restores_model_mode():
    _seed_all(6)
    board = chess.Board()
    model = _RecordingSamplingPolicy()
    model.train()

    sample = batched_policy_step(model, [board], temperature=1.0)

    assert sample is not None
    assert model.seen_training_flags
    assert all(flag is False for flag in model.seen_training_flags)
    assert model.training is True


def test_train_mode_without_eval_guard_is_dropout_noisy():
    _seed_all(7)
    model = _build_model(action_dim=80)
    states, actions, legal_masks = _make_trajectory_batch(seed=8, action_dim=80)

    model.policy_model.train()
    outputs = []
    with torch.no_grad():
        for _ in range(8):
            outputs.append(
                model.policy_model.get_group_log_probs(
                    states,
                    actions,
                    legal_masks,
                    temperature=1.0,
                )
            )

    all_equal = all(torch.allclose(outputs[0], out) for out in outputs[1:])
    assert not all_equal


def test_ppo_step_skips_when_loss_has_no_grad(monkeypatch):
    _seed_all(9)
    model = _build_model(action_dim=64)
    log_calls: list[dict] = []

    def _record_log(*_args, **kwargs):
        log_calls.append(kwargs)

    model.log = _record_log  # type: ignore[method-assign]

    states, actions, legal_masks = _make_trajectory_batch(
        seed=10,
        action_dim=64,
        batch_size=1,
        num_trajectories=2,
        depth=3,
    )
    old_log_probs = torch.zeros((1, 2, 3))
    step_rewards = torch.zeros((1, 2, 3))
    pad_mask = torch.ones((1, 2, 3), dtype=torch.bool)

    def fake_grpo_ppo_loss(*_args, **_kwargs):
        zero = torch.tensor(0.0)  # Deliberately no grad_fn / requires_grad=False
        info = GRPOLossInfo(
            kl_div=zero,
            mean_ratio=zero,
            mean_clip_fraction=zero,
            ppo_loss=zero,
            entropy=zero,
            advantage_mean=zero,
            advantage_std=zero,
        )
        return zero, info

    # Patch the exact global symbol used by _ppo_step to avoid module-alias
    # mismatches in notebook/reload-heavy environments.
    monkeypatch.setitem(model._ppo_step.__globals__, "grpo_ppo_loss", fake_grpo_ppo_loss)
    assert model._ppo_step.__globals__["grpo_ppo_loss"] is fake_grpo_ppo_loss

    loss, loss_info = model._ppo_step(
        states,
        actions,
        old_log_probs,
        legal_masks,
        step_rewards,
        pad_mask,
    )

    assert loss is None
    assert loss_info is None
    assert any(call.get("batch_size") == 1 for call in log_calls)
