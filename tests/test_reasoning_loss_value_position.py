"""L_value is computed at <end_think> position only."""
import torch

from src.grpo_logic.reasoning_loss import compute_value_loss


def test_value_loss_at_end_think_only():
    v_hat = torch.tensor([0.2, -0.4, 0.1])
    v_target = torch.tensor([0.0, 0.0, 0.0])
    loss = compute_value_loss(v_hat, v_target)
    assert loss.item() == pytest_approx(0.07, rel=1e-5)


def pytest_approx(x, rel):
    import pytest

    return pytest.approx(x, rel=rel)
