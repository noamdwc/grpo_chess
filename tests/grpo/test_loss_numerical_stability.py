import torch

from src.grpo_logic.loss import grpo_ppo_loss, importance_ratio


def test_importance_ratio_clamps_extreme_log_ratio() -> None:
    logprobs_old = torch.tensor([0.0, 0.0, 0.0])
    logprobs_new = torch.tensor([1000.0, -1000.0, float("nan")])

    ratio = importance_ratio(logprobs_new, logprobs_old)

    assert torch.isfinite(ratio).all()
    assert torch.allclose(ratio[0], torch.exp(torch.tensor(20.0)))
    assert torch.allclose(ratio[1], torch.exp(torch.tensor(-20.0)))
    assert torch.allclose(ratio[2], torch.tensor(1.0))


def test_grpo_ppo_loss_returns_zero_on_all_invalid_entries() -> None:
    logprobs_new = torch.full((1, 2, 3), float("nan"))
    logprobs_old = torch.zeros((1, 2, 3))
    step_rewards = torch.randn(1, 2, 3)
    pad_mask = torch.ones((1, 2, 3), dtype=torch.bool)

    loss, info = grpo_ppo_loss(
        logprobs_new,
        logprobs_old,
        step_rewards,
        pad_mask=pad_mask,
        return_info=True,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.isfinite(info.kl_div)
    assert torch.isfinite(info.mean_ratio)
    assert torch.isfinite(info.mean_clip_fraction)
    assert torch.isfinite(info.ppo_loss)
    assert torch.isfinite(info.entropy)
    assert torch.isfinite(info.advantage_mean)
    assert torch.isfinite(info.advantage_std)


def test_grpo_ppo_loss_stays_finite_with_extreme_logprob_deltas() -> None:
    torch.manual_seed(0)
    logprobs_old = torch.randn(2, 3, 4) - 2.0
    logprobs_new = logprobs_old.clone()
    logprobs_new[0, 0, 0] = 1000.0
    logprobs_new[0, 1, 1] = -1000.0
    step_rewards = torch.randn(2, 3, 4)
    pad_mask = torch.ones((2, 3, 4), dtype=torch.bool)

    loss, info = grpo_ppo_loss(
        logprobs_new,
        logprobs_old,
        step_rewards,
        pad_mask=pad_mask,
        kl_coef=0.001,
        return_info=True,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(info.ppo_loss)
    assert torch.isfinite(info.kl_div)
    assert torch.isfinite(info.mean_ratio)
