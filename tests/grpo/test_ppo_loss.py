import torch

from src.grpo_logic.ppo_loss import (
    entropy_loss,
    kl_loss,
    safe_ppo_actor_loss,
    safe_ppo_critic_loss,
    safe_ppo_loss,
)
from src.searchless_chess_imports import ACTION_DIM


def _uniform_logprobs(batch_size: int = 1) -> torch.Tensor:
    logits = torch.zeros((batch_size, ACTION_DIM))
    return torch.log_softmax(logits, dim=-1)


def test_actor_loss_uses_only_selected_actions_for_ppo_term() -> None:
    logprobs_old = _uniform_logprobs()
    logprobs_new = logprobs_old.clone()
    advantages = torch.zeros_like(logprobs_new)
    selected_mask = torch.zeros_like(logprobs_new, dtype=torch.bool)
    selected_mask[0, 7] = True
    advantages[0, 7] = 2.0
    advantages[0, 8] = 100.0

    loss, info = safe_ppo_actor_loss(
        logprobs_new,
        logprobs_old,
        advantages,
        selected_mask=selected_mask,
        kl_coef=0.0,
        entropy_coef=0.0,
        return_info=True,
    )

    assert torch.allclose(loss, torch.tensor(-2.0))
    assert torch.allclose(info.ppo_loss, torch.tensor(-2.0))
    assert torch.allclose(info.mean_ratio, torch.tensor(1.0))


def test_actor_loss_matches_manual_clipped_calculation() -> None:
    logprobs_old = _uniform_logprobs()
    logprobs_new = logprobs_old.clone()
    selected_mask = torch.zeros_like(logprobs_new, dtype=torch.bool)
    selected_mask[0, 4] = True
    logprobs_new[0, 4] = logprobs_old[0, 4] + torch.log(torch.tensor(1.5))
    advantages = torch.zeros_like(logprobs_new)
    advantages[0, 4] = 2.0

    loss, info = safe_ppo_actor_loss(
        logprobs_new,
        logprobs_old,
        advantages,
        selected_mask=selected_mask,
        clip_eps=0.2,
        kl_coef=0.0,
        entropy_coef=0.0,
        return_info=True,
    )

    manual_ratio = torch.tensor(1.5)
    manual_unclipped = manual_ratio * torch.tensor(2.0)
    manual_clipped = torch.tensor(1.2) * torch.tensor(2.0)
    manual_ppo = -torch.minimum(manual_unclipped, manual_clipped)

    assert torch.allclose(loss, manual_ppo)
    assert torch.allclose(info.ppo_loss, manual_ppo)
    assert torch.allclose(info.mean_ratio, manual_ratio)


def test_kl_loss_matches_manual_masked_calculation() -> None:
    logprobs_old = torch.log(torch.tensor([[0.25, 0.75, 1.0]]))
    logprobs_new = torch.log(torch.tensor([[0.50, 0.50, 1.0]]))
    valid_mask = torch.tensor([[True, True, False]])

    kl = kl_loss(logprobs_new, logprobs_old, valid_mask)

    expected_kl = (
        0.25 * torch.log(torch.tensor(0.25 / 0.50))
        + 0.75 * torch.log(torch.tensor(0.75 / 0.50))
    )
    assert torch.allclose(kl, expected_kl)


def test_entropy_loss_matches_manual_masked_calculation() -> None:
    logprobs = torch.log(torch.tensor([[0.25, 0.75, 1.0]]))
    valid_mask = torch.tensor([[True, True, False]])

    entropy = entropy_loss(logprobs, valid_mask)

    expected_entropy = -(
        0.25 * torch.log(torch.tensor(0.25))
        + 0.75 * torch.log(torch.tensor(0.75))
    )
    assert torch.allclose(entropy, expected_entropy)


def test_kl_loss_stays_finite_with_masked_negative_infinity() -> None:
    logprobs = torch.tensor([[-float("inf"), torch.log(torch.tensor(1.0))]])
    valid_mask = torch.tensor([[False, True]])

    kl = kl_loss(logprobs, logprobs, valid_mask)

    assert torch.isfinite(kl)
    assert kl.item() == 0.0


def test_entropy_loss_stays_finite_with_masked_negative_infinity() -> None:
    logprobs = torch.tensor([[-float("inf"), torch.log(torch.tensor(1.0))]])
    valid_mask = torch.tensor([[False, True]])

    entropy = entropy_loss(logprobs, valid_mask)

    assert torch.isfinite(entropy)
    assert entropy.item() == 0.0


def test_critic_loss_matches_manual_masked_calculation() -> None:
    values = torch.tensor([1.0, 10.0, float("nan")])
    targets = torch.tensor([3.0, 0.0, 5.0])
    policy_valid_mask = torch.tensor(
        [
            [True, False],
            [False, False],
            [True, False],
        ]
    )

    loss = safe_ppo_critic_loss(values, targets, policy_valid_mask)

    manual_loss = (torch.tensor(1.0) - torch.tensor(3.0)).pow(2)
    assert torch.allclose(loss, manual_loss)


def test_safe_ppo_loss_composes_actor_and_critic_terms_and_detaches_info() -> None:
    logprobs_old = _uniform_logprobs()
    logprobs_new = logprobs_old.clone().detach().requires_grad_(True)
    advantages = torch.zeros_like(logprobs_new)
    selected_mask = torch.zeros_like(logprobs_new, dtype=torch.bool)
    selected_mask[0, 3] = True
    advantages[0, 3] = 1.5
    values = torch.tensor([1.0], requires_grad=True)
    targets = torch.tensor([3.0])

    loss, info = safe_ppo_loss(
        logprobs_new,
        logprobs_old,
        advantages,
        values,
        targets,
        selected_mask=selected_mask,
        kl_coef=0.0,
        entropy_coef=0.0,
        critic_coef=0.5,
        return_info=True,
    )

    assert torch.allclose(loss, torch.tensor(0.5))
    assert info.actor_loss is not None
    assert info.critic_loss is not None
    assert torch.allclose(info.actor_loss, torch.tensor(-1.5))
    assert torch.allclose(info.critic_loss, torch.tensor(4.0))
    assert not info.ppo_loss.requires_grad
    assert not info.actor_loss.requires_grad
    assert not info.critic_loss.requires_grad


def test_safe_ppo_loss_matches_manual_calculation() -> None:
    logprobs_old = _uniform_logprobs()
    logprobs_new = logprobs_old.clone()
    selected_mask = torch.zeros_like(logprobs_new, dtype=torch.bool)
    selected_mask[0, 0] = True
    advantages = torch.zeros_like(logprobs_new)
    advantages[0, 0] = 2.0
    values = torch.tensor([1.0])
    targets = torch.tensor([4.0])

    loss, info = safe_ppo_loss(
        logprobs_new,
        logprobs_old,
        advantages,
        values,
        targets,
        selected_mask=selected_mask,
        ppo_coef=2.0,
        kl_coef=3.0,
        entropy_coef=0.5,
        critic_coef=0.25,
        return_info=True,
    )

    manual_ratio = torch.tensor(1.0)
    manual_ppo = -(manual_ratio * torch.tensor(2.0))
    manual_kl = torch.tensor(0.0)
    manual_entropy = torch.log(torch.tensor(float(ACTION_DIM)))
    manual_actor = 2.0 * manual_ppo + 3.0 * manual_kl - 0.5 * manual_entropy
    manual_critic = (torch.tensor(1.0) - torch.tensor(4.0)).pow(2)
    manual_total = manual_actor + 0.25 * manual_critic

    assert torch.allclose(info.ppo_loss, manual_ppo)
    assert torch.allclose(info.kl_loss, manual_kl)
    assert torch.allclose(info.entropy, manual_entropy)
    assert torch.allclose(info.critic_loss, manual_critic)
    assert torch.allclose(loss, manual_total)
