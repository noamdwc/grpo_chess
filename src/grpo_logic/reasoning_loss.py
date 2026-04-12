"""Reasoning-GRPO loss: group-normalized PPO + KL + auxiliary value MSE."""
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from src.reasoning.sampler import RolloutResult


@dataclass
class ReasoningLossConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 0.001
    value_coef: float = 0.5
    adv_eps: float = 1e-6


class ReasoningLossOutput(NamedTuple):
    total: torch.Tensor
    ppo: torch.Tensor
    kl: torch.Tensor
    value: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_std: torch.Tensor


def build_final_move_mask(result: RolloutResult) -> torch.Tensor:
    """Mask only the real move decision that receives the environment reward."""
    num_samples, seq_len = result.token_sequences.shape
    mask = torch.zeros((num_samples, seq_len), dtype=torch.bool, device=result.token_sequences.device)
    batch_idx = torch.arange(num_samples, device=result.token_sequences.device)
    mask[batch_idx, result.final_move_positions] = True
    return mask & result.sampled_mask


def compute_group_advantages(rewards: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Reshape to [B, K], normalize per row, reshape back to [B*K]."""
    total = rewards.shape[0]
    if total % group_size != 0:
        raise ValueError("rewards length must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    raw_std = grouped.std(dim=1, keepdim=True, unbiased=True)
    std = raw_std.clamp_min(eps)
    advantages = (grouped - mean) / std
    advantages = advantages.masked_fill(raw_std < eps, 0.0)
    return advantages.view(-1)


def compute_value_loss(v_hat: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """MSE at <end_think>. v_hat and v_target are [N]."""
    return F.mse_loss(v_hat, v_target)


def ppo_surrogate(
    *,
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    advantages: torch.Tensor,
    sampled_mask: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    ratio = torch.exp(log_probs_new - log_probs_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    per_pos = -torch.minimum(unclipped, clipped)
    mask = sampled_mask.float()
    return (per_pos * mask).sum() / mask.sum().clamp_min(1.0)


def reasoning_loss(
    *,
    result: RolloutResult,
    log_probs_new: torch.Tensor,
    v_hat: torch.Tensor,
    rewards: torch.Tensor,
    v_target: torch.Tensor,
    cfg: ReasoningLossConfig,
    group_size: int,
) -> ReasoningLossOutput:
    num_samples, seq_len = result.token_sequences.shape
    if log_probs_new.shape != (num_samples, seq_len):
        raise ValueError("log_probs_new shape must match rollout token shape")

    adv_per_sample = compute_group_advantages(rewards, group_size, eps=cfg.adv_eps)
    policy_mask = build_final_move_mask(result)
    advantages = adv_per_sample[:, None].expand(num_samples, seq_len)

    ppo = ppo_surrogate(
        log_probs_old=result.log_probs_old,
        log_probs_new=log_probs_new,
        advantages=advantages,
        sampled_mask=policy_mask,
        clip_ratio=cfg.clip_ratio,
    )

    mask = policy_mask.float()
    per_pos_kl = result.log_probs_old - log_probs_new
    kl = (per_pos_kl * mask).sum() / mask.sum().clamp_min(1.0)
    value = compute_value_loss(v_hat, v_target)
    total = ppo + cfg.kl_coef * kl + cfg.value_coef * value

    with torch.no_grad():
        ratio = torch.exp(log_probs_new - result.log_probs_old)
        clip_lo = 1.0 - cfg.clip_ratio
        clip_hi = 1.0 + cfg.clip_ratio
        clipped = ((ratio < clip_lo) | (ratio > clip_hi)) & policy_mask
        clip_fraction = clipped.float().sum() / mask.sum().clamp_min(1.0)
        ratio_mean = (ratio * mask).sum() / mask.sum().clamp_min(1.0)
        advantage_mean = adv_per_sample.mean()
        advantage_std = adv_per_sample.std(unbiased=False)

    return ReasoningLossOutput(
        total=total,
        ppo=ppo.detach(),
        kl=kl.detach(),
        value=value.detach(),
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )
