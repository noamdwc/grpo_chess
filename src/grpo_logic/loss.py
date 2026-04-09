import torch
from typing import Tuple
from dataclasses import dataclass

MIN_LOG_RATIO = -20.0
MAX_LOG_RATIO = 20.0


@dataclass
class GRPOLossInfo:
    """Information about GRPO loss components for logging and debugging."""
    kl_div: torch.Tensor
    mean_ratio: torch.Tensor
    mean_clip_fraction: torch.Tensor
    ppo_loss: torch.Tensor
    entropy: torch.Tensor
    advantage_mean: torch.Tensor
    advantage_std: torch.Tensor


def importance_ratio(logprobs_new: torch.Tensor, logprobs_old: torch.Tensor) -> torch.Tensor:
    """Compute importance ratio pi_new / pi_old from log-probabilities."""
    if logprobs_new.shape != logprobs_old.shape:
        raise ValueError(
            f"logprobs_new and logprobs_old must have matching shapes, "
            f"got {logprobs_new.shape} vs {logprobs_old.shape}"
        )
    # Clamp log-ratio before exp to prevent overflow during unstable early training.
    log_ratio = logprobs_new - logprobs_old
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=MAX_LOG_RATIO, neginf=MIN_LOG_RATIO)
    log_ratio = log_ratio.clamp(min=MIN_LOG_RATIO, max=MAX_LOG_RATIO)
    return log_ratio.exp()


def grpo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    beta_kl: float = 0.0,   # β in the formula (0 = no explicit KL penalty)
    eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO chess loss (legacy function, consider using grpo_ppo_loss instead).
    
    Args:
        logprobs_new: New policy log probabilities [G, T]
        logprobs_old: Old policy log probabilities [G, T]
        advantages: Advantage values [G, T]
        clip_eps: PPO clipping epsilon
        beta_kl: KL penalty coefficient
        eps: Numerical stability epsilon
        
    Returns:
        Tuple of (loss, approximate_kl_divergence)
    """

    # ------------------------------------------------------------
    # 3. Probability ratio r_{g,k,t}(θ)
    #
    #    r_{g,k,t}(θ) = πθ(a_{g,k,t}|s_{g,k,t}) / πold(a_{g,k,t}|s_{g,k,t})
    #                 = exp( logπθ - logπold )
    # ------------------------------------------------------------
    ratio = importance_ratio(logprobs_new, logprobs_old)  # [G, T]
    pg_unclipped = -advantages * ratio  # [G, T]
    pg_clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) # [G, T]

    # Surrogate policy gradient loss (PPO-clip part)
    # This corresponds to the -E[min(...)] in the formula.
    policy_loss = torch.max(pg_unclipped, pg_clipped).mean()
    approx_kl = (logprobs_old - logprobs_new).mean()

    # KL penalty: β * E[ KL(...) ]
    kl_loss = beta_kl * approx_kl
    loss = policy_loss + kl_loss

    return loss, approx_kl


# Utils functions for GRPO
def group_advantage(group_rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized advantages from group rewards using standardization.

    Args:
        group_rewards: Group rewards tensor [B, G] or [G]

    Returns:
        Normalized advantages with same shape as input
    """
    mean_reward = group_rewards.mean(dim=-1, keepdim=True)
    std_reward = group_rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
    advantages = (group_rewards - mean_reward) / std_reward
    return advantages


def step_group_advantage(step_rewards: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute per-step z-score normalized advantages from step rewards.
    For each timestep t, z-score normalizes across the G dimension (trajectories):

        advantage[b, g, t] = (reward[b,g,t] - mean_g[b,t]) / max(std_g[b,t], 1e-8)

    eps=1e-8 is a denominator floor (clamp_min). When all G rewards at a timestep
    are equal, std=0 and the numerator is also 0, so advantages are exactly 0.
    Padded entries (pad_mask=False) are zeroed in the output.

    Args:
        step_rewards: Per-step rewards tensor [B, G, T]
        pad_mask: Optional mask for valid steps [B, G, T], True=valid

    Returns:
        Z-score normalized advantages [B, G, T] where each timestep is normalized
        across the G dimension independently.
    """
    eps = 1e-8
    if pad_mask is None:
        mean_t = step_rewards.mean(dim=1, keepdim=True)  # [B, 1, T]
        std_t = step_rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)  # [B, 1, T]
        advantages = (step_rewards - mean_t) / std_t  # [B, G, T]
    else:
        valid = pad_mask.float()
        valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1, T]
        mean_t = (step_rewards * valid).sum(dim=1, keepdim=True) / valid_count  # [B, 1, T]
        sq_diff = ((step_rewards - mean_t) ** 2) * valid
        var_t = sq_diff.sum(dim=1, keepdim=True) / valid_count
        std_t = var_t.sqrt().clamp_min(eps)  # [B, 1, T]
        advantages = ((step_rewards - mean_t) / std_t) * valid  # [B, G, T]

    return advantages


def ppo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    pad_mask: torch.Tensor | None = None,  # [G, T], True = real, False = pad
    return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO-clip loss for chess policy optimization.
    
    Args:
        logprobs_new: New policy log probabilities [B, G, T] or [G, T]
        logprobs_old: Old policy log probabilities [B, G, T] or [G, T]
        advantages: Advantage values [B, G, T] or [G, T]
        clip_eps: PPO clipping epsilon (default: 0.2)
        pad_mask: Mask indicating valid steps, True=valid, False=padding
        return_info: If True, return additional statistics
        
    Returns:
        If return_info=False: policy loss tensor [B, G, T] or [G, T]
        If return_info=True: tuple of (policy_loss, mean_ratio, mean_clip_fraction)
    """
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    valid_mask = (
        pad_mask
        & torch.isfinite(logprobs_new)
        & torch.isfinite(logprobs_old)
        & torch.isfinite(advantages)
    )
    safe_logprobs_new = torch.where(valid_mask, logprobs_new, torch.zeros_like(logprobs_new))
    safe_logprobs_old = torch.where(valid_mask, logprobs_old, torch.zeros_like(logprobs_old))
    safe_advantages = torch.where(valid_mask, advantages, torch.zeros_like(advantages))

    ratio = importance_ratio(safe_logprobs_new, safe_logprobs_old)  # [G, T]
    pg_unclipped = -safe_advantages * ratio  # [G, T]
    pg_clipped = -safe_advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) # [G, T]
    # Surrogate policy gradient loss (PPO-clip part)
    # This corresponds to the -E[min(...)] in the formula.
    policy_loss = torch.max(pg_unclipped, pg_clipped) * valid_mask.float()
    if return_info:
        valid_steps = valid_mask.sum().clamp_min(1.0)
        mean_padded_ratio = (ratio * valid_mask.float()).sum() / valid_steps
        clip_fraction_mask = (ratio > (1.0 + clip_eps)) | (ratio < (1.0 - clip_eps))
        mean_clip_fraction = (clip_fraction_mask.float() * valid_mask.float()).sum() / valid_steps
        return policy_loss, mean_padded_ratio, mean_clip_fraction # [G, T], scalar, scalar
    return policy_loss # [G, T]


def kl_penalty(logprobs_new: torch.Tensor,
               logprobs_old: torch.Tensor,
               pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute KL divergence penalty between old and new policies.
    
    Args:
        logprobs_new: New policy log probabilities
        logprobs_old: Old policy log probabilities
        pad_mask: Optional mask for valid steps
        
    Returns:
        Mean KL divergence over valid steps
    """
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    valid_kl = (logprobs_old - logprobs_new)[pad_mask]
    if valid_kl.numel() == 0:
        return torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
    finite_kl = valid_kl[torch.isfinite(valid_kl)]
    if finite_kl.numel() == 0:
        return torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
    return finite_kl.mean()


def grpo_ppo_loss(
    logprobs_new: torch.Tensor,     # [B, G, T] or [G, T]
    logprobs_old: torch.Tensor,     # [B, G, T] or [G, T]
    step_rewards: torch.Tensor,     # [B, G, T] or [G, T] - per-step rewards
    pad_mask: torch.Tensor | None = None,  # [B, G, T] or [G, T]
    clip_ratio: float = 0.2,        # PPO clipping ratio (epsilon in paper)
    kl_coef: float = 0.01,          # KL penalty coefficient (beta in paper)
    return_info: bool = False,      # Return extra info for logging
    ) -> torch.Tensor | Tuple[torch.Tensor, GRPOLossInfo]:
    """
    Compute GRPO (Group Relative Policy Optimization) loss with PPO clipping.

    This combines PPO-clip loss with KL divergence penalty.
    Advantages are computed per-step by normalizing step rewards across trajectories
    (G dimension) for each timestep.

    Args:
        logprobs_new: New policy log probabilities [B, G, T] or [G, T]
        logprobs_old: Old policy log probabilities [B, G, T] or [G, T]
        step_rewards: Per-step rewards [B, G, T] or [G, T]
        pad_mask: Mask indicating valid steps, True=valid, False=padding
        clip_ratio: PPO clipping ratio (default: 0.2)
        kl_coef: KL divergence penalty coefficient (default: 0.01)
        return_info: If True, return GRPOLossInfo for logging

    Returns:
        If return_info=False: scalar loss tensor
        If return_info=True: tuple of (loss, GRPOLossInfo)
    """
    # Handle 2D input (no batch dimension) by adding batch dimension
    if logprobs_new.ndim == 2:
        logprobs_new = logprobs_new.unsqueeze(0)
        logprobs_old = logprobs_old.unsqueeze(0)
        step_rewards = step_rewards.unsqueeze(0)
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(0)

    if pad_mask is None:
        pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)

    valid_mask = (
        pad_mask
        & torch.isfinite(logprobs_new)
        & torch.isfinite(logprobs_old)
        & torch.isfinite(step_rewards)
    )
    if not bool(valid_mask.any()):
        zero = torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
        if return_info:
            return zero, GRPOLossInfo(
                kl_div=zero.detach(),
                mean_ratio=zero.detach(),
                mean_clip_fraction=zero.detach(),
                ppo_loss=zero.detach(),
                entropy=zero.detach(),
                advantage_mean=zero.detach(),
                advantage_std=zero.detach(),
            )
        return zero

    # Compute per-step advantages (normalized across G for each timestep)
    advantages = step_group_advantage(step_rewards, valid_mask).detach()  # [B, G, T]

    ppo_loss, mean_ratio, mean_clip_fraction = ppo_chess_loss(logprobs_new,
                                                              logprobs_old,
                                                              advantages,
                                                              clip_ratio,
                                                              valid_mask,
                                                              return_info=True)
    valid_steps = valid_mask.sum().clamp_min(1)
    ppo_loss = ppo_loss.sum() / valid_steps
    kl_div = kl_penalty(logprobs_new, logprobs_old, valid_mask)

    # Entropy: H(π) ≈ -E[log π(a|s)] — logged for monitoring, not part of loss
    valid_logprobs_new = logprobs_new[valid_mask]
    entropy = -valid_logprobs_new.mean() if valid_logprobs_new.numel() > 0 else torch.zeros_like(ppo_loss)

    loss = ppo_loss + kl_coef * kl_div

    if return_info:
        valid_advantages = advantages[valid_mask]
        if valid_advantages.numel() > 0:
            adv_mean = valid_advantages.mean().detach()
            adv_std = valid_advantages.std(unbiased=False).detach()
        else:
            adv_mean = torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
            adv_std = torch.zeros((), dtype=logprobs_new.dtype, device=logprobs_new.device)
        return loss, GRPOLossInfo(
            kl_div=kl_div.detach(),
            mean_ratio=mean_ratio.detach(),
            mean_clip_fraction=mean_clip_fraction.detach(),
            ppo_loss=ppo_loss.detach(),
            entropy=entropy.detach(),
            advantage_mean=adv_mean,
            advantage_std=adv_std,
        )
    return loss
    
