import torch
from typing import Tuple
from dataclasses import dataclass


@dataclass
class GRPOLossInfo:
    kl_div: torch.Tensor
    mean_ratio: torch.Tensor
    mean_clip_fraction: torch.Tensor
    ppo_lose: torch.Tensor

def grpo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    beta_kl: float = 0.0,   # β in the formula (0 = no explicit KL penalty)
    eps: float = 1e-8):

    # ------------------------------------------------------------
    # 3. Probability ratio r_{g,k,t}(θ)
    #
    #    r_{g,k,t}(θ) = πθ(a_{g,k,t}|s_{g,k,t}) / πold(a_{g,k,t}|s_{g,k,t})
    #                 = exp( logπθ - logπold )
    # ------------------------------------------------------------
    ratio = (logprobs_new - logprobs_old).exp() # [G, T]
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
def group_advantage(group_rewards):
    mean_reward = group_rewards.mean(dim=-1, keepdim=True)
    std_reward = group_rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
    advantages = (group_rewards - mean_reward) / std_reward
    return advantages


def ppo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    pad_mask: torch.Tensor | None = None,  # [G, T], True = real, False = pad
    return_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    ratio = (logprobs_new - logprobs_old).exp() # [G, T]
    pg_unclipped = -advantages * ratio  # [G, T]
    pg_clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) # [G, T]
    # Surrogate policy gradient loss (PPO-clip part)
    # This corresponds to the -E[min(...)] in the formula.
    policy_loss = torch.max(pg_unclipped, pg_clipped) * pad_mask.float()
    if return_info:
        valid_steps = pad_mask.sum().clamp_min(1.0)
        mean_padded_ratio = (ratio * pad_mask.float()).sum() / valid_steps
        clip_fraction_mask = (ratio > (1.0 + clip_eps)) | (ratio < (1.0 - clip_eps))
        mean_clip_fraction = (clip_fraction_mask.float() * pad_mask.float()).sum() / valid_steps
        return policy_loss, mean_padded_ratio, mean_clip_fraction # [G, T], scalar, scalar
    return policy_loss # [G, T]


def kl_penalty(logprobs_new: torch.Tensor,
               logprobs_old: torch.Tensor,
               pad_mask=None):
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    return (logprobs_old - logprobs_new)[pad_mask].mean()


def grpo_ppo_loss(
    logprobs_new: torch.Tensor,     # [G, T]
    logprobs_old: torch.Tensor,     # [G, T]
    group_rewards: torch.Tensor,    # [G,]
    pad_mask: torch.Tensor | None = None,  # [G, T]
    clip_ratio: float = 0.2,        # in paper this epsilon
    kl_coef: float = 0.01,          # in paper this is beta
    return_info: bool = False, # return extra info foe logging
    ) -> torch.Tensor | Tuple[torch.Tensor, GRPOLossInfo]:
    if logprobs_new.ndim == 2: # No batch input - unsqueeze
        logprobs_new = logprobs_new.unsqueeze(0)
        logprobs_old = logprobs_old.unsqueeze(0)
        group_rewards = group_rewards.unsqueeze(0)
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(0)

    if pad_mask is None:
        pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)

    B, G, T = logprobs_new.shape
    advantages_2d = group_advantage(group_rewards).detach()
    advantages = advantages_2d.unsqueeze(-1).expand(B, G, T)
    advantages = advantages * pad_mask.float()
    ppo_lose, mean_ratio, mean_clip_fraction = ppo_chess_loss(logprobs_new,
                                                              logprobs_old,
                                                              advantages,
                                                              clip_ratio,
                                                              pad_mask,
                                                              return_info=True)
    valid_steps = pad_mask.sum().clamp_min(1)
    ppo_lose = ppo_lose.sum() / valid_steps
    kl_div = kl_penalty(logprobs_new, logprobs_old, pad_mask)
    loss = ppo_lose + kl_coef * kl_div
    if return_info:
        return loss, GRPOLossInfo(kl_div.detach(),
                                  mean_ratio.detach(),
                                  mean_clip_fraction.detach(),
                                  ppo_lose.detach())
    return loss
    