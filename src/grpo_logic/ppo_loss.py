import torch
from typing import Tuple
from dataclasses import dataclass
from src.searchless_chess_imports import ACTION_DIM   

MIN_LOG_RATIO = -20.0
MAX_LOG_RATIO = 20.0

@dataclass
class PPOLossInfo:
    """Information about PPO loss components for logging and debugging."""
    kl_loss: torch.Tensor
    ppo_loss: torch.Tensor
    entropy: torch.Tensor
    mean_ratio: torch.Tensor
    actor_loss: torch.Tensor | None = None
    critic_loss: torch.Tensor | None = None

    def _detach(self) -> PPOLossInfo:
        """Make sure all tensors are detached"""
        self.kl_loss = self.kl_loss.detach()
        self.ppo_loss = self.ppo_loss.detach()
        self.entropy = self.entropy.detach()
        self.mean_ratio = self.mean_ratio.detach()
        if self.actor_loss is not None:
            self.actor_loss = self.actor_loss.detach()
        if self.critic_loss is not None:
            self.critic_loss = self.critic_loss.detach()
        return self


def safe_mean(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute mean of tensor with valid mask."""
    if mask is None:
        return tensor.mean()
    tensor = tensor * mask.to(dtype=tensor.dtype, device=tensor.device)
    return tensor.sum() / mask.sum().clamp_min(1.0)


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
    return log_ratio.exp() # [B, A]



def clipped_ppo_loss(logprobs_new: torch.Tensor, 
                     logprobs_old: torch.Tensor, 
                     advantages: torch.Tensor, 
                     selected_mask: torch.Tensor | None = None,
                     valid_mask: torch.Tensor | None = None,
                     clip_eps: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PPO loss.
       L_ppo = E[min(ratio * advantages, clip(ratio, 1-epsilon, 1+epsilon) * advantages)]
       returns the mean ration detached for monitoring 
    """
    if valid_mask is None:
      valid_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    if selected_mask is None:
      selected_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    ppo_mask = (valid_mask & selected_mask & torch.isfinite(advantages))
    # probs of the selected actions for ppo loss, kl_loss and entorpy_loss use full distribution, not just the selected actions
    safe_logprobs_new = torch.where(ppo_mask, logprobs_new, torch.zeros_like(logprobs_new))
    safe_logprobs_old = torch.where(ppo_mask, logprobs_old, torch.zeros_like(logprobs_old))
    safe_advantages = torch.where(ppo_mask, advantages, torch.zeros_like(advantages))

    # calculate the ratio of the selected actions
    ratio = importance_ratio(safe_logprobs_new, safe_logprobs_old)
    unclipped = ratio * safe_advantages
    clipped = ratio.clamp(min=1.0 - clip_eps, max=1.0 + clip_eps) * safe_advantages

    ppo_loss = safe_mean(-torch.minimum(unclipped, clipped), ppo_mask)
    ratio_mean = safe_mean(ratio, ppo_mask).detach()
    return ppo_loss, ratio_mean # scalar, scalar - the mean ratio for monitoring


def kl_loss(logprobs_new: torch.Tensor, logprobs_old: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute KL loss.
       KL(p_old || p_new) = sum(p_old * (log(p_old) - log(p_new)))
    """
    if valid_mask is None:
      valid_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    safe_logprobs_old = torch.where(valid_mask, logprobs_old, torch.zeros_like(logprobs_old))
    safe_logprobs_new = torch.where(valid_mask, logprobs_new, torch.zeros_like(logprobs_new))
    kl_per_action = safe_logprobs_old.exp() * (safe_logprobs_old - safe_logprobs_new) # [B, A]
    kl_per_batch = kl_per_action.sum(dim=-1) # [B]
    valid_boards = valid_mask.any(dim=-1) # [B]
    return safe_mean(kl_per_batch, valid_boards) # scalar


def entropy_loss(logprobs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute entropy loss.
       H(p) = -sum(p * log(p))
    """
    if valid_mask is None:
      valid_mask = torch.ones_like(logprobs, dtype=torch.bool)
    safe_logprobs = torch.where(valid_mask, logprobs, torch.zeros_like(logprobs))
    probs = safe_logprobs.exp() * valid_mask.to(dtype=safe_logprobs.dtype, device=safe_logprobs.device) # [B, A]
    entropy_per_batch = -(probs * safe_logprobs).sum(dim=-1) # [B]
    valid_boards = valid_mask.any(dim=-1) # [B]
    return safe_mean(entropy_per_batch, valid_boards) # scalar


def assert_ppo_actor_input_shapes(logprobs_new: torch.Tensor, 
                            logprobs_old: torch.Tensor, 
                            advantages: torch.Tensor,
                            selected_mask: torch.Tensor | None = None,
                            policy_valid_mask: torch.Tensor | None = None) -> None:
    """Assert input shapes are compatible."""
    if logprobs_new.shape != logprobs_old.shape:
        raise ValueError(f"logprobs_new and logprobs_old must have matching shapes, got {logprobs_new.shape} vs {logprobs_old.shape}")
    if advantages.shape != logprobs_new.shape:
        raise ValueError(f"advantages must have matching shapes with logprobs_new, got {advantages.shape} vs {logprobs_new.shape}")
    if logprobs_new.ndim != 2:
        raise ValueError(f"logprobs_new must be 2D, got {logprobs_new.ndim}D")
    if logprobs_new.shape[1] != ACTION_DIM:
        raise ValueError(f"logprobs_new must have {ACTION_DIM} actions, got {logprobs_new.shape[1]}")
    if advantages.shape[1] != ACTION_DIM:
        raise ValueError(f"advantages must have {ACTION_DIM} actions, got {advantages.shape[1]}")
    if selected_mask is not None and selected_mask.shape != logprobs_new.shape:
        raise ValueError(f"selected_mask must have matching shapes with logprobs_new, got {selected_mask.shape} vs {logprobs_new.shape}")
    if policy_valid_mask is not None and policy_valid_mask.shape != logprobs_new.shape:
        raise ValueError(f"policy_valid_mask must have matching shapes with logprobs_new, got {policy_valid_mask.shape} vs {logprobs_new.shape}")


def safe_ppo_actor_loss(logprobs_new: torch.Tensor,
                        logprobs_old: torch.Tensor,
                        advantages: torch.Tensor,
                        selected_mask: torch.Tensor | None = None,
                        policy_valid_mask: torch.Tensor | None = None,
                        clip_eps: float = 0.2,
                        ppo_coef: float = 1.0,
                        kl_coef: float = 0.01,
                        entropy_coef: float = 0.0, # default to 0 for no entropy loss
                        return_info: bool = False) -> torch.Tensor | Tuple[torch.Tensor, PPOLossInfo]:
    """Compute PPO actor loss with safe log probabilities."""
    assert_ppo_actor_input_shapes(logprobs_new, logprobs_old, advantages, selected_mask, policy_valid_mask)
    if policy_valid_mask is None:
      policy_valid_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    if selected_mask is None:
      selected_mask = torch.ones_like(logprobs_new, dtype=torch.bool)

    # probs of the selected actions for ppo loss, kl_loss and entorpy_loss use
    ppo_loss_scalar, mean_ratio = clipped_ppo_loss(logprobs_new, 
                                                   logprobs_old, 
                                                   advantages,
                                                   selected_mask, 
                                                   policy_valid_mask, 
                                                   clip_eps)
    kl_loss_scalar = kl_loss(logprobs_new, logprobs_old, policy_valid_mask)
    entropy_loss_scalar = entropy_loss(logprobs_new, policy_valid_mask)
    loss = ppo_coef * ppo_loss_scalar + kl_coef * kl_loss_scalar - entropy_coef * entropy_loss_scalar
    if return_info:
        return loss, PPOLossInfo(kl_loss=kl_loss_scalar, ppo_loss=ppo_loss_scalar, entropy=entropy_loss_scalar, mean_ratio=mean_ratio)
    return loss



def assert_ppo_critic_input_shapes(values: torch.Tensor, targets: torch.Tensor, policy_valid_mask: torch.Tensor | None = None) -> None:
    """Assert input shapes are compatible."""
    if values.shape != targets.shape:
        raise ValueError(f"values and targets must have matching shapes, got {values.shape} vs {targets.shape}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1D, got {values.ndim}D")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D, got {targets.ndim}D")
    if policy_valid_mask is not None and policy_valid_mask.ndim != 2:
        raise ValueError(f"policy_valid_mask must be 2D, got {policy_valid_mask.ndim}D")
    if policy_valid_mask is not None and policy_valid_mask.shape[0] != values.shape[0]:
        raise ValueError(f"policy_valid_mask must have matching batch dimension with values, got {policy_valid_mask.shape[0]} vs {values.shape[0]}")

def safe_ppo_critic_loss(values: torch.Tensor, targets: torch.Tensor, policy_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute PPO critic loss with safe values and targets."""
    assert_ppo_critic_input_shapes(values, targets, policy_valid_mask)
    if policy_valid_mask is None:
        valid_boards = torch.ones_like(values, dtype=torch.bool) # [B]
    else:
        valid_boards = policy_valid_mask.any(dim=-1) # [B]
    valid_mask = (valid_boards & torch.isfinite(values) & torch.isfinite(targets)) # [B]
    safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    critic_loss = (safe_values - safe_targets).pow(2)
    return safe_mean(critic_loss, valid_mask) # scalar


def assert_pad_mask_shape(pad_mask: torch.Tensor | None, logprobs_new: torch.Tensor, logprobs_old: torch.Tensor):
    if pad_mask is None:
        return
    if logprobs_new.shape != logprobs_old.shape:
        raise ValueError(f"logprobs_new and logprobs_old must have matching shapes, got {logprobs_new.shape} vs {logprobs_old.shape}")
    if pad_mask.shape != logprobs_old.shape:
        raise ValueError(f"pad_mask and logprobs_old must have matching shapes, got {pad_mask.shape} vs {logprobs_old.shape}")

def safe_ppo_loss(logprobs_new: torch.Tensor,
                  logprobs_old: torch.Tensor,
                  advantages: torch.Tensor,
                  values: torch.Tensor,
                  targets: torch.Tensor,
                  selected_mask: torch.Tensor | None = None,
                  pad_mask: torch.Tensor | None = None,
                  clip_eps: float = 0.2,
                  ppo_coef: float = 1.0,
                  kl_coef: float = 0.01,
                  entropy_coef: float = 0.0, # default to 0 for no entropy loss
                  return_info: bool = False,
                  critic_coef: float = 1.0) -> torch.Tensor | Tuple[torch.Tensor, PPOLossInfo]:
    """Compute PPO loss with safe log probabilities, advantages, values and targets."""
    assert_pad_mask_shape(pad_mask, logprobs_new, logprobs_old)
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    policy_valid_mask = (
        pad_mask
        & torch.isfinite(logprobs_new)
        & torch.isfinite(logprobs_old) 
    )
    if return_info:
        actor_loss, actor_info = safe_ppo_actor_loss(logprobs_new, 
                                                     logprobs_old,
                                                     advantages,
                                                     selected_mask,
                                                     policy_valid_mask,
                                                     clip_eps,
                                                     ppo_coef,
                                                     kl_coef,
                                                     entropy_coef,
                                                     return_info)
        critic_loss = safe_ppo_critic_loss(values, targets, policy_valid_mask)
        actor_info.actor_loss = actor_loss
        actor_info.critic_loss = critic_loss
        return actor_loss + critic_coef * critic_loss, actor_info._detach()
    actor_loss = safe_ppo_actor_loss(logprobs_new,
                                     logprobs_old,
                                     advantages,
                                     selected_mask,
                                     policy_valid_mask, 
                                     clip_eps,
                                     ppo_coef, 
                                     kl_coef, 
                                     entropy_coef, 
                                     return_info)
    critic_loss = safe_ppo_critic_loss(values, targets, policy_valid_mask)
    return actor_loss + critic_coef * critic_loss