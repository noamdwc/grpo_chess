"""
Research-logic invariant tests for step_group_advantage (z-score normalization).

These tests protect the algorithmic decision to z-score normalize per-step group
advantages in the GRPO training loop. They should fail if the normalization
semantics are accidentally reverted to mean-only or removed entirely.

All tests are deterministic, CPU-only, and use toy tensors — no model, Stockfish,
or external data required.
"""

import torch
import pytest

from src.grpo_logic.loss import step_group_advantage, grpo_ppo_loss


# ---------------------------------------------------------------------------
# A — Kept invariants (now verified under z-score semantics)
# ---------------------------------------------------------------------------

def test_group_advantages_are_zero_when_all_rewards_equal():
    """Equal rewards within a group => normalized advantages are all exactly 0.

    Catches: std normalization bugs where equal-reward case produces NaN or
    non-zero values. With z-score, numerator = r - mean = 0 exactly.
    """
    rewards = torch.ones(2, 4, 3)  # [B=2, G=4, T=3]

    # No mask
    out = step_group_advantage(rewards)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), (
        f"Equal rewards should give zero advantages; got max abs {out.abs().max():.2e}"
    )

    # With full mask (all valid)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    out_masked = step_group_advantage(rewards, mask)
    assert torch.allclose(out_masked, torch.zeros_like(out_masked), atol=1e-7), (
        f"Equal rewards with mask should give zero advantages; got max abs {out_masked.abs().max():.2e}"
    )


def test_group_advantages_invariant_to_constant_reward_shift():
    """Adding a constant c to all rewards in a group leaves advantages unchanged.

    With z-score: (r+c - mean(r+c)) / std(r+c) = (r - mean(r)) / std(r).
    Catches: accidental absolute reward dependence, baseline subtraction bugs.
    """
    torch.manual_seed(0)
    rewards = torch.randn(2, 4, 5)
    base = step_group_advantage(rewards)

    for c in [5.0, -3.0, 0.0]:
        shifted = step_group_advantage(rewards + c)
        assert torch.allclose(shifted, base, atol=1e-6), (
            f"Shift c={c}: max diff {(shifted - base).abs().max():.2e}"
        )


def test_group_advantages_permutation_invariant():
    """Reordering trajectories within G should only reorder advantages, not change their values.

    Catches: order-dependent accumulation bugs, wrong group-axis mean/std.
    """
    torch.manual_seed(1)
    rewards = torch.randn(2, 6, 4)  # [B=2, G=6, T=4]
    perm = torch.randperm(6)

    out_original = step_group_advantage(rewards)
    out_permuted = step_group_advantage(rewards[:, perm, :])

    # Un-permute the output and compare
    inv_perm = torch.argsort(perm)
    assert torch.allclose(out_permuted[:, inv_perm, :], out_original, atol=1e-6), (
        "Permuting G and un-permuting should match original advantages"
    )


# ---------------------------------------------------------------------------
# B — Replaced test: z-score scale invariance (was "scale-sensitivity" under mean-only)
# ---------------------------------------------------------------------------

def test_group_zscore_advantages_invariant_to_positive_reward_scaling():
    """Scaling all rewards in a group by alpha > 0 leaves z-score advantages unchanged.

    With z-score: (alpha*r - mean(alpha*r)) / std(alpha*r) = (r - mean(r)) / std(r).
    Catches: missing std normalization (mean-only path would fail this test).

    Note: invariant holds when alpha * std >> eps=1e-8. We use rewards with
    spread ~1.0, so even alpha=0.1 gives std ~0.1 >> 1e-8. Tolerance is loose
    (1e-5) to accommodate floating-point rounding.
    """
    torch.manual_seed(2)
    # Use rewards with sufficient spread so eps doesn't dominate
    rewards = torch.randn(2, 5, 6)  # std across G ≈ 1.0
    base = step_group_advantage(rewards)

    for alpha in [2.0, 0.1, 100.0]:
        scaled = step_group_advantage(alpha * rewards)
        assert torch.allclose(scaled, base, atol=1e-5), (
            f"alpha={alpha}: max diff {(scaled - base).abs().max():.2e}. "
            f"Fails if std normalization is missing."
        )


# ---------------------------------------------------------------------------
# C — Robustness: zero-variance group (std=0 edge case)
# ---------------------------------------------------------------------------

def test_group_zscore_advantages_zero_or_finite_when_group_std_is_zero():
    """When all rewards in a group are equal (std=0), output is finite and ≈ 0.

    The eps=1e-8 denominator floor prevents NaN/Inf. Since the numerator is
    also 0, the result is 0 / eps = 0 exactly.
    Catches: missing epsilon handling, divide-by-zero bugs.
    """
    # Case 1: all equal rewards, G=4
    rewards_equal = torch.full((2, 4, 3), fill_value=3.7)
    out = step_group_advantage(rewards_equal)
    assert torch.isfinite(out).all(), "NaN/Inf with equal rewards (no mask)"
    assert out.abs().max().item() < 1e-5, f"Expected ~0, got {out.abs().max():.2e}"

    # Case 2: G=1 (single trajectory, std always 0 by definition)
    rewards_g1 = torch.randn(3, 1, 5)
    out_g1 = step_group_advantage(rewards_g1)
    assert torch.isfinite(out_g1).all(), "NaN/Inf with G=1"
    assert out_g1.abs().max().item() < 1e-5, f"G=1 should give 0 advantages, got {out_g1.abs().max():.2e}"

    # Case 3: with pad_mask that leaves only 1 valid G entry per (b,t)
    rewards_masked = torch.randn(1, 4, 3)
    mask = torch.zeros(1, 4, 3, dtype=torch.bool)
    mask[0, 0, :] = True  # Only first trajectory is valid for all timesteps
    out_m = step_group_advantage(rewards_masked, mask)
    assert torch.isfinite(out_m).all(), "NaN/Inf with single-valid-G masked case"
    # Padded entries must be 0
    assert out_m[~mask].abs().max().item() == 0.0, "Padded entries should be exactly 0"


# ---------------------------------------------------------------------------
# D — Sign and rank preservation
# ---------------------------------------------------------------------------

def test_group_zscore_advantages_preserve_sign_relative_to_group_mean():
    """Rewards above group mean → positive advantage; below → negative.

    Also verifies rank ordering: the highest-reward trajectory in each group
    has the highest advantage.
    Catches: sign inversion, wrong normalization axis.
    """
    torch.manual_seed(3)
    B, G, T = 3, 6, 4
    rewards = torch.randn(B, G, T)
    out = step_group_advantage(rewards)

    mean_r = rewards.mean(dim=1, keepdim=True)  # [B, 1, T]

    # Sign test: exclude near-mean entries (ties within 1e-4 of mean)
    above_mean = (rewards - mean_r) > 1e-4
    below_mean = (rewards - mean_r) < -1e-4

    assert (out[above_mean] > 0).all(), "Rewards above group mean should have positive advantages"
    assert (out[below_mean] < 0).all(), "Rewards below group mean should have negative advantages"

    # Rank test: highest reward in group has highest advantage (per B, T)
    best_r_idx = rewards.argmax(dim=1)   # [B, T]
    best_a_idx = out.argmax(dim=1)       # [B, T]
    assert torch.equal(best_r_idx, best_a_idx), (
        "Trajectory with highest reward should have highest advantage in each group"
    )


# ---------------------------------------------------------------------------
# E — Masked-path: padded entries are exactly 0, valid entries are z-scored
# ---------------------------------------------------------------------------

def test_step_group_advantage_pad_mask_zeroes_padded_entries():
    """Padded entries (mask=False) are zeroed; valid entries are z-score normalized.

    Catches: mask not applied to output, masking only the mean computation.
    """
    torch.manual_seed(4)
    B, G, T = 1, 4, 3

    rewards = torch.randn(B, G, T)
    # Last timestep: only first 2 of 4 trajectories are valid
    mask = torch.ones(B, G, T, dtype=torch.bool)
    mask[0, 2:, 2] = False  # Pad last 2 trajectories at t=2

    out = step_group_advantage(rewards, mask)

    # Padded positions must be exactly 0
    padded = out[~mask]
    assert padded.abs().max().item() == 0.0, (
        f"Padded entries should be 0, got max abs {padded.abs().max():.2e}"
    )

    # Valid positions at fully-valid timesteps should be z-scored (std ≈ 1)
    for t in range(T - 1):  # t=0 and t=1 are fully valid
        group_adv = out[0, :, t]  # [G]
        assert torch.isfinite(group_adv).all(), f"Non-finite at t={t}"
        # Population std of z-scored values should be ≈ 1
        std_adv = group_adv.std(unbiased=False)
        assert abs(std_adv.item() - 1.0) < 1e-5, (
            f"At t={t}, std of z-scored advantages should be ≈1, got {std_adv:.6f}"
        )

    # Valid entries at the partial timestep (t=2, only G=0,1 valid) should be finite
    valid_t2 = out[0, :2, 2]
    assert torch.isfinite(valid_t2).all()


# ---------------------------------------------------------------------------
# F — Integration: scale invariance survives the full production loss pipeline
# ---------------------------------------------------------------------------

def test_grpo_loss_is_invariant_to_positive_reward_scaling_when_kl_disabled():
    """Policy loss from grpo_ppo_loss is unchanged when step_rewards are scaled by alpha>0,
    provided KL is disabled (kl_coef=0).

    With z-score normalization, alpha * rewards → same advantages → same PPO loss.
    With KL disabled the only term is the PPO clip loss, so the full loss is also
    scale-invariant. This test validates that the invariant survives the complete
    production pipeline, not just the advantage helper.

    Catches: z-score normalization being bypassed or overridden downstream.
    """
    torch.manual_seed(5)
    B, G, T = 2, 4, 5

    # Fixed log-probs — kept identical for both runs (only rewards change)
    logprobs_old = torch.randn(B, G, T) - 2.0   # plausible log-prob range
    # new log-probs slightly different from old to get non-trivial ratios
    logprobs_new = logprobs_old + 0.1 * torch.randn(B, G, T)
    logprobs_new = logprobs_new.detach().requires_grad_(False)

    # Non-degenerate rewards with clear spread across G
    step_rewards = torch.randn(B, G, T)
    pad_mask = torch.ones(B, G, T, dtype=torch.bool)

    base_loss = grpo_ppo_loss(
        logprobs_new, logprobs_old, step_rewards,
        pad_mask=pad_mask, kl_coef=0.0,
    )

    for alpha in [2.0, 0.1, 50.0]:
        scaled_loss = grpo_ppo_loss(
            logprobs_new, logprobs_old, alpha * step_rewards,
            pad_mask=pad_mask, kl_coef=0.0,
        )
        assert torch.allclose(scaled_loss, base_loss, atol=1e-5), (
            f"alpha={alpha}: policy loss changed from {base_loss:.6f} to {scaled_loss:.6f} "
            f"(diff {(scaled_loss - base_loss).abs():.2e}). "
            f"Fails if z-score normalization is missing from the production path."
        )
