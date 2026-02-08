---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Clean Run Stuck: Temperature Mismatch Bug Between Sampling and PPO Causes 55% Clip Fraction at Step 0"
date: "2026-02-07 12:00:00 UTC"
agent: "claude-opus-4-6"

# Git State - Critical for reproducibility
git_commit: "150788d638e77e423959879428606628b2fabdcb"
git_branch: "self_play"
uncommitted_changes: false

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py    # training_step, _ppo_step, on_train_epoch_start
  - src/grpo_self_play/grpo_logic/loss.py      # grpo_ppo_loss, step_group_advantage
  - src/grpo_self_play/grpo_logic/sampling.py   # batched_policy_step, sample_trajectories_batched
  - src/grpo_self_play/models.py               # get_group_log_probs, get_legal_moves_probs, get_legal_moves_logits
  - src/grpo_self_play/chess/rewards.py        # normalize_cp
  - src/grpo_self_play/configs/default.yaml    # Clean run config
  - src/grpo_self_play/trainer.py              # Trainer setup
  - src/grpo_self_play/train_self_play.py      # Entry point
  - src/grpo_self_play/pretrain/pretrain.py    # Pretrain pipeline (no eval_stockfish)

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "p9u63rgu"
    run_name: "chess-grpo-20260205-2306-lkav"
  - run_id: "xkbkwo4s"
    run_name: "chess-grpo-20260204-1956-1iaa"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary, get_plots, compare_runs)

# Original Prompt - The exact prompt given to the agent
prompt: |
  Look at the last research document. The last run is the "clean run" — the learning is
  still stuck and the model doesn't learn to play chess at all.

# Tags for categorization
tags:
  - training
  - bug
  - temperature
  - ppo
  - clip-fraction
  - critical-fix
---

# Clean Run Stuck: Temperature Mismatch Bug Between Sampling and PPO

## Executive Summary

**TL;DR:** The model fails to learn because of a temperature mismatch bug: trajectory sampling computes old log probs at `temperature=1.3` while the PPO step computes new log probs at `temperature=1.0` (the default). This causes the PPO ratio to deviate from 1.0 even when model weights are identical, producing a **55% clip fraction at step 0** before any learning has occurred. Nearly half of all gradient signal is destroyed by clipping an artifact, not real policy divergence. Fixing this single bug should dramatically reduce clip fraction and allow PPO to actually transmit reward signal.

## Context

### Background
The previous research document (`2026-02-06_loss-budget-and-monitor-analysis.md`) identified entropy dominance (95% of gradient) and a saturated adaptive KL controller as the reasons run `1iaa` failed to learn. A "clean run" (`lkav`, run `p9u63rgu`) was launched with all recommendations applied: `entropy_coef=0.0`, `kl_coef=0.001`, `lr=1e-6`, monitors disabled. Despite these fixes, the model still shows zero chess improvement over 6800 steps.

### Research Question
1. Why does the clean run still fail to learn despite fixing entropy dominance?
2. Why is clip fraction ~48% throughout the run?
3. What is the root cause of the model being stuck at ~3% eval score?

### Scope
- Clean run analysis (`p9u63rgu` / `chess-grpo-20260205-2306-lkav`)
- Code analysis of temperature handling across sampling and PPO
- Comparison with previous run `1iaa`

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `p9u63rgu` / `chess-grpo-20260205-2306-lkav` (~6800 steps) |
| WandB Run (prev) | `xkbkwo4s` / `chess-grpo-20260204-1956-1iaa` (~6800 steps) |
| Code | At commit `150788d` |

### Clean Run Hyperparameters

| Parameter | Value | Change from 1iaa |
|-----------|-------|-------------------|
| lr | 1e-6 | Reduced from 3e-6 |
| entropy_coef | 0.0 | Reduced from 0.1 |
| kl_coef | 0.001 | Reduced from 0.01 (was 0.1 via adaptive) |
| adaptive_kl | false | Was true (saturated at max) |
| use_entropy_floor | false | Was true (never triggered) |
| rollout_temperature | 1.3 | Unchanged |
| clip_ratio | 0.2 | Unchanged |
| ppo_steps | 1 | Unchanged |

## Findings

### Finding 1 (CRITICAL): Temperature mismatch between sampling and PPO evaluation

**The Bug:**

During trajectory sampling (`sampling.py:96-100`):
```python
probs = model.get_legal_moves_probs(states_tensor, legal_mask, temperature)  # temperature=1.3
chosen_log_probs = torch.log(chosen_probs + 1e-12)
```

During PPO step (`model.py:513-514`):
```python
new_log_probs = self.policy_model.get_group_log_probs(
    trajectories_states, trajectories_actions, trajectories_legal_masks
)  # temperature defaults to 1.0!
```

Both functions call `get_legal_moves_logits` (`models.py:118`):
```python
logits = self(tensor_state) / temperature
```

The old log probs use `logits / 1.3` (flatter distribution), while new log probs use `logits / 1.0` (sharper distribution). The PPO ratio `exp(new_log_prob - old_log_prob)` is therefore systematically biased even when model weights are identical.

**Evidence — Step 0 metrics (old_model == new_model):**
- `mean_clip_fraction` at step 0: **0.556** (55.6% of ratios outside [0.8, 1.2])
- `mean_ratio` at step 0: **1.012** (close to 1.0 because increases and decreases cancel)

If both used the same temperature, clip fraction at step 0 would be **0.0** and mean ratio would be **1.0**.

**Why mean_ratio ≈ 1.0 despite the bug:**

For a given state with logits `z`, the temperature mismatch creates:
- High-prob actions: `softmax(z/1.0)[a] > softmax(z/1.3)[a]` → ratio > 1
- Low-prob actions: `softmax(z/1.0)[a] < softmax(z/1.3)[a]` → ratio < 1
- Since probabilities sum to 1, these deviations balance out in expectation → mean ratio ≈ 1.0

But individual ratios deviate significantly, causing massive clipping.

**Example:** For 3 legal moves with logits [2, 1, 0]:

| | temp=1.3 (old) | temp=1.0 (new) | ratio |
|---|---|---|---|
| Move A | 0.596 | 0.665 | 1.116 |
| Move B | 0.276 | 0.245 | 0.888 |
| Move C | 0.128 | 0.090 | **0.703** (clipped!) |

With ~30 legal moves in a typical chess position, most low-probability moves get ratios well below 0.8. Since sampling at temp=1.3 explores these moves more often, a large fraction of sampled actions get clipped.

### Finding 2: All reward and training metrics are completely flat

**Rolling 500-step averages for key metrics:**

| Metric | Steps 0-500 | Steps 3000-3500 | Steps 6500-7000 | Trend |
|--------|-------------|-----------------|-----------------|-------|
| train/avg_reward | -0.158 | -0.132 | -0.188 | Flat (noise) |
| ppo_loss | -0.002 | -0.006 | -0.006 | Flat |
| mean_clip_fraction | 0.536 | 0.474 | 0.475 | Slight decrease |
| train/reward_std | 0.869 | 0.854 | 0.867 | Flat |

No learning signal in any metric over 6800 steps. The model is not improving at chess.

**Full summary statistics:**

| Metric | Mean | Min | Max | First | Last |
|--------|------|-----|-----|-------|------|
| train/avg_reward | -0.151 | -1.180 | 0.282 | 0.003 | 0.061 |
| ppo_loss | -0.005 | -0.016 | 0.008 | -0.001 | -0.007 |
| mean_clip_fraction | 0.483 | 0.407 | 0.587 | 0.556 | 0.478 |
| entropy | 2.871 | 2.588 | 3.160 | 2.816 | 2.918 |
| mean_kl_divergence | 0.045 | 0.023 | 0.077 | 0.046 | 0.036 |
| train/reward_std | 0.865 | 0.639 | 1.236 | 0.817 | 0.962 |
| train/reward_best | 2.183 | 1.0 | 4.0 | 1.795 | 2.269 |
| train/reward_p90 | 0.901 | -0.081 | 1.580 | 0.975 | 1.274 |

### Finding 3: Entropy remains healthy without entropy bonus

Entropy stayed at 2.59-3.16 (mean 2.87) with `entropy_coef=0.0`. Legal move masking naturally maintains entropy — the model can't collapse to a few moves because it only sees legal moves. This confirms removing the entropy bonus was correct.

### Finding 4: Eval scores are identical to the broken run

| Run | eval_stockfish/score values |
|-----|---------------------------|
| lkav (clean) | 0.016, 0.023, 0.031 |
| 1iaa (broken) | 0.016, 0.047, 0.031, fluctuated, 0.031 |

Both runs hover at ~3% win rate against Stockfish. The clean run config changes had zero effect on chess performance because the temperature bug was present in both runs.

### Finding 5: Reward variance exists but advantage metrics are not logged

`train/reward_std` averages 0.87 across the run, indicating there IS meaningful variance in trajectory rewards (reward range is approximately [-2, 2]). This suggests the advantage signal is not degenerate — the problem is the gradient not reaching the model due to clipping.

However, `train/advantage_mean` and `train/advantage_std` are not logged, making it impossible to verify the actual advantage magnitudes after group normalization.

### Finding 6: Pretrain pipeline has no eval_stockfish logging

The pretrain pipeline (`pretrain/pretrain.py`) only logs supervised metrics (loss, accuracy, top5_accuracy, entropy, perplexity). There is no evaluation against Stockfish during pretraining, so we have no baseline for how well the pretrained model plays chess before GRPO begins.

## Analysis

### Root Cause: The Temperature Mismatch Bug

The primary reason the model doesn't learn is the temperature mismatch between sampling (temp=1.3) and PPO evaluation (temp=1.0):

1. **55% of all gradient signal is destroyed at step 0.** Before any learning happens, half the state-action pairs are clipped due to the temperature artifact, not policy divergence.

2. **The remaining 45% of gradient signal is contaminated.** Non-clipped ratios are biased by the temperature difference — they push the policy toward the temp=1.0 shape regardless of the advantage signal.

3. **This bias is constant and does not diminish.** Unlike real policy staleness (which resets each epoch when old policy syncs), the temperature mismatch is a permanent systematic error present at every step.

4. **The clip fraction slightly decreases over training (55% → 48%).** This suggests the model is slowly adapting its weight distribution to partially compensate for the temperature mismatch, rather than learning chess.

### Why Previous Fixes Had No Effect

The entropy and KL fixes from the previous research document were correct diagnoses, but they address secondary problems. Even with perfect loss weighting, the PPO gradient is mostly destroyed or corrupted by the temperature mismatch. It's like fixing the fuel mixture on an engine with a broken crankshaft.

### Severity Assessment

This is a **critical correctness bug** that has been present since the PPO implementation was written. Every GRPO training run has been affected. No meaningful chess learning is possible until this is fixed.

## Recommendations

### Immediate Action: Fix the temperature mismatch

**In `model.py:_ppo_step` (line 513)**, pass the rollout temperature:

```python
# Before (buggy):
new_log_probs = self.policy_model.get_group_log_probs(
    trajectories_states, trajectories_actions, trajectories_legal_masks
)

# After (fixed):
new_log_probs = self.policy_model.get_group_log_probs(
    trajectories_states, trajectories_actions, trajectories_legal_masks,
    temperature=self.hparams.grpo_config.rollout_temperature,
)
```

**Expected impact:**
- Clip fraction at step 0 should drop to ~0% (since old_model == new_model at epoch start)
- Clip fraction during epoch should reflect only real policy divergence (likely 5-15%)
- PPO gradient signal increases ~10x (from ~45% useful to ~85-95% useful)

### Secondary: Add advantage logging

Log advantage statistics for future debugging:
- `train/advantage_mean`: Mean of group-normalized advantages
- `train/advantage_std`: Std of group-normalized advantages

### Secondary: Add eval_stockfish to pretrain pipeline

The pretrain pipeline needs Stockfish evaluation to establish a baseline. Without knowing the pretrained model's chess ability, we can't tell whether GRPO is improving on a reasonable starting point or trying to bootstrap from random play.

### After the fix

With the temperature bug fixed, re-run with the current clean config (`entropy_coef=0.0`, `kl_coef=0.001`, `lr=1e-6`). Monitor:
1. **Clip fraction**: Should start near 0% each epoch, increase within epoch. If it stays > 20%, consider syncing old policy more frequently.
2. **train/avg_reward**: Should show an upward trend if GRPO is working.
3. **eval_stockfish/score**: Should improve over time.

If the model still doesn't learn after the fix, the next hypotheses to investigate are:
- Model capacity (256 embed, 4 layers, 2 frozen — only ~3.3M trainable params)
- Pretrained base quality (is the starting policy better than random?)
- Advantage normalization (currently no std normalization in `step_group_advantage`)

## Open Questions

- [ ] After fixing the temperature bug, what is the "natural" clip fraction within an epoch? Is 16 steps between old policy syncs too many?
- [ ] Should rollout_temperature be 1.0 (matching PPO evaluation) instead of fixing PPO evaluation to match rollout? This is a design question about whether temperature-based exploration is desirable in GRPO.
- [ ] How good is the pretrained model? We need eval_stockfish in the pretrain pipeline.
- [ ] Is the reward signal (dense step rewards via Stockfish) actually informative for a near-random policy?

## Appendix

### A. The Temperature Bug in Detail

In `models.py:104-119`:
```python
def get_legal_moves_logits(self, tensor_state, legal_moves_mask, temperature=1.0):
    logits = self(tensor_state) / temperature
    return logits.masked_fill(~legal_moves_mask, -float('inf'))
```

The temperature divides the raw logits before softmax. Higher temperature → flatter distribution. The issue is that `get_group_log_probs` (used in PPO) defaults to `temperature=1.0` while sampling uses the configured `rollout_temperature=1.3`.

### B. Related Documents

- [Loss Budget Analysis](./2026-02-06_loss-budget-and-monitor-analysis.md) — identified entropy/KL issues (now fixed), but temperature bug was not detected
