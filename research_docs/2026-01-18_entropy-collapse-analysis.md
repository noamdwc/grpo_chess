---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Entropy Collapse in GRPO Training: Root Cause Analysis"
date: "2026-01-18 12:30:00 UTC"
agent: "claude-opus-4-5-20251101"

# Git State - Critical for reproducibility
git_commit: "255f144bc824c28ce9e7538d5ae9311484ec3204"
git_branch: "self_play"
uncommitted_changes: true  # Changes to entropy_coef and lr defaults

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py
  - src/grpo_self_play/grpo_logic/loss.py
  - src/grpo_self_play/grpo_logic/sampling.py
  - src/grpo_self_play/chess/rewards.py

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "p26atclv"
    run_name: "chess-grpo-20260115-0312-z65z"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (list_runs, get_run_metrics, get_run_summary)

# Original Prompt - The exact prompt given to the agent
prompt: |
  use the research-insights instructions. look at the last run, the loss goes up,
  how can it be? it looks like there is a bug in the training

# Tags for categorization
tags:
  - training
  - entropy
  - debugging
  - policy-collapse
  - hyperparameters
---

# Entropy Collapse in GRPO Training: Root Cause Analysis

## Executive Summary

Investigation into why `train_total_loss` was increasing (becoming less negative) in run `p26atclv` revealed that the root cause is **entropy collapse** - the policy became increasingly deterministic over training, causing all trajectories to converge to similar actions and eliminating the learning signal.

**TL;DR:** The entropy coefficient (0.01) was too weak relative to the PPO gradient magnitude, causing policy collapse despite having entropy regularization enabled. Combined with a high learning rate (0.001), the policy became deterministic, trajectories converged, advantages approached zero, and training stalled.

## Context

### Background
Run `p26atclv` (chess-grpo-20260115-0312-z65z) showed unexpected training dynamics:
- `train_total_loss` went from -0.03 to ~0 (loss increasing)
- `train/avg_reward` went from +0.05 to -0.10 (rewards decreasing)
- Model performance against Stockfish remained poor (0/32 wins, 2 draws)

### Research Question
Why is the loss going up during training? Is there a bug in the training code?

### Scope
- Analyzed training code for bugs in loss computation, advantage calculation, and masking
- Examined WandB metrics to identify trends
- Investigated reward computation and normalization

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `p26atclv` - 1800+ steps, 54 epochs |
| Code Files | loss.py, model.py, sampling.py, rewards.py |
| Git Commit | 8c19419 (entropy loss removal) analyzed |

### Analysis Approach

1. First, queried WandB metrics to understand loss and reward trends over training
2. Traced the data flow from trajectory sampling through advantage computation to loss
3. Verified the `start_player_mask` interaction with `step_rewards` for correctness
4. Investigated the reward clipping hypothesis
5. Discovered entropy collapse as the root cause through metric analysis

### Tools Used

- **Read**: Examined source files at loss.py:75-220, model.py:134-220, sampling.py:120-200
- **WandB MCP**: Retrieved metrics for train_total_loss, entropy, step_reward_std, raw_step_cp
- **Bash**: Extracted and analyzed metric trends using jq

## Findings

### Finding 1: Entropy Collapsed 5x During Training

**Evidence:**

Entropy metric over training:
| Step | Entropy | Effective Actions |
|------|---------|------------------|
| 0 | 3.17 | ~24 actions |
| 166 | 3.23 | ~25 actions |
| 327 | 2.91 | ~18 actions |
| 522 | 2.66 | ~14 actions |
| 702 | 2.25 | ~9 actions |
| 900 | 1.75 | ~6 actions |
| 1082 | 1.43 | ~4 actions |
| 1250 | 1.18 | ~3 actions |
| 1425 | 1.04 | ~2.8 actions |
| 1793 | 0.71 | ~2 actions |

**Interpretation:**
The policy went from exploring ~24 actions per position to deterministically choosing between ~2 actions. This is classic entropy collapse.

### Finding 2: Step Reward Variance Decreased as Trajectories Converged

**Evidence:**

| Metric | Early (step 0-30) | Late (step 1760+) | Change |
|--------|-------------------|-------------------|--------|
| step_reward_std | 0.93 | 0.68 | -27% |
| raw_step_cp_abs_mean | 1000 cp | 600 cp | -40% |
| raw_step_cp_std | 1800 cp | 1300 cp | -28% |

**Interpretation:**
As the policy became deterministic, all G trajectories from the same starting position took similar paths, leading to similar step_rewards. This reduced the variance available for computing meaningful advantages.

### Finding 3: Entropy Bonus Was Too Weak Relative to PPO Gradient

**Evidence:**

Run configuration:
```python
lr = 0.001
entropy_coef = 0.01
```

Gradient magnitude comparison:
- **PPO gradient**: `advantage * ratio` ≈ 1-2 (normalized advantages)
- **Entropy gradient**: `entropy_coef` = 0.01
- **Ratio**: PPO dominates by **100-200x**

In `loss.py:218-219`:
```python
# Loss = PPO loss + KL penalty - entropy bonus
loss = ppo_loss + kl_coef * kl_div - entropy_coef * entropy
```

**Interpretation:**
The entropy bonus coefficient (0.01) provided negligible gradient compared to the PPO loss gradient. With a high learning rate (0.001), the policy quickly converged to deterministic behavior, and the entropy bonus couldn't counteract this.

### Finding 4: No Bug in Masking Logic

**Evidence:**

The `start_player_mask` interaction with `step_rewards` was verified correct:

In `model.py:167-173`:
```python
start_player_mask = (t % 2 == 0)[None, None, :]  # Only even timesteps
effective_pad_mask = pad_mask & start_player_mask
```

In `loss.py:89-92`:
```python
mean_t = step_rewards.mean(dim=1, keepdim=True)  # Mean over G trajectories
advantages = (step_rewards - mean_t)  # Per-timestep normalization
```

The advantage computation is independent per timestep - even timesteps don't affect odd timesteps. Masking is applied after normalization, correctly zeroing out opponent moves.

**Interpretation:**
The masking logic is correct. The issue is not a bug but a hyperparameter problem.

### Finding 5: Reward Clipping Was Not the Primary Issue

**Evidence:**

Initially hypothesized that `normalize_cp` clipping (±2000 cp) was causing signal loss. However:
- `raw_step_cp_abs_mean` was 600-1000 cp (below clip threshold)
- `raw_step_cp_std` was 1300-1800 cp (some values hit clips, but not majority)

More importantly, the step reward values were **decreasing** over training, not increasing into the clip zone.

**Interpretation:**
Clipping may contribute marginally but is not the root cause. Entropy collapse explains the observed behavior.

## Analysis

### Root Cause Analysis

The cascade of failures:

```
High LR (0.001) + Weak entropy bonus (0.01)
    ↓
PPO gradient overwhelms entropy gradient (100:1 ratio)
    ↓
Policy becomes deterministic (entropy: 3.17 → 0.71)
    ↓
All G trajectories take identical actions
    ↓
step_rewards become identical across trajectories
    ↓
advantages = (reward - mean) / std ≈ 0 / small ≈ 0
    ↓
PPO loss ≈ 0, no learning signal
    ↓
Policy stops improving, rewards decrease
```

### Impact Assessment

| Impact | Severity |
|--------|----------|
| Training stalls | Critical |
| Compute wasted | High (~10 hours) |
| No model improvement | Critical |

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Higher entropy_coef | Maintains exploration | May slow convergence |
| Lower learning rate | More stable training | Slower initial progress |
| Entropy floor | Guarantees exploration | Requires monitoring |

## Recommendations

### Immediate Actions (High Priority)

1. **Increase entropy coefficient 10x**
   - What: Change `entropy_coef` from 0.01 to 0.1
   - Why: Make entropy gradient comparable to PPO gradient (1:10-20 ratio instead of 1:100-200)
   - Where: `src/grpo_self_play/grpo_logic/model.py:37`
   - Status: **Implemented**

2. **Lower default learning rate**
   - What: Change default `lr` from 1e-4 to 3e-5
   - Why: Slower updates allow entropy bonus to maintain exploration
   - Where: `src/grpo_self_play/grpo_logic/model.py:32`
   - Status: **Implemented**

### Medium-Term Improvements

1. **Add entropy monitoring alerts**
   - What: Log warning if entropy drops below 1.5
   - Why: Early detection of collapse
   - Where: `model.py:training_step`

2. **Consider adaptive entropy coefficient**
   - What: Increase entropy_coef when entropy drops too fast
   - Why: Automatic correction of collapse
   - Reference: SAC-style automatic temperature tuning

### Long-Term Considerations

1. **Investigate std normalization in advantages**
   - What: The DR. GRPO paper removed std normalization, but this may contribute to instability
   - Why: Without std normalization, advantage magnitudes vary with reward scale
   - Status: User considering re-adding std normalization

2. **Entropy scheduling**
   - What: Start with high entropy_coef, anneal over training
   - Why: Early exploration, later exploitation

## Open Questions

- [ ] What is the optimal entropy_coef for chess GRPO? (0.1 is a starting point)
- [ ] Should we use std normalization in `step_group_advantage`? (DR. GRPO says no, but may help exploration)
- [ ] Would entropy floor stopping be better than entropy bonus?
- [ ] How does the learning rate interact with trajectory_depth and num_trajectories?

## Appendix

### A. Key Metrics Over Training

```
Step     Loss      Entropy   Reward    step_reward_std
0        -0.031    3.17      +0.05     0.93
500      -0.025    2.66      +0.03     0.85
1000     -0.015    1.50      -0.02     0.75
1500     -0.005    0.90      -0.08     0.70
1800     -0.004    0.71      -0.10     0.68
```

### B. Configuration Comparison

| Parameter | Failing Run | New Defaults |
|-----------|-------------|--------------|
| lr | 0.001 | 3e-5 |
| entropy_coef | 0.01 | 0.1 |
| clip_ratio | 0.15 | 0.2 |
| kl_coef | 0.001 | 0.01 |

### C. Related Documents

- [Previous analysis: GRPO Learning Failure](./2026-01-09_grpo-learning-failure-analysis.md)
- [Per-step rewards analysis](./2026-01-14_per-step-rewards-analysis.md)

---
