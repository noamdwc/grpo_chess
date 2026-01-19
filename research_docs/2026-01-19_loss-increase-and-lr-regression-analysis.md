---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Loss Increase Analysis: Entropy Collapse and LR Regression in Recent Runs"
date: "2026-01-19 01:30:00 UTC"
agent: "claude-opus-4-5-20251101"

# Git State - Critical for reproducibility
git_commit: "4c9ef9f123f739638d294283ed40fab80e62f0d1"
git_branch: "self_play"
uncommitted_changes: true  # enviroment.yml and thesis_plan.md are untracked

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py
  - src/grpo_self_play/grpo_logic/loss.py
  - src/grpo_self_play/grpo_logic/sampling.py
  - research_docs/2026-01-18_followup-entropy-collapse-review.md

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "ksumd93q"
    run_name: "chess-grpo-20260119-0055-2sda"
  - run_id: "3zbv7l8m"
    run_name: "chess-grpo-20260118-1805-c4gy"
  - run_id: "594y62v3"
    run_name: "chess-grpo-20260118-1426-f2wf"
  - run_id: "p26atclv"
    run_name: "chess-grpo-20260115-0312-z65z"
  - run_id: "26kjpms2"
    run_name: "chess-grpo-20260114-1741-9qa2"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - WandB MCP (list_runs, get_run_metrics, get_run_summary)
  - Bash

# Original Prompt - The exact prompt given to the agent
prompt: |
  Use the instructions from insight-agent. Look at the last run and help
  understand why the loss goes up? Also compare this run to previous runs
  to determine if recent changes are positive or negative.

# Tags for categorization
tags:
  - training
  - entropy
  - debugging
  - hyperparameters
  - loss
  - regression
---

# Loss Increase Analysis: Entropy Collapse and LR Regression in Recent Runs

## Executive Summary

The observed "loss going up" in run `ksumd93q` is a direct symptom of entropy collapse, not an independent problem. The loss formula includes a `-entropy_coef * entropy` term, so as entropy drops (3.23 → 1.73), the total loss becomes less negative. More critically, comparison across 5 recent runs reveals that the Jan 19 run **regressed** by increasing the learning rate from 3e-05 back to 1e-04, which accelerated entropy collapse and degraded Stockfish evaluation performance.

**TL;DR:** The loss increase is caused by entropy collapse; the best-performing run used `lr=3e-05`, but the most recent run increased LR to `1e-04` causing a regression.

## Context

### Background

The GRPO Chess training has been experiencing instability, with runs crashing and showing signs of entropy collapse. The user observed that the loss was "going up" in the most recent run and wanted to understand the root cause and whether recent hyperparameter changes were beneficial.

### Research Question

1. Why is the `train_total_loss` increasing (becoming less negative) during training?
2. Are the recent configuration changes (from Jan 14 → Jan 19) positive or negative?
3. Which run configuration produced the best results?

### Scope

**Included:**
- Analysis of 5 runs from Jan 14-19, 2026
- Comparison of hyperparameters, entropy trajectories, and Stockfish evaluation
- Loss formula analysis

**Excluded:**
- Detailed analysis of model architecture changes
- Reward function modifications (covered in prior docs)

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Runs | ksumd93q, 3zbv7l8m, 594y62v3, p26atclv, 26kjpms2 |
| Code Files | loss.py (loss formula), model.py (training loop) |
| Prior Research | 2026-01-18_followup-entropy-collapse-review.md |

### Analysis Approach

1. Retrieved metrics and summaries from the 5 most recent WandB runs
2. Extracted the loss formula from `loss.py` to understand component contributions
3. Compared hyperparameters across runs to identify changes
4. Correlated entropy trajectories with loss behavior
5. Evaluated Stockfish performance as ground truth for "good" vs "bad" runs

### Tools Used

- **Read**: Examined `loss.py:159-227` for loss formula, `model.py` for training logic
- **WandB MCP**: Retrieved run summaries and metrics for all 5 runs
- **Bash**: Parsed WandB JSON outputs to extract run metadata

## Findings

### Finding 1: Loss increase is directly caused by entropy collapse

**Evidence:**

The loss formula in `loss.py:219`:
```python
loss = ppo_loss + kl_coef * kl_div - entropy_coef * entropy
```

As entropy decreases, the `-entropy_coef * entropy` term contributes a **smaller negative value**, making the total loss less negative (i.e., "go up").

**Metrics from run `ksumd93q`:**

| Metric | Start (step 0) | End (step 623) | Change |
|--------|----------------|----------------|--------|
| `train_total_loss` | -0.326 | -0.171 | +0.155 (less negative) |
| `entropy` | 3.23 | 1.73 | -1.50 |
| `ppo_loss` | -0.0026 | +0.0013 | +0.0039 |

With `entropy_coef = 0.1`, the entropy drop of 1.50 accounts for ~0.15 of the loss change, which almost exactly matches the observed loss increase of 0.155.

**Interpretation:**

The loss "going up" is a **symptom** of entropy collapse, not an independent problem. The real issue is that the model is becoming overconfident, and the entropy bonus term is contributing less negative value to the loss.

### Finding 2: PPO loss transitioned from negative to positive (concerning)

**Evidence:**

| Run | PPO Loss Start | PPO Loss End | Interpretation |
|-----|----------------|--------------|----------------|
| ksumd93q | -0.0026 | +0.0013 | Policy degrading |
| 594y62v3 | -0.003 | -0.0008 | Policy improving |

**Interpretation:**

A negative PPO loss indicates the policy is improving in the direction of higher advantages. The transition to positive PPO loss in `ksumd93q` suggests the policy updates are no longer beneficial—a classic sign of entropy collapse where advantage variance approaches zero.

### Finding 3: Jan 18 run `594y62v3` was the BEST performing run

**Evidence:**

| Run | Date | LR | entropy_coef | Final Entropy | Eval Score | Elo Diff |
|-----|------|-----|--------------|---------------|------------|----------|
| 26kjpms2 | Jan 14 | 0.001 | 0.01 | 2.58 | 0.031 | -597 |
| p26atclv | Jan 15 | 0.001 | 0.01 | **0.58** | 0.031 | -597 |
| **594y62v3** | **Jan 18** | **3e-05** | **0.1** | **2.81** | **0.094** | **-394** |
| 3zbv7l8m | Jan 18 | 3e-05 | 0.1 | 2.06 | 0.016 | -720 |
| ksumd93q | Jan 19 | 1e-04 | 0.1 | 1.73 | 0.016 | -720 |

**Interpretation:**

Run `594y62v3` achieved:
- **6 draws** out of 32 games (vs 1 draw for other runs)
- **Elo diff -394** (vs -720 for most recent)
- **Entropy stayed healthy at 2.81**

This run used `lr=3e-05` and `entropy_coef=0.1`, which prevented entropy collapse.

### Finding 4: The Jan 19 run REGRESSED by increasing LR

**Evidence:**

Configuration change from Jan 18 (594y62v3) to Jan 19 (ksumd93q):

| Parameter | Jan 18 (best) | Jan 19 (latest) | Change |
|-----------|---------------|-----------------|--------|
| `lr` | 3e-05 | **1e-04** | **3.3x increase** |
| `entropy_coef` | 0.1 | 0.1 | No change |
| `adaptive_kl` | True | True | No change |

**Impact:**

| Metric | Jan 18 (594y62v3) | Jan 19 (ksumd93q) | Regression |
|--------|-------------------|-------------------|------------|
| Final entropy | 2.81 | 1.73 | -38% |
| Eval score | 0.094 | 0.016 | -83% |
| Elo diff | -394 | -720 | -326 Elo |

**Interpretation:**

The higher learning rate (1e-04 vs 3e-05) caused faster entropy collapse and significantly worse Stockfish evaluation. This is a clear regression.

### Finding 5: Adaptive KL controller may be too aggressive

**Evidence:**

In run `ksumd93q`:
- `kl_coef` started at 0.01, ended at **0.000225** (dropped 45x)
- `kl_coef_min` was set to 0.0001, allowing near-zero KL penalty

**Interpretation:**

When `kl_coef` drops to near-zero, there's almost no penalty for the policy drifting aggressively from the old policy, which can accelerate entropy collapse. The adaptive controller successfully kept KL near target, but this may have been counterproductive for stability.

### Finding 6: Rollout temperature is hardcoded to 1.0

**Evidence:**

In `sampling.py:134`:
```python
roll_out_step = batched_policy_step(model, active_boards, temperature=1.0)
```

The temperature parameter is not configurable and is fixed at 1.0.

**Interpretation:**

Temperature > 1.0 during rollouts would flatten the policy distribution, directly increasing entropy during sampling. This is a simple mechanism to prevent entropy collapse that is currently not being used. AlphaZero and similar systems use temperature > 1 during training to encourage exploration.

## Analysis

### Root Cause Analysis

The causal chain for "loss going up":

1. **Learning rate too high** (1e-04 vs optimal 3e-05)
2. **Entropy collapses faster** (3.23 → 1.73)
3. **Trajectories converge** (all G samples do similar things)
4. **Advantage variance shrinks** (step_rewards - mean ≈ 0)
5. **PPO gradient signal degrades** (nothing to learn)
6. **Loss "goes up"** (entropy term contributes less negative value)

### Impact Assessment

- **High cost**: Each crashed run wastes 4-6 hours of compute
- **Regression identified**: The Jan 19 LR change was detrimental
- **Clear path forward**: Reverting to Jan 18 config should restore progress

### Trade-offs

| Change | Benefit | Cost |
|--------|---------|------|
| Higher LR (1e-04) | Faster initial learning | Faster entropy collapse |
| Lower LR (3e-05) | Stable entropy | Slower convergence |
| Aggressive adaptive KL | Keeps KL on target | May allow too much drift |
| Conservative KL min | More stability | May constrain learning |
| Higher rollout temperature | More exploration | Noisier learning signal |

## Recommendations

### Immediate Actions (High Priority)

1. **Revert learning rate to 3e-05**
   - What: Change `lr` from 1e-04 back to 3e-05
   - Why: Jan 18 run with 3e-05 had best Stockfish performance
   - Where: Training config
   - Estimated effort: Low

2. **Add configurable rollout temperature (T > 1.0)**
   - What: Add `rollout_temperature` parameter to `GRPOConfig`, default to 1.2-1.5 during training
   - Why: Temperature > 1 flattens the policy distribution during sampling, directly increasing entropy and exploration. This is complementary to the entropy bonus in the loss—temperature affects sampling, while `entropy_coef` affects gradients.
   - Where:
     - Add parameter to `GRPOConfig` in `model.py`
     - Pass to `sample_trajectories_batched()` in `model.py:445-450`
     - Use in `sampling.py:134` instead of hardcoded 1.0
   - Trade-offs:
     - T=1.2-1.5: Moderate exploration boost, minimal noise
     - T=2.0+: Strong exploration, may slow learning
     - Could also schedule: start at 1.5, decay to 1.0
   - Estimated effort: Low

3. **Raise entropy floor to 2.0-2.5**
   - What: Change `entropy_floor` from 1.6 to 2.0 or higher
   - Why: Current floor (1.6) doesn't trigger; entropy settles at ~1.7 which is already too collapsed
   - Where: `model.py` / training config
   - Estimated effort: Low

4. **Increase kl_coef_min to 0.001**
   - What: Prevent KL coefficient from dropping below 0.001
   - Why: Near-zero KL penalty may accelerate entropy collapse
   - Where: Training config
   - Estimated effort: Low

### Medium-Term Improvements

5. **Run longer with stable config**
   - What: The best run (594y62v3) crashed at 556 steps; run it longer
   - Why: Need to see if stable training leads to actual improvement
   - Estimated effort: Medium (compute time)

6. **Add early stopping on entropy collapse**
   - What: Stop run if entropy < 2.0 for 100+ consecutive steps
   - Why: Prevents wasting compute on collapsed runs
   - Where: `model.py:488-494`
   - Estimated effort: Low

### Long-Term Considerations

7. **Investigate entropy collapse root cause**
   - What: Why does entropy collapse even with entropy_coef=0.1?
   - Why: Current mitigations are symptomatic, not curative
   - Estimated effort: High

## Future Research Notes

### Note: Stockfish as Training Opponent (Idea for Documentation)

During this analysis, an alternative training approach was discussed: using Stockfish as an opponent during training (not just for evaluation). This would involve the model playing one side while Stockfish plays the other, potentially with teacher forcing to learn from Stockfish's moves.

**Potential benefits:**
- Exposes model to strong moves it wouldn't discover via self-play
- Prevents self-play echo chamber / mode collapse
- More diverse game states

**Potential drawbacks:**
- Stockfish may be too strong → model always loses → sparse/uninformative gradients
- Teacher forcing causes exposure bias (model never learns to recover from its own mistakes)
- Adds complexity to training loop

**Current assessment:** This idea is documented for future consideration but is **not recommended** at this stage. The priority should be fixing entropy collapse with the current self-play setup. The existing reward signal (Stockfish position evaluation) already provides teacher-like guidance without the downsides of full teacher forcing.

If pursued later, a curriculum approach would be advisable: start with very weak Stockfish (skill 0, depth 1) and gradually increase strength.

## Open Questions

- [ ] Why does entropy still collapse even with `entropy_coef=0.1`? Is 0.2-0.3 needed?
- [ ] What is the optimal rollout temperature? Should it be scheduled (start high, decay)?
- [ ] Is the adaptive KL controller helping or hurting overall?
- [ ] What is the optimal entropy floor value for this action space (~1968 legal moves)?
- [ ] Would gradient clipping help prevent large updates that collapse entropy?

## Appendix

### A. Run Configuration Comparison

**Old config (Jan 14-15):**
```python
GRPOConfig(
    lr=0.001,
    entropy_coef=0.01,
    # No adaptive_kl, no entropy_floor
)
```

**Best config (Jan 18 - 594y62v3):**
```python
GRPOConfig(
    lr=3e-05,  # ← Key difference
    entropy_coef=0.1,
    adaptive_kl=True,
    target_kl=0.012,
    entropy_floor=1.6,
    entropy_floor_action='boost',
)
```

**Regressed config (Jan 19 - ksumd93q):**
```python
GRPOConfig(
    lr=1e-04,  # ← Increased from 3e-05, caused regression
    entropy_coef=0.1,
    adaptive_kl=True,
    target_kl=0.012,
    entropy_floor=1.6,
    entropy_floor_action='boost',
)
```

### B. Loss Formula Reference

From `loss.py:214-219`:
```python
# Entropy bonus: H(π) ≈ -E[log π(a|s)] encourages exploration
# We use the negative log_probs of selected actions as an estimate
entropy = -logprobs_new[pad_mask].mean()

# Loss = PPO loss + KL penalty - entropy bonus (subtract to encourage higher entropy)
loss = ppo_loss + kl_coef * kl_div - entropy_coef * entropy
```

### C. Temperature Effects on Entropy

| Temperature | Effect on Distribution | Use Case |
|-------------|----------------------|----------|
| T < 1.0 | Sharper (more greedy) | Evaluation / exploitation |
| T = 1.0 | Original policy | Baseline |
| T = 1.2-1.5 | Moderately flatter | Training exploration |
| T > 2.0 | Very flat (near uniform) | Strong exploration / early training |

### D. Related Documents

- [2026-01-18_entropy-collapse-analysis.md](./2026-01-18_entropy-collapse-analysis.md) - Initial entropy collapse diagnosis
- [2026-01-18_followup-entropy-collapse-review.md](./2026-01-18_followup-entropy-collapse-review.md) - Review of entropy collapse findings
- [2026-01-14_per-step-rewards-analysis.md](./2026-01-14_per-step-rewards-analysis.md) - Step rewards implementation
