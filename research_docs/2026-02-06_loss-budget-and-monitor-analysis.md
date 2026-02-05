---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Loss Budget Analysis: Entropy Dominance, Dead Monitors, and Why the Model Is Not Learning"
date: "2026-02-06 00:45:00 UTC"
agent: "claude-opus-4-6"

# Git State - Critical for reproducibility
git_commit: "70a63b840bba6d47c893ef8328c9ce628a9a79a8"
git_branch: "self_play"
uncommitted_changes: false

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py    # EntropyFloorMonitor, AdaptiveKLController, GRPOConfig, training_step
  - src/grpo_self_play/grpo_logic/loss.py      # grpo_ppo_loss, loss component computation
  - src/grpo_self_play/grpo_logic/sampling.py   # Trajectory sampling, reward computation
  - src/grpo_self_play/chess/rewards.py         # Stockfish reward, normalize_cp

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "xkbkwo4s"
    run_name: "chess-grpo-20260204-1956-1iaa"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary)

# Original Prompt - The exact prompt given to the agent
prompt: |
  The last run "chess-grpo-20260204-1956-1iaa" had a peak eval_stockfish/score of 0.1,
  and then dropped. Analyze this run and try to understand what happened. Did the entropy
  monitor or AdaptiveKLController hurt the training? Maybe we don't really need them.
  Also analyze why the model mean is capped around 0.03.

# Tags for categorization
tags:
  - training
  - loss-analysis
  - entropy
  - kl-divergence
  - hyperparameters
  - debugging
  - monitors
---

# Loss Budget Analysis: Entropy Dominance, Dead Monitors, and Why the Model Is Not Learning

## Executive Summary

**TL;DR:** The model made zero chess skill improvement over 6800 steps because 95% of the loss gradient comes from the entropy bonus, the KL penalty fights the remaining signal, and both monitors (EntropyFloorMonitor & AdaptiveKLController) are either dead weight or actively harmful. The peak eval score of 0.1 was stochastic noise, not a real skill peak.

## Context

### Background
Run `chess-grpo-20260204-1956-1iaa` trained for 214 epochs (~6800 steps) before crashing due to Colab runtime limit. The eval_stockfish/score briefly peaked at 0.1 around step 5300, then dropped back to ~0.03. The question is whether the entropy floor monitor or adaptive KL controller caused this regression, and more broadly why average performance is capped around 0.03.

### Research Question
1. Did the EntropyFloorMonitor or AdaptiveKLController hurt training?
2. Why is eval_stockfish/score capped at ~0.03 on average?
3. Is the model actually learning anything?

### Scope
- Single run analysis (chess-grpo-20260204-1956-1iaa)
- Focus on loss component breakdown and monitor behavior
- Code analysis of loss computation and logging

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `xkbkwo4s` / `chess-grpo-20260204-1956-1iaa` (214 epochs, ~6800 steps, crashed) |
| Code | Loss computation (`loss.py`), training loop (`model.py`), at commit `70a63b8` |

### Run Hyperparameters (key values)

| Parameter | Value |
|-----------|-------|
| lr | 3e-6 |
| num_trajectories (G) | 16 |
| trajectory_depth (T) | 16 |
| clip_ratio | 0.2 |
| kl_coef (initial) | 0.01 |
| entropy_coef | 0.1 |
| rollout_temperature | 1.3 |
| adaptive_kl | True (target_kl=0.012, adapt_rate=1.2, max=0.1) |
| use_entropy_floor | True (floor=1.5, action=boost, boost_factor=1.5) |
| frozen_layers | 2 (of 4 total) |
| ppo_steps | 1 |

### Analysis Approach

1. Queried WandB for all key metrics over time
2. Read loss computation code to verify what's logged raw vs with coefficients
3. Computed loss component budget breakdown at multiple training phases
4. Analyzed reward trends in rolling windows to check for learning signal
5. Examined monitor behavior throughout training

## Findings

### Finding 1: EntropyFloorMonitor never triggered — complete dead weight

**Evidence:**
- `entropy_floor/below_floor = 0.0` for **all 6800+ steps**
- `entropy_floor/consecutive_low_steps = 0` throughout
- `entropy_floor/current_entropy_coef = 0.1` (unchanged) throughout
- Entropy stayed in range 2.56-3.15, far above floor of 1.5

**Interpretation:** The entropy_coef=0.1 already keeps entropy healthy (2.7-3.1 range). The floor of 1.5 is never approached. The monitor is pure overhead — it adds logging, code complexity, and config parameters for zero benefit.

### Finding 2: AdaptiveKLController saturated at max immediately and stayed there

**Evidence:**
- `adaptive_kl/current_kl_coef`: 0.012 at step 0 → **0.100 (max) by step ~500** → 0.100 for remaining 6300 steps
- `adaptive_kl/kl_ratio` (= kl_div / target_kl): consistently **2.9x to 6.6x** above target
- target_kl=0.012, but actual mean_kl_divergence ranged 0.025-0.079

**The controller logic** (from `model.py:192-195`):
```python
if kl_div > self.target_kl:
    self.current_kl_coef = min(self.current_kl_coef * self.adapt_rate, self.kl_coef_max)
else:
    self.current_kl_coef = max(self.current_kl_coef / self.adapt_rate, self.kl_coef_min)
```

With adapt_rate=1.2 and KL always 3-6x above target, the coefficient multiplied by 1.2 every step until hitting kl_coef_max=0.1, which took only ~30 steps. It never came back down because actual KL never dropped below 0.012.

**Interpretation:** target_kl=0.012 is far too low for this training regime. The controller immediately maxed out and became a **large fixed KL penalty** (kl_coef=0.1) for the entire run. Not adaptive at all — just an accidentally large constant.

### Finding 3: The loss gradient is 95% entropy, 3% PPO, 2% KL penalty

**Evidence — loss component breakdown:**

The total loss is computed at `loss.py:222-223`:
```python
loss_without_entropy = ppo_loss + kl_coef * kl_div
loss = loss_without_entropy - entropy_coef * entropy
```

Logged values: `ppo_loss`, `mean_kl_divergence`, and `entropy` are **raw** (no coefficients). Only `train/loss_without_entropy` has KL coef baked in, and `train_total_loss` has both baked in.

| Step | PPO loss | KL penalty (0.1 * kl) | Entropy bonus (-0.1 * H) | Total loss |
|------|----------|----------------------|--------------------------|------------|
| 0 | -0.0006 | +0.0005 | -0.283 | -0.283 |
| 2000 | -0.0061 | +0.0063 | -0.298 | -0.298 |
| 4000 | -0.0018 | +0.0044 | -0.302 | -0.300 |
| 5300 | -0.0042 | +0.0037 | -0.288 | -0.288 |
| 6800 | -0.0136 | +0.0053 | -0.294 | -0.302 |

**Budget allocation:**
- Entropy bonus: **~-0.28** (93-96% of total gradient)
- PPO loss: **~-0.005** (2-4% of total gradient)
- KL penalty: **~+0.004** (1-2%, and it **opposes** the PPO signal)

**Interpretation:** The model spends ~95% of its gradient budget maintaining high entropy rather than learning to play chess. The KL penalty then fights the small remaining PPO signal. The effective chess-learning gradient is tiny.

### Finding 4: The model made ZERO chess skill improvement

**Evidence — rolling 1000-step averages of train/avg_reward:**

| Steps | avg_reward | reward_p90 | reward_best |
|-------|-----------|------------|-------------|
| 0-1000 | -0.162 | 0.902 | 2.229 |
| 1000-2000 | -0.110 | 0.930 | 2.207 |
| 2000-3000 | -0.155 | 0.891 | 2.120 |
| 3000-4000 | -0.109 | 0.913 | 2.119 |
| 4000-5000 | -0.114 | 0.904 | 2.187 |
| 5000-6000 | -0.113 | 0.899 | 2.210 |
| 6000-7000 | -0.123 | 0.879 | 2.163 |

All reward metrics are **completely flat**. No trend over 6800 steps. The model is not getting better at chess.

### Finding 5: ~47% clip fraction is very high

**Evidence:** mean_clip_fraction stable at 0.46-0.52 throughout the entire run (all windows).

Healthy PPO typically sees 5-15% clip fraction. At 47%, **nearly half of all gradient updates are being clipped**, meaning the policy ratio falls outside [0.8, 1.2] for half the steps.

**Interpretation:** The old policy (synced once per epoch via `on_train_epoch_start`) diverges enough from current policy within each epoch to cause widespread clipping. This is a separate issue from the entropy/KL problem but further reduces effective learning.

### Finding 6: The peak at 0.1 was stochastic noise, not a regime change

**Evidence:**
- No corresponding change in any training metric around step 5300 (entropy, KL, rewards, clip fraction all stable)
- eval_stockfish/score is evaluated on 64 games — score=0.1 means ~6 wins+draws out of 64
- The model scored 0.047 at step ~700, fluctuated 0.02-0.04, then 0.1 at 5300, then back to 0.03
- No mechanism (entropy, KL, reward) shows a regime change at the peak

**Interpretation:** The 0.1 peak was a lucky evaluation. With 64 games and ~3% average win rate, a single evaluation hitting 10% is within normal stochastic variation.

## Analysis

### Root Cause: Why the model is capped at ~0.03

The fundamental problem is a **loss budget imbalance**:

1. **entropy_coef=0.1** with entropy~2.8 creates a -0.28 gradient signal pushing the model toward maximum entropy (uniform policy)
2. The PPO signal (actual chess improvement) contributes only -0.005 of gradient
3. The KL penalty (kl_coef maxed at 0.1) adds +0.004, actively fighting the PPO signal
4. Net chess-learning gradient: ~-0.001 to -0.006 (essentially noise)

The model is trapped: it optimizes primarily for entropy while the tiny remaining policy gradient is half-cancelled by the KL penalty and half-clipped by PPO.

### Impact of Monitors
- **EntropyFloorMonitor**: Zero impact. Never triggered. Dead code.
- **AdaptiveKLController**: Negative impact. Saturated at max kl_coef=0.1 within 500 steps and stayed there, imposing a large constant penalty that fights the PPO signal.

## Recommendations

### Immediate Actions — Clean Run Configuration

1. **Remove EntropyFloorMonitor**
   - What: Set `use_entropy_floor=False` or remove the code entirely
   - Why: Never triggered, zero value, adds config complexity
   - Where: `model.py:19-81` (class), `model.py:350-358` (init), `model.py:647-652` (check)

2. **Remove AdaptiveKLController**
   - What: Set `adaptive_kl=False` or remove the code entirely
   - Why: Saturated at max immediately, became a large constant penalty that hurts learning
   - Where: `model.py:160-201` (class), `model.py:362-370` (init), `model.py:655-658` (update)

3. **Reduce `entropy_coef` from 0.1 to 0.01**
   - What: 10x reduction in entropy bonus weight
   - Why: Currently 95% of gradient is entropy maintenance. At 0.01, entropy bonus becomes ~-0.028 vs PPO ~-0.005, giving PPO about 15-20% of the gradient budget instead of 3%
   - Risk: Entropy might decline faster. Monitor entropy in the clean run — if it drops below ~1.5, consider bumping to 0.02

4. **Reduce `kl_coef` from 0.01 to 0.001**
   - What: 10x reduction in KL penalty weight
   - Why: With kl_div~0.04, this gives KL penalty of ~0.00004 instead of ~0.004. The KL penalty was half-canceling the PPO signal
   - Note: Without the adaptive controller, this is the fixed kl_coef used directly

5. **Adjust LR after reducing coefficients**
   - What: Consider reducing LR from 3e-6 to ~1e-6
   - Why: With entropy_coef reduced 10x and kl_coef reduced 10x, the PPO loss will dominate the gradient much more. The effective learning rate for chess-relevant gradients will increase significantly. A lower LR may be needed to maintain stability and prevent the clip fraction from getting worse.
   - Alternative: Keep LR at 3e-6 initially and reduce only if clip fraction increases

### Proposed Clean Run Config

```python
GRPOConfig(
    lr=1e-6,                    # Reduced from 3e-6 (PPO signal now dominates)
    num_trajectories=16,
    trajectory_depth=16,
    clip_ratio=0.2,
    kl_coef=0.001,              # Reduced from 0.01 (was being overridden to 0.1 by adaptive)
    entropy_coef=0.01,          # Reduced from 0.1 (was 95% of gradient)
    eval_every_n_epochs=10,
    use_entropy_floor=False,    # REMOVED - never triggered
    adaptive_kl=False,          # REMOVED - saturated at max, harmful
    ppo_steps=1,
    rollout_temperature=1.3,
    enable_safety_checks=False,
    teacher_forcing_prob=0.1,
    teacher_forcing_depth=4,
)
```

### After Clean Run

- Monitor entropy — if it drops below ~1.5, increase entropy_coef to 0.02
- If clip fraction is still ~47%, consider more frequent policy syncs (every N steps instead of every epoch) or reducing clip_ratio
- If rewards are improving, consider LR warmup or scheduling

## Open Questions

- [ ] With entropy_coef=0.01, will entropy remain stable? Or will the model collapse without sufficient entropy pressure?
- [ ] Is the 47% clip fraction a separate bottleneck that needs independent attention?
- [ ] How many steps per epoch are there? If epochs are long, the old policy may be too stale regardless of coefficient tuning
- [ ] Should we consider removing the KL penalty entirely (kl_coef=0) for the initial clean run, relying only on PPO clipping for stability?

## Appendix

### A. Logged Metric Definitions

| Metric | Formula | Coefficient included? |
|--------|---------|----------------------|
| `ppo_loss` | PPO-clip surrogate loss | No (raw) |
| `mean_kl_divergence` | mean(log_old - log_new) | No (raw) |
| `entropy` | -mean(log_new) | No (raw) |
| `train/loss_without_entropy` | ppo_loss + kl_coef * kl_div | KL coef baked in |
| `train_total_loss` | loss_without_entropy - entropy_coef * entropy | Both coefs baked in |

### B. Related Documents

- [Entropy Collapse Recovery Mechanisms](./2026-01-20_entropy-collapse-recovery-mechanisms.md) — introduced the monitors analyzed here
- [Entropy Collapse Analysis](./2026-01-18_entropy-collapse-analysis.md) — original entropy concern
- [Loss Increase and LR Regression](./2026-01-19_loss-increase-and-lr-regression-analysis.md) — prior LR analysis
