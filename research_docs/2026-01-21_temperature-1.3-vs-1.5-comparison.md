---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Rollout Temperature 1.3 vs 1.5: Faster Peak, Same Ceiling"
date: "2026-01-21 16:30:00 UTC"
agent: "claude-opus-4-5-20251101"

# Git State - Critical for reproducibility
git_commit: "89536463eed175107aee36bfa0ef9aceacf551ae"
git_branch: "self_play"
uncommitted_changes: false

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - research_docs/2026-01-21_rollout-temperature-experiment-analysis.md

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "pjjfjmis"
    run_name: "chess-grpo-20260121-0108-q6qc"
    description: "Experiment run with rollout_temperature=1.3"
  - run_id: "z7s89e1z"
    run_name: "chess-grpo-20260120-1220-g61h"
    description: "Baseline run with rollout_temperature=1.5"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary)

# Original Prompt - The exact prompt given to the agent
prompt: |
  use the instractions of research-insights, read rollout-temperature-experiment-analysis.
  and compare run chess-grpo-20260121-0108-q6qc with run chess-grpo-20260120-1220-g61h

# Tags for categorization
tags:
  - training
  - entropy
  - temperature
  - hyperparameters
  - experiment
  - grpo
---

# Rollout Temperature 1.3 vs 1.5: Faster Peak, Same Ceiling

## Executive Summary

This analysis compares run `q6qc` (temperature=1.3) against the previous experiment `g61h` (temperature=1.5), following the recommendation from the prior analysis to lower temperature for better exploitation. **Key finding: Temperature=1.3 reaches peak performance 3x faster but doesn't improve the performance ceiling.**

**TL;DR:** Temperature=1.3 achieves -470 ELO at epoch 10 (vs epoch 30 at temp=1.5), but both runs stabilize at -720 ELO with identical training stability metrics.

## Context

### Background

Prior analysis ([2026-01-21_rollout-temperature-experiment-analysis.md](./2026-01-21_rollout-temperature-experiment-analysis.md)) found that temperature=1.5 successfully prevented entropy collapse but hypothesized the entropy was "too high" (3.1-3.2) for optimal exploitation. The recommendation was:

> **Lower rollout_temperature to 1.3**: Temperature=1.5 maintained entropy at 3.1-3.2, possibly too exploratory. Temperature=1.3 should maintain enough diversity while allowing sharper policies.

### Research Question

1. Does lowering temperature from 1.5 to 1.3 improve chess performance?
2. Does temperature=1.3 reduce entropy to the "sweet spot" of ~1.5-2.0?
3. Is training stability maintained at the lower temperature?

### Scope

**Included:**
- Comparison of experiment (q6qc, temp=1.3) vs baseline (g61h, temp=1.5)
- Analysis of entropy, KL, action diversity, and Stockfish evaluation over time
- Assessment of whether the "lower entropy = better performance" hypothesis holds

**Excluded:**
- Testing other temperature values (1.1, 1.2, 1.4)
- Investigation of train/eval temperature mismatch

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| Experiment Run | `pjjfjmis` (chess-grpo-20260121-0108-q6qc) - 872 steps, 53 epochs, crashed |
| Baseline Run | `z7s89e1z` (chess-grpo-20260120-1220-g61h) - 1022 steps, 63 epochs, crashed |
| Prior Research | 2026-01-21_rollout-temperature-experiment-analysis.md |

### Analysis Approach

1. Retrieved run summaries from WandB to compare final metrics
2. Extracted evaluation checkpoints (every 10 epochs) to track performance over time
3. Compared training stability metrics (entropy, KL, action agreement)
4. Assessed whether the entropy hypothesis from prior research was validated

## Findings

### Finding 1: Temperature=1.3 Reaches Peak Performance 3x Faster

| Metric | g61h (temp=1.5) | q6qc (temp=1.3) |
|--------|-----------------|-----------------|
| Best ELO | -470 | -470 |
| Epoch at best | 30 | **10** |
| Steps to best | 485 | **161** |

**Evaluation progression (q6qc, temp=1.3):**

| Epoch | Step | ELO | Draws |
|-------|------|-----|-------|
| 10 | 161 | **-470** | ~2 |
| 20 | 323 | -2400 | 0 |
| 30 | 485 | -720 | 1 |
| 40 | 647 | -720 | 1 |
| 50 | 809 | -720 | 1 |

**Evaluation progression (g61h, temp=1.5):**

| Epoch | Step | ELO | Draws |
|-------|------|-----|-------|
| 10 | 161 | -597 | 1 |
| 20 | 323 | -523 | ~1.5 |
| 30 | 485 | **-470** | 2 |
| 40 | 647 | -2400 | 0 |
| 50 | 809 | -2400 | 0 |
| 63 | ~1008 | -720 | 1 |

**Interpretation:** Lower temperature allows the policy to sharpen faster, reaching peak performance in 1/3 of the training time. However, the peak performance ceiling (-470 ELO) is identical.

### Finding 2: Both Runs Exhibit Performance Collapse After Peak

Both runs show dramatic performance degradation after their respective peaks:

- **q6qc (temp=1.3):** Peak at epoch 10 (-470) → collapse to -2400 at epoch 20 → recovery to -720 by epoch 30
- **g61h (temp=1.5):** Peak at epoch 30 (-470) → collapse to -2400 at epoch 40-50 → recovery to -720 by epoch 63

**Interpretation:** The performance collapse is a consistent pattern regardless of temperature. This suggests the issue is not temperature-related but may be:
1. Evaluation noise (32 games is statistically noisy)
2. Train/eval temperature mismatch (training at 1.3-1.5, eval at 0.8 greedy)
3. Overfitting to training positions

### Finding 3: Entropy Did NOT Decrease at Temperature=1.3

| Metric | g61h (temp=1.5) | q6qc (temp=1.3) | Expected |
|--------|-----------------|-----------------|----------|
| Final entropy | 3.15 | 3.28 | ~2.0-2.5 |
| Entropy min | 3.06 | 3.04 | - |
| Entropy max | 3.33 | 3.32 | - |

**Interpretation:** The hypothesis that "temperature=1.3 would lower entropy to 2.0-2.5" was **not validated**. Entropy remained at ~3.1-3.3 for both temperatures. This indicates:

1. Entropy is determined primarily by the learned policy distribution, not rollout temperature
2. Rollout temperature affects sampling diversity during trajectory collection, not the policy's inherent entropy
3. The "sweet spot entropy of 1.5-2.0" may require different interventions (e.g., higher entropy coefficient penalty)

### Finding 4: Training Stability Equivalent

| Metric | g61h (temp=1.5) | q6qc (temp=1.3) |
|--------|-----------------|-----------------|
| Action agreement mean | 12.3% | 12.0% |
| low_entropy_steps | 0 | 0 |
| high_kl_steps | 0 | 0 |
| Final kl_coef | 0.001 (min) | 0.001 (min) |
| Mean KL divergence | 0.004 | 0.001 |

**Interpretation:** Both temperatures maintain excellent training stability with zero safety violations. Temperature=1.3 is equally safe as 1.5.

### Finding 5: Same Final Performance

| Metric | g61h (temp=1.5) | q6qc (temp=1.3) |
|--------|-----------------|-----------------|
| Final ELO | -720 | -720 |
| Final draws | 1 | 1 |
| Final score | 0.016 | 0.016 |
| Epochs completed | 63 | 53 |

**Interpretation:** Both runs converged to identical final performance. Temperature=1.3 gets there faster but doesn't improve the ceiling.

## Analysis

### Root Cause: Temperature Controls Sampling, Not Learned Entropy

```
rollout_temperature=1.5                    rollout_temperature=1.3
         ↓                                          ↓
Softer sampling distribution              Slightly sharper sampling
         ↓                                          ↓
More trajectory diversity                 Same trajectory diversity (~12%)
         ↓                                          ↓
Policy entropy: 3.1-3.2                   Policy entropy: 3.0-3.3 (unchanged)
         ↓                                          ↓
Slower policy sharpening                  Faster policy sharpening
         ↓                                          ↓
Peak at epoch 30                          Peak at epoch 10
```

The rollout temperature affects how actions are sampled during trajectory generation, but the policy's learned entropy is a function of training dynamics (entropy coefficient, learning rate, number of PPO steps) rather than sampling temperature.

### Impact Assessment

| Impact | Assessment |
|--------|------------|
| Training speed | ✅ **Improved** - 3x faster to peak |
| Peak performance | ➡️ **Unchanged** - same -470 ELO best |
| Final performance | ➡️ **Unchanged** - same -720 ELO final |
| Training stability | ➡️ **Unchanged** - zero safety violations |
| Entropy reduction | ❌ **Not achieved** - still 3.0-3.3 |

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| temp=1.5 | More exploration time | Slower to peak |
| temp=1.3 | Faster peak performance | Same ceiling, entropy unchanged |

## Recommendations

### Immediate Actions (High Priority)

#### 1. Keep rollout_temperature at 1.3

**What:** Use temperature=1.3 as the new default

**Why:**
- Reaches peak performance 3x faster
- Equivalent stability and final performance
- No downside observed

**Where:** `src/grpo_self_play/configs/default.yaml:25`

**Effort:** Low (single config change)

#### 2. Increase eval games to 64

**What:** Change `eval_cfg.games` from 32 to 64

**Why:**
- Both runs show unexplained ELO swings (-470 → -2400 → -720)
- 32 games has high variance; need more statistical significance
- Would help distinguish real performance changes from noise

**Where:** `src/grpo_self_play/configs/default.yaml`

**Effort:** Low (increases eval time ~2x)

### Medium-Term Improvements

#### 3. Investigate entropy reduction mechanisms

**What:** If lower entropy is truly beneficial, explore:
- Increasing `entropy_coef` penalty (currently 0.1)
- Using entropy scheduling (high early, low later)
- Lowering `entropy_floor` threshold

**Why:** Temperature alone doesn't control policy entropy. Need direct entropy interventions.

**Effort:** Medium

#### 4. Test eval temperature matching

**What:** Try eval temperature closer to training temperature (e.g., 1.0 or 1.2 instead of 0.8)

**Why:** Train/eval mismatch may explain performance variance

**Where:** `src/grpo_self_play/configs/default.yaml` (policy_cfg.temperature)

**Effort:** Low

### Long-Term Considerations

#### 5. Checkpoint saving at peak performance

**What:** Save checkpoints when eval performance peaks, not just final checkpoint

**Why:** Peak performance (-470 ELO) is better than final (-720 ELO); need to capture best model

**Effort:** Medium

## Open Questions

- [ ] Why does performance collapse after peak despite healthy training metrics?
- [ ] Is the -470 → -2400 swing real or evaluation noise?
- [ ] Would even lower temperature (1.1, 1.2) provide faster convergence?
- [ ] What mechanism would actually reduce learned policy entropy?
- [ ] Does the train/eval temperature mismatch (1.3 vs 0.8) cause the performance variance?

## Appendix

### A. Run Configurations

**Experiment Run (q6qc):**
```python
GRPOConfig(
    lr=3e-05,
    num_trajectories=16,
    trajectory_depth=16,
    clip_ratio=0.1,
    kl_coef=0.01,
    entropy_coef=0.1,
    rollout_temperature=1.3,     # ← Changed from 1.5
    use_entropy_floor=True,
    entropy_floor=2.0,
    entropy_floor_steps=150,
    adaptive_kl=True,
    target_kl=0.012,
    kl_coef_min=0.001,
    kl_coef_max=0.2,
)
```

**Baseline Run (g61h):**
```python
# Same as above except:
rollout_temperature=1.5          # ← Original value
```

### B. Metric Comparison Summary

| Metric | g61h (temp=1.5) | q6qc (temp=1.3) | Change |
|--------|-----------------|-----------------|--------|
| Epochs completed | 63 | 53 | -16% |
| Steps to peak | 485 | 161 | **-67%** |
| Best ELO | -470 | -470 | 0 |
| Final ELO | -720 | -720 | 0 |
| Final entropy | 3.15 | 3.28 | +4% |
| Action agreement | 12.3% | 12.0% | -2% |
| Safety violations | 0 | 0 | 0 |

### C. Related Documents

- [2026-01-21_rollout-temperature-experiment-analysis.md](./2026-01-21_rollout-temperature-experiment-analysis.md) - Prior analysis recommending temperature=1.3
- [2026-01-20_entropy-collapse-recovery-mechanisms.md](./2026-01-20_entropy-collapse-recovery-mechanisms.md) - Original entropy collapse diagnosis

---
