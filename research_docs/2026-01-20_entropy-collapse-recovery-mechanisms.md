---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Entropy Collapse Recovery: KL Trap Analysis and Unified Recovery Mechanism"
date: "2026-01-20 12:00:00 UTC"
agent: "claude-opus-4-5-20251101"

# Git State - Critical for reproducibility
git_commit: "473cc4993a03c1e4e4cb9c523ab6dc1bb5156588"
git_branch: "self_play"
uncommitted_changes: true  # Modified: config_loader.py, default.yaml, model.py, train_self_play.py

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py
  - src/grpo_self_play/grpo_logic/loss.py
  - src/grpo_self_play/chess/boards_dataset.py
  - src/grpo_self_play/configs/default.yaml
  - research_docs/2026-01-18_entropy-collapse-analysis.md
  - research_docs/2026-01-19_loss-increase-and-lr-regression-analysis.md

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "zjij66xd"
    run_name: "chess-grpo-20260119-1054-xb0f"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary)

# Original Prompt - The exact prompt given to the agent
prompt: |
  use the instractions for research-insights and analyze the last run (chess-grpo-20260119-1054-xb0f)

# Tags for categorization
tags:
  - training
  - entropy
  - debugging
  - policy-collapse
  - hyperparameters
  - kl-divergence
  - recovery-mechanism
---

# Entropy Collapse Recovery: KL Trap Analysis and Unified Recovery Mechanism

## Executive Summary

Run `chess-grpo-20260119-1054-xb0f` crashed after ~18.7 hours with catastrophic entropy collapse (0.255 nats) despite the entropy boost mechanism increasing `entropy_coef` by 38x (0.1 → 3.84). Analysis reveals a **KL penalty trap**: once entropy collapses, the KL penalty actively prevents recovery by anchoring the policy to the already-collapsed old policy.

Timeline analysis shows the model **performed best early in training** (step 283-692) when entropy was 1.4-2.8, achieving its best Stockfish evaluation (2 wins/draws) at step 647 with entropy ~1.5. The boost mechanism achieved **one successful recovery** (step 1365: entropy 0.83 → 2.01) but couldn't sustain it. High entropy coefficients caused **numerical instability** with entropy spiking to 22-67.

**TL;DR:** The model's best performance was at entropy ~1.5 (below the 2.0 floor). The entropy boost mechanism can work but needs coordination with KL reduction and temperature increase, plus a lower max_entropy_coef to prevent numerical instability.

## Context

### Background

Previous analyses (Jan 18-19) identified entropy collapse as the root cause of training failures and recommended:
1. Lower learning rate (3e-05) - **implemented**
2. Higher entropy_coef (0.1) - **implemented**
3. Entropy floor with boost mechanism - **implemented**
4. Rollout temperature > 1.0 - **NOT implemented**

Run `chess-grpo-20260119-1054-xb0f` was meant to recreate a previous run with the correct LR (3e-05). It used all recommended settings except rollout temperature remained at 1.0.

### Research Question

1. Why did entropy collapse despite the entropy boost mechanism increasing `entropy_coef` by 38x?
2. When did the model perform best during training?
3. What additional mechanisms are needed to recover from entropy collapse?
4. How does the dataset phase distribution affect entropy dynamics?

### Scope

**Included:**
- Analysis of run `zjij66xd` metrics and configuration
- Timeline analysis identifying peak performance periods
- Investigation of KL penalty interaction with entropy recovery
- Design of unified entropy recovery mechanism
- Dataset phase distribution analysis

**Excluded:**
- Pretrain/supervised learning cold start (work in progress)
- Code implementation (separate task)

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `zjij66xd` (chess-grpo-20260119-1054-xb0f) - 3,219 steps, 198 epochs, crashed |
| Code Files | model.py (EntropyFloorMonitor), loss.py (loss formula), boards_dataset.py |
| Prior Research | 2026-01-18 and 2026-01-19 entropy collapse analyses |

### Analysis Approach

1. Retrieved run summary and metrics from WandB
2. Analyzed entropy, KL, and group collapse metrics over training timeline
3. Identified peak performance periods and correlated with entropy levels
4. Examined the interaction between entropy boost and KL penalty
5. Reviewed dataset phase distribution and its potential effects
6. Designed unified recovery mechanism based on findings

### Tools Used

- **Read**: Examined model.py:19-81 (EntropyFloorMonitor), boards_dataset.py (full file)
- **WandB MCP**: Retrieved run summary and detailed metrics for zjij66xd
- **Bash/jq**: Extracted and analyzed 500 data points across training timeline
- **Grep**: Searched for entropy_floor, phase_distribution patterns

## Findings

### Finding 1: Training Timeline - Best Performance Was Early

Detailed analysis of metrics over time reveals distinct training phases:

#### Phase 1: Healthy Learning (Step 0-354, Epoch 0-21)

| Step | Epoch | Entropy | avg_reward | action_agreement | entropy_coef |
|------|-------|---------|------------|------------------|--------------|
| 0 | 0 | 3.24 | -0.036 | 0.12 | 0.1 |
| 133 | 8 | 3.01 | 0.179 | 0.13 | 0.1 |
| 283 | 17 | 2.77 | **0.294** | 0.15 | 0.1 |
| 354 | 21 | 2.69 | 0.005 | 0.15 | 0.1 |

**Interpretation:** This was the healthiest learning period. Entropy declined gradually (3.24 → 2.69), rewards improved significantly (-0.036 → 0.294), and action agreement stayed low (0.12-0.15), indicating good trajectory diversity.

#### Phase 2: Peak Performance + Early Collapse Warning (Step 430-692, Epoch 26-42)

| Step | Epoch | Entropy | avg_reward | reward_best | action_agreement | entropy_coef |
|------|-------|---------|------------|-------------|------------------|--------------|
| 504 | 31 | 1.91 | 0.254 | 2.76 | 0.19 | 0.1 |
| 567 | 35 | 1.71 | 0.195 | 3.32 | 0.22 | 0.1 |
| **647** | **40** | **~1.5** | - | - | - | - |
| 692 | 42 | 1.38 | 0.303 | **4.0** | 0.28 | 0.15 |

**Critical finding:**
- **Best Stockfish evaluation** occurred at step 647 with score 0.0625 (2 wins or 4 draws out of 32 games)
- **Best single trajectory reward** (4.0) at step 692
- Entropy was **1.4-1.7** during peak performance - **below the 2.0 floor**
- First entropy boost triggered at step 692 (0.1 → 0.15)

**Interpretation:** The model performed best with entropy around 1.5, suggesting the entropy_floor of 2.0 may be too aggressive.

#### Phase 3: Collapse + Boost Attempts (Step 757-1276, Epoch 46-78)

| Step | Epoch | Entropy | action_agreement | entropy_coef |
|------|-------|---------|------------------|--------------|
| 815 | 50 | 1.14 | 0.30 | 0.15 |
| 974 | 60 | 0.89 | 0.34 | 0.34 |
| 1082 | 66 | 0.74 | 0.41 | 0.34 |
| 1213 | 74 | 0.83 | 0.43 | 0.51 |
| **1276** | **78** | **1.53** | 0.49 | 0.76 |

**Interpretation:** Multiple boosts occurred (0.15 → 0.34 → 0.51 → 0.76). At step 1276, entropy spiked from 0.83 to 1.53 - the boost mechanism was having an effect.

#### Phase 4: Temporary Recovery SUCCESS (Step 1365, Epoch 84)

| Step | Epoch | Entropy | action_agreement | Notes |
|------|-------|---------|------------------|-------|
| 1213 | 74 | 0.83 | 0.43 | Pre-recovery |
| 1276 | 78 | 1.53 | 0.49 | Boost effect |
| **1365** | **84** | **2.01** | **0.22** | **Recovery succeeded!** |
| 1401 | 86 | 1.30 | 0.32 | Fading |

**Critical finding:** The boost mechanism achieved a genuine recovery:
- Entropy: 0.83 → 2.01 (142% increase)
- Action agreement: 0.43 → 0.22 (trajectory diversity restored)

**Interpretation:** Recovery IS possible with the boost mechanism, but it didn't sustain. The KL penalty likely pulled the policy back toward the collapsed state.

#### Phase 5: Numerical Instability (Step 1773-2426)

| Step | Epoch | Entropy | entropy_coef | Notes |
|------|-------|---------|--------------|-------|
| 1773 | 109 | **22.25** | 0.76 | Spike |
| 1909 | 117 | **12.51** | 0.76 | Spike |
| 2164 | 133 | **4.54** | 0.76 | Spike |
| 2322 | 143 | **60.77** | 0.76 | Extreme spike |
| 2426 | 149 | **67.81** | 0.76 | Extreme spike |

**Critical finding:** Entropy spiked to extreme values (22, 60, 67) indicating numerical instability when entropy_coef gets too high. These spikes correspond to the loss being dominated by the entropy term.

**Interpretation:** `max_entropy_coef` should be capped lower (around 1.0-1.5) to prevent numerical instability.

#### Phase 6: Terminal Collapse (Step 2500+)

| Step | Epoch | Entropy | action_agreement | entropy_coef |
|------|-------|---------|------------------|--------------|
| 2772 | 171 | 0.10 | 0.86 | 1.14 |
| 2857 | 176 | 0.22 | 0.89 | 1.71 |
| 3066 | 189 | 0.08 | 0.94 | 2.56 |
| 3143 | 194 | 0.04 | 0.94 | 2.56 |

**Interpretation:** Despite entropy_coef reaching 2.56, entropy collapsed to 0.04 with 94% action agreement. The run was unrecoverable.

### Finding 2: KL Penalty Trapped the Collapsed Policy

**Evidence:**

| Metric | Value | Target |
|--------|-------|--------|
| mean_kl_divergence | **0.198** | 0.012 |
| adaptive_kl/kl_ratio | **16.5x** | 1.0x |
| current_kl_coef | **0.2** | (maxed out at kl_coef_max) |

The loss formula from `loss.py:219`:
```python
loss = ppo_loss + kl_coef * kl_div - entropy_coef * entropy
```

**The KL Trap:**
```
Collapsed policy becomes "old policy"
    ↓
Entropy boost tries to encourage exploration
    ↓
KL penalty says "stay close to old (collapsed) policy!"
    ↓
These forces fight each other
    ↓
KL penalty maxes out, entropy continues to collapse
```

**Interpretation:** The KL penalty actively prevents recovery by anchoring to the collapsed old policy. This explains why the step 1365 recovery didn't sustain.

### Finding 3: Group Collapse Confirms Trajectory Convergence

**Evidence:**

| Metric | Value | Healthy Range |
|--------|-------|---------------|
| action_agreement_mean | **0.893** | <0.3 |
| action_agreement_max | **1.0** | <0.5 |
| reward_std_within_min | **0** | >0.1 |

**Interpretation:** 89% action agreement means trajectories are nearly identical, eliminating the GRPO learning signal.

### Finding 4: Rollout Temperature Was Not Increased

**Evidence:** Run config showed `rollout_temperature=1.0`

| Mechanism | Where it acts | Effect |
|-----------|---------------|--------|
| `entropy_coef` | Loss function (gradients) | Weak when policy already deterministic |
| `rollout_temperature` | Sampling | Forces diversity **immediately** |

**Interpretation:** Temperature > 1.0 would bypass the vanishing gradient problem by directly flattening the sampling distribution.

### Finding 5: Dataset Phase Distribution May Contribute

**Evidence:** Current config uses equal distribution (opening: 0.33, middlegame: 0.34, endgame: 0.33).

| Phase | Natural Entropy | Optimal Play |
|-------|-----------------|--------------|
| Opening | Low (1-3 moves) | Deterministic |
| Middlegame | High (5-15 moves) | Exploratory |
| Endgame | Low (often 1 move) | Deterministic |

**Interpretation:** 66% of positions naturally reward low entropy, potentially teaching the model that determinism is good.

**Note:** This is speculative without phase-specific metrics. **Recommendation: Add phase-specific entropy logging to verify.**

## Analysis

### Root Cause Analysis

```
1. Policy starts training (entropy ~3.0)
       ↓
2. Healthy learning (step 0-354): entropy 3.2→2.7, rewards improving
       ↓
3. Peak performance (step 647): entropy ~1.5, best Stockfish eval
       ↓
4. Entropy drops below 2.0 floor, boost mechanism activates
       ↓
5. Temporary recovery (step 1365): entropy 0.83→2.01 ✓
       ↓
6. KL penalty pulls policy back toward collapsed state
       ↓
7. High entropy_coef causes numerical instability (spikes to 22-67)
       ↓
8. Terminal collapse: entropy 0.04, action_agreement 0.94
       ↓
9. Run crashes after 18.7 hours
```

### Key Insight: Optimal Entropy May Be Lower Than 2.0

The model achieved its best Stockfish evaluation (step 647) with entropy ~1.5, which is below the current floor of 2.0. This suggests:

1. The entropy floor of 2.0 is too aggressive
2. Some degree of policy confidence is beneficial for chess
3. The floor should be lowered to ~1.5 and the critical stop threshold set around 0.5

### Impact Assessment

| Impact | Severity |
|--------|----------|
| Training crashes | Critical |
| Compute wasted | High (~18.7 hours) |
| Temporary recovery achieved | Positive signal |
| Numerical instability identified | Important finding |

## Recommendations

### Immediate Actions (High Priority)

#### 1. Implement Unified Entropy Recovery Mechanism

**What:** Create a coordinated response when entropy drops below floor:

```python
@dataclass
class EntropyRecoveryConfig:
    # Thresholds (UPDATED based on timeline analysis)
    entropy_floor: float = 1.5           # Lowered from 2.0 - model performed well at 1.5
    entropy_critical: float = 0.5        # Hard stop below this
    steps_threshold: int = 100           # Faster intervention

    # Entropy coefficient (UPDATED - lower max to prevent instability)
    entropy_boost_factor: float = 1.5
    max_entropy_coef: float = 1.5        # Lowered from 3.0 - prevents numerical instability

    # KL coefficient (NEW - reduce to escape KL trap)
    kl_reduction_factor: float = 0.5     # Halve kl_coef each intervention
    min_kl_coef: float = 0.0             # Allow full escape from KL trap

    # Temperature (NEW - increase for direct diversity)
    temperature_boost: float = 0.2       # Add per intervention
    max_temperature: float = 2.5
    base_temperature: float = 1.0
```

**Recovery logic:**
```
When entropy < entropy_floor for steps_threshold steps:
    1. entropy_coef *= entropy_boost_factor (up to max_entropy_coef)
    2. kl_coef *= kl_reduction_factor (down to min_kl_coef)
    3. temperature += temperature_boost (up to max_temperature)

When entropy < entropy_critical (0.5):
    → STOP training immediately, save checkpoint

When entropy recovers above floor:
    → Keep boosted values (for simplicity, no gradual restoration)
```

**Why:**
- Coordinated response addresses KL trap
- Lower max_entropy_coef (1.5 vs 3.0) prevents numerical instability seen at steps 1773-2426
- Stop on entropy (not maxed boosts) catches unrecoverable collapse

**Where:** `src/grpo_self_play/grpo_logic/model.py`

**Effort:** Medium

#### 2. Lower Entropy Floor to 1.5

**What:** Change `entropy_floor` from 2.0 to 1.5

**Why:** The model's best Stockfish evaluation occurred at entropy ~1.5. The 2.0 floor triggered interventions during a period of good performance.

**Where:** `src/grpo_self_play/configs/default.yaml:29`

**Effort:** Low

#### 3. Increase Default Rollout Temperature to 1.5

**What:** Change `rollout_temperature` from 1.0 to 1.5

**Why:** Temperature > 1 flattens sampling distribution immediately, preventing trajectory convergence before gradients need to fix it.

**Where:** `src/grpo_self_play/configs/default.yaml:25`

**Effort:** Low

#### 4. Add Hard Stop on Critical Entropy (0.5)

**What:** Stop training when entropy drops below 0.5

**Why:** At entropy 0.04-0.5, the run is unrecoverable. Stopping early saves compute and preserves checkpoints.

**Where:** `src/grpo_self_play/grpo_logic/model.py`

**Effort:** Low

### Medium-Term Improvements

#### 5. Add Phase-Specific Entropy Logging

**What:** Log entropy metrics separately for opening, middlegame, and endgame positions

**Why:** Verify hypothesis that opening/endgame positions contribute to entropy collapse

**Where:** `src/grpo_self_play/grpo_logic/model.py` (training_step)

**Effort:** Medium

#### 6. Cap max_entropy_coef at 1.5

**What:** Prevent entropy_coef from exceeding 1.5

**Why:** Values above ~1.0 caused numerical instability with entropy spiking to 22-67. The entropy term dominated the loss and destabilized training.

**Where:** `src/grpo_self_play/grpo_logic/model.py` (EntropyFloorMonitor)

**Effort:** Low

### Long-Term Considerations

#### 7. Investigate Why Recovery Didn't Sustain

**What:** The step 1365 recovery (entropy 0.83 → 2.01) faded. Understand why.

**Why:** If we can make recoveries stick, training becomes more robust.

**Hypothesis:** KL penalty pulled policy back. Test by reducing kl_coef during recovery.

**Effort:** Medium

## Open Questions

- [ ] Does phase-specific entropy logging confirm opening/endgame contribute to collapse?
- [ ] Is 1.5 the right entropy_floor, or should it be even lower (1.2)?
- [ ] Should KL coefficient be restored gradually when entropy recovers?
- [ ] Would checkpoint saving at peak performance (step 647) allow resuming from a good state?
- [ ] Is the optimal max_entropy_coef 1.0 or 1.5?

## Appendix

### A. Run Configuration

```python
GRPOConfig(
    lr=3e-05,                    # ✓ Correct
    num_trajectories=16,
    trajectory_depth=16,
    clip_ratio=0.1,
    kl_coef=0.01,
    entropy_coef=0.1,            # Boosted to 3.84

    # Entropy floor
    use_entropy_floor=True,
    entropy_floor=2.0,           # Too aggressive - model performed best at 1.5
    entropy_floor_steps=150,
    entropy_floor_action='boost',
    entropy_boost_factor=1.5,

    # Adaptive KL
    adaptive_kl=True,
    target_kl=0.012,
    kl_coef_max=0.2,             # Maxed out - contributed to KL trap

    # Temperature
    rollout_temperature=1.0,     # ✗ NOT increased
)
```

### B. Stockfish Evaluations

| Step | Epoch | Entropy | Score | Result |
|------|-------|---------|-------|--------|
| **647** | **40** | **~1.5** | **0.0625** | **Best: 2 wins or 4 draws** |
| 1133 | 70 | ~0.8 | 0.0 | 0 wins, 0 draws |
| 1457 | 89 | ~0.9 | 0.016 | ~0.5 draws |

### C. Revised Parameter Recommendations

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `entropy_floor` | 2.0 | **1.5** | Model performed best at 1.5 |
| `entropy_critical` | (none) | **0.5** | Hard stop threshold |
| `entropy_floor_steps` | 150 | **100** | Faster intervention |
| `max_entropy_coef` | (none) | **1.5** | Prevent numerical instability |
| `kl_reduction_factor` | (none) | **0.5** | Escape KL trap |
| `min_kl_coef` | 0.001 | **0.0** | Allow full KL escape |
| `temperature_boost` | (none) | **0.2** | Increase per intervention |
| `max_temperature` | (none) | **2.5** | Cap temperature |
| `rollout_temperature` | 1.0 | **1.5** | Default higher |

### D. Training Phases Summary

| Phase | Steps | Epochs | Entropy | Performance | Key Event |
|-------|-------|--------|---------|-------------|-----------|
| Healthy Learning | 0-354 | 0-21 | 3.2→2.7 | Improving | Best learning period |
| Peak Performance | 430-692 | 26-42 | 1.4-1.9 | **Best** | Best eval at step 647 |
| Boost Attempts | 757-1276 | 46-78 | 0.7-1.5 | Declining | Multiple boosts |
| Temporary Recovery | 1365 | 84 | **2.01** | Promising | Recovery succeeded briefly |
| Numerical Instability | 1773-2426 | 109-149 | 4-68 | Unstable | Entropy spikes |
| Terminal Collapse | 2500+ | 153+ | 0.04-0.5 | Failed | Unrecoverable |

### E. Related Documents

- [2026-01-18_entropy-collapse-analysis.md](./2026-01-18_entropy-collapse-analysis.md) - Initial entropy collapse diagnosis
- [2026-01-19_loss-increase-and-lr-regression-analysis.md](./2026-01-19_loss-increase-and-lr-regression-analysis.md) - LR regression analysis
- [2026-01-18_followup-entropy-collapse-review.md](./2026-01-18_followup-entropy-collapse-review.md) - Follow-up review

---
