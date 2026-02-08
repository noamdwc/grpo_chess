---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Post-Temperature-Fix Run Analysis: Model Still Not Learning — Root Causes and Experiment Plan"
date: "2026-02-08 21:00:00 UTC"
agent: "claude-opus-4-6"

# Git State - Critical for reproducibility
git_commit: "3be9036c64577496360a561ca3e740b2cb79b297"
git_branch: "cleanup/main-ready"
uncommitted_changes: false

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_logic/model.py        # GRPOConfig, _ppo_step, training_step, _freeze_transformer_layers
  - src/grpo_logic/loss.py         # grpo_ppo_loss, step_group_advantage, ppo_chess_loss
  - src/grpo_logic/sampling.py     # TrajectoriesSample, sample_trajectories_batched
  - src/chess/rewards.py           # reward_board, evaluate_fen, normalize_cp
  - research_docs/grpo_chess_debug_research.md  # GPT-5.2 Thinking analysis (v2)

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "dk88cqs1"
    run_name: "chess-grpo-20260207-2238-bxp7"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary, get_plots)

# Original Prompt - The exact prompt given to the agent
prompt: |
  analyze run chess-grpo-20260207-2238-bxp7

# Tags for categorization
tags:
  - training
  - debugging
  - reward-signal
  - capacity
  - grpo
---

# Post-Temperature-Fix Run Analysis: Model Still Not Learning

## Executive Summary

Run `chess-grpo-20260207-2238-bxp7` is the first clean run after fixing the temperature mismatch bug (clip fraction dropped from ~55% to ~27%). PPO mechanics are now healthy (ratio ~1.0, KL ~0.02, clip ~0.27), but the model shows **zero improvement in chess strength** over 151 epochs (~4800 steps, ~17 hours). Average reward oscillates around zero with no upward trend, and eval vs Stockfish shows 0 wins out of 64 games.

**TL;DR:** The PPO optimization is working correctly, but the model can't learn because of (1) frozen layers limiting capacity, (2) noisy reward signal from weak self-play, (3) missing advantage normalization, and (4) insufficient teacher forcing.

## Context

### Background
The temperature mismatch bug discovered on 2026-02-07 caused a 55% clip fraction at step 0 by using different temperatures for sampling (1.3) and log-prob computation (1.0). This was fixed by passing `rollout_temperature` consistently. Run `chess-grpo-20260207-2238-bxp7` is the first post-fix run to validate whether the training loop can now produce learning.

### Research Question
With the temperature bug fixed, does the model learn to play better chess? If not, what are the remaining blockers?

### Scope
- Full metric analysis of run `dk88cqs1` (4800+ steps)
- Code review of loss computation, advantage normalization, reward shaping, and model capacity
- Cross-reference with independent GPT-5.2 analysis (`grpo_chess_debug_research.md`)
- Excluded: architecture changes, search-based solutions

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `dk88cqs1` (`chess-grpo-20260207-2238-bxp7`), state=running, 151 epochs |
| Code Files | `model.py`, `loss.py`, `sampling.py`, `rewards.py` at commit `3be9036` |
| External Analysis | GPT-5.2 Thinking document `grpo_chess_debug_research.md` (v2) |

### Analysis Approach

1. Retrieved run summary and hyperparameters from WandB
2. Pulled full time-series for all training metrics (sampled at 20 evenly-spaced points)
3. Verified temperature fix is applied in code (`model.py:239,299`)
4. Analyzed loss computation, advantage normalization, and reward shaping in source
5. Cross-referenced findings with independent GPT-5.2 analysis
6. Identified consensus root causes and ranked by likely impact

## Findings

### Finding 1: PPO mechanics are now healthy (temperature fix confirmed working)

**Evidence:**

Temperature is correctly passed in `src/grpo_logic/model.py:237-239`:
```python
new_log_probs = self.policy_model.get_group_log_probs(
    trajectories_states, trajectories_actions, trajectories_legal_masks,
    temperature=self.hparams.grpo_config.rollout_temperature,
)
```

And in sampling at `model.py:298-299`:
```python
temperature=self.hparams.grpo_config.rollout_temperature,
```

**Metrics from WandB:**

| Metric | Start (step 0) | Mid (~step 2400) | End (~step 4800) | Healthy Range |
|--------|---------------|-------------------|-------------------|---------------|
| `mean_clip_fraction` | 0.27 | 0.28 | 0.28 | 0.1–0.3 |
| `mean_ratio` | 1.005 | 0.997 | 1.005 | ~1.0 |
| `mean_kl_divergence` | 0.015 | 0.024 | 0.015 | <0.05 |

**Interpretation:** All PPO health indicators are within normal bounds. The temperature fix resolved the 55% clip fraction problem. The optimization machinery is functioning correctly — the issue is elsewhere.

---

### Finding 2: No learning — average reward oscillates around zero

**Evidence:**

`train/avg_reward` over 4800+ steps (sampled at regular intervals):

| Step | avg_reward |
|------|-----------|
| 0 | -0.387 |
| 636 | -0.340 |
| 1015 | +0.236 |
| 1563 | -0.076 |
| 2199 | -0.194 |
| 2687 | -0.425 |
| 3451 | +0.099 |
| 4346 | +0.131 |
| 4661 | +0.077 |
| 6080 | -0.249 |

No monotonic improvement visible. The reward oscillates between -0.44 and +0.24 with no trend.

**Interpretation:** The model is not learning to play better chess. The gradient signal from PPO is not translating into policy improvement.

---

### Finding 3: Eval vs Stockfish confirms zero improvement

**Evidence:**

Only 3 non-NaN eval points were recorded (most eval steps logged NaN):

| Step | eval_stockfish/score | eval_stockfish/elo_diff |
|------|---------------------|------------------------|
| 321 | 0.039 (2.5 draws/64) | -556 |
| 1609 | 0.031 (2 draws/64) | -597 |
| 2253 | 0.047 (3 draws/64) | -523 |

Latest eval: 0 wins, 4 draws, 60 losses out of 64 games. The model plays essentially random chess.

**Interpretation:** The model's actual playing strength has not improved at all during training. The slight score variations are within noise.

---

### Finding 4: Only 2 of 4 transformer layers are trainable

**Evidence:**

Config: `freeze_layers=2` with a 4-layer transformer.

In `src/grpo_logic/model.py:169-190`:
```python
def _freeze_transformer_layers(self, num_layers: int) -> None:
    # Freeze embedding and positional encoding
    for param in self.policy_model.embedding.parameters():
        param.requires_grad = False
    self.policy_model.pos_encoding.requires_grad = False

    # Freeze specified number of transformer layers
    for i, layer in enumerate(self.policy_model.transformer.layers):
        if i < num_layers:
            for param in layer.parameters():
                param.requires_grad = False
```

Embeddings + positional encoding + layers 0,1 are all frozen. Only layers 2-3 and the output head can learn.

**Interpretation:** This severely limits the model's capacity to adapt. The pretrained representations in the frozen layers were learned for supervised position evaluation, not for RL-driven policy improvement. The model cannot reshape its lower-level features.

---

### Finding 5: Advantages are not normalized by standard deviation

**Evidence:**

In `src/grpo_logic/loss.py:78-99`:
```python
def step_group_advantage(step_rewards, pad_mask=None):
    """NOTE: No std normalization is applied here, Using DR. GRPO paper."""
    mean_t = step_rewards.mean(dim=1, keepdim=True)  # [B, 1, T]
    advantages = (step_rewards - mean_t)  # [B, G, T]
    if pad_mask is not None:
        advantages = advantages * pad_mask.float()
    return advantages
```

Only mean-centering, no division by std. The `advantage_std` metric shows values of 0.35–0.49 throughout the run, meaning raw advantage magnitudes vary significantly.

**Interpretation:** Without std normalization, positions with high reward variance produce disproportionately large gradient updates, while positions with low variance contribute almost nothing. This creates unstable, noisy learning dynamics — especially problematic with the already-noisy self-play reward signal.

---

### Finding 6: Reward signal is dominated by opponent noise in self-play

**Evidence:**

In `src/chess/rewards.py:87-108`:
```python
def reward_board(env, board_start, movetime_ms=0, depth=16):
    # ...
    r_t = evaluate_fen(fen_t, pov_is_white, movetime_ms, depth)
    r_0 = evaluate_fen(fen_0, pov_is_white, movetime_ms, depth)
    return r_t - r_0  # Reward is the change in eval
```

Step rewards measure Stockfish eval delta between consecutive positions. In self-play, both players are the same weak model. The opponent's random blunders create large positive eval swings that are not due to the player's skill.

Supporting metrics:
- `train/step_reward_std` ranges 0.43–0.76 (high variance)
- `train/reward_gap_best_minus_mean` ranges 1.4–4.1 (best trajectory is far from mean)
- `train/reward_p50` oscillates around 0 (median trajectory is noise)

**Interpretation:** The reward signal has a very low signal-to-noise ratio. The model gets high rewards when the opponent blunders, not when it plays well. This makes it hard to learn meaningful chess strategy.

---

### Finding 7: Teacher forcing at 10% is insufficient to bootstrap

**Evidence:**

Config: `teacher_forcing_prob=0.1`, `teacher_forcing_depth=4`

Only 10% of opponent moves are replaced with Stockfish moves at depth 4 (shallow). With 16-ply trajectories where only even-indexed steps are the player's moves, this means ~0.8 opponent moves per trajectory are Stockfish quality. The remaining 7+ opponent moves are random.

**Interpretation:** The small fraction of teacher-forced moves is not enough to move trajectories into "reasonable chess" territory where the reward signal becomes informative.

---

### Finding 8: Entropy is declining slowly — not a current concern

**Evidence:**

| Step | entropy |
|------|---------|
| 0 | 2.80 |
| 1000 | 2.49 |
| 2000 | 2.26 |
| 3000 | 2.12 |
| 4800 | 2.07 |

Min observed: 1.94.

**Interpretation:** Entropy is declining gradually, which is expected as the policy becomes more confident. It is well above any collapse threshold (1.0 would be concerning). Entropy regularization is NOT needed at this time.

## Analysis

### Root Cause Analysis

The PPO optimization machinery is working correctly after the temperature fix. The remaining blockers are ranked by likely impact:

1. **Limited model capacity (freeze_layers=2)**: Half the network is frozen. The model cannot learn new representations needed for RL-driven improvement.

2. **Poor reward signal quality in weak self-play**: Both players are the same weak model. Reward is dominated by random opponent blunders, not player skill. The signal-to-noise ratio is too low for learning.

3. **Missing advantage std normalization**: Without whitening, gradient magnitudes vary wildly across positions, making optimization unstable in a high-variance reward setting.

4. **Insufficient teacher forcing (10%, depth 4)**: Not enough to bootstrap trajectories into regions where reward is informative.

These four factors compound: limited capacity + noisy signal + unstable gradients + no bootstrapping = no learning.

### Impact Assessment

- The model has trained for 17 hours with zero improvement in playing strength
- PPO health metrics look fine, which means the issue is NOT in the optimization loop — it's in the signal the optimizer receives and the model's ability to use it
- Additional training time with current settings will not help

### Cross-Reference with GPT-5.2 Analysis

An independent analysis by GPT-5.2 Thinking (`grpo_chess_debug_research.md`) initially attributed the problem to stale old-policy sync (512 steps per epoch). After correction with actual metrics (~32 steps/epoch, healthy ratio/KL), the updated v2 document converged on the same four root causes in the same priority order. This convergence across independent analyses strengthens confidence in the diagnosis.

## Recommendations

### Immediate Actions (High Priority)

1. **Unfreeze all transformer layers**
   - What: Set `freeze_layers=0` in config
   - Why: With only 2/4 layers trainable, the model cannot adapt its representations. The pretrained features may not be suitable for RL-driven policy improvement.
   - Where: `src/configs/default.yaml` → `pretrain.freeze_layers: 0`
   - Effort: Low (config change)

2. **Add advantage std normalization**
   - What: Divide centered advantages by std in `step_group_advantage()`
   - Why: Stabilizes gradient magnitudes across positions with different reward variance. Critical in high-variance reward settings.
   - Where: `src/grpo_logic/loss.py:78-99`
   - Effort: Low (3-line code change)

3. **Increase learning rate to 3e-6**
   - What: Change `lr` from `1e-6` to `3e-6`
   - Why: Current LR is very conservative. With small per-step rewards and 27% clipping, the effective learning signal is tiny. Monitor KL divergence to ensure stability.
   - Where: `src/configs/default.yaml` → `grpo.lr: 3e-6`
   - Effort: Low (config change)

4. **Sync old policy every N steps instead of per epoch**
   - What: Add a `sync_every_n_steps` config parameter and call `_sync_old_policy()` at that cadence in `training_step`, replacing the current `on_train_epoch_start` sync.
   - Why: The current sync is tied to Lightning's epoch boundary, which conflates a framework concept (epoch) with an RL algorithm invariant (behavior policy freshness). Decoupling sync frequency from epoch size makes the code cleaner, the PPO semantics explicit, and prevents future regressions if epoch size changes. While ~32 steps/epoch is not causing severe staleness today, a configurable sync cadence is better PPO hygiene and allows tuning independently.
   - Where: `src/grpo_logic/model.py` — `GRPOConfig` (new field), `training_step`, remove `on_train_epoch_start` sync
   - Effort: Low

### Medium-Term Improvements

5. **Increase teacher forcing to 30%, depth 8 for bootstrap**
   - What: Set `teacher_forcing_prob=0.3`, `teacher_forcing_depth=8` initially, anneal to 0.05 over time
   - Why: Gives the model exposure to reasonable chess positions where reward deltas are meaningful, not just random blunder noise
   - Where: `src/configs/default.yaml` → `grpo.teacher_forcing_prob`, `grpo.teacher_forcing_depth`
   - Effort: Low (config change), but needs annealing logic if automated

6. **Add easier eval tiers (Stockfish skill 0 and 1)**
   - What: Evaluate against multiple Stockfish skill levels to detect early improvements invisible at skill 2
   - Why: A skill 2 opponent may be too strong to show initial gains from a small transformer
   - Where: `src/evaluator.py`, eval config
   - Effort: Medium

### Long-Term Considerations

7. **Fixed-opponent curriculum instead of pure self-play**
   - What: Replace or supplement self-play with a pool of fixed opponents (frozen checkpoints, varying Stockfish levels)
   - Why: Pure self-play creates non-stationary rewards and prevents monotonic progress measurement. A fixed reference opponent gives a stable learning signal.
   - Effort: Medium-High

8. **Per-step reward attribution cleanup**
   - What: Ensure step rewards only credit the player's move, not the subsequent opponent move
   - Why: Current step rewards at even indices include the eval change from the opponent's response, which dilutes the credit assignment
   - Effort: Medium (requires careful reward computation refactoring)

## Recommended First Experiment

Combine recommendations 1-3 in a single run to test whether the training loop can produce learning with adequate capacity and signal:

```yaml
# Changes from current config:
pretrain:
  freeze_layers: 0        # was 2
grpo:
  lr: 3e-6                # was 1e-6
```

Plus add advantage std normalization in `loss.py`.

**Expected outcome:** If the training loop is fundamentally correct, this should show reward improvement within 50-100 epochs, and possible early eval gains at lower Stockfish skill levels.

## Open Questions

- [ ] Why are most `eval_stockfish/score` datapoints NaN? Is this a logging issue or does eval crash/timeout?
- [ ] What fraction of positions does the model encounter more than once (LRU cache hit rate on reward evals)?
- [ ] Would a cosine or warmup LR schedule help compared to constant LR?
- [ ] Is the 16-trajectory group size sufficient for stable advantage estimation, or would 32/64 help?
- [ ] How does the model's per-move centipawn loss compare to the pretrained baseline?

## Appendix

### A. Run Configuration

```
lr: 1e-6
num_trajectories: 16
trajectory_depth: 16
clip_ratio: 0.2
kl_coef: 0.001
rollout_temperature: 1.3
ppo_steps: 1
teacher_forcing_prob: 0.1
teacher_forcing_depth: 4
freeze_layers: 2
model: 4L-8H-256d (vocab=300, action=1968)
pretrain_checkpoint: pretrain-20260128-1215-16es-epoch=12
stockfish: skill_level=2, movetime=50ms
eval: 64 games, every 10 epochs
```

### B. Full Metric Trajectories

**train/avg_reward (sampled every ~250 steps):**
```
step=    0  val=-0.387    step= 1015  val=+0.236    step= 2199  val=-0.194
step=  273  val=-0.348    step= 1340  val=+0.168    step= 2363  val=-0.353
step=  636  val=-0.340    step= 1563  val=-0.076    step= 2687  val=-0.425
                          step= 1920  val=-0.065    step= 3146  val=-0.274
step= 3451  val=+0.099    step= 4346  val=+0.131    step= 5197  val=-0.426
step= 3731  val=-0.438    step= 4661  val=+0.077    step= 5528  val=-0.081
step= 4087  val=-0.122    step= 4953  val=+0.061    step= 5816  val=+0.187
                                                     step= 6080  val=-0.249
```

**entropy:**
```
step=    0  val=2.799    step= 2363  val=2.332    step= 4953  val=2.094
step= 1015  val=2.493    step= 3146  val=2.120    step= 6080  val=2.067
step= 1920  val=2.401    step= 4087  val=2.100
```

**mean_clip_fraction:**
```
step=    0  val=0.275    step= 2687  val=0.251    step= 5528  val=0.269
step= 1340  val=0.280    step= 4087  val=0.275    step= 6080  val=0.276
```

### C. Related Documents

- [Temperature Mismatch Bug](./2026-02-07_clean-run-temperature-mismatch-bug.md) — Root cause of 55% clip fraction, fixed before this run
- [Loss Budget and Monitor Analysis](./2026-02-06_loss-budget-and-monitor-analysis.md) — Analysis of entropy dominance and dead monitors (led to cleanup)
- [GPT-5.2 Independent Analysis](./grpo_chess_debug_research.md) — Converges on same root causes after correcting stale-policy hypothesis
