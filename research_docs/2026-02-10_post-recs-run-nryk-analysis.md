---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Post-Recommendations Run Analysis: Faint Learning Signal, Entropy Collapse Risk"
date: "2026-02-10 12:00:00 UTC"
agent: "claude-opus-4-6"

# Git State - Critical for reproducibility
git_commit: "b5897d8188e915acbb9d0481c329b3726ca79da8"
git_branch: "self_play"
uncommitted_changes: false

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/grpo_logic/model.py        # training_step, start_player_mask, step-based sync
  - src/grpo_logic/loss.py         # step_group_advantage (std normalization)
  - src/grpo_logic/sampling.py     # step reward computation
  - src/chess/rewards.py           # reward_board
  - research_docs/2026-02-08_post-tempfix-run-analysis.md  # Previous analysis with recs
  - research_docs/postfix_grpo_run_review_2026-02-10.md    # GPT-5.2 analysis

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "f11u4id9"
    run_name: "chess-grpo-20260208-2025-nryk"
  - run_id: "dk88cqs1"
    run_name: "chess-grpo-20260207-2238-bxp7"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - Bash
  - WandB MCP (get_run_metrics, get_run_summary, compare_runs)

# Original Prompt - The exact prompt given to the agent
prompt: |
  Look at the last run "chess-grpo-20260208-2025-nryk", in this run we
  implemented recommendations 1,2,3,4,5,6,8 from
  "2026-02-08_post-tempfix-run-analysis.md"

tags:
  - training
  - evaluation
  - entropy
  - self-play
  - grpo
---

# Post-Recommendations Run Analysis: Faint Learning Signal, Entropy Collapse Risk

## Executive Summary

Run `chess-grpo-20260208-2025-nryk` implemented 7 of 8 recommendations from the post-tempfix analysis (unfreeze layers, std-normalize advantages, 3x LR, step-based sync, 30% teacher forcing, multi-skill eval, start-player-only loss). PPO mechanics remain healthy. For the **first time**, eval shows a faint upward trend (~0.02 → 0.06-0.075 score over 5000 steps), but the model still has 0 wins and plays at -500 elo vs Stockfish skill 0. Entropy is collapsing 2x faster than the previous run (2.8 → 1.4), creating a risk of premature policy lock-in.

**TL;DR:** First sign of learning, but painfully slow. Entropy collapse is the new top concern. Next run should increase rollout temperature and PPO update budget.

## Context

### Background
The previous run (`chess-grpo-20260207-2238-bxp7`) showed healthy PPO mechanics but zero chess improvement after the temperature-mismatch fix. The post-tempfix analysis identified 8 recommendations to address limited capacity, noisy rewards, missing normalization, and insufficient bootstrapping. This run implemented 7 of those 8 (skipping rec 7: fixed-opponent curriculum).

### Research Question
Did the 7 implemented recommendations produce measurable chess learning? If not, what are the remaining blockers?

### Scope
- Full metric analysis of run `f11u4id9` (4877 steps, 151 epochs, ~24 hours, crashed due to Colab timeout)
- Comparison with previous run `dk88cqs1`
- Code verification of implemented recommendations
- WandB plots provided by user for eval data (API sampling missed sparse eval points)

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Run | `f11u4id9` (`chess-grpo-20260208-2025-nryk`), state=crashed (Colab timeout), 151 epochs |
| WandB Run (baseline) | `dk88cqs1` (`chess-grpo-20260207-2238-bxp7`), previous run without recommendations |
| Code Files | `model.py`, `loss.py`, `sampling.py`, `rewards.py` at commit `b5897d8` |
| WandB Plots | User-provided screenshots showing eval trends with ~13 data points (API sampling missed sparse eval steps) |

### Analysis Approach

1. Retrieved run summary and full time-series metrics from WandB
2. Verified all 7 recommendations were implemented in code and hyperparameters
3. Sampled key metrics at 15 evenly-spaced intervals across the run
4. Compared with previous run metrics
5. Examined user-provided WandB plots for eval data (API sampling at 500 points missed most eval-only steps)
6. Cross-referenced with independent GPT-5.2 analysis

## Findings

### Finding 1: All 7 recommendations confirmed implemented

| Rec | Description | Evidence |
|-----|-------------|----------|
| 1 | Unfreeze all layers | `freeze_layers=0` in hyperparams |
| 2 | Advantage std normalization | `loss.py:94` divides by `std_t` |
| 3 | LR 3e-6 | `lr=3e-06` in hyperparams |
| 4 | Step-based sync | `sync_every_n_steps=1` in hyperparams, `model.py:371-375` |
| 5 | Teacher forcing 30%, depth 8 | `teacher_forcing_prob=0.3, teacher_forcing_depth=8` |
| 6 | Multi-skill eval | `eval_skill_levels=[0, 1, 2]` |
| 8 | Start-player-only loss | `model.py:318-322` masks odd-indexed steps |

**Rec 7 (fixed-opponent curriculum) was intentionally skipped.**

---

### Finding 2: PPO mechanics still healthy

**Metrics (sampled):**

| Metric | Start | Mid (~2500) | End (~4800) | Healthy Range |
|--------|-------|-------------|-------------|---------------|
| clip_fraction | 0.26 | 0.23 | 0.25 | 0.1–0.3 |
| ratio | 1.008 | 0.996 | 0.994 | ~1.0 |
| kl_divergence | 0.010 | 0.022 | 0.030 | <0.05 |
| advantage_std | 0.97 | 0.96 | 0.96 | ~1.0 (with normalization) |

Advantage std ~0.96 confirms recommendation 2 (std normalization) is working correctly.

---

### Finding 3: First-ever faint eval improvement (from plots)

**From user-provided WandB plots (~13 eval data points):**

| Metric | Early (~step 500) | Late (~step 4500) | Trend |
|--------|-------------------|-------------------|-------|
| eval_stockfish/score | ~0.02 | 0.06-0.075 | Weak upward |
| eval_stockfish_skill0/score | ~0.02 | 0.05-0.08 | Weak upward |
| eval_stockfish_skill0/elo_diff | -700 | -470 | Improving |

**Interpretation:** The model went from ~1 draw/64 games to ~4-5 draws/64 games. This is the first measurable eval improvement across all GRPO runs. However, 0 wins and -470 elo vs skill 0 is still extremely weak. At this rate (~200 elo gain per 24 hours), reaching even 0% winrate against skill 0 would require many more days.

**Note on API data:** The WandB MCP returned 500 evenly-sampled points across all steps. Since eval only logs every ~320 steps (every 10 epochs), the sampling missed most eval data points, returning only 3 non-NaN values. The user-provided plots show the full ~13 eval points and are the authoritative source for eval trends.

---

### Finding 4: Entropy collapsing 2x faster than previous run

| | Previous run (dk88cqs1) | This run (f11u4id9) |
|---|---|---|
| Start | 2.80 | 2.77 |
| Step ~1000 | 2.49 | 2.15 |
| Step ~2400 | 2.33 | 1.81 |
| End (~4800) | 2.07 | 1.73 |
| Min observed | 1.94 | 1.40 |

Entropy is declining approximately 2x faster. Contributing factors:
- **All layers unfrozen** (rec 1): more parameters can shift the policy
- **3x higher LR** (rec 3): faster parameter updates
- **Std-normalized advantages** (rec 2): more consistent gradient magnitudes across positions

The policy is becoming more confident (lower entropy = more deterministic action selection) without proportional skill improvement. The divergence between increasing confidence and stagnant skill is a warning sign for premature policy collapse.

---

### Finding 5: Training reward shows zero trend

`train/reward_mean` oscillates between -1.34 and +0.34 with no upward trend (mean ~-0.27). This is nearly identical to the previous run. The eval improvement is happening despite the training reward providing no clear signal.

**Possible explanation:** The faint eval improvement may be driven by:
1. The 30% teacher-forced moves (rec 5) bootstrapping slightly better patterns
2. The model memorizing a few good responses from teacher-forced positions
3. Not from PPO reward optimization (reward signal too noisy to drive improvement)

---

### Finding 6: Start-player mask halves effective training signal

`model.py:318-322` masks odd-indexed steps (opponent moves):
```python
t = torch.arange(T, device=pad_mask.device)
start_player_mask = (t % 2 == 0)[None, None, :]  # [1, 1, T]
effective_pad_mask = pad_mask & start_player_mask  # [B, G, T]
```

With trajectory_depth=16, only 8 steps per trajectory contribute to the loss. Combined with ppo_steps=1 and 16 trajectories, the effective training data per step is: 16 trajectories × 8 player moves = 128 policy gradient samples.

## Analysis

### Root Cause Analysis

The 7 recommendations together produced the first faint learning signal, but three bottlenecks remain:

1. **Entropy collapsing toward a mediocre policy** (NEW, HIGH PRIORITY)
   - 2.8 → 1.4 entropy with almost no skill gain means the model is locking in bad habits
   - Driven by unfrozen layers + higher LR — the same changes that enabled the faint learning also accelerate collapse
   - Fix: increase rollout temperature to maintain exploration

2. **Self-play reward signal too noisy for PPO to extract learning**
   - reward_mean has zero trend despite 5000 steps
   - The eval improvement likely comes from teacher forcing, not from the PPO reward signal
   - The model cannot learn from self-play when both players are equally bad

3. **Update budget too small**
   - ppo_steps=1 with 128 effective policy gradient samples is very low
   - More PPO steps per batch would extract more learning from each costly trajectory sampling

### Comparison with Previous Run

| Aspect | Previous (dk88cqs1) | This run (f11u4id9) | Change |
|--------|---------------------|---------------------|--------|
| Eval score (skill 0) | N/A | 0.02 → 0.08 | First time tracked |
| Eval score (skill 2) | 0.031-0.047 | 0.02 → 0.075 | Slight improvement |
| Entropy (final) | 2.07 | 1.73 | Dropping faster |
| Entropy (min) | 1.94 | 1.40 | Significantly lower |
| Clip fraction | 0.27 | 0.25 | Similar |
| Reward mean | -0.39 to +0.24 | -1.34 to +0.34 | Similar noise |

### Cross-Reference with GPT-5.2 Analysis

The GPT-5.2 analysis (`postfix_grpo_run_review_2026-02-10.md`) made similar observations:
- PPO mechanics are stable
- Eval shows "plausible upward trend" (confirmed by plots)
- Main bottlenecks: eval variance and small update budget

Our analysis adds: entropy collapse is a **new** high-priority concern not present in the previous run, and the reward signal shows zero trend (suggesting teacher forcing, not PPO, is driving the faint improvement).

## Recommendations

### Immediate Actions (High Priority) — Bundle for Next Run

1. **Increase rollout temperature (1.3 → 1.6)**
   - What: Set `rollout_temperature: 1.6` in config
   - Why: Counter the fast entropy collapse (2.8 → 1.4). Higher temperature increases action randomness during sampling, maintaining exploration. This is simpler than re-adding entropy regularization and directly addresses the root cause.
   - Where: `src/configs/default.yaml` → `grpo.rollout_temperature: 1.6`
   - Effort: Low (config change)
   - Monitor: entropy should stabilize in the 1.8-2.2 range instead of continually declining

2. **Increase ppo_steps (1 → 4)**
   - What: Set `ppo_steps: 4` in config
   - Why: Each trajectory batch is expensive to sample. With ppo_steps=1, we only make one optimization pass over the data. Increasing to 4 extracts more learning per sample batch, improving sample efficiency by ~4x.
   - Where: `src/configs/default.yaml` → `grpo.ppo_steps: 4`
   - Effort: Low (config change)
   - Monitor: clip_fraction and KL should stay healthy. If KL spikes above 0.05 or clip_fraction exceeds 0.4, reduce back to 2.

3. **Increase eval games (64 → 128)**
   - What: Set `eval.games: 128` in config
   - Why: With 64 games, the difference between 3 draws and 5 draws is within noise. Doubling to 128 gives better statistical power to detect real progress.
   - Where: `src/configs/default.yaml` → `eval.games: 128`
   - Effort: Low (config change, takes more eval time)

### Medium-Term Improvements

4. **Increase num_trajectories (16 → 32)**
   - What: Set `num_trajectories: 32`
   - Why: More trajectories per position = more stable advantage estimates and less noisy gradients. Especially important with std normalization (G=16 gives rough std estimates).
   - Effort: Low (config change, ~2x slower per step)

5. **Switch to fixed Stockfish opponent for training** (deferred per user preference)
   - What: Replace self-play with Stockfish skill 0-1 as opponent
   - Why: The self-play reward signal is pure noise (reward_mean has zero trend). A fixed opponent gives stationary, meaningful rewards.
   - This is deferred but remains the most likely high-impact change if temperature + ppo_steps don't produce faster learning.

## Open Questions

- [ ] Is the faint eval improvement driven by teacher forcing or PPO? An ablation with `teacher_forcing_prob=0` would test this.
- [ ] Would rollout_temperature=1.6 or 1.8 be better? Previous experiments showed 1.3 vs 1.5 had different entropy profiles.
- [ ] Does increasing ppo_steps destabilize training with sync_every_n_steps=1? The old policy sync happens every step but ppo_steps=4 means 4 gradient updates between syncs.
- [ ] Why do some eval points log as NaN in the WandB time-series? (Eval appears to run but metrics don't always reach the logger.)
- [ ] Would gradient accumulation over 2-4 microbatches be better than larger trajectory count?

## Appendix

### A. Run Configuration (this run)

```
lr: 3e-6                          # was 1e-6
num_trajectories: 16
trajectory_depth: 16
clip_ratio: 0.2
kl_coef: 0.001
rollout_temperature: 1.3
ppo_steps: 1
sync_every_n_steps: 1              # NEW (was epoch-based)
teacher_forcing_prob: 0.3           # was 0.1
teacher_forcing_depth: 8            # was 4
freeze_layers: 0                    # was 2
eval_skill_levels: [0, 1, 2]       # was [2]
model: 4L-8H-256d (vocab=300, action=1968)
pretrain_checkpoint: pretrain-20260128-1215-16es-epoch=12
stockfish: skill_level=2, movetime=50ms
eval: 64 games, every 10 epochs
```

### B. Recommended Config for Next Run

```yaml
grpo:
  rollout_temperature: 1.6    # was 1.3
  ppo_steps: 4                # was 1
eval:
  games: 128                  # was 64
```

### C. Full Metric Trajectories (15-point samples)

**train/reward_mean:**
```
step=    0  val=-0.206    step= 1856  val=-0.119    step= 3915  val=-0.240
step=  332  val=-0.738    step= 2147  val=-0.149    step= 4278  val=-0.417
step=  762  val=-0.630    step= 2354  val=-0.617    step= 4576  val=-0.105
step= 1106  val=-0.224    step= 2732  val=-0.483    step= 4874  val=+0.022
step= 1456  val=-0.053    step= 3183  val=-0.454
                          step= 3525  val=+0.188
```

**train/entropy:**
```
step=    0  val=2.775    step= 1856  val=1.929    step= 3915  val=1.719
step=  332  val=2.156    step= 2147  val=1.941    step= 4278  val=1.684
step=  762  val=2.146    step= 2354  val=1.807    step= 4576  val=1.642
step= 1106  val=2.254    step= 2732  val=1.744    step= 4874  val=1.734
step= 1456  val=1.893    step= 3183  val=1.730
                          step= 3525  val=1.651
```

**train/clip_fraction:**
```
step=    0  val=0.265    step= 1856  val=0.243    step= 3915  val=0.247
step=  332  val=0.263    step= 2147  val=0.271    step= 4278  val=0.229
step=  762  val=0.249    step= 2354  val=0.239    step= 4576  val=0.258
step= 1106  val=0.285    step= 2732  val=0.229    step= 4874  val=0.254
step= 1456  val=0.274    step= 3183  val=0.239
                          step= 3525  val=0.244
```

### D. Related Documents

- [Post-Temperature-Fix Analysis (recs implemented)](./2026-02-08_post-tempfix-run-analysis.md) — Source of recommendations 1-8
- [Temperature Mismatch Bug](./2026-02-07_clean-run-temperature-mismatch-bug.md) — Root cause of original 55% clip fraction
- [GPT-5.2 Post-Fix Review](./postfix_grpo_run_review_2026-02-10.md) — Independent analysis of same run
