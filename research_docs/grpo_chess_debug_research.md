---
title: "GRPO Self-Play Chess: Updated Root-Cause Analysis (v2)"
date: "2026-02-08 20:15:00 UTC"
agent: "GPT-5.2 Thinking"

status: "update"
notes: |
  This document updates the earlier write-up after reconciling claims against actual run metrics:
  epoch=151, global_step=4839 (~32 steps/epoch), mean_ratio≈1.0, mean_clip_fraction≈0.27, mean_kl≈0.015–0.024.

tags:
  - grpo
  - ppo
  - self-play
  - evaluation
  - debugging
  - reward-signal
---

> Historical note
>
> This document was imported from the `self_play` branch during branch consolidation on 2026-04-09. That branch was intentionally not merged into `main` yet. Treat this file as archived research context, not as documentation of code currently present on the unified branch.

# GRPO Self-Play Chess: Updated Root-Cause Analysis (v2)

## 1) What changed vs the prior diagnosis

### Correction: “512 stale steps” was not supported by the run
Earlier, the hypothesis was that `old_policy_model` stayed stale for ~512 optimizer steps (an epoch).
Your actual run indicates:
- **epoch = 151, global_step = 4839 → ~32 steps/epoch**
- PPO health metrics look **reasonable**:
  - `mean_ratio ≈ 1.0`
  - `mean_clip_fraction ≈ 0.27`
  - `mean_kl_divergence ≈ 0.015–0.024`

So “severely off-policy due to epoch-level syncing” is **much weaker** as a primary root cause.

### Still true (but smaller): syncing per iteration is best practice
Even with ~32 steps/epoch, syncing old policy **per rollout-optimization iteration** is still cleaner PPO hygiene.
But given your ratio/KL stats, it is unlikely to be the *main* reason the model doesn’t improve vs Stockfish.

---

## 2) Most likely root causes (ranked)

### A) Limited learning capacity: freezing 2 of 4 transformer layers
Freezing half the network can drastically reduce the policy’s ability to reshape representation and strategy,
especially in early stages when the model needs to move far from the pretrained distribution.

**Prediction:** unfreezing more layers should immediately increase the slope of learning curves (even on easier eval tiers).

**Recommended experiment:**
- Unfreeze **all layers**, or freeze only embeddings / first block for the first N steps, then unfreeze.
- Pair with a modest LR increase (see Section 4).

---

### B) Reward signal quality is poor in weak self-play
In symmetric self-play where both sides are the same weak model, the reward signal is dominated by:
- random blunders by either side,
- short tactical noise,
- non-stationary opponent behavior,
- and “lucky wins” rather than robust skill.

This is especially problematic if your reward is **Stockfish-eval deltas per step**: early policies generate chaotic positions,
so eval swings are high-variance and credit assignment is unstable.

**Prediction:** you’ll see little/no improvement vs Stockfish skill 2 even if training “moves”.

**Recommended experiment (highest value):**
- Add a **fixed-opponent curriculum**:
  1) train/eval vs a frozen older checkpoint or a simple scripted policy
  2) progressively update the opponent (league / pool of checkpoints)
  3) keep one opponent permanently fixed for “true progress” measurement

---

### C) Advantage scaling / normalization
If you’re using a DR-GRPO-like variant without advantage std-normalization (“whitening”), gradients can be unstable:
- too big on a few noisy trajectories,
- too small on most steps,
- and sensitive to reward scale drift.

Your PPO ratio/KL looking fine does **not** guarantee the *advantage-weighted* update has good SNR.

**Recommended experiment:**
- Add **advantage standardization** per batch (or per group) before loss:
  - `A ← (A - mean(A)) / (std(A) + eps)`
- Log:
  - `adv_mean`, `adv_std`, and importantly `effective_adv_snr = mean(|A|)/std(A)` or similar.
- If you already center advantages (mean ~ 0), std normalization is the missing stabilizer.

---

### D) Teacher forcing is too weak to bootstrap (10% at depth=4)
Teacher forcing at 10% for only 4 plies of a 16-ply rollout may not move the trajectory distribution
enough to enter “reasonable chess” regions where reward is informative.

**Recommended experiment:**
- Increase forcing to **p=0.2–0.4** and **depth=8** for a bootstrap phase.
- Then **anneal** down to p≈0.05→0.0 over time.

**What to log (to ensure it helps, not masks issues):**
- `%forced_moves`
- `reward_forced_mean` vs `reward_unforced_mean`
- `eval_vs_easy_opponent` improvement (see Section 3)

---

## 3) Evaluation: how to see learning earlier (diagnostics)

### A) Add easier tiers (curriculum eval)
Stockfish skill 2 @ 50ms can be too hard to show early wins for a small transformer without strong search.
Add:
- Stockfish skill **0 / 1 / 2**
- and/or a fixed-depth with low strength

Track:
- W/D/L (score)
- average centipawn loss (ACPL) if available
- “illegal/no-move” termination count

### B) Fix the “no move => draw” edge case
Count “policy returned None / illegal” as a **loss**, not a draw.
This is minor, but improves diagnostic fidelity.

---

## 4) Hyperparameter moves that are worth trying (in this order)

### 1) Unfreeze more layers
- Unfreeze all, or unfreeze after a short warmup.

### 2) Increase LR modestly
Current: `lr = 1e-6` is extremely conservative.
Try:
- `3e-6`, then `1e-5` if KL remains controlled.

### 3) Add advantage whitening
Even if everything else is correct, this often helps a lot in high-variance reward settings.

### 4) Teacher forcing bootstrap schedule
Example schedule:
- Steps 0–10k: `p=0.3, depth=8`
- 10k–50k: linearly decay to `p=0.05`
- 50k+: `p=0.0–0.02`

### 5) Add entropy regularization if policy collapses
Your `entropy_coef=0.0`. If entropy collapses early, exploration dies.
If you observe entropy dropping fast, try a small coef (e.g. `1e-3` to `1e-2`),
or use your existing entropy-floor mechanism.

---

## 5) Minimal experiment plan (fastest path to clarity)

**Run 1 (capacity + scaling):**
- Unfreeze all layers
- Add advantage std-normalization
- LR = 3e-6
- Keep everything else the same
- Add eval tiers skill 0/1/2

**Expected outcome:** you should see improvement at skill 0/1 quickly if the training loop is fundamentally correct.

**Run 2 (bootstrap reward SNR):**
- Keep Run 1 changes
- Teacher forcing: p=0.3, depth=8 for first phase
- Add forced/unforced reward logging

**Expected outcome:** trajectories become less chaotic; reward variance drops; easier-tier eval improves faster.

**Run 3 (opponent quality):**
- Keep Run 1 changes
- Replace pure self-play with a fixed opponent (older checkpoint pool)
- Measure progress vs fixed reference

**Expected outcome:** progress becomes monotonic and measurable; reduces non-stationarity.

---

## 6) Bottom line

Given your reported metrics (ratio/KL/clip healthy-ish) and the small steps/epoch (~32),
the most plausible blockers are **(1) limited capacity from freezing 2/4 layers** and **(2) poor reward SNR in weak self-play**,
with **(3) missing advantage whitening** and **(4) insufficient teacher forcing** as strong secondary factors.

Teacher forcing is a good lever — but I’d treat it as part of a bootstrap/curriculum strategy,
not the only fix.
