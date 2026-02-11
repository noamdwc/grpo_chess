---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Post-fix GRPO/PPO Run Review: What the Plots Say, Why Elo Is Still Low, and the Next 3–6 Experiments"
date: "2026-02-10 02:10:00 UTC"
agent: "GPT-5.2 Thinking"

# Git State - Critical for reproducibility
git_commit: "UNKNOWN (no .git repo available in provided artifacts)"
git_branch: "UNKNOWN"
uncommitted_changes: "UNKNOWN"

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - TEMPLATE.md
  - SKILL.md
  - loss.py
  - stockfish.py
  - 2026-02-08_post-tempfix-run-analysis.md
  - 2026-02-07_clean-run-temperature-mismatch-bug.md

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "UNKNOWN (not provided)"
    run_name: "UNKNOWN (not provided)"

# Tools Used - What capabilities did you use?
tools_used:
  - Manual review of plots + provided configs
  - Static code review of provided .py files

# Original Prompt - The exact prompt given to the agent
prompt: |
  You are an expert ML/RL researcher reviewing my chess RL training run (PPO-style) where a “Policy” agent plays against Stockfish.
  I want: (1) interpret plots re: optimization/exploration/stability, (2) bottlenecks, (3) 3–6 high-impact experiments with hypotheses,
  (4) what extra info you need. Then: write conclusions as a research document using TEMPLATE.md. Also: note I use depth instead of movetime_ms.

tags:
  - grpo
  - ppo
  - chess
  - evaluation
  - entropy
  - kl
  - clip-fraction
  - reproducibility
---

# Post-fix GRPO/PPO Run Review: What the Plots Say, Why Elo Is Still Low, and the Next 3–6 Experiments

## Executive Summary

Your PPO mechanics look *healthy enough to learn* (no obvious collapse): KL is bounded, clip fraction is moderate and improving, losses are stable. Meanwhile, eval vs Stockfish (skill=2) shows a *real upward trend* but is still extremely far behind, and the curve is noisy largely because you’re only playing 64 games per eval with randomized openings.

The biggest actionable issues I see now are: (1) **evaluation variance & protocol mismatch**, and (2) **update budget is tiny** (`ppo_steps=1`, only 16 trajectories × 16 plies per update), which strongly limits sample efficiency and makes progress slow/noisy.

**TL;DR:** The run is no longer “broken”; it’s “underpowered + high-variance eval”. Fix eval methodology and increase effective PPO update budget before chasing exotic changes.

---

## Context

### Background

You’re training a chess transformer with GRPO/PPO-like updates and evaluating periodically vs Stockfish. This run includes key fixes/changes after the temperature mismatch investigation and post-fix analysis docs.

### Research Question

Given the current plots + current config, what do the training dynamics imply (stability/exploration/optimization), what’s the most likely bottleneck for Elo, and what are the 3–6 highest expected value experiments to improve Elo?

### Scope

Included:
- Plot-level diagnostics (entropy/KL/clip fraction/reward/eval trends)
- Code-level verification for advantage normalization + Stockfish limit behavior

Excluded:
- Full WandB metric pulls (run IDs not provided)
- Deep review of reward shaping logic (not requested here, but I list the exact questions needed)

---

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| Plots (screenshots) | train/* curves + eval_stockfish/* trends (provided in-chat) |
| Configs | EvalConfig / GRPOConfig / PolicyConfig / StockfishConfig / model size (provided in-chat) |
| Code | `loss.py` (advantage normalization, loss logging definition), `stockfish.py` (time-vs-depth limit) |
| Prior research docs | Temperature mismatch bug + post-tempfix analysis |

### Analysis Approach

1. Interpreted each plot as a PPO “health” indicator (KL, clip fraction, entropy, reward mean/var).
2. Cross-checked whether code matches the intended fixes (step-wise advantage standardization, entropy logging definition).
3. Identified protocol mismatches that can inflate noise or hide progress (64-game eval, opening randomization, depth vs time).
4. Proposed a minimal set of experiments maximizing information gain and Elo impact.

---

## Findings

### Finding 1: PPO mechanics look stable (not obviously over-regularized, not exploding)

**Evidence (loss implementation / logged metrics definitions):**
- Advantage standardization is implemented per-timestep across trajectories in `step_group_advantage`.
- Loss uses PPO clipping + KL penalty; “entropy” is logged as `-E[log π(a|s)]` on sampled actions (monitoring only), not true categorical entropy.

**Interpretation (from plots):**
- `train/kl_divergence` sits in a narrow band (~0.02–0.03-ish) with spikes but no drift → updates aren’t diverging.
- `train/clip_fraction` trends downward (~0.30 → ~0.24) → updates are becoming smaller/more consistent, or the policy is stabilizing.
- `train/loss` and `train/ppo_loss` are stationary → optimizer is not unstable.

**Suspicious/Watch-out:**
- Stable PPO doesn’t mean “learning chess”; it can also mean “tiny updates”.

---

### Finding 2: Exploration is tightening steadily (entropy falls), which may be good *or* may prematurely reduce search diversity

**Key nuance:** the logged “entropy” is a *surprisal proxy* (`-log π(a|s)` on sampled actions), not full legal-move entropy. A downward curve mainly means “sampled actions are becoming higher-probability under the policy”.

You also observed previously that increasing temperature reversed this behavior; that’s consistent with temperature increasing action randomness.

---

### Finding 3: Eval improvement is plausible but variance is dominating with only 64 games + randomized openings

- `eval_stockfish/score` trends upward (roughly ~0.02 → ~0.06–0.08 peak) but is jagged.
- `eval_stockfish_skill0/elo_diff` swings by hundreds of Elo.

**Most likely explanation:** statistical noise dominates at this sample size, especially when the true score is small.

---

### Finding 4: Protocol mismatch risk: Stockfish player code uses time-based limits

You stated you use depth instead of movetime, but the provided `StockfishPlayer.act` implementation uses a time limit derived from `movetime_ms`. If eval uses this path, opponent strength is time-budgeted, not depth-budgeted.

Action item: confirm the exact “go” limit used in *eval matches* and in *training reward eval*, then pin it deterministically (prefer fixed depth or fixed nodes).

---

### Finding 5: Update budget is extremely small → slow Elo progress is expected

From your config: `num_trajectories=16`, `trajectory_depth=16`, `ppo_steps=1`. This is low effective optimization per batch. With such small batches, PPO often needs multiple epochs/steps to efficiently extract learning signal.

---

## Analysis

### Root Cause Analysis (most likely bottlenecks)

1. **Eval variance & protocol mismatch** (64 games + randomized openings; depth/time confusion; searcher hides raw policy improvements).
2. **Underpowered PPO update budget** (`ppo_steps=1`, small batch).
3. **Exploration schedule not controlled** (entropy collapses unless temperature tuned; train behavior differs from eval behavior).
4. **Opponent too strong / compute not pinned** (hard to measure & learn consistently if opponent budget drifts).

### Trade-offs

- More eval games → slower iteration but dramatically better signal-to-noise.
- More PPO steps / larger batch → more compute per iteration; likely higher Elo per hour.

---

## Recommendations

### Immediate Actions (High Priority)

1) **Make evaluation decision-grade (variance reduction)**
- Increase `games`: 64 → **256** (or 4 seeds × 128)
- Use paired openings + both colors
- Report mean score with 95% CI

Expected: smoother eval curves; Elo becomes interpretable.

2) **Pin Stockfish compute deterministically & log it**
- Ensure eval uses a fixed **depth** (or fixed nodes), not implicit time.
- Log the exact limit used at eval/training time.

Expected: less unexplained eval drift; more reliable reward shaping and opponent strength.

3) **Increase PPO optimization per batch (highest expected Elo per hour)**
- Set `ppo_steps: 1 → 4` (or 8 if stable)
- Monitor KL/clip_fraction; add KL early-stop if needed

Expected: faster improvement, especially at skill0/1.

---

### Medium-Term Experiments (next 2–3)

4) **Train/eval alignment ablation**
- Log eval in three modes: pure policy (sampled), pure policy (greedy), searcher (current).
- This isolates whether gains come from the network or from search.

5) **Increase batch size / reduce gradient variance**
- Increase `num_trajectories: 16 → 32` or use gradient accumulation.

6) **Opponent curriculum**
- Track progress at skill0/1 frequently; keep skill2 as “north star”.
- Optionally anneal teacher forcing / opponent strength.

---

## Open Questions (what I need from you)

1) Eval: how do you pair openings, and are openings fixed across eval checkpoints?
2) Reward: exact definition (cp-delta? terminal? clipping?), and credit assignment (your move only vs both plies).
3) PPO: do you have minibatching / multiple epochs within `ppo_steps`, or is it truly one pass?
4) Stockfish: confirm the exact depth-based limit used in eval and training reward eval (and where in code it is set).
5) Searcher: confirm algorithm (beam/minimax/rollout) and whether it uses engine eval or only policy/value.

---

## Appendix

### A. Suggested logging additions (low effort, high leverage)
- True legal-move entropy (categorical), not surprisal proxy
- Illegal move probability mass
- Mean centipawn loss / blunder rate on your moves
- Termination reason distribution + game length histogram
- Pure-policy vs searcher eval scores
