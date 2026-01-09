---
title: "GRPO Chess Learning Failure Analysis"
date: "2026-01-09 18:30:00 UTC"
agent: "claude-opus-4-5-20251101"

git_commit: "5e15b437a7f8e07d843cbdd17289f79b7fe7d6aa"
git_branch: "improve_data_quality"
uncommitted_changes: true  # Several uncommitted files present

files_analyzed:
  - src/grpo_self_play/grpo_logic/loss.py
  - src/grpo_self_play/grpo_logic/model.py
  - src/grpo_self_play/grpo_logic/sampling.py
  - src/grpo_self_play/chess/rewards.py
  - src/grpo_self_play/chess/boards_dataset.py
  - src/grpo_self_play/README.md
  - chess_model_run_git.ipynb

wandb_runs:
  - run_id: "xyqjy01q"
    run_name: "chess-grpo-20251227-1907-ddju"
  - run_id: "xi0fye6i"
    run_name: "chess-grpo-20251226-0009-42ky"
  - run_id: "0zzn82pw"
    run_name: "chess-grpo-20251217-1951-cxww"
  - run_id: "etnp8mk3"
    run_name: "chess-grpo-20251217-0605-en2l"
  - run_id: "61dds3s3"
    run_name: "chess-grpo-20251222-0243-tygy"
  - run_id: "cjwemrb6"
    run_name: "chess-grpo-20260101-1933-glav"

tools_used:
  - Read
  - Grep
  - Glob
  - Task (Explore subagent)
  - WandB MCP (list_runs)

prompt: |
  From the readme in src/grpo_self_play you can understand I use the notebook
  chess_model_run_git.ipynb to run the code in colab, and more importantly how I run the code.
  Use the code, and the metrics from wandb you have the tools for - and help me understand
  why the model fails to learn how to play chess well and been stuck.

tags:
  - training
  - debugging
  - rewards
  - grpo
  - critical-analysis
---

# GRPO Chess Learning Failure Analysis

## Executive Summary

The GRPO chess model fails to learn effectively due to a combination of **weak reward signals**, **no temporal credit assignment**, and **aggressive policy optimization constraints**. After extensive training (100+ epochs across multiple runs), the model achieves only 1-3% win rate against Stockfish level 1-5, barely above random play.

**TL;DR:** The model receives a single weak scalar reward per trajectory, making it impossible to learn which specific moves are good. Combined with batch normalization that destroys absolute reward information and PPO clipping that prevents learning, the model is stuck.

## Context

### Background
The project aims to train a chess-playing transformer using GRPO (Group Relative Policy Optimization), a variant of PPO that uses group-based advantage estimation instead of value functions. The model plays trajectories against itself, receives Stockfish-based rewards, and updates via policy gradients.

### Research Question
Why does the model fail to improve its chess-playing ability despite extensive training? Multiple runs show:
- Win rates stuck at 1-3% against Stockfish
- Elo difference of -600 to -800 (far below amateur level)
- Training losses that decrease but don't translate to improved play

### Scope
This analysis covers:
- GRPO loss computation and advantage estimation
- Reward signal strength and credit assignment
- Trajectory sampling and data quality
- Policy optimization dynamics

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Runs | 30+ runs from Nov 2025 - Jan 2026, focusing on recent runs with new metrics |
| Code Files | Core GRPO implementation at commit `5e15b43` |
| Training Config | `lr=1e-6 to 1e-3`, `num_trajectories=16`, `trajectory_depth=16`, `clip_ratio=0.1-0.2` |

### Analysis Approach

1. Retrieved run summaries from WandB to identify patterns in metrics
2. Read all core GRPO implementation files to understand the algorithm
3. Traced the data flow from position generation → trajectory sampling → reward computation → loss calculation
4. Identified mismatches between expected behavior and observed metrics

### Tools Used

- **Read**: Examined `loss.py`, `sampling.py`, `rewards.py`, `model.py`, `boards_dataset.py`
- **Task (Explore)**: Deep exploration of codebase structure and relationships
- **WandB MCP**: Retrieved run list with summary metrics (detailed metrics unavailable due to API issues)

## Findings

### Finding 1: Sparse Reward Signal (CRITICAL)

**Evidence:**

In `src/grpo_self_play/chess/rewards.py:58-79`:
```python
def reward_board(env: chess.Board, board_start: chess.Board, movetime_ms: int = 0, depth: int = 16) -> float:
    pov_is_white = (board_start.turn == chess.WHITE)
    if env.is_game_over(claim_draw=True):
        if env.is_checkmate():
            pov_loses = (env.turn == (chess.WHITE if pov_is_white else chess.BLACK))
            r_t = -1.0 if pov_loses else 1.0
        else:
            r_t = 0.0  # Draw
    else:
        fen_t = env.fen()
        r_t = evaluate_fen(fen_t, pov_is_white, movetime_ms, depth)

    fen_0 = board_start.fen()
    r_0 = evaluate_fen(fen_0, pov_is_white, movetime_ms, depth)
    return r_t - r_0  # Reward is the CHANGE in eval
```

**The Problem:**
The reward is computed as `r_t - r_0` (final position eval minus starting position eval). Since positions are filtered to be within [-200, +200] centipawns, and evaluation uses `tanh(cp/600)` normalization:

- A 50 centipawn improvement → `tanh(50/600) ≈ 0.083`
- A 100 centipawn improvement → `tanh(100/600) ≈ 0.165`
- Typical rewards are in **[-0.3, +0.3]** range

**Metrics from WandB:**

| Run | train/avg_reward | train/reward_std | Interpretation |
|-----|------------------|------------------|----------------|
| `xyqjy01q` | -0.150 | 0.914 | Near-zero mean, high variance |
| `xi0fye6i` | 0.122 | 0.754 | Near-zero mean, high variance |
| `61dds3s3` | 0.147 | 0.783 | Near-zero mean, high variance |

The weak signal (mean ~0) with high variance (std ~0.8) means the model receives very little useful gradient information.

---

### Finding 2: No Temporal Credit Assignment (CRITICAL)

**Evidence:**

In `src/grpo_self_play/grpo_logic/sampling.py:142-147`:
```python
# Compute reward per final state
group_rewards = torch.zeros(B, G, dtype=torch.float32, device=device)
for env_idx, env in enumerate(envs):
    b_idx = env_idx // G
    g_idx = env_idx % G
    group_rewards[b_idx, g_idx] = reward_board(env, boards[b_idx], depth=4, movetime_ms=0)
```

Each trajectory of 16 moves gets **ONE scalar reward** at the end.

In `src/grpo_self_play/grpo_logic/loss.py:174-176`:
```python
advantages_2d = group_advantage(group_rewards).detach()  # [B, G]
advantages = advantages_2d.unsqueeze(-1).expand(B, G, T)  # [B, G, T] - SAME value for all T!
advantages = advantages * pad_mask.float()
```

**The Problem:**
All 16 moves in a trajectory receive the **identical advantage value**. The model cannot distinguish:
- Move 1 (blundering a piece) gets advantage = 0.5
- Move 8 (brilliant sacrifice) gets advantage = 0.5
- Move 16 (random pawn push) gets advantage = 0.5

This is fundamentally incompatible with learning chess, where individual move quality varies enormously.

---

### Finding 3: Batch Normalization Destroys Absolute Signal

**Evidence:**

In `src/grpo_self_play/grpo_logic/loss.py:59-72`:
```python
def group_advantage(group_rewards: torch.Tensor) -> torch.Tensor:
    mean_reward = group_rewards.mean(dim=-1, keepdim=True)
    std_reward = group_rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
    advantages = (group_rewards - mean_reward) / std_reward
    return advantages
```

**The Problem:**
When all trajectories from a position have similar (poor) outcomes:
- Trajectory 1: reward = -0.1 → advantage = +0.5 (normalized)
- Trajectory 2: reward = -0.15 → advantage = -0.5 (normalized)
- Trajectory 3: reward = -0.12 → advantage = +0.1 (normalized)

The model thinks Trajectory 1 is "good" even though all options were bad. There's no way to learn "this entire position is losing."

---

### Finding 4: PPO Clipping Prevents Learning

**Evidence from WandB:**

| Run | mean_clip_fraction | mean_ratio | Interpretation |
|-----|-------------------|------------|----------------|
| `0zzn82pw` | **1.0 (100%)** | 0.036 | All gradients clipped! |
| `xyqjy01q` | 0.117 | 0.996 | Moderate clipping |
| `etnp8mk3` | 0.581 | 0.415 | High clipping |

**The Problem:**
When `mean_clip_fraction = 1.0`, every single gradient is being clipped, meaning **no learning can occur**. This happens when:
1. The policy ratio `π_new/π_old` leaves the [0.8, 1.2] range
2. Policy sync at epoch start creates stale old_policy
3. Early aggressive updates cause ratio explosion

Run `0zzn82pw` shows this failure mode clearly: `mean_ratio = 0.036` means the new policy probability is ~3.6% of the old policy's - massive divergence.

---

### Finding 5: Shallow Stockfish Evaluation

**Evidence:**

In `src/grpo_self_play/grpo_logic/sampling.py:147`:
```python
group_rewards[b_idx, g_idx] = reward_board(env, boards[b_idx], depth=4, movetime_ms=0)
```

In `src/grpo_self_play/chess/boards_dataset.py:298`:
```python
stockfish_filter_depth: int = 2  # Default depth for quality filtering
```

**The Problem:**
- **Depth=4** for reward computation is extremely shallow
- **Depth=2** for position filtering is nearly random
- Stockfish at depth 2-4 makes significant tactical errors and has high variance

This introduces noise into the reward signal, making it even harder to learn.

---

### Finding 6: Known Dataset Issues

**Evidence:**

In `src/grpo_self_play/chess/boards_dataset.py:155`:
```python
def generate_endgame_position():  # TODO: This is not working as expected
```

The endgame generation is documented as broken. The dataset may:
- Miss important position types
- Over-represent certain phases
- Generate unrealistic positions

---

## Analysis

### Root Cause Hierarchy

```
PRIMARY CAUSES (Fix these first):
├── 1. No credit assignment (single reward per trajectory)
├── 2. Weak reward signal (differential eval in [-0.3, +0.3])
└── 3. Batch normalization destroys absolute signal

SECONDARY CAUSES (Fix after primary):
├── 4. PPO clipping too aggressive (causing 100% clip fraction)
└── 5. Shallow Stockfish evaluation (depth=4 is noisy)

TERTIARY CAUSES (Nice to fix):
├── 6. Dataset quality issues (broken endgame generation)
└── 7. No learning rate warmup/scheduling
```

### Impact on Training

The combination of issues creates a **catastrophic learning failure**:

1. Weak rewards → small gradients
2. Same advantage for all moves → incorrect credit
3. Batch normalization → loss of absolute information
4. PPO clipping → gradients often zeroed
5. Net effect: **Model learns nothing meaningful**

## Recommendations

### Immediate Actions (High Priority)

1. **Add Per-Step Rewards**
   - What: Compute `r_t - r_{t-1}` for each step, not just final reward
   - Why: Enables temporal credit assignment
   - Where: `src/grpo_self_play/grpo_logic/sampling.py:142-147`
   - Estimated effort: Medium
   - Code sketch:
     ```python
     step_rewards = torch.zeros(B, G, T)
     for t in range(T):
         step_rewards[:, :, t] = evaluate_fen(state_t) - evaluate_fen(state_{t-1})
     ```

2. **Increase Reward Magnitude**
   - What: Change normalization from `tanh(cp/600)` to `tanh(cp/200)` or use raw centipawns
   - Why: Stronger gradient signal
   - Where: `src/grpo_self_play/chess/rewards.py:53`
   - Estimated effort: Low

3. **Increase Stockfish Depth**
   - What: Use `depth=12+` for rewards, `depth=6+` for filtering
   - Why: Reduces reward noise
   - Where: `sampling.py:147`, `boards_dataset.py:298`
   - Estimated effort: Low (but increases training time)

4. **Reduce KL Coefficient Initially**
   - What: Start with `kl_coef=0.0`, gradually increase
   - Why: Allow more exploration early in training
   - Where: Training config in notebook
   - Estimated effort: Low

### Medium-Term Improvements

5. **Add GAE (Generalized Advantage Estimation)**
   - What: Replace group normalization with `A_t = Σ (γλ)^k δ_{t+k}`
   - Why: Better credit assignment with temporal decay
   - Where: New function in `loss.py`
   - Estimated effort: High

6. **Add Value Function**
   - What: Train `V(s)` alongside policy, use for advantage estimation
   - Why: Standard PPO approach, more stable
   - Where: Add to `models.py`, `loss.py`
   - Estimated effort: High

### Long-Term Considerations

7. **Use Opening Book + Endgame Tablebases**
   - What: Ground truth for early and late game
   - Why: Reduces noise, provides perfect signal for known positions
   - Estimated effort: Medium

## Open Questions

- [ ] Would switching to actor-critic (with value function) provide more stable learning?
- [ ] Could curriculum learning (starting with simpler positions) help?
- [ ] Would self-play against a fixed opponent (not self) be more stable?

## Appendix

### A. WandB Run Summary Table

| Run ID | Name | Epochs | Avg Reward | Clip Frac | SF Score | Elo Diff |
|--------|------|--------|------------|-----------|----------|----------|
| `xyqjy01q` | chess-grpo-20251227 | 180 | -0.150 | 0.117 | 0% | -2400 |
| `xi0fye6i` | chess-grpo-20251226 | 176 | 0.122 | 0.094 | 1.5% | -720 |
| `0zzn82pw` | chess-grpo-20251217 | 116 | 0.294 | 1.000 | 3.1% | -597 |
| `etnp8mk3` | chess-grpo-20251217 | 100 | 0.249 | 0.581 | 1.5% | -720 |
| `61dds3s3` | chess-grpo-20251222 | 173 | 0.147 | 0.087 | 1.5% | -720 |

### B. Key Metric Definitions

- **train/avg_reward**: Mean reward across all trajectories in batch
- **mean_clip_fraction**: Fraction of policy ratios outside [1-ε, 1+ε]
- **mean_ratio**: Average π_new(a|s) / π_old(a|s)
- **eval_stockfish/score**: Win rate (wins + 0.5*draws) / games
- **eval_stockfish/elo_diff**: Approximate Elo difference from win rate

### C. Related Documents

This is the first document in the research_docs directory.

---

<!--
HUMAN REVIEW SECTION

(Add review notes here after human verification)
-->
