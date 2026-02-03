---
title: "Pretrain + GRPO Training Failure Analysis"
date: "2026-01-29 12:30:00 UTC"
agent: "claude-opus-4-5-20251101"

git_commit: "18f604785a542d0573c51ed1e93be017934945c7"
git_branch: "self_play"
uncommitted_changes: false

files_analyzed:
  - src/grpo_self_play/grpo_logic/model.py

wandb_runs:
  - run_id: "gnt9fhq0"
    run_name: "chess-grpo-20260128-1834-79ai (WITH pretrain)"
  - run_id: "pjjfjmis"
    run_name: "chess-grpo-20260121-0108-q6qc (no pretrain)"
  - run_id: "z7s89e1z"
    run_name: "chess-grpo-20260120-1220-g61h (no pretrain)"

tools_used:
  - Read
  - WandB MCP (list_runs, get_run_metrics, get_run_summary)
  - Bash

prompt: |
  Analyze the last run that used pretrain weights - why did the model fail to learn chess?

tags:
  - pretrain
  - training-failure
  - hyperparameters
---

# Pretrain + GRPO Training Failure Analysis

## Executive Summary

The GRPO run initialized with pretrained weights failed to learn chess due to excessively aggressive policy updates. The same hyperparameters that work for training from scratch cause catastrophic forgetting when fine-tuning from pretrain.

**TL;DR:** Pretrained weights need 3-10x lower learning rate and fewer PPO steps to prevent destroying learned features before RL can guide improvement.

## Context

### Research Question
Why did run `gnt9fhq0` (using pretrained weights) fail to improve at chess after 80 epochs, while maintaining healthy entropy?

### Scope
Compared the pretrain run against recent non-pretrain runs to isolate the effect of initialization.

## Findings

### Finding 1: Clip Fraction 4-10x Higher With Pretrain

| Run | Pretrain | mean_clip_fraction |
|-----|----------|-------------------|
| gnt9fhq0 | **Yes** | **37-78%** |
| pjjfjmis | No | 0.8-30% |
| z7s89e1z | No | 3.6-30% |

**Interpretation:** The pretrained policy changes too much per step, causing most gradients to be clipped and wasted.

### Finding 2: KL Coefficient Maxed Out Immediately

| Run | Pretrain | adaptive_kl/current_kl_coef |
|-----|----------|----------------------------|
| gnt9fhq0 | **Yes** | **0.015 → 0.20 (max) by step 124** |
| pjjfjmis | No | stayed at 0.001 (min) |
| z7s89e1z | No | stayed at 0.001 (min) |

**Interpretation:** KL divergence consistently exceeded target, indicating unstable policy updates. The adaptive controller maxed out the penalty, dominating the loss.

### Finding 3: No Learning Despite 80 Epochs

| Metric | Start | End |
|--------|-------|-----|
| eval_stockfish/score | 3.9% | 2.3% |
| train/avg_reward | -0.02 | 0.12 |

**Interpretation:** Eval score flat/declining. The model couldn't translate training signal into improved chess play.

## Root Cause

When starting from pretrained weights:
1. Policy has existing structure (~37% move accuracy from pretrain)
2. Same LR (3e-5) causes larger *relative* policy changes
3. `ppo_steps=4` compounds drift (4 updates per batch)
4. High clip fraction → wasted gradients
5. KL penalty maxes out → dominates loss
6. Result: Pretrained features destroyed before RL can guide improvement

## Recommendations

### Hyperparameter Changes for Pretrain + GRPO

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **lr** | 3e-5 | **3e-6 to 1e-5** | 3-10x lower to preserve pretrained features |
| **ppo_steps** | 4 | **1** | Single update prevents compounding policy drift |
| **clip_ratio** | 0.1 | **0.2** | Wider clip allows meaningful updates when starting from structured policy |
| **freeze_layers** | 0 | **1-2** | Preserve early transformer layers; fine-tune upper layers only |
| **kl_coef_max** | 0.2 | **0.05-0.1** | Prevent KL penalty from dominating loss |
| **kl_adapt_rate** | 1.5 | **1.2** | Slower adaptation prevents KL coef spikes |

### Priority

1. **High**: Lower LR to 3e-6 or 1e-5
2. **High**: Set ppo_steps=1
3. **Medium**: Set freeze_layers=1 or 2
4. **Medium**: Increase clip_ratio to 0.2

## Open Questions

- [ ] What is the optimal number of layers to freeze for this architecture?
- [ ] Should we use a warmup period with very low LR before ramping up?
- [ ] Would a separate optimizer/LR for the action head vs transformer help?
