---
title: "Distill-to-GRPO Collapse Analysis (run chess-grpo-20260218-2255-d1xh)"
date: "2026-02-20 00:00:00 UTC"
agent: "codex-gpt-5"

git_commit: "666db78185b3649aa695172e11b530f7bdbc0b55"
git_branch: "main"
uncommitted_changes: true

files_analyzed:
  - src/distill/distill.py:99
  - src/distill/teacher.py:214
  - src/distill/convert_deepmind_data.py:143
  - src/distill/distill_dataset.py:117
  - src/evaluator.py:110

wandb_runs:
  - run_id: "27bkfy9l"
    run_name: "chess-grpo-20260218-2255-d1xh"
  - run_id: "uztyx8jw"
    run_name: "distill-20260216-0909-m9kq"

tools_used:
  - Bash
  - WandB MCP (list_runs, get_run_summary, sample_metrics)

prompt: |
  Distillation appears to fail; GRPO run from distilled checkpoint lost 62/64 vs Stockfish level 2.
  Identify likely root causes and define a concrete fix path.

tags:
  - distillation
  - grpo
  - checkpoint-quality
  - data-integrity
---

# Distill-to-GRPO Collapse Analysis

## Executive Summary

The failure is consistent with weak or misaligned distillation supervision feeding a highly aggressive GRPO setup.

Evidence from `distill-20260216-0909-m9kq` (`uztyx8jw`) shows near-random imitation quality throughout training (`train/top1_match ≈ 0.05`, `train/entropy ≈ 3.42`, `train/loss ≈ 3.42`).

Then `chess-grpo-20260218-2255-d1xh` (`27bkfy9l`) starts from that checkpoint with unstable fine-tune settings (`ppo_steps=4`, `rollout_temperature=1.6`, `teacher_forcing_prob=0.3`, `freeze_layers=0`), and evaluation collapses to near-zero score (0 wins, 2 draws, 62 losses in 64 games).

## Findings

### Finding 1: Distillation checkpoint quality is near-random

- Evidence:
  - `uztyx8jw` summary: `train/top1_match=0.051`, `val/top1_match=0.049`, `train/top5_match=0.239`, `val/top5_match=0.210`.
  - `uztyx8jw` sampled metrics remain around these values across run duration.
- Interpretation:
  - The distilled policy never learned a strong move ranking signal; the output distribution remains broad/high-entropy.

### Finding 2: Distill loss path has no guard for teacher/legal misalignment

- Evidence:
  - `src/distill/distill.py:120-127` computes loss directly from `teacher_indices` without checking if each teacher move is legal under `legal_masks`.
  - `src/distill/distill.py:154-157` KL metric also assumes teacher entries are valid.
- Interpretation:
  - Any bad teacher index (e.g., from conversion artifacts or action-space mismatch) can inject invalid supervision and degrade signal.

### Finding 3: DeepMind conversion path does not validate move legality

- Evidence:
  - `src/distill/convert_deepmind_data.py:143-148` filters by `MOVE_TO_ACTION` only, not `move in board.legal_moves`.
- Interpretation:
  - If source records include non-legal actions for a FEN, dataset targets can be contaminated.

### Finding 4: GRPO setup after distillation is too aggressive for fragile warm-start

- Evidence:
  - `27bkfy9l` hyperparams: `lr=3e-6`, `ppo_steps=4`, `rollout_temperature=1.6`, `teacher_forcing_prob=0.3`, `freeze_layers=0`.
  - `27bkfy9l` training entropy declines strongly (sampled from ~2.93 to ~1.25), reward mean unstable/negative, and final eval from user report is 0/2/62.
- Interpretation:
  - Even a moderate initial policy can be damaged quickly with these settings; with weak distilled initialization, collapse risk is very high.

## Recommendations (Priority Order)

1. Add legality-aware filtering and renormalization inside distill loss.
2. Add legality filtering in DeepMind conversion to prevent illegal teacher actions entering shards.
3. Add distill diagnostics (`teacher_legal_fraction`, `teacher_valid_sample_fraction`) to detect silent label corruption.
4. Re-run distillation with conservative settings (lower LR, fewer epochs, freeze lower layers from pretrain).
5. Re-run GRPO from the new distilled checkpoint with safer warm-start settings (`ppo_steps=1`, lower rollout temp, lower teacher forcing, partial freezing).

## Open Questions

- Which dataset path produced the checkpoint used in `distill_final.pt` (teacher-generated vs converted DeepMind shards)?
- Should we tune teacher softmax temperature (<1) to sharpen targets once legality filtering is fixed?

## → Handoff to /plan-experiment

**Recommended experiment direction:**
- Implement legality-safe distillation loss and dataset conversion filtering.
- Train a new distill checkpoint with conservative hyperparameters and explicit diagnostics.
- Run a GRPO warm-start ablation from that checkpoint with `ppo_steps=1` and partial freezing.

**Key metrics to beat:**
- Distill: `val/top1_match > 0.10` (baseline ~0.049 in `uztyx8jw`).
- GRPO eval: `eval_stockfish/score > 0.03` at skill 2 over 64 games (baseline from user report is 0.015625 for `27bkfy9l`).

**Relevant prior run(s):**
- `uztyx8jw` (distill baseline)
- `27bkfy9l` (failed GRPO warm-start)
