---
title: "DeepMind Teacher Model Baselines vs Stockfish (9M and 136M)"
date: "2026-02-26"
agent: "claude-sonnet-4-6"

git_commit: "26681b4033e396431f4ad9e03e8d904217a444c9"
git_branch: "feature/lightning_training"
uncommitted_changes: false

files_analyzed:
  - src/distill/teacher.py
  - src/distill/eval_teacher.py
  - src/eval_utils.py
  - src/configs/distill.yaml

wandb_runs: []

tools_used:
  - Read
  - Bash

prompt: |
  Run the DeepMind 136M teacher model against Stockfish to establish a baseline
  win rate for comparison with distilled student models.

tags:
  - distillation
  - baseline
  - evaluation
  - teacher-model
---

# DeepMind Teacher Model Baselines vs Stockfish (9M and 136M)

## Executive Summary

Both DeepMind searchless chess models (9M and 136M parameters) were evaluated against Stockfish skill level 2 over 32 games each. Neither model ever loses. The 136M scores **0.688** (+137 Elo) and the 9M scores **0.625** (+89 Elo). These are the distillation success thresholds for student models trained from each teacher.

**TL;DR:** 136M is stronger (+48 Elo over 9M), but 9M is still solidly above Stockfish skill 2. A distilled 9M student should target ≥0.625.

## Eval Configuration

From `src/configs/distill.yaml`:

| Parameter | Value |
|-----------|-------|
| Stockfish skill level | 2 |
| Stockfish movetime | 50ms |
| Games | 32 |
| Max plies | 400 |
| Randomize opening | true |
| Opening plies | 6 |
| Seed | 0 |

## Results

| Model | Score | W / D / L | Elo diff | Checkpoint step |
|-------|-------|-----------|----------|-----------------|
| DeepMind 136M | **0.688** | 12 / 20 / 0 | **+137** | 6,400,000 |
| DeepMind 9M | **0.625** | 8 / 24 / 0 | **+89** | 6,400,000 |

Run via:
```bash
python -m src.distill.eval_teacher --config distill.yaml --teacher_model 136M
python -m src.distill.eval_teacher --config distill.yaml --teacher_model 9M
```

Checkpoints: `searchless_chess/checkpoints/{9M,136M}/`, bfloat16 inference on CPU.

## Interpretation

- **0 losses** for both models — both are reliably above Stockfish skill 2.
- The higher draw rate for 9M (75% vs 62.5% for 136M) suggests 9M plays more defensively / reaches more drawn endgames rather than converting to wins.
- **Distillation thresholds**: a student distilled from 9M should target ≥0.625; from 136M ≥0.688.
- The 48 Elo gap between 9M and 136M quantifies the cost of the 15x size reduction in the teacher itself — the distilled student can only hope to match, not exceed, its teacher.

## Open Questions

- [ ] How do both models score vs Stockfish skill 5, 10, 15? Would give absolute strength context.
- [ ] Does a distilled 9M student (post GRPO fine-tuning) approach the 9M teacher's 0.625?
