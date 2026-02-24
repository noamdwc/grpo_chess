---
title: "DeepMind 136M Teacher Baseline vs Stockfish"
date: "2026-02-24"
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

# DeepMind 136M Teacher Baseline vs Stockfish

## Executive Summary

The DeepMind 136M searchless chess teacher model scores **0.688** (12W / 20D / 0L) against Stockfish skill level 2, corresponding to approximately **+137 Elo** above that opponent. This is the target performance ceiling for distilled student models.

**TL;DR:** Teacher wins 37.5%, draws 62.5%, loses 0% vs Stockfish skill 2. Any distilled student scoring below ~0.69 has degraded teacher strength.

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

| Metric | Value |
|--------|-------|
| Score | **0.688** |
| Wins | 12 |
| Draws | 20 |
| Losses | 0 |
| Elo diff vs Stockfish skill=2 | **+137** |

Run via:
```bash
python -m src.distill.eval_teacher --config distill.yaml
```

Model: `136M` checkpoint at step 6,400,000 (`searchless_chess/checkpoints/136M`), bfloat16 inference on CPU.

## Interpretation

- **0 losses** indicates the teacher is reliably above Stockfish skill 2.
- The high draw rate (62.5%) is expected: the teacher plays accurately but Stockfish defends well even at skill 2, leading to many drawn endgames.
- This score is the **distillation success threshold**: a student model that achieves ≥0.688 has preserved teacher playing strength.

## Open Questions

- [ ] How does the teacher score vs Stockfish skill levels 5, 10, 15? Would give a clearer picture of its absolute strength.
- [ ] Does the distilled 9M student approach this score after GRPO fine-tuning?
