# `src/` Module Guide

This directory contains the project-owned PyTorch/Lightning implementation around a fixed-action-space chess policy. The code is experimental: it supports training and evaluation workflows, but the research reports do not establish that GRPO improves playing strength.

## Main pipeline

```text
ChessStartStatesDataset
    -> FEN tokenization + legal-action masks
    -> ChessTransformer policy
    -> grouped trajectory sampling
    -> Stockfish reward deltas
    -> group-relative advantages
    -> PPO clipping + KL penalty
    -> optimizer update
    -> Stockfish evaluation callback
```

Supervised pretraining and teacher distillation are separate initialization paths that use the same policy family. The `searchless_chess` directory is an upstream Git submodule; this `src/` tree contains the surrounding training, evaluation, conversion, and testing infrastructure.

## Module map

| Path | Responsibility |
| --- | --- |
| `models.py` | Encoder-only transformer policy over 1,968 move actions, including legal-action probability helpers. |
| `models_9m.py` | DeepMind Searchless Chess 9M-compatible transformer and action-value policy path. |
| `grpo_logic/model.py` | Lightning module that samples rollouts, computes rewards, and performs manual PPO updates. |
| `grpo_logic/loss.py` | Group-relative advantages, importance ratios, PPO clipping, KL penalty, and logging diagnostics. |
| `grpo_logic/sampling.py` | Batched self-play trajectory generation, Stockfish opponent moves, padding, and reward deltas. |
| `chess/chess_logic.py` | FEN/board encoding, action mapping, and legal move masks. |
| `chess/rewards.py` | Stockfish-backed position evaluation and normalized/cached rewards. |
| `chess/stockfish.py` | Stockfish process lifecycle, configuration, path resolution, and concurrency helpers. |
| `chess/boards_dataset.py` | Synthetic opening, middlegame, and endgame starting-position generation. |
| `evaluator.py` / `eval_utils.py` | Policy-vs-Stockfish games, score calculation, PGN capture, and Lightning callback integration. |
| `distill/` | DeepMind teacher loading, dataset generation/conversion, soft-label training, and checkpoint export. |
| `pretrain/` | Supervised training on sampled positions from high-Elo game data. |
| `configs/` | YAML configurations loaded into typed dataclasses by `configs/config_loader.py`. |
| `checkpoint_compat.py` | Checkpoint format compatibility and safe loading helpers. |

## Commands

From the repository root, after installing `requirements.txt` and Stockfish:

```bash
python -m pytest tests/ -q
python -m src.train_self_play --config default.yaml
python -m src.pretrain.pretrain --config pretrain.yaml
python -m src.distill.distill --config distill.yaml
python -m src.distill.eval_teacher --config distill.yaml --teacher_model 136M
```

Relative config names resolve under `src/configs/`; absolute paths are also accepted. Training and distillation require external datasets/checkpoints and are not quick local commands. Use the smoke configurations and the dated reports for bounded experiments.

## Configuration

`src/configs/config_loader.py` converts YAML sections into typed dataclasses. The main GRPO configuration includes trajectory count/depth, rollout temperature, PPO steps, clipping, KL coefficient, teacher-forcing controls, evaluation cadence, and checkpoint settings. Stockfish path, skill, move limit, and thread settings are also configuration-driven.

The default evaluation score is:

```text
(wins + 0.5 * draws) / games
```

The accompanying Elo field is an approximate logistic transform of that score. It is useful for comparing runs under one protocol, but it is not a calibrated rating.

## Testing signal

The tests cover:

- group-advantage centering, padding behavior, clipping, and finite-loss invariants;
- legal-action masking and board/action mapping;
- model readout and checkpoint compatibility;
- pretraining/distillation dataset and loss guards;
- Stockfish process safety and evaluation callbacks; and
- small training and data-pipeline smoke paths.

Some tests skip or require external Stockfish, JAX/DeepMind dependencies, checkpoints, or datasets. The test suite therefore demonstrates substantial checked behavior without implying that every external training workflow is reproducible on every machine.

## Research context

Use [`research_docs/`](../research_docs/) for the evidence trail. In particular:

- [`2026-02-24_deepmind-136m-teacher-baseline.md`](../research_docs/2026-02-24_deepmind-136m-teacher-baseline.md) records the teacher reference measurements.
- [`2026-02-20_distill-grpo-collapse-analysis.md`](../research_docs/2026-02-20_distill-grpo-collapse-analysis.md) analyzes weak distillation and GRPO warm-start failure.
- [`2026-01-14_per-step-rewards-analysis.md`](../research_docs/2026-01-14_per-step-rewards-analysis.md) records the strongest documented early GRPO result and its limitations.
