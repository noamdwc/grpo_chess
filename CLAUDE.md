# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **This is a searchless chess project.** Do not suggest MCTS, tree search, or similar solutions.
- **Always use the miniconda environment** for any Python command: `~/miniconda3/envs/grpo_chess/bin/python`
- **Edit `default.yaml` for config changes** — never create new config files or hardcode values in Python. Ask which config file if unclear.
- **Keep implementations simple and minimal.** No unnecessary abstractions, utility classes, or wrapper layers. If it can be done in 10 lines, don't write 50.
- **Investigate before concluding.** When analyzing ML experiment data, pull and show the actual metrics over time before drawing any conclusions. Do not jump to premature interpretations.
- **Check compatibility** when modifying the training pipeline — verify that existing Trainer/Lightning config options (gradient clipping, greedy eval, etc.) don't conflict with new changes.

## Working Style

- Start implementation quickly when the request is clear; do minimal exploration first.
- Check existing repo scripts/files before proposing new utilities or external lookup.
- Keep solutions minimal and in scope; avoid unrelated cleanup and overengineering.
- Before major edits, verify execution context (env, config target, relevant paths/projects).
- Make reasonable assumptions and proceed; ask questions only when truly blocked.
- Run focused validation (targeted tests/syntax checks) before handoff whenever feasible.
- For larger tasks, use short plan-then-execute flow with explicit file targets.

## Commands

```bash
# Run all tests
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/

# Run a single test
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_real_grpo_training.py -k "test_name"

# Syntax check
~/miniconda3/envs/grpo_chess/bin/python -m py_compile src/grpo_logic/model.py

# GRPO training (primarily done in Google Colab via chess_model_run_git.ipynb)
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play --config my_experiment.yaml

# Distillation training
~/miniconda3/envs/grpo_chess/bin/python -m src.distill.distill --config distill.yaml
```

Tests require Stockfish installed locally (`brew install stockfish` on macOS). Some tests use subprocess invocation to mirror the Colab environment.

## Architecture

### Training Pipeline Flow

1. **`train_self_play.py`** — Entry point. Loads YAML config via `config_loader.py`, builds dataloader and Lightning model, starts training.
2. **`grpo_logic/model.py`** (`GRPOChessTransformer`, Lightning module) — Orchestrates the GRPO loop in `training_step`: samples trajectories with old policy, computes Stockfish rewards, calculates group-normalized advantages, runs PPO updates. Syncs old policy each epoch.
3. **`grpo_logic/sampling.py`** — Batched trajectory sampling from starting positions using the policy network.
4. **`grpo_logic/loss.py`** — `grpo_ppo_loss` is the main loss function (PPO surrogate + KL penalty). `train/loss` = PPO loss + kl_coef * KL divergence. Loss components are logged raw (no coefficients baked in).
5. **`chess/rewards.py`** — Dense rewards via Stockfish evaluation at each position.
6. **`chess/boards_dataset.py`** — Generates starting positions with phase distribution (opening/middlegame/endgame) and quality filtering.

### Model

`models.py` (`ChessTransformer`) — Transformer encoder processing FEN-tokenized board states, outputting logits over 1968 possible moves with legal move masking (`-inf` for illegal moves).

### Configuration System

YAML configs in `src/configs/`. `config_loader.py` loads YAML into typed dataclasses (`ExperimentConfig` → `TrainingConfig`, `GRPOConfig`, `ChessTransformerConfig`, `EvalConfig`, `StockfishConfig`, etc.). Programmatic overrides:
```python
train(config_path="default.yaml", overrides={"grpo": {"lr": 1e-4}})
```

### Distillation Pipeline

`distill/` trains a student ChessTransformer on soft labels from the DeepMind searchless chess 136M teacher model. Entry point: `python -m src.distill.distill --config distill.yaml`. Key files: `distill.py` (Lightning module + training), `distill_dataset.py` (dataset/collation), `teacher.py` (teacher model wrapper), `generate_dataset.py` / `convert_deepmind_data.py` (data generation).

### Pretrain Pipeline

`pretrain/` provides supervised pretraining on position-evaluation data before GRPO. Configured via the `pretrain` section in YAML config.

### Evaluation

`evaluator.py` benchmarks against Stockfish at configurable skill levels. `StockfishEvalCallback` runs evaluation as a Lightning callback (used by GRPO, pretrain, and distill). Key WandB metrics: `eval_stockfish/score` (win rate), `eval_stockfish/elo_diff`. Supporting utilities in `eval_utils.py`.

### WandB MCP Servers

Two MCP server instances: `wandb-pretrain` (chess-grpo-pretrain project) and `wandb-grpo` (Chess-GRPO-Bot project). Server code in `mcp_wandb_server/`.

## Research Workflow

The experiment loop runs through four skills in sequence, each producing a structured handoff artifact:

```
/research-insights  →  /plan-experiment  →  /code-implementation  →  /run-experiment  →  (loop)
```

Use `/experiment-cycle` to orchestrate the full loop automatically.

| Skill | Purpose | Output artifact |
|-------|---------|-----------------|
| `/research-insights` | Analyze runs/code, write findings | `research_docs/YYYY-MM-DD_*.md` |
| `/plan-experiment` | Design experiment, write config | `research_docs/experiments/YYYY-MM-DD_<slug>.md` + `src/configs/<name>.yaml` |
| `/code-implementation` | Implement plan doc changes | Code edits + git commit |
| `/run-experiment` | Submit to Lightning.ai, monitor, report | `research_docs/runs/YYYY-MM-DD_<job>.md` |
| `/experiment-cycle` | Orchestrate all four steps end-to-end | All of the above |
| `/code-auditor` | Audit proposed changes | Checks `research_docs/KNOWN_NOT_IMPLEMENTED_CHANGES.md` |

**Document storage conventions:**
- Research findings: `research_docs/YYYY-MM-DD_*.md`
- Experiment plans: `research_docs/experiments/YYYY-MM-DD_<slug>.md`
- Run reports: `research_docs/runs/YYYY-MM-DD_<job>.md`

Directories are created lazily on first write. Research docs follow the template at `research_docs/TEMPLATE.md`.

## Key Files

| Task | Files |
|------|-------|
| GRPO training loop | `grpo_logic/model.py` |
| Loss computation | `grpo_logic/loss.py` |
| Trajectory sampling | `grpo_logic/sampling.py` |
| Reward computation | `chess/rewards.py` |
| Dataset generation | `chess/boards_dataset.py` |
| Model architecture | `models.py` |
| Evaluation | `evaluator.py`, `eval_utils.py` |
| Stockfish interface | `chess/stockfish.py` |
| Policy player | `chess/policy_player.py`, `chess/searcher.py` |
| Distillation | `distill/distill.py`, `distill/distill_dataset.py`, `distill/teacher.py` |
| Trainer utilities | `trainer.py` |
| Config | `configs/default.yaml`, `configs/distill.yaml`, `configs/pretrain.yaml`, `configs/config_loader.py` |

All paths relative to `src/`.

## Troubleshooting

- **Reasoning-GRPO failure modes:** see `docs/reasoning_grpo/failure_modes.md`.
  When you add a safety check, a metric, or a handling path for an observed
  failure, also write it up in that file with symptom / how you'll notice /
  recovery / history. Leave an inline comment at the code site that says the
  same thing in its own words.
