# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **This is a searchless chess project.** Do not suggest MCTS, tree search, or similar solutions.
- **Always use the miniconda environment** for any Python command: `~/miniconda3/envs/grpo_chess/bin/python`
- **Edit `default.yaml` for config changes** — never create new config files or hardcode values in Python. Ask which config file if unclear.
- **Keep implementations simple and minimal.** No unnecessary abstractions, utility classes, or wrapper layers. If it can be done in 10 lines, don't write 50.
- **Investigate before concluding.** When analyzing ML experiment data, pull and show the actual metrics over time before drawing any conclusions. Do not jump to premature interpretations.
- **Check compatibility** when modifying the training pipeline — verify that existing Trainer/Lightning config options (gradient clipping, greedy eval, etc.) don't conflict with new changes.

## Commands

```bash
# Run all tests
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/

# Run a single test
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_real_grpo_training.py -k "test_name"

# Syntax check
~/miniconda3/envs/grpo_chess/bin/python -m py_compile src/grpo_logic/model.py

# Training (primarily done in Google Colab via chess_model_run_git.ipynb)
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play --config my_experiment.yaml
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

### Pretrain Pipeline

`pretrain/` provides supervised pretraining on position-evaluation data before GRPO. Configured via the `pretrain` section in YAML config.

### Evaluation

`evaluator.py` benchmarks against Stockfish at configurable skill levels. Key WandB metrics: `eval_stockfish/score` (win rate), `eval_stockfish/elo_diff`.

### WandB MCP Servers

Two MCP server instances: `wandb-pretrain` (chess-grpo-pretrain project) and `wandb-grpo` (Chess-GRPO-Bot project). Server code in `mcp_wandb_server/`.

## Research Workflow

Research documents in `research_docs/` follow a structured template with YAML frontmatter. Use `/research-insights` for analysis, `/code-implementation` for implementing recommendations, and `/code-auditor` to check against `research_docs/KNOWN_NOT_IMPLEMENTED_CHANGES.md`.

## Key Files

| Task | Files |
|------|-------|
| Training loop | `grpo_logic/model.py` |
| Loss computation | `grpo_logic/loss.py` |
| Trajectory sampling | `grpo_logic/sampling.py` |
| Reward computation | `chess/rewards.py` |
| Dataset generation | `chess/boards_dataset.py` |
| Model architecture | `models.py` |
| Evaluation | `evaluator.py` |
| Config | `configs/default.yaml`, `configs/config_loader.py` |

All paths relative to `src/`.
