# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **This is a searchless chess project.** Do not suggest MCTS, tree search, or similar solutions.
- **Always use the miniconda environment** for any Python command: `~/miniconda3/envs/grpo_chess/bin/python`
- **Edit `default.yaml` for config changes** — never create new config files or hardcode values in Python. Ask which config file if unclear.
- **Keep implementations simple and minimal.** No unnecessary abstractions, utility classes, or wrapper layers. If it can be done in 10 lines, don't write 50.
- **Surgical changes.** Touch only what the request requires. Match existing style. Clean up only the orphans your changes created — flag pre-existing dead code, don't delete it.
- **Goal-driven execution.** Restate the task as a verifiable success criterion (failing test to pass, metric to hit) before coding. For multi-step work, list steps each with a verify-check, then loop until each passes.
- **Investigate before concluding.** State assumptions explicitly; surface tradeoffs; ask when unclear rather than silently picking one interpretation. When analyzing ML experiment data, pull and show the actual metrics over time before drawing conclusions.
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

# Reasoning-GRPO training (primarily done in Google Colab via chess_model_run_git.ipynb)
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play
~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play --config my_experiment.yaml

# Distillation training
~/miniconda3/envs/grpo_chess/bin/python -m src.distill.distill --config distill.yaml
```

Tests require Stockfish installed locally (`brew install stockfish` on macOS). Some tests use subprocess invocation to mirror the Colab environment.

## Architecture

### Training Pipeline Flow

1. **`train_self_play.py`** — **Reasoning-GRPO** entry point. Loads YAML via `config_loader.py`, builds the reasoning model + Lightning trainer, wires `StockfishEvalCallback`.
2. **`grpo_logic/model.py`** (Lightning module) — Orchestrates the GRPO loop in `training_step`: samples trajectories with old policy, computes Stockfish rewards, group-normalized advantages, PPO updates. Syncs old policy each epoch. `grpo_logic/mode_utils.py` holds mode helpers.
3. **`reasoning/sampler.py`** (`rollout_batch`) — Batched think→move trajectory sampling. Token ids in `reasoning/tokens.py` (`THINK_ID`, `END_THINK_ID`, `MOVE_ID`, `BASE_VOCAB_SIZE`, `FEN_LEN`). `reasoning/attention_mask.py` builds the causal+segment mask over think/move tokens.
4. **`grpo_logic/loss.py` / `reasoning_loss.py`** — PPO surrogate + KL penalty; reasoning variant applies log-prob masking over think vs. move tokens. `train/loss` = PPO + kl_coef * KL. Components logged raw.
5. **`chess/rewards.py`** — Dense rewards via Stockfish. `chess/dm_leaf_oracle.py` provides DM-teacher leaf evals; `chess/chess_logic.py` holds shared board utilities.
6. **`chess/boards_dataset.py`** — Starting positions with phase distribution (opening/middlegame/endgame) and quality filtering.

### Reasoning stack

`src/reasoning/` — the think-then-move model used by GRPO. `model.py` wraps the DM-ported transformer with think/move heads; `sampler.py` drives rollouts; `tokens.py` defines the extended vocab; `attention_mask.py` the segment mask; `real_play.py` the eval adapter used by `StockfishEvalCallback`.

### DeepMind port

`src/dm_port/` — PyTorch port of the DeepMind searchless-chess 136M transformer (`transformer.py`), with `convert_jax.py` turning JAX checkpoints into PyTorch state dicts (layout notes in `convert_jax_layout.md`). Serves as the base architecture for the reasoning stack.

### Legacy model

`models.py` (`ChessTransformer`) — older transformer outputting logits over 1968 moves with legal-move masking. Still used by distillation/pretrain; the reasoning stack has superseded it for GRPO.

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

### Colab experiment ladder

`colab_experiment_ladder.py` drives multi-stage Colab GRPO runs (probe → bracket → main), gating stage transitions on WandB windows. Contract tests in `tests/test_colab_experiment_ladder.py` and `tests/test_experiment_ladder_notebook_contract.py`.

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
| Reasoning-GRPO entry | `train_self_play.py` |
| GRPO loop / loss | `grpo_logic/model.py`, `grpo_logic/loss.py`, `grpo_logic/reasoning_loss.py`, `grpo_logic/mode_utils.py` |
| Reasoning stack | `reasoning/{model,sampler,tokens,attention_mask,real_play}.py` |
| DeepMind port | `dm_port/{transformer,convert_jax}.py` |
| Rewards / chess logic | `chess/rewards.py`, `chess/dm_leaf_oracle.py`, `chess/chess_logic.py` |
| Dataset generation | `chess/boards_dataset.py` |
| Legacy model | `models.py` |
| Evaluation | `evaluator.py`, `eval_utils.py` |
| Stockfish / policy | `chess/stockfish.py`, `chess/policy_player.py` |
| Distillation | `distill/{distill,distill_dataset,teacher}.py` |
| Trainer / utils | `trainer.py`, `checkpoint_compat.py`, `logging_utils.py`, `parameter_counter.py`, `constants.py` |
| Colab ladder | `colab_experiment_ladder.py` |
| Config | `configs/{default,grpo_colab_main,distill,pretrain}.yaml`, `configs/config_loader.py` |

All paths relative to `src/`.

## Troubleshooting

- **JAX ↔ PyTorch DM-port parity (local only):** run `scripts/compare_jax_pytorch_port.py` — see `docs/parity_check.md`.
- **Reasoning-GRPO failure modes:** see `docs/reasoning_grpo/failure_modes.md`.
  When you add a safety check, a metric, or a handling path for an observed
  failure, also write it up in that file with symptom / how you'll notice /
  recovery / history. Leave an inline comment at the code site that says the
  same thing in its own words.
