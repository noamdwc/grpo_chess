# Repository Guidelines

## Project Invariant
- This is a **searchless chess** project.
- Do not propose or implement MCTS, tree-search, or related search-based chess approaches.

## Project Structure & Module Organization
- `src/`: core code for GRPO self-play training, evaluation, distillation, and pretraining.
- `src/grpo_logic/`: GRPO sampling, loss, and Lightning module internals.
- `src/chess/`: chess-specific logic, Stockfish integration, rewards, and datasets.
- `src/configs/`: YAML experiment configs (`default.yaml`, `distill.yaml`, `pretrain.yaml`).
- `tests/`: pytest suite (unit + integration-style tests), plus fixtures in `tests/resources/`.
- `chess_model_run_git.ipynb`: primary Colab workflow for end-to-end runs.
- `research_docs/`: experiment analyses and debugging notes. Subdirs: `experiments/` (plan docs from `/plan-experiment`), `runs/` (run reports from `/run-experiment`).

## Build, Test, and Development Commands
Use the project Python environment (`~/miniconda3/envs/grpo_chess/bin/python`) for consistency.

- `~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/`
Runs the full test suite.
- `~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_pretrain_pipeline.py -v`
Runs a targeted test module.
- `~/miniconda3/envs/grpo_chess/bin/python -m src.train_self_play`
Starts GRPO training with default config.
- `~/miniconda3/envs/grpo_chess/bin/python -m src.distill.distill --config distill.yaml`
Runs distillation training.
- `~/miniconda3/envs/grpo_chess/bin/python -m src.pretrain.pretrain --config pretrain.yaml`
Runs supervised pretraining.

## Codex Working Style
- Prioritize implementation over exploration: if a request is clear, start coding after minimal file/context checks.
- Search the repository before external lookup; prefer existing scripts/utilities over creating new ones.
- Keep changes minimal and in-scope; avoid unrelated cleanup or extra abstractions.
- Before major edits, confirm execution context: active Python env, target config/module, and relevant data/project paths.
- Use reasonable assumptions and proceed; ask follow-up questions only when blocked by a true ambiguity.
- Validate changes with focused checks (`pytest`, `py_compile`, or targeted tests) before handoff when feasible.
- For larger tasks, follow a short plan-then-execute flow with concrete file targets and incremental verification.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, type hints where practical, small focused functions.
- Naming: `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep training behavior configurable in YAML; prefer updating `src/configs/*.yaml` over hardcoding.
- Follow existing module boundaries (e.g., keep GRPO logic in `src/grpo_logic/`, chess engine code in `src/chess/`).

## Testing Guidelines
- Framework: `pytest`.
- Test files follow `tests/test_*.py`; test functions should be descriptive (e.g., `test_chess_dataset_multi_epoch_multi_worker`).
- For engine-dependent tests, ensure Stockfish is installed (`brew install stockfish` on macOS).
- Add/update tests for behavior changes in training loops, config loading, dataset iteration, or reward logic.

## Commit & Pull Request Guidelines
- Commit messages in history are short imperative summaries (examples: `Fix ...`, `Add ...`, `refactor ...`). Keep that style.
- Keep commits scoped to one logical change and include config updates when relevant.
- PRs should include: purpose, key code paths touched, test evidence (command + result), and linked issue/run context when applicable.
- For training/eval changes, include impacted metrics (for example from Weights & Biases) and any required environment assumptions.
