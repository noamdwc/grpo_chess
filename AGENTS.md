# AGENTS.md - GRPO Chess Project Instructions

This file provides instructions for AI coding agents working on this repository.
Compatible with: Claude Code, GitHub Copilot, Cursor, OpenAI Codex, and other tools supporting the AGENTS.md standard.

## Project Overview

**GRPO Chess** trains a chess-playing transformer using Group Relative Policy Optimization (GRPO).

**Key constraint**: This is a **searchless chess** project. Do not suggest MCTS, tree search, or similar solutions.

## Repository Structure

```
grpo_chess/
├── chess_model_run_git.ipynb    # Main training notebook (Google Colab)
├── src/grpo_self_play/          # Core GRPO implementation
├── research_docs/               # Agent research documentation
│   ├── TEMPLATE.md              # Template for new documents
│   └── *.md                     # Research insights (dated)
└── .claude/agents/              # Specialized agent prompts
    ├── research-insights.md     # For analysis tasks
    └── code-implementation.md   # For coding tasks
```

## Commands

```bash
# Run tests (if available)
python -m pytest

# Check syntax
python -m py_compile src/grpo_self_play/grpo_logic/model.py

# Training is done via Google Colab notebook
# See chess_model_run_git.ipynb
```

## Code Style

- Python with type hints where they exist
- snake_case for functions and variables
- PascalCase for classes
- Dataclasses for configuration
- PyTorch Lightning for training modules
- Follow existing patterns in the codebase

## Key Files for Common Tasks

| Task | Primary Files |
|------|--------------|
| Training loop | `src/grpo_self_play/grpo_logic/model.py` |
| Loss computation | `src/grpo_self_play/grpo_logic/loss.py` |
| Trajectory sampling | `src/grpo_self_play/grpo_logic/sampling.py` |
| Reward computation | `src/grpo_self_play/chess/rewards.py` |
| Dataset generation | `src/grpo_self_play/chess/boards_dataset.py` |
| Model architecture | `src/grpo_self_play/models.py` |
| Evaluation | `src/grpo_self_play/evaluator.py` |

## Git Workflow

- Main branch: `main`
- Development branch: `improve_data_quality` (current)
- Commit messages: Clear, concise descriptions
- Never commit secrets or API keys

## Boundaries

### DO NOT
- Suggest MCTS or tree search solutions
- Modify files outside `src/` without explicit request
- Commit to main without approval
- Change code style in lines you're not modifying
- Over-engineer solutions

### DO
- Read files before editing
- Make minimal, targeted changes
- Follow existing code patterns
- Include file paths with line numbers when discussing code
- Use research_docs/ for analysis documentation

## Experiment Tracking

Training is tracked with Weights & Biases. Key metrics:
- `train/avg_reward` - Mean trajectory reward
- `eval_stockfish/score` - Win rate vs Stockfish
- `mean_clip_fraction` - PPO clipping statistics

## Specialized Agents

For complex tasks, see specialized prompts in `.claude/agents/`:
- **research-insights.md** - Analyzing training issues, writing research docs
- **code-implementation.md** - Implementing specific code changes

## Documentation

- Research findings go in `research_docs/` using the template
- Code documentation follows existing docstring patterns
- README updates only when explicitly requested
