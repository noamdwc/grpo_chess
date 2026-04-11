# GRPO Chess

Training chess-playing transformers using **Group Relative Policy Optimization (GRPO)** through self-play.

## Overview

This project implements a reinforcement learning pipeline for training neural network chess policies. The system learns to play chess by:

1. **Self-Play**: Sampling multiple trajectory groups from diverse starting positions
2. **Reward Computation**: Using Stockfish evaluations to compute rewards
3. **Policy Optimization**: Applying GRPO with PPO clipping and KL divergence penalties
4. **Evaluation**: Benchmarking against Stockfish at multiple skill levels

## Model Architecture

The policy model in `src/models.py` is an encoder-only transformer that maps tokenized board states to logits over the fixed chess action space (`action_dim=1968`).

1. **Input tokens**: `board_tokens` with shape `[B, T]` (padding token id `0` by default)
2. **Embeddings**: token embedding + learnable absolute positional embedding
3. **Transformer encoder**: configurable depth/width (`embed_dim`, `num_layers`, `num_heads`, `ffn_mult`, `activation`, `dropout`)
4. **Sequence readout** (`transformer.readout`):
   - `"mean"`: masked mean pooling over non-pad tokens
   - `"last"`: hidden state of the last non-pad token
   - `"cls"`: prepends a CLS token and reads position `0`
5. **Policy head**: MLP from readout vector to move logits
   - hidden size = `head_mult * embed_dim`
   - activation = `transformer.activation`
   - output shape = `[B, action_dim]`

Notes:
- Legal-move masking is applied outside the base forward pass during training/eval loss and action selection.
- The same core architecture is used across GRPO, pretraining, and distillation entry points, configured from YAML (`src/configs/*.yaml`).
- Warm-start compatibility: `transformer.readout: "cls"` can increase `embedding.weight` rows (default `cls_token_id=vocab_size`), so loading older non-CLS checkpoints will fail with an explicit compatibility error.

## Quick Start

### Running in Google Colab (Primary Method)

The main way to run this branch in Google Colab is through `chess_model_run_git.ipynb`.

This notebook handles:
- Google Drive mounting for persistent checkpoints and caches
- repository clone and dependency installation
- Drive-local checkpoint preparation
- reasoning-GRPO fine-tuning through `src.train_self_play`
- `smoke`, `full`, and checkpoint-based `resume` modes

`grpo_9m_colab.ipynb` and `full_pipeline_notebook.ipynb` are legacy notebooks kept only for reference.

### Local Development

```bash
# Clone the repository
git clone https://github.com/noamdwc/grpo_chess.git
cd grpo_chess
git submodule update --init

# Install dependencies
pip install torch pytorch-lightning wandb python-chess jaxtyping datasets huggingface_hub

# Install Stockfish
# Ubuntu/Debian: sudo apt-get install stockfish
# macOS: brew install stockfish
```

### Automatic File Metadata Normalization

Use this when you want files moved into the project folder to get fresh `mtime` and `atime`.

```bash
# Install dependencies (includes watchdog)
pip install -r requirements.txt

# Watch project folder recursively and normalize incoming files
~/miniconda3/envs/grpo_chess/bin/python scripts/watch_and_touch_metadata.py --root /Users/noamc/repos/grpo_chess

# Preview actions without modifying files
~/miniconda3/envs/grpo_chess/bin/python scripts/watch_and_touch_metadata.py --root /Users/noamc/repos/grpo_chess --dry-run

# One-shot mode for a single file (no watcher loop)
~/miniconda3/envs/grpo_chess/bin/python scripts/watch_and_touch_metadata.py --root /Users/noamc/repos/grpo_chess --once path/to/file.txt
```

## Project Structure

```
grpo_chess/
├── chess_model_run_git.ipynb    # Main reasoning-GRPO Colab notebook
├── grpo_9m_colab.ipynb          # Legacy notebook (historical reference)
├── full_pipeline_notebook.ipynb # Legacy notebook (historical reference)
├── src/
│   ├── models.py                # Chess transformer architecture
│   ├── trainer.py               # PyTorch Lightning trainer
│   ├── evaluator.py             # Stockfish evaluation
│   ├── train_self_play.py       # Training entry point
│   ├── grpo_logic/              # GRPO algorithm
│   │   ├── model.py             # Lightning module
│   │   ├── loss.py              # GRPO/PPO loss functions
│   │   └── sampling.py          # Trajectory sampling
│   ├── chess/                   # Chess utilities
│   │   ├── rewards.py           # Stockfish reward computation
│   │   ├── boards_dataset.py    # Position generation
│   │   └── stockfish.py         # Engine integration
│   ├── distill/                 # Knowledge distillation pipeline
│   ├── pretrain/                # Supervised pretraining pipeline
│   └── configs/                 # YAML configuration files
├── searchless_chess/            # DeepMind submodule (tokenizer, move tables)
├── research_docs/               # Research documentation
│   ├── experiments/             # Experiment plan docs (per /plan-experiment)
│   ├── runs/                    # Run reports (per /run-experiment)
│   └── TEMPLATE.md              # Research doc template
└── tests/                       # Test suite
```

## Documentation

### For Detailed Module Documentation
See [src/README.md](src/README.md) for comprehensive documentation of the GRPO implementation.

### For Lightning Cloud Job Operations
See [docs/lightning_cloud_jobs.md](docs/lightning_cloud_jobs.md) for setup, submission, monitoring, GPU usage, and troubleshooting on Lightning.ai.

### For Research Insights (AI Agents & Humans)
See [research_docs/](research_docs/) for structured analysis documents, debugging insights, and research findings.

## Configuration System

This project uses a **YAML-based configuration system** to manage all hyperparameters and experiment settings.

### Configuration Files

Configuration files are located in `src/configs/`. The default configuration is `default.yaml`, which contains all hyperparameters for a training run:

- **Training hyperparameters**: Learning rate, batch size, number of epochs, steps per epoch
- **GRPO algorithm settings**: Trajectory count, depth, PPO clipping ratio, KL coefficient
- **Model architecture**: Transformer dimensions, layers, attention heads, vocabulary size
- **Evaluation settings**: Number of games, Stockfish skill levels, evaluation frequency
- **Dataset configuration**: Position sampling, phase distribution, quality filters
- **Stockfish settings**: Engine path, skill level, time limits, resource usage

### Using Configurations

```python
from src.train_self_play import train

# Use default configuration
train()

# Override specific hyperparameters
train(
    config_path="default.yaml",
    overrides={
        "grpo": {"lr": 1e-4},
        "training": {"num_epochs": 100},
        "stockfish": {"skill_level": 5},
    }
)
```

#### Command-Line Usage

```bash
# Use default config
python -m src.train_self_play

# Use custom config file
python -m src.train_self_play --config my_experiment.yaml
```

### Key Hyperparameters

- **Learning rate** (`grpo.lr`): Controls optimization step size
- **Trajectory settings** (`grpo.num_trajectories`, `grpo.trajectory_depth`): Number and depth of self-play trajectories
- **PPO clipping** (`grpo.clip_ratio`): Prevents large policy updates
- **KL penalty** (`grpo.kl_coef`): Regularizes policy divergence
- **Model size** (`transformer.embed_dim`, `transformer.num_layers`): Transformer architecture dimensions

See `src/configs/default.yaml` for the complete list of all hyperparameters and their default values.

## Baselines

### DeepMind 136M Teacher (2026-02-24)

Evaluated against Stockfish skill level 2 (32 games, randomized openings):

| Model | Score | W / D / L | Elo diff |
|-------|-------|-----------|----------|
| DeepMind 136M teacher | **0.688** | 12 / 20 / 0 | **+137** |
| DeepMind 270M teacher | 0.656 | 10 / 22 / 0 | +112 |
| DeepMind 9M teacher | 0.625 | 8 / 24 / 0 | +89 |

These are the distillation **success thresholds**: a student model should match its teacher's score to demonstrate preserved playing strength. See [`research_docs/2026-02-24_deepmind-136m-teacher-baseline.md`](research_docs/2026-02-24_deepmind-136m-teacher-baseline.md) for full details.

To reproduce:
```bash
python -m src.distill.eval_teacher --config src/configs/distill.yaml
```

## Distillation v2 Update

The focused v2 distillation upgrade is captured in:
- `src/configs/distill_labelsafe_v2.yaml`

### Student model changes (`src/models.py`)

- Added configurable readout mode via `transformer.readout`:
  - `"mean"`: legacy masked mean pooling
  - `"last"`: last non-pad token readout (v2 default)
  - `"cls"`: prepends CLS token and reads position 0
  - Note: warm-starting CLS models from legacy non-CLS checkpoints is not shape-compatible unless `cls_token_id` keeps embedding size unchanged.
- Added configurable transformer FFN width/activation:
  - `transformer.ffn_mult` controls `dim_feedforward = ffn_mult * embed_dim`
  - `transformer.activation` supports `"relu"` and `"gelu"`
- Added configurable policy head width:
  - `transformer.head_mult` controls head hidden size `head_mult * embed_dim`
- Existing configs remain backward-compatible (legacy behavior if new fields are unset).

### Distillation config changes (`distill_labelsafe_v2.yaml`)

- Unified label regime:
  - `generate.top_k = deepmind_data.top_k = dataset.top_k = 32`
  - `generate.teacher_temperature = deepmind_data.temperature = 2.0`
- Disabled score shaping for ablation:
  - `generate.teacher_target_score_norm: "plain"`
- Rebalanced loss weights:
  - `distill_lambda_soft: 0.50`
  - `distill_lambda_hard: 0.50`
  - `distill_target_top1_mix_alpha: 0.05`
- Increased position diversity:
  - `deepmind_data.min_win_prob: 0.50`
- Strengthened quality gate:
  - `quality_gate_min_val_top1: 0.20`
  - `quality_gate_epoch: 3`

### Additional distillation diagnostics (`src/distill/distill.py`)

New logged metrics:
- `teacher_topk_mass`
- `teacher_entropy_mean`
- `teacher_top1_prob_mean`
- (Existing) `teacher_legal_fraction`, `teacher_valid_sample_fraction`

### Parameter counting utility

A reusable wrapper is available at `src/parameter_counter.py`:

```python
from src.parameter_counter import ParameterCounter
from src.models import ChessTransformer, ChessTransformerConfig

model = ChessTransformer(ChessTransformerConfig())
counter = ParameterCounter(model)
print(counter.summary())  # total/trainable/frozen
```

## Experiment Tracking

Training runs are tracked with [Weights & Biases](https://wandb.ai). Key metrics:
- `train/reward_mean`: Average trajectory reward
- `eval_stockfish/score`: Win rate vs Stockfish
- `eval_stockfish/elo_diff`: Approximate Elo difference
- `train/clip_fraction`: PPO clipping statistics
- `train/kl_divergence`: Policy divergence

## Experiment Workflow

The project uses a structured four-step experiment loop, driven by Claude Code skills:

```
/research-insights  →  /plan-experiment  →  /code-implementation  →  /run-experiment
        ↑                                                                      |
        └──────────────────────── /experiment-cycle ────────────────────────────┘
```

Each step produces a structured artifact:
- **Research**: `research_docs/YYYY-MM-DD_*.md`
- **Plan**: `research_docs/experiments/YYYY-MM-DD_<slug>.md` + `src/configs/<name>.yaml`
- **Run report**: `research_docs/runs/YYYY-MM-DD_<job>.md`

Use `/experiment-cycle` to orchestrate all steps automatically, or invoke each skill standalone.

## Contributing

1. Check existing [research_docs/](research_docs/) for context on current issues
2. Use `/research-insights` to analyze a question and produce a findings document
3. Use `/plan-experiment` to design an experiment and save a plan doc before writing any config
4. Reference specific code with file paths and line numbers
5. Include WandB run IDs for metric references

## Acknowledgments

- Based on DeepMind's [Searchless Chess](https://github.com/google-deepmind/searchless_chess) project
- Uses Stockfish for evaluation and reward computation
