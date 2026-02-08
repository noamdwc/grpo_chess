# GRPO Chess

Training chess-playing transformers using **Group Relative Policy Optimization (GRPO)** through self-play.

## Overview

This project implements a reinforcement learning pipeline for training neural network chess policies. The system learns to play chess by:

1. **Self-Play**: Sampling multiple trajectory groups from diverse starting positions
2. **Reward Computation**: Using Stockfish evaluations to compute rewards
3. **Policy Optimization**: Applying GRPO with PPO clipping and KL divergence penalties
4. **Evaluation**: Benchmarking against Stockfish at multiple skill levels

## Quick Start

### Running in Google Colab (Primary Method)

The main way to run this code is through the `chess_model_run_git.ipynb` notebook in Google Colab:

1. Open `chess_model_run_git.ipynb` in Google Colab
2. The notebook handles:
   - Repository cloning and setup
   - Dependency installation
   - Stockfish installation
   - Model training and evaluation

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

## Project Structure

```
grpo_chess/
├── chess_model_run_git.ipynb    # Main training notebook (Colab)
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
│   ├── configs/                 # YAML configuration files
│   └── pretrain/                # Supervised pretraining pipeline
├── searchless_chess/            # DeepMind submodule (tokenizer, move tables)
├── research_docs/               # Agent research documentation
└── tests/                       # Test suite
```

## Documentation

### For Detailed Module Documentation
See [src/README.md](src/README.md) for comprehensive documentation of the GRPO implementation.

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

## Experiment Tracking

Training runs are tracked with [Weights & Biases](https://wandb.ai). Key metrics:
- `train/reward_mean`: Average trajectory reward
- `eval_stockfish/score`: Win rate vs Stockfish
- `eval_stockfish/elo_diff`: Approximate Elo difference
- `train/clip_fraction`: PPO clipping statistics
- `train/kl_divergence`: Policy divergence

## Contributing

1. Check existing [research_docs/](research_docs/) for context on current issues
2. Use the template at `research_docs/TEMPLATE.md` for documenting findings
3. Reference specific code with file paths and line numbers
4. Include WandB run IDs for metric references

## Acknowledgments

- Based on DeepMind's [Searchless Chess](https://github.com/google-deepmind/searchless_chess) project
- Uses Stockfish for evaluation and reward computation
