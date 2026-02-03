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
│   ├── grpo_self_play/          # Core GRPO implementation
│   │   ├── models.py            # Chess transformer architecture
│   │   ├── trainer.py           # PyTorch Lightning trainer
│   │   ├── evaluator.py         # Stockfish evaluation
│   │   ├── grpo_logic/          # GRPO algorithm
│   │   │   ├── model.py         # Lightning module
│   │   │   ├── loss.py          # GRPO/PPO loss functions
│   │   │   └── sampling.py      # Trajectory sampling
│   │   └── chess/               # Chess utilities
│   │       ├── rewards.py       # Stockfish reward computation
│   │       ├── boards_dataset.py # Position generation
│   │       └── stockfish.py     # Engine integration
│   └── searchless_chess_model/  # DeepMind model (downloaded)
├── research_docs/               # Agent research documentation
│   ├── README.md                # Documentation guide
│   ├── TEMPLATE.md              # Template for new documents
│   └── *.md                     # Research insights
└── plan_searchless_chess_distillation.md
```

## Documentation

### For Detailed Module Documentation
See [src/grpo_self_play/README.md](src/grpo_self_play/README.md) for comprehensive documentation of the GRPO implementation.

### For Research Insights (AI Agents & Humans)
See [research_docs/](research_docs/) for structured analysis documents, debugging insights, and research findings.

The `research_docs/` directory is designed for:
- **AI agents** to document findings in a structured, reproducible format
- **Human researchers** to review and build upon agent analyses
- **Future contributors** to understand project history and decisions

## Configuration System

This project uses a **YAML-based configuration system** to manage all hyperparameters and experiment settings. All training hyperparameters, model architecture settings, and evaluation configurations are defined in YAML files.

### Configuration Files

Configuration files are located in `src/grpo_self_play/configs/`. The default configuration is `default.yaml`, which contains all hyperparameters for a training run:

- **Training hyperparameters**: Learning rate, batch size, number of epochs, steps per epoch
- **GRPO algorithm settings**: Trajectory count, depth, PPO clipping ratio, KL coefficient, entropy regularization
- **Model architecture**: Transformer dimensions, layers, attention heads, vocabulary size
- **Evaluation settings**: Number of games, Stockfish skill levels, evaluation frequency
- **Dataset configuration**: Position sampling, phase distribution, quality filters
- **Stockfish settings**: Engine path, skill level, time limits, resource usage

### Using Configurations

#### Basic Usage

```python
from src.grpo_self_play.train_self_play import train

# Use default configuration
train()

# Use a custom config file
train(config_path="my_experiment.yaml")
```

#### Programmatic Overrides

You can override specific config values programmatically:

```python
from src.grpo_self_play.train_self_play import train

# Override specific hyperparameters
train(
    config_path="default.yaml",
    overrides={
        "grpo": {"lr": 1e-4, "entropy_coef": 0.2},
        "training": {"num_epochs": 100},
        "stockfish": {"skill_level": 5},
    }
)
```

#### Command-Line Usage

```bash
# Use default config
python -m src.grpo_self_play.train_self_play

# Use custom config file
python -m src.grpo_self_play.train_self_play --config my_experiment.yaml
```

### Configuration Structure

The `default.yaml` file is organized into sections:

- **`training`**: Training loop settings (epochs, batch size, steps per epoch)
- **`grpo`**: GRPO algorithm hyperparameters (learning rate, trajectories, clipping, KL penalty, entropy)
- **`transformer`**: Model architecture (embedding dimension, layers, heads, vocabulary)
- **`eval`**: Evaluation settings (number of games, max plies, opening randomization)
- **`stockfish`**: Stockfish engine configuration (path, skill level, time limits)
- **`policy`**: Policy player settings (temperature, greedy mode, search depth)
- **`searcher`**: Optional trajectory search configuration
- **`dataset`**: Dataset generation settings (position phases, quality filters)

### Creating Custom Configurations

1. Copy `default.yaml` to create a new experiment config:
   ```bash
   cp src/grpo_self_play/configs/default.yaml src/grpo_self_play/configs/my_experiment.yaml
   ```

2. Modify the values you want to change in `my_experiment.yaml`

3. Run training with your custom config:
   ```python
   train(config_path="my_experiment.yaml")
   ```

### Configuration Loading Logic

The configuration system (`src/grpo_self_play/configs/config_loader.py`) works as follows:

1. **Load YAML file**: Reads the specified YAML configuration file
2. **Apply overrides**: Merges any programmatic overrides with the YAML values
3. **Convert to dataclasses**: Transforms YAML dictionaries into typed dataclass objects
4. **Validate**: Ensures all required fields are present and valid

The config loader provides type-safe access to all configuration values through the `ExperimentConfig` dataclass, which contains all sub-configs (training, grpo, transformer, eval, stockfish, policy, searcher, dataset).

### Key Hyperparameters

All hyperparameters are defined in the configuration files. Key settings include:

- **Learning rate** (`grpo.lr`): Controls optimization step size
- **Trajectory settings** (`grpo.num_trajectories`, `grpo.trajectory_depth`): Number and depth of self-play trajectories
- **PPO clipping** (`grpo.clip_ratio`): Prevents large policy updates
- **KL penalty** (`grpo.kl_coef`): Regularizes policy divergence
- **Entropy coefficient** (`grpo.entropy_coef`): Encourages exploration
- **Model size** (`transformer.embed_dim`, `transformer.num_layers`): Transformer architecture dimensions

See `src/grpo_self_play/configs/default.yaml` for the complete list of all hyperparameters and their default values.

## Experiment Tracking

Training runs are tracked with [Weights & Biases](https://wandb.ai). Key metrics:
- `train/avg_reward`: Average trajectory reward
- `eval_stockfish/score`: Win rate vs Stockfish
- `eval_stockfish/elo_diff`: Approximate Elo difference
- `mean_clip_fraction`: PPO clipping statistics
- `mean_kl_divergence`: Policy divergence

## Contributing

1. Check existing [research_docs/](research_docs/) for context on current issues
2. Use the template at `research_docs/TEMPLATE.md` for documenting findings
3. Reference specific code with file paths and line numbers
4. Include WandB run IDs for metric references

## License

[Specify license]

## Acknowledgments

- Based on DeepMind's [Searchless Chess](https://github.com/google-deepmind/searchless_chess) project
- Uses Stockfish for evaluation and reward computation
