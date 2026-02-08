# GRPO Self-Play Chess Module

An experimental, research-grade implementation of **Group Relative Policy Optimization (GRPO)** for training transformer-based chess policies through self-play. This module implements a full reinforcement learning pipeline for chess, but training stability and final strength are still under active investigation.

## Overview

This module trains neural network chess policies using GRPO, a variant of Proximal Policy Optimization (PPO) that uses group-based advantage estimation. The system learns to play chess by:

1. **Self-Play**: Sampling multiple trajectory groups from diverse starting positions
2. **Reward Computation**: Using Stockfish evaluations to compute dense rewards
3. **Policy Optimization**: Applying GRPO with PPO clipping and KL divergence penalties
4. **Evaluation**: Comprehensive benchmarking against Stockfish at multiple skill levels

## Key Features

### 🎯 Core Capabilities

- **Transformer-Based Policy Network**: Deep neural network architecture that processes FEN-encoded board states
- **GRPO Training Algorithm**: Group-relative advantage estimation with PPO-style clipping
- **Self-Play Training Loop**: Infinite dataset of diverse chess positions for robust learning
- **Stockfish Integration**: Professional-grade evaluation and reward computation
- **Comprehensive Evaluation**: Multi-level skill ladder evaluation against Stockfish
- **PyTorch Lightning Integration**: Scalable training with automatic mixed precision, gradient clipping, and checkpointing
- **Weights & Biases Logging**: Full experiment tracking and visualization

### 🏗️ Architecture Highlights

- **Modular Design**: Clean separation between model, training logic, chess rules, and evaluation
- **Efficient Batching**: Parallel trajectory sampling across multiple board positions
- **Legal Move Masking**: Proper handling of chess rules with action space masking
- **Trajectory Search**: Optional trajectory search wrapper for improved play strength
- **Resource Management**: Efficient Stockfish engine pooling and caching

## Installation

```bash
# Install dependencies
pip install torch pytorch-lightning wandb chess python-chess

# Ensure Stockfish is available
# On Ubuntu/Debian: sudo apt-get install stockfish
# On macOS: brew install stockfish
# Or download from: https://stockfishchess.org/download/
```

## Quick Start

### Basic Training

The easiest way to start training is using the YAML-based configuration system:

```python
from src.train_self_play import train

# Use default configuration (loads from configs/default.yaml)
train()

# Use a custom config file
train(config_path="my_experiment.yaml")

# Override specific hyperparameters programmatically
train(
    config_path="default.yaml",
    overrides={
        "grpo": {"lr": 1e-4, "num_trajectories": 8},
        "training": {"num_epochs": 100},
    }
)
```

All hyperparameters (learning rate, model architecture, training settings, etc.) are defined in YAML configuration files. See the [Configuration](#configuration) section below for details.

### Running Training in Google Colab

**Note for AI agents and contributors**: The primary way this code is run is through the `chess_model_run_git.ipynb` notebook in Google Colab. This notebook is the actual workflow used for training and evaluation.

The `chess_model_run_git.ipynb` notebook provides:

- **Automated Setup**: Clones the repository, installs dependencies, and downloads the searchless chess model
- **Complete Configuration**: Pre-configured settings for GRPO training, dataset generation, and evaluation
- **Phase-Aware Dataset**: Example configuration using `ChessDatasetConfig` with `phase_distribution` for balanced training across opening, middlegame, and endgame positions
- **Evaluation Pipeline**: Integrated evaluation against Stockfish at multiple skill levels

The notebook handles all setup steps including:
1. Repository cloning and branch checkout
2. Dependency installation (PyTorch Lightning, WandB, python-chess, etc.)
3. Downloading the searchless chess model from HuggingFace
4. Stockfish installation
5. Training configuration with phase-distributed dataset sampling
6. Model training and periodic evaluation

### Evaluation

```python
from src import Evaluator, EvalConfig
from src.chess.stockfish import StockfishConfig

# Create evaluator
evaluator = Evaluator(
    eval_cfg=EvalConfig(games=50),
    stockfish_cfg=StockfishConfig(skill_level=10, movetime_ms=100)
)

# Single evaluation
results, policy = evaluator.single_evaluation(model)
print(f"Win rate: {results['score']:.2%}")
print(f"Approx Elo diff: {results['elo_diff_vs_stockfish_approx']:.0f}")

# Skill ladder evaluation
skill_results = evaluator.eval_ladder(model)
for skill, score in skill_results.items():
    print(f"Skill {skill}: {score:.2%} win rate")
```

## Architecture

### Model Architecture

The `ChessTransformer` processes chess positions using:

- **Input Encoding**: FEN strings tokenized using DeepMind's chess tokenizer
- **Transformer Encoder**: Multi-head self-attention with learnable positional encodings
- **Policy Head**: Dense layers outputting logits over 1968 possible moves
- **Legal Move Masking**: Automatic filtering of illegal moves during inference

### GRPO Algorithm

Group Relative Policy Optimization extends PPO by:

1. **Group-Based Sampling**: Sample G trajectories per starting position
2. **Group Rewards**: Compute final reward for each trajectory group
3. **Relative Advantages**: Normalize advantages within each batch using group statistics
4. **PPO Clipping**: Prevent large policy updates with clipped importance ratios
5. **KL Penalty**: Regularize policy updates to prevent divergence

The loss function combines:
- **PPO Surrogate Loss**: `L_clip = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]`
- **KL Divergence Penalty**: `β * KL(π_old || π_new)`

### Training Pipeline

```
1. Sample random starting positions (FEN strings)
2. For each position:
   - Sample G trajectory groups using old policy
   - Compute group rewards using Stockfish evaluation
3. Compute advantages via group normalization
4. Update policy using GRPO loss
5. Sync old policy every epoch
6. Periodic evaluation against Stockfish
```

## Module Structure

```
src/
├── models.py              # ChessTransformer architecture
├── trainer.py             # PyTorch Lightning trainer setup
├── train_self_play.py     # Main training script
├── evaluator.py           # Evaluation framework
├── eval_utils.py          # Evaluation utilities
├── constants.py           # Configuration constants
├── grpo_logic/
│   ├── model.py           # GRPOChessTransformer (Lightning module)
│   ├── loss.py            # GRPO loss computation
│   └── sampling.py        # Trajectory sampling logic
└── chess/
    ├── chess_logic.py     # Board encoding, legal moves
    ├── policy_player.py   # Policy-based player
    ├── searcher.py        # Trajectory search wrapper
    ├── rewards.py         # Stockfish reward computation
    └── stockfish.py       # Stockfish engine integration
```

## Key Design Decisions

### 1. Group-Based Advantage Estimation

Instead of using value functions or Monte Carlo returns, GRPO computes advantages by normalizing rewards within trajectory groups. This approach:
- Eliminates the need for value function approximation
- Provides stable learning signals through relative comparisons
- Reduces variance in advantage estimates

### 2. Stockfish-Based Rewards

Using Stockfish for reward computation provides:
- **Dense Rewards**: Evaluation at every position, not just terminal states
- **High-Quality Signals**: Professional-grade position evaluation
- **Caching**: LRU cache for efficient reward computation during training

### 3. Legal Move Masking

The action space (1968 moves) is larger than legal moves in any position. The system:
- Masks illegal moves with `-inf` in logits
- Ensures policy only samples legal moves
- Handles edge cases (no legal moves, promotion moves)

### 4. Trajectory Padding and Masking

Trajectories have variable lengths due to game terminations. The implementation:
- Pads trajectories to fixed length for batching
- Uses attention masks to ignore padding
- Only considers moves from the starting player's perspective

## Configuration

This module uses a **YAML-based configuration system** to manage all hyperparameters and experiment settings. All training hyperparameters, model architecture settings, and evaluation configurations are centralized in YAML files located in `configs/`.

### Configuration Files

The default configuration file is `configs/default.yaml`, which contains all hyperparameters organized into sections:

- **`training`**: Training loop settings (epochs, batch size, steps per epoch)
- **`grpo`**: GRPO algorithm hyperparameters (learning rate, trajectories, clipping, KL penalty, entropy regularization, adaptive KL control)
- **`transformer`**: Model architecture (embedding dimension, layers, attention heads, vocabulary size, action space)
- **`eval`**: Evaluation settings (number of games, max plies, opening randomization)
- **`stockfish`**: Stockfish engine configuration (path, skill level, time limits, resource usage)
- **`policy`**: Policy player settings (temperature, greedy mode, branching factor, search depth)
- **`searcher`**: Optional trajectory search configuration
- **`dataset`**: Dataset generation settings (position phases, quality filters, evaluation bounds)

### Using Configurations

#### Loading Configurations

```python
from src.configs.config_loader import load_experiment_config

# Load default config
config = load_experiment_config("default.yaml")

# Load with overrides
config = load_experiment_config("default.yaml", overrides={
    "grpo": {"lr": 1e-4, "entropy_coef": 0.2},
    "training": {"num_epochs": 100},
})

# Access config values
print(config.grpo.lr)
print(config.training.batch_size)
print(config.transformer.embed_dim)
```

#### Training with Configurations

```python
from src.train_self_play import train

# Use default config
train()

# Use custom config file
train(config_path="my_experiment.yaml")

# Override specific values
train(
    config_path="default.yaml",
    overrides={
        "grpo": {"lr": 1e-4},
        "training": {"num_epochs": 50},
    },
    dataloader_kwargs={"num_workers": 4}  # Override DataLoader args
)
```

### Creating Custom Configurations

1. Copy the default config:
   ```bash
   cp configs/default.yaml configs/my_experiment.yaml
   ```

2. Edit `my_experiment.yaml` to modify hyperparameters

3. Use your custom config:
   ```python
   train(config_path="my_experiment.yaml")
   ```

### Configuration Dataclasses

The configuration system converts YAML files into typed dataclasses:

- **`TrainingConfig`**: Training loop settings
- **`GRPOConfig`**: GRPO algorithm hyperparameters
- **`ChessTransformerConfig`**: Model architecture
- **`EvalConfig`**: Evaluation settings
- **`StockfishConfig`**: Stockfish engine settings
- **`PolicyConfig`**: Policy player settings
- **`SearchConfig`**: Trajectory search settings (optional)
- **`ChessDatasetConfig`**: Dataset generation settings

All configs are combined into an `ExperimentConfig` object that provides type-safe access to all settings.

### Key Hyperparameters

All hyperparameters are defined in YAML files. Key settings include:

**GRPO Algorithm:**
- `grpo.lr`: Learning rate for policy optimization
- `grpo.num_trajectories`: Number of trajectory groups per starting position
- `grpo.trajectory_depth`: Maximum moves per trajectory
- `grpo.clip_ratio`: PPO clipping epsilon (prevents large policy updates)
- `grpo.kl_coef`: KL divergence penalty coefficient
- `grpo.entropy_coef`: Entropy regularization coefficient
- `grpo.adaptive_kl`: Enable adaptive KL coefficient adjustment
- `grpo.use_entropy_floor`: Monitor and respond to entropy collapse

**Model Architecture:**
- `transformer.embed_dim`: Transformer embedding dimension
- `transformer.num_layers`: Number of transformer layers
- `transformer.num_heads`: Number of attention heads
- `transformer.vocab_size`: Token vocabulary size
- `transformer.action_dim`: Action space size (1968 for chess)

**Training:**
- `training.num_epochs`: Total number of training epochs
- `training.batch_size`: Batch size for training
- `training.steps_per_epoch`: Number of training steps per epoch

See `configs/default.yaml` for the complete list of all hyperparameters and their default values.

## Advanced Usage

### Custom Reward Function

```python
from src.chess.rewards import reward_board

def custom_reward(board, start_board):
    # Your custom reward logic
    return reward_board(board, start_board, depth=8, movetime_ms=50)
```

### Trajectory Search

```python
from src.chess.searcher import TrajectorySearcher, SearchConfig
from src.chess.policy_player import PolicyPlayer

policy = PolicyPlayer(model)
searcher = TrajectorySearcher(
    policy,
    cfg=SearchConfig(n_trajectories=10, trajectory_depth=3)
)
```

### Custom Training Loop

```python
import pytorch_lightning as pl
from src.grpo_logic.model import GRPOChessTransformer

model = GRPOChessTransformer(transformer_config, grpo_config)
trainer = pl.Trainer(
    max_epochs=1000,
    gradient_clip_val=1.0,
    accelerator="gpu",
    devices=1
)
trainer.fit(model, dataloader)
```

## Performance Considerations

- **Batch Size**: Larger batches improve advantage normalization quality
- **Trajectory Depth**: Deeper trajectories provide more learning signal but increase compute
- **Stockfish Depth**: Higher depth = better rewards but slower training
- **Caching**: Reward caching significantly speeds up training
- **Gradient Clipping**: Prevents exploding gradients in transformer training

## Monitoring and Logging

The module logs comprehensive metrics to Weights & Biases:

- **Training Metrics**: Loss, KL divergence, policy ratios, reward statistics
- **Evaluation Metrics**: Win rate, Elo difference, game outcomes
- **System Metrics**: Trajectory lengths, padding fractions, gradient norms

## Research Background

GRPO (Group Relative Policy Optimization) is inspired by:
- **PPO (Proximal Policy Optimization)**: Clipped surrogate objective
- **REINFORCE**: Policy gradient methods
- **Self-Play**: Learning through playing against oneself
- **AlphaZero**: Combining deep learning with game tree search

This implementation adapts these ideas specifically for chess, using Stockfish for reward signals and evaluation.

## Technical Highlights

- ✅ **Practical Infrastructure**: Error handling, resource management, logging
- ✅ **Scalable Design**: Efficient batching, parallel trajectory sampling
- ✅ **Extensible**: Modular design allows easy customization
- ✅ **Documented**: Type hints, docstrings, clear structure
- ⚠️ **Status**: This is a research system, not a production-ready chess engine

## Future Enhancements

Potential improvements:
- Value function approximation for better advantage estimates
- More robust entropy and KL control for GRPO
- Multi-GPU training support
- Distributed self-play
- Opening book integration
- Endgame tablebase integration

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{grpo_chess,
  title = {GRPO Self-Play Chess Module},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/grpo_chess}
}
```

## Contact

For questions or contributions, please open an issue or contact [your email].

