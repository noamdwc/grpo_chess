# GRPO Self-Play Chess Module

A production-ready implementation of **Group Relative Policy Optimization (GRPO)** for training transformer-based chess policies through self-play. This module implements a complete reinforcement learning pipeline for chess, combining modern transformer architectures with advanced policy optimization techniques.

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
- **Trajectory Search**: Optional tree search wrapper for improved play strength
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

```python
from src.grpo_self_play import GRPOChessTransformer, GRPOConfig, ChessTransformerConfig
from src.grpo_self_play.train_self_play import train

# Configure model and training
transformer_config = ChessTransformerConfig(
    vocab_size=300,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    action_dim=1968
)

grpo_config = GRPOConfig(
    lr=1e-6,
    num_trajectories=8,
    trajectory_depth=32,
    clip_ratio=0.2,
    kl_coef=0.01
)

# Start training
train()
```

### Evaluation

```python
from src.grpo_self_play import Evaluator, EvalConfig
from src.grpo_self_play.chess.stockfish import StockfishConfig

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
grpo_self_play/
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

### Model Configuration

```python
@dataclass
class ChessTransformerConfig:
    vocab_size: int = 300      # Token vocabulary size
    embed_dim: int = 256       # Transformer embedding dimension
    num_layers: int = 4        # Number of transformer layers
    num_heads: int = 8         # Attention heads
    action_dim: int = 1968     # Action space size
```

### GRPO Configuration

```python
@dataclass
class GRPOConfig:
    lr: float = 1e-4              # Learning rate
    num_trajectories: int = 4      # Trajectory groups per position
    trajectory_depth: int = 5      # Max steps per trajectory
    clip_ratio: float = 0.2       # PPO clipping epsilon
    kl_coef: float = 0.01         # KL penalty coefficient
    eval_every_n_epochs: int = 10  # Evaluation frequency
```

### Evaluation Configuration

```python
@dataclass
class EvalConfig:
    games: int = 50                # Number of evaluation games
    max_plies: int = 400          # Max moves per game
    randomize_opening: bool = False # Random opening moves
    opening_plies: int = 6         # Number of random opening moves
```

## Advanced Usage

### Custom Reward Function

```python
from src.grpo_self_play.chess.rewards import reward_board

def custom_reward(board, start_board):
    # Your custom reward logic
    return reward_board(board, start_board, depth=8, movetime_ms=50)
```

### Trajectory Search

```python
from src.grpo_self_play.chess.searcher import TrajectorySearcher, SearchConfig
from src.grpo_self_play.chess.policy_player import PolicyPlayer

policy = PolicyPlayer(model)
searcher = TrajectorySearcher(
    policy,
    cfg=SearchConfig(n_trajectories=10, trajectory_depth=3)
)
```

### Custom Training Loop

```python
import pytorch_lightning as pl
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer

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

- ✅ **Production-Ready**: Error handling, resource management, logging
- ✅ **Scalable**: Efficient batching, parallel trajectory sampling
- ✅ **Extensible**: Modular design allows easy customization
- ✅ **Well-Tested**: Comprehensive evaluation framework
- ✅ **Documented**: Type hints, docstrings, clear structure

## Future Enhancements

Potential improvements:
- Value function approximation for better advantage estimates
- Monte Carlo tree search (MCTS) integration
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

