# GRPO Self-Play Module

This module implements **Group Relative Policy Optimization (GRPO)** for training chess policies through self-play. It combines transformer-based neural networks with PPO (Proximal Policy Optimization) to learn chess strategies without requiring search algorithms.

## Overview

The module trains a chess policy network that:
- Takes FEN-encoded board states as input
- Outputs probability distributions over legal moves
- Learns through self-play using GRPO, which samples multiple trajectories per position and uses relative rewards within groups
- Evaluates against Stockfish at various skill levels

## Directory Structure

```
grpo_self_play/
├── __init__.py              # Module exports and version info
├── train_self_play.py       # Main training script entry point
├── trainer.py               # PyTorch Lightning trainer setup with WandB
├── evaluator.py             # High-level evaluation interface
├── models.py                # ChessTransformer model definition
├── constants.py             # Shared constants and defaults
├── eval_utils.py            # Evaluation utilities (game playing, ELO estimation)
├── searchless_chess_imports.py  # Imports from external searchless_chess model
│
├── grpo_logic/              # GRPO training logic
│   ├── model.py             # GRPOChessTransformer Lightning module
│   ├── loss.py              # GRPO loss functions (PPO-clip + KL penalty)
│   └── sampling.py          # Trajectory sampling from policy
│
└── chess/                   # Chess game logic and players
    ├── chess_logic.py       # Board encoding, legal moves, datasets
    ├── policy_player.py     # PolicyPlayer wrapper for models
    ├── searcher.py          # Trajectory search wrapper (optional)
    ├── rewards.py           # Reward computation using Stockfish
    └── stockfish.py         # Stockfish player and engine management
```

## Key Components

### Models (`models.py`)

**`ChessTransformer`**: Core transformer-based policy network
- Input: Tokenized FEN strings `[batch, seq_len]` (seq_len ≈ 77)
- Output: Action logits `[batch, action_dim]` (action_dim = 1968)
- Architecture: Transformer encoder with learnable positional encodings
- Key methods:
  - `forward(x)`: Standard forward pass
  - `get_legal_moves_logits()`: Returns logits with illegal moves masked to -inf
  - `get_legal_moves_probs()`: Returns probability distribution over legal moves
  - `get_group_log_probs()`: Computes log probabilities for batched trajectories `[B, G, T]`
  - `select_action()`: Samples a move from the policy for a single board

**`ChessTransformerConfig`**: Configuration dataclass
- `vocab_size`: Token vocabulary size (default: 300)
- `embed_dim`: Embedding dimension (default: 256)
- `num_layers`: Transformer layers (default: 4)
- `num_heads`: Attention heads (default: 8)
- `action_dim`: Action space size (default: 1968)

### GRPO Logic (`grpo_logic/`)

**`GRPOChessTransformer`** (`grpo_logic/model.py`): PyTorch Lightning module
- Maintains two models: `policy_model` (trainable) and `old_policy_model` (frozen, synced each epoch)
- Implements `training_step()`: Samples trajectories, computes GRPO loss
- Handles evaluation against Stockfish periodically
- Key hyperparameters from `GRPOConfig`:
  - `lr`: Learning rate
  - `num_trajectories`: Number of trajectory groups per batch (G)
  - `trajectory_depth`: Maximum trajectory length (T)
  - `clip_ratio`: PPO clipping epsilon (default: 0.2)
  - `kl_coef`: KL divergence penalty coefficient (default: 0.01)

**`grpo_ppo_loss()`** (`grpo_logic/loss.py`): Main loss function
- Computes advantages by normalizing group rewards within each batch
- Applies PPO-clip loss with KL penalty
- Input shapes: `logprobs_new/old [B, G, T]`, `group_rewards [B, G]`, `pad_mask [B, G, T]`
- Returns: scalar loss + optional `GRPOLossInfo` for logging

**`sample_trajectories_batched()`** (`grpo_logic/sampling.py`): Trajectory sampling
- Samples G trajectories of depth T from each starting board
- Returns `TrajectoriesSample` containing:
  - `trajectories_log_probs [B, G, T]`: Log probabilities of sampled actions
  - `trajectories_actions [B, G, T]`: Action indices
  - `trajectories_states [B, G, T, SEQ]`: State tensors
  - `group_rewards [B, G]`: Final rewards per trajectory group
  - `pad_mask [B, G, T]`: Boolean mask for valid steps
  - `trajectories_legal_masks [B, G, T, A]`: Legal moves masks

### Chess Logic (`chess/`)

**`chess_logic.py`**: Core chess utilities
- `board_to_tensor()`: Converts `chess.Board` to tokenized tensor
- `get_legal_moves_indices()`: Maps legal moves to action indices
- `get_legal_moves_mask()`: Creates boolean mask `[action_dim]` for legal moves
- `ChessStartStatesDataset`: Infinite dataset yielding random FEN strings

**`policy_player.py`**: `PolicyPlayer` class
- Wraps a model to play chess games
- `act(board)`: Returns a move from the policy
- Supports temperature sampling and greedy selection
- Tracks statistics (illegal moves, fallbacks)

**`rewards.py`**: Reward computation
- `reward_board()`: Computes reward for a board position using Stockfish
- Uses cached Stockfish evaluations for efficiency
- Returns normalized reward in [-1, 1] range

**`stockfish.py`**: Stockfish integration
- `StockfishPlayer`: Wrapper for Stockfish engine
- `StockfishManager`: Manages engine lifecycle (creation, reuse, cleanup)
- Supports configurable skill levels and time limits

### Evaluation (`evaluator.py`, `eval_utils.py`)

**`Evaluator`**: High-level evaluation interface
- `single_evaluation()`: Plays games against Stockfish, returns results dict
- `eval_ladder()`: Evaluates at multiple Stockfish skill levels [1, 3, 5, 8, 10]
- Results include: wins, draws, losses, score, estimated ELO difference

**`eval_utils.py`**: Low-level evaluation utilities
- `play_one_game()`: Plays a single game between policy and Stockfish
- `evaluate_policy_vs_stockfish()`: Runs multiple games, aggregates results
- `estimate_elo_diff()`: Converts win rate to approximate ELO difference

### Training (`trainer.py`, `train_self_play.py`)

**`get_trainer()`**: Creates PyTorch Lightning trainer
- Configures WandB logging with unique run names
- Sets up checkpointing (saves top 2 models by `train_total_loss`)
- Gradient clipping (norm = 1.0)

**`train_self_play.py`**: Main training script
- Creates `GRPOChessTransformer` model
- Sets up `ChessStartStatesDataset` dataloader
- Runs training loop

## Data Flow

### Training Step Flow

1. **Data Loading**: `ChessStartStatesDataset` yields random FEN strings
2. **Trajectory Sampling** (`sampling.py`):
   - For each board, sample G trajectories of depth T using `old_policy_model`
   - Each trajectory: states → policy → actions → next states
   - Compute final reward for each trajectory using Stockfish
3. **Loss Computation** (`loss.py`):
   - Compute new policy log probabilities using `policy_model`
   - Normalize group rewards to get advantages
   - Apply PPO-clip loss with KL penalty
4. **Backpropagation**: Update `policy_model` parameters
5. **Policy Sync**: At epoch start, copy `policy_model` → `old_policy_model`

### Evaluation Flow

1. Create `PolicyPlayer` from model (optionally wrap with `TrajectorySearcher`)
2. Create `StockfishPlayer` with desired skill level
3. Play N games alternating colors
4. Aggregate results (wins/draws/losses) and compute score/ELO

## Key Patterns and Conventions

### Tensor Shapes

- **States**: `[batch, seq_len]` where seq_len = 77 (tokenized FEN)
- **Actions**: Integer indices in range [0, action_dim-1] where action_dim = 1968
- **Trajectories**: `[B, G, T, ...]` where B=batch, G=groups, T=time steps
- **Legal Masks**: `[batch, action_dim]` boolean tensors

### Device Management

- Models infer device from their parameters: `next(model.parameters()).device`
- Always pass `device` parameter to tensor creation functions
- Use `model.device` property when available

### Masking Conventions

- **Padding mask**: `True` = valid step, `False` = padding
- **Legal moves mask**: `True` = legal move, `False` = illegal
- Illegal moves are masked to `-inf` before softmax
- Padding steps are excluded from loss computation

### Reward Computation

- Rewards are computed only at the end of trajectories
- Uses Stockfish evaluation with caching for efficiency
- Rewards normalized to [-1, 1] range
- Group rewards `[B, G]` are normalized within each batch to compute advantages

### Model State Management

- `policy_model`: Always trainable, updated by optimizer
- `old_policy_model`: Frozen copy, synced at epoch start
- Both models share the same architecture but independent parameters

## Common Tasks

### Adding a New Model Architecture

1. Create new model class inheriting from `nn.Module` in `models.py`
2. Implement `forward()`, `get_legal_moves_logits()`, `get_group_log_probs()`
3. Update `GRPOChessTransformer` to accept new model type
4. Ensure model outputs logits of shape `[batch, action_dim]`

### Modifying the Loss Function

1. Edit `grpo_ppo_loss()` in `grpo_logic/loss.py`
2. Ensure output is scalar loss tensor
3. Update `GRPOLossInfo` dataclass if adding new metrics
4. Update logging in `GRPOChessTransformer.training_step()`

### Changing Reward Function

1. Modify `reward_board()` in `chess/rewards.py`
2. Ensure output is scalar reward (typically normalized)
3. Consider caching if reward computation is expensive
4. Update reward normalization in `group_advantage()` if needed

### Adding Evaluation Metrics

1. Add metric computation in `Evaluator` or `eval_utils.py`
2. Include in results dictionary returned by evaluation functions
3. Log metrics in `GRPOChessTransformer._log_stockfish_eval()` if using Lightning

### Debugging Trajectory Sampling

- Check `pad_mask` to ensure trajectories aren't all padding
- Verify `group_rewards` are reasonable (not all zeros/NaNs)
- Ensure `trajectories_legal_masks` has at least one True per step
- Check that sampled actions are actually legal moves

### Debugging Loss Issues

- Monitor `mean_ratio` (should be near 1.0, not exploding)
- Check `mean_clip_fraction` (high values indicate policy changing too fast)
- Watch `mean_kl_divergence` (should be small, < 0.1 typically)
- Ensure loss is finite (check for NaNs/Infs)

## Dependencies

- **PyTorch**: Neural network framework
- **PyTorch Lightning**: Training framework
- **python-chess**: Chess board representation and move generation
- **chess.engine**: Stockfish integration
- **WandB**: Experiment tracking (optional but recommended)

## Configuration

Key configuration classes:
- `ChessTransformerConfig`: Model architecture
- `GRPOConfig`: Training hyperparameters
- `EvalConfig`: Evaluation settings
- `PolicyConfig`: Policy player behavior
- `StockfishConfig`: Stockfish engine settings

Default values are in `constants.py` and can be overridden when creating instances.

## Notes for Agents

When editing this module:

1. **Preserve tensor shapes**: Always verify tensor dimensions match expected shapes
2. **Handle padding**: Use `pad_mask` to exclude padding from computations
3. **Legal moves**: Always mask illegal moves to -inf before softmax
4. **Device consistency**: Ensure all tensors are on the same device
5. **Gradient flow**: Only `policy_model` should have `requires_grad=True`
6. **Reward normalization**: Group rewards are normalized within batches for advantages
7. **Model sync**: `old_policy_model` is synced at epoch start, not during training
8. **Evaluation mode**: Set model to `eval()` and use `torch.no_grad()` during evaluation
9. **Stockfish cleanup**: Always close Stockfish engines after use to free resources
10. **Error handling**: Check for game over states, empty legal moves, and finite losses

