# Chess Model Pretraining

This module provides supervised pretraining on expert chess moves from Lichess games before GRPO reinforcement learning fine-tuning.

## Overview

The pretraining pipeline:
1. Streams chess games from HuggingFace (`Lichess/standard-chess-games`)
2. Filters by player ELO rating
3. Extracts positions and moves from games
4. Trains the ChessTransformer with cross-entropy loss on expert moves
5. Saves checkpoints compatible with GRPO training

## Quick Start

```bash
# Run pretraining with default config
python -m src.grpo_self_play.pretrain.pretrain --config pretrain.yaml

# With custom parameters
python -m src.grpo_self_play.pretrain.pretrain --config pretrain.yaml \
    --lr 1e-4 --batch_size 512 --min_elo 1800

# Disable wandb logging
python -m src.grpo_self_play.pretrain.pretrain --no_wandb
```

## Configuration

Configuration is in `src/grpo_self_play/configs/pretrain.yaml`:

```yaml
pretrain:
  lr: 0.0001                    # Learning rate
  batch_size: 256               # Batch size
  num_epochs: 1                 # Number of epochs
  warmup_steps: 1000            # Linear warmup steps
  weight_decay: 0.01            # AdamW weight decay
  max_grad_norm: 1.0            # Gradient clipping
  label_smoothing: 0.1          # Prevents overconfidence
  val_check_interval: 0.1       # Validate every 10% of epoch

dataset:
  min_elo: 1800                 # Minimum player rating
  skip_first_n_moves: 5         # Skip opening moves
  skip_last_n_moves: 5          # Skip endgame moves
  sample_positions_per_game: 3  # Positions per game
  eval_fraction: 0.05           # 5% held out for evaluation

transformer:
  embed_dim: 256
  num_layers: 4
  num_heads: 8
```

## Train/Eval Split

The dataset uses a **hash-based deterministic split** to ensure:
- No data leakage between training and evaluation
- Consistent splits across runs
- Process-safe multi-worker data loading

Games are assigned to train or eval based on:
```python
is_eval = hash(game_site_url) % 10000 < (eval_fraction * 10000)
```

This means the same game always goes to the same split, regardless of worker or epoch.

## Using Pretrained Weights in GRPO

After pretraining, use the checkpoint for GRPO fine-tuning by updating `default.yaml`:

```yaml
pretrain:
  checkpoint_path: "checkpoints/pretrain/pretrain_final.pt"
  freeze_layers: 0  # Optional: freeze first N transformer layers
```

Or pass the path when running training:
```bash
python -m src.grpo_self_play.train_self_play --config default.yaml
```

## Module Structure

```
pretrain/
├── __init__.py              # Package exports
├── pretrain.py              # PyTorch Lightning training module
├── pretrain_dataset.py      # Streaming dataset from HuggingFace
├── pretrain_load_config.py  # Config for loading pretrained weights
└── README.md                # This file
```

## Key Classes

### PretrainChessTransformer

PyTorch Lightning module that wraps the ChessTransformer for supervised learning.

```python
from src.grpo_self_play.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
from src.grpo_self_play.models import ChessTransformerConfig

model = PretrainChessTransformer(
    transformer_config=ChessTransformerConfig(embed_dim=256, num_layers=4, num_heads=8),
    pretrain_config=PretrainConfig(lr=1e-4, batch_size=256),
)
```

### ChessPretrainDataset

Streaming dataset that yields (board_tokens, action, legal_mask) tuples.

```python
from src.grpo_self_play.pretrain import ChessPretrainDataset, PretrainDatasetConfig

dataset = ChessPretrainDataset(PretrainDatasetConfig(
    min_elo=1800,
    is_eval=False,  # True for evaluation set
))
```

## Metrics

The following metrics are logged during training:

| Metric | Description |
|--------|-------------|
| `train/loss` | Cross-entropy loss with label smoothing |
| `train/accuracy` | Top-1 move prediction accuracy |
| `train/top5_accuracy` | Top-5 move prediction accuracy |
| `train/entropy` | Policy entropy (confidence measure) |
| `train/perplexity` | Exponential of loss |

## Tests

Run the test suite:
```bash
pytest tests/test_pretrain_pipeline.py -v
```

Tests cover:
- Configuration dataclasses
- PGN move parsing
- Position extraction from games
- UCI to action conversion
- Collate function
- Model creation and forward pass
- Training and validation steps
- Hash-based train/eval splitting
- Integration with PyTorch Lightning
