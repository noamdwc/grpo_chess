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

## Key Configuration

```python
GRPO_CONFIG = GRPOConfig(
    lr=1e-6,                    # Learning rate
    num_trajectories=16,        # Trajectories per position
    trajectory_depth=16,        # Max moves per trajectory
    clip_ratio=0.15,            # PPO clipping
    kl_coef=0.001,              # KL penalty coefficient
    eval_every_n_epochs=10,     # Evaluation frequency
)

TRANSFORMER_CONFIG = ChessTransformerConfig(
    vocab_size=300,
    embed_dim=256,
    num_layers=4,
    num_heads=8,
    action_dim=1968,
)
```

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
