"""GRPO Self-Play Module for Chess.

This module implements Group Relative Policy Optimization (GRPO) for training
chess policies through self-play. It includes:
- Transformer-based chess policy models
- GRPO training logic with PPO clipping
- Trajectory sampling and reward computation
- Evaluation against Stockfish
"""

__version__ = "0.1.0"

# Main exports
from src.models import ChessTransformer, ChessTransformerConfig
from src.grpo_logic.model import GRPOChessTransformer, GRPOConfig
from src.grpo_logic.loss import grpo_ppo_loss, GRPOLossInfo
from src.evaluator import Evaluator
from src.eval_utils import EvalConfig

__all__ = [
    "ChessTransformer",
    "ChessTransformerConfig",
    "GRPOChessTransformer",
    "GRPOConfig",
    "grpo_ppo_loss",
    "GRPOLossInfo",
    "Evaluator",
    "EvalConfig",
]
