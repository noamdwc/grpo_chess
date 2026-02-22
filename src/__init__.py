"""Top-level package exports for GRPO Chess.

Keep this module lightweight: importing ``src`` should not eagerly import
PyTorch Lightning or other heavy training stacks. This is important for
distillation setup scripts that only need submodules such as ``src.distill``.
"""

import importlib
import sys
from typing import Dict, Tuple

__version__ = "0.1.0"

# Backward-compat alias for checkpoints saved before package rearrangement.
sys.modules.setdefault("src.grpo_self_play", sys.modules[__name__])

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ChessTransformer": ("src.models", "ChessTransformer"),
    "ChessTransformerConfig": ("src.models", "ChessTransformerConfig"),
    "GRPOChessTransformer": ("src.grpo_logic.model", "GRPOChessTransformer"),
    "GRPOConfig": ("src.grpo_logic.model", "GRPOConfig"),
    "grpo_ppo_loss": ("src.grpo_logic.loss", "grpo_ppo_loss"),
    "GRPOLossInfo": ("src.grpo_logic.loss", "GRPOLossInfo"),
    "Evaluator": ("src.evaluator", "Evaluator"),
    "EvalConfig": ("src.eval_utils", "EvalConfig"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'src' has no attribute '{name}'")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
