"""Pretrain load configuration - separated to avoid circular imports."""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class PretrainLoadConfig:
    """Configuration for loading pretrained weights.

    Attributes:
        checkpoint_path: Path to pretrained checkpoint file
        freeze_layers: Number of transformer layers to freeze (0 = train all)
        use_9m_direct: Use Searchless9MGRPOPolicy (9M body + new action head) instead of ChessTransformer
    """
    checkpoint_path: Optional[str] = None
    freeze_layers: int = 0
    use_9m_direct: bool = False


# Register as safe for torch.load with weights_only=True (PyTorch 2.6+ compatibility)
torch.serialization.add_safe_globals([PretrainLoadConfig])
