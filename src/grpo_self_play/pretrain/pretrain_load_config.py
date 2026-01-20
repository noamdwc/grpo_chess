"""Pretrain load configuration - separated to avoid circular imports."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PretrainLoadConfig:
    """Configuration for loading pretrained weights.

    Attributes:
        checkpoint_path: Path to pretrained checkpoint file
        freeze_layers: Number of transformer layers to freeze (0 = train all)
    """
    checkpoint_path: Optional[str] = None
    freeze_layers: int = 0
