"""PyTorch module wrapper for parameter counting."""

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    frozen: int


class ParameterCounter(nn.Module):
    """Wrap a model and expose parameter counting helpers.

    This is a transparent wrapper: ``forward`` delegates to the wrapped model.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def summary(self) -> ParameterSummary:
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return ParameterSummary(
            total=total,
            trainable=trainable,
            frozen=total - trainable,
        )
