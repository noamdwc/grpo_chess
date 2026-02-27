"""Pretraining module for chess model.

Keep package import lightweight so config-only imports do not pull dataset
dependencies (e.g., pandas/datasets) unless they are actually needed.
"""

from src.pretrain.pretrain_load_config import PretrainLoadConfig

__all__ = [
    "PretrainLoadConfig",
    "ChessPretrainDataset",
    "PretrainDatasetConfig",
    "collate_pretrain_batch",
]


def __getattr__(name: str):
    if name in {"ChessPretrainDataset", "PretrainDatasetConfig", "collate_pretrain_batch"}:
        from src.pretrain.pretrain_dataset import (
            ChessPretrainDataset,
            PretrainDatasetConfig,
            collate_pretrain_batch,
        )

        exports = {
            "ChessPretrainDataset": ChessPretrainDataset,
            "PretrainDatasetConfig": PretrainDatasetConfig,
            "collate_pretrain_batch": collate_pretrain_batch,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
