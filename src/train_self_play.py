"""Entry point for reasoning-GRPO fine-tuning."""
from __future__ import annotations

import argparse
from typing import Any

from torch.utils.data import DataLoader

from src.chess.boards_dataset import ChessStartStatesDataset
from src.configs.config_loader import load_experiment_config
from src.grpo_logic.model import ReasoningGRPOLightningModule
from src.trainer import get_trainer


def train(
    config_path: str = "default.yaml",
    overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    cfg = load_experiment_config(config_path, overrides=overrides)

    dataset = ChessStartStatesDataset(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, num_workers=0)
    module = ReasoningGRPOLightningModule(cfg)
    trainer = get_trainer(
        num_epochs=cfg.training.num_epochs,
        checkpoint_dir=cfg.training.checkpoint_dir,
        checkpoint_every_n_epochs=cfg.training.checkpoint_every_n_epochs,
        keep_n_checkpoints=cfg.training.keep_n_checkpoints,
        use_wandb=cfg.training.use_wandb,
        wandb_project=cfg.training.wandb_project,
    )
    trainer.fit(module, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.yaml")
    args = parser.parse_args()
    train(args.config)
