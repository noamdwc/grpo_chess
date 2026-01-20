"""Training script for GRPO chess self-play."""
import argparse
import warnings
from typing import Any
from torch.utils.data import DataLoader

from src.grpo_self_play.trainer import get_trainer
from src.grpo_self_play.chess.boards_dataset import ChessStartStatesDataset
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer
from src.grpo_self_play.configs.config_loader import load_experiment_config


def train(
    config_path: str = "default.yaml",
    overrides: dict[str, dict[str, Any]] | None = None,
    dataloader_kwargs: dict[str, Any] | None = None
) -> None:
    """Main training function for GRPO chess self-play.
    
    Args:
        config_path: Path to the YAML config file (relative to configs directory)
        overrides: Optional dict of overrides per section. Example:
            {
                "grpo": {"lr": 1e-4, "entropy_coef": 0.2},
                "training": {"num_epochs": 100},
                "stockfish": {"skill_level": 5},
            }
        dataloader_kwargs: Optional dict of arguments to pass to DataLoader constructor.
            These override config values. Example: {"batch_size": 64, "num_workers": 4}
    """
    config = load_experiment_config(config_path, overrides=overrides)
    
    # Build dataloader kwargs from config, with defaults
    dataloader_config = {
        "batch_size": config.training.batch_size,
        "num_workers": 2,
    }
    
    # Apply dataloader_kwargs overrides and warn if overriding config values
    if dataloader_kwargs:
        for key, value in dataloader_kwargs.items():
            if key in dataloader_config:
                warnings.warn(
                    f"Overriding DataLoader '{key}' from config ({dataloader_config[key]}) "
                    f"with provided value ({value})",
                    UserWarning,
                    stacklevel=2
                )
            dataloader_config[key] = value
    
    trainer = get_trainer(num_epochs=config.training.num_epochs)
    dataset = ChessStartStatesDataset(config.dataset)
    dataloader = DataLoader(dataset, **dataloader_config)
    model = GRPOChessTransformer(
        transformer_config=config.transformer,
        grpo_config=config.grpo,
        eval_cfg=config.eval,
        stockfish_cfg=config.stockfish,
        policy_cfg=config.policy,
        searcher_cfg=config.searcher,
        pretrain_cfg=config.pretrain,
    )

    print("Starting Training with WandB Tracking...")
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="default.yaml")
    args = parser.parse_args()
    train(config_path=args.config)