"""
Config module for GRPO Chess experiments.

Provides YAML-based configuration loading with override support.

Usage:
    from src.grpo_self_play.configs import load_experiment_config

    # Load default config
    config = load_experiment_config()

    # Load with overrides
    config = load_experiment_config("default.yaml", overrides={
        "grpo": {"lr": 1e-4},
        "training": {"num_epochs": 100},
    })
"""

from src.grpo_self_play.configs.config_loader import (
    ExperimentConfig,
    TrainingConfig,
    load_experiment_config,
    load_grpo_config,
    load_transformer_config,
    load_eval_config,
    load_stockfish_config,
    load_dataset_config,
    list_available_configs,
    print_config_summary,
)

__all__ = [
    "ExperimentConfig",
    "TrainingConfig",
    "load_experiment_config",
    "load_grpo_config",
    "load_transformer_config",
    "load_eval_config",
    "load_stockfish_config",
    "load_dataset_config",
    "list_available_configs",
    "print_config_summary",
]
