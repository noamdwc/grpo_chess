"""
Config loader for GRPO Chess experiments.

This module provides utilities to load experiment configurations from YAML files
and convert them to the appropriate dataclass objects.

Usage:
    from src.grpo_self_play.configs.config_loader import load_experiment_config

    # Load a complete experiment config
    config = load_experiment_config("default.yaml")

    # Load with overrides
    config = load_experiment_config("default.yaml", overrides={
        "grpo": {"lr": 1e-4, "entropy_coef": 0.2},
        "training": {"num_epochs": 100},
    })

    # Access configs
    grpo_config = config.grpo
    transformer_config = config.transformer
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, TypeVar, Type
import yaml

# Import all config dataclasses
from src.grpo_self_play.grpo_logic.model import GRPOConfig
from src.grpo_self_play.models import ChessTransformerConfig
from src.grpo_self_play.eval_utils import EvalConfig
from src.grpo_self_play.chess.stockfish import StockfishConfig
from src.grpo_self_play.chess.policy_player import PolicyConfig
from src.grpo_self_play.chess.searcher import SearchConfig
from src.grpo_self_play.chess.boards_dataset import ChessDatasetConfig
from src.grpo_self_play.pretrain.pretrain_load_config import PretrainLoadConfig


# Directory containing config YAML files
CONFIGS_DIR = Path(__file__).parent


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    num_epochs: int = 400
    batch_size: int = 32
    steps_per_epoch: int = 512
    checkpoint_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3


@dataclass
class ExperimentConfig:
    """Complete experiment configuration containing all sub-configs."""
    training: TrainingConfig
    grpo: GRPOConfig
    transformer: ChessTransformerConfig
    eval: EvalConfig
    stockfish: StockfishConfig
    policy: PolicyConfig
    searcher: Optional[SearchConfig]
    dataset: ChessDatasetConfig
    pretrain: PretrainLoadConfig


T = TypeVar('T')


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Deep merge two dictionaries, with overrides taking precedence.

    Args:
        base: Base dictionary
        overrides: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def dict_to_dataclass(cls: Type[T], data: dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass, ignoring extra keys.

    Args:
        cls: The dataclass type to instantiate
        data: Dictionary with field values

    Returns:
        Instance of the dataclass with values from data
    """
    if data is None:
        return None

    # Get valid field names for this dataclass
    valid_fields = {f.name for f in fields(cls)}

    # Filter to only include valid fields
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return cls(**filtered_data)


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file (absolute or relative to configs dir)

    Returns:
        Dictionary containing the parsed YAML
    """
    path = Path(path)

    # If not absolute, look in configs directory
    if not path.is_absolute():
        path = CONFIGS_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, dict[str, Any]] | None = None
) -> ExperimentConfig:
    """Load a complete experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file (absolute or relative to configs dir)
        overrides: Optional dict of overrides per section. Example:
            {
                "grpo": {"lr": 1e-4, "entropy_coef": 0.2},
                "training": {"num_epochs": 100},
                "stockfish": {"skill_level": 5},
            }

    Returns:
        ExperimentConfig containing all sub-configs
    """
    data = load_yaml_file(path)

    # Apply overrides if provided
    if overrides:
        data = _deep_merge(data, overrides)

    # Convert each section to its dataclass
    training = dict_to_dataclass(TrainingConfig, data.get('training', {}))
    grpo = dict_to_dataclass(GRPOConfig, data.get('grpo', {}))
    transformer = dict_to_dataclass(ChessTransformerConfig, data.get('transformer', {}))
    eval_cfg = dict_to_dataclass(EvalConfig, data.get('eval', {}))
    stockfish = dict_to_dataclass(StockfishConfig, data.get('stockfish', {}))
    policy = dict_to_dataclass(PolicyConfig, data.get('policy', {}))
    dataset = dict_to_dataclass(ChessDatasetConfig, data.get('dataset', {}))
    pretrain = dict_to_dataclass(PretrainLoadConfig, data.get('pretrain', {}))

    # Searcher is optional (can be null)
    searcher_data = data.get('searcher')
    searcher = dict_to_dataclass(SearchConfig, searcher_data) if searcher_data else None

    return ExperimentConfig(
        training=training,
        grpo=grpo,
        transformer=transformer,
        eval=eval_cfg,
        stockfish=stockfish,
        policy=policy,
        searcher=searcher,
        dataset=dataset,
        pretrain=pretrain,
    )


def load_grpo_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None
) -> GRPOConfig:
    """Load just the GRPO config from a YAML file.

    Args:
        path: Path to the YAML file
        overrides: Optional dict of field overrides. Example: {"lr": 1e-4}
    """
    data = load_yaml_file(path)
    grpo_data = data.get('grpo', {})
    if overrides:
        grpo_data = _deep_merge(grpo_data, overrides)
    return dict_to_dataclass(GRPOConfig, grpo_data)


def load_transformer_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None
) -> ChessTransformerConfig:
    """Load just the transformer config from a YAML file."""
    data = load_yaml_file(path)
    cfg_data = data.get('transformer', {})
    if overrides:
        cfg_data = _deep_merge(cfg_data, overrides)
    return dict_to_dataclass(ChessTransformerConfig, cfg_data)


def load_eval_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None
) -> EvalConfig:
    """Load just the eval config from a YAML file."""
    data = load_yaml_file(path)
    cfg_data = data.get('eval', {})
    if overrides:
        cfg_data = _deep_merge(cfg_data, overrides)
    return dict_to_dataclass(EvalConfig, cfg_data)


def load_stockfish_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None
) -> StockfishConfig:
    """Load just the stockfish config from a YAML file."""
    data = load_yaml_file(path)
    cfg_data = data.get('stockfish', {})
    if overrides:
        cfg_data = _deep_merge(cfg_data, overrides)
    return dict_to_dataclass(StockfishConfig, cfg_data)


def load_dataset_config(
    path: str | Path = "default.yaml",
    overrides: dict[str, Any] | None = None
) -> ChessDatasetConfig:
    """Load just the dataset config from a YAML file."""
    data = load_yaml_file(path)
    cfg_data = data.get('dataset', {})
    if overrides:
        cfg_data = _deep_merge(cfg_data, overrides)
    return dict_to_dataclass(ChessDatasetConfig, cfg_data)


def list_available_configs() -> list[str]:
    """List all available YAML config files in the configs directory."""
    return [f.name for f in CONFIGS_DIR.glob("*.yaml")]


def print_config_summary(config: ExperimentConfig) -> None:
    """Print a summary of the experiment configuration."""
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    print("\n[Training]")
    print(f"  epochs: {config.training.num_epochs}")
    print(f"  batch_size: {config.training.batch_size}")
    print(f"  steps_per_epoch: {config.training.steps_per_epoch}")

    print("\n[GRPO]")
    print(f"  lr: {config.grpo.lr}")
    print(f"  num_trajectories: {config.grpo.num_trajectories}")
    print(f"  trajectory_depth: {config.grpo.trajectory_depth}")
    print(f"  entropy_coef: {config.grpo.entropy_coef}")
    print(f"  rollout_temperature: {config.grpo.rollout_temperature}")
    print(f"  adaptive_kl: {config.grpo.adaptive_kl}")
    print(f"  use_entropy_floor: {config.grpo.use_entropy_floor}")

    print("\n[Transformer]")
    print(f"  embed_dim: {config.transformer.embed_dim}")
    print(f"  num_layers: {config.transformer.num_layers}")
    print(f"  num_heads: {config.transformer.num_heads}")

    print("\n[Eval]")
    print(f"  games: {config.eval.games}")
    print(f"  max_plies: {config.eval.max_plies}")

    print("\n[Stockfish]")
    print(f"  skill_level: {config.stockfish.skill_level}")

    print("\n[Searcher]")
    print(f"  enabled: {config.searcher is not None}")

    print("=" * 60)
