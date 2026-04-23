"""Config loader for the reasoning-GRPO refactor."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

import yaml

from src.chess.boards_dataset import ChessDatasetConfig
from src.chess.stockfish import StockfishConfig

CONFIGS_DIR = Path(__file__).parent


@dataclass
class ValueHeadConfig:
    enabled: bool = True
    hidden_dim: int = 256
    coef: float = 0.5


@dataclass
class ModelConfig:
    base_checkpoint: str = "checkpoints/dm_port/9M.pt"
    dm_av_embedding_source: str | None = None
    freeze_body: bool = False
    max_seq_len: int = 120
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)


@dataclass
class ReasoningConfig:
    min_think_tokens: int = 2
    max_think_tokens: int = 12
    allow_early_end_think: bool = True
    legal_mask_think: bool = True
    exclude_specials_from_gradient: bool = False


@dataclass
class GRPOConfig:
    k_samples_per_root: int = 8
    lr: float = 1e-6
    clip_ratio: float = 0.20
    kl_coef: float = 0.001
    ppo_epochs: int = 1
    rollout_temperature: float = 1.0
    move_sampling_temperature: float = 1.0
    eval_every_n_epochs: int = 10


@dataclass
class SelfPlayRivalConfig:
    think_enabled: bool = False
    temperature: float = 0.0


@dataclass
class FrozenDmRivalConfig:
    checkpoint_path: str = "checkpoints/dm_port/9M.pt"
    dm_av_embedding_source: str | None = None


@dataclass
class StockfishRivalConfig:
    movetime_ms: int = 50
    skill_level: int = 2


@dataclass
class RivalConfig:
    mode: str = "self_play"
    self_play: SelfPlayRivalConfig = field(default_factory=SelfPlayRivalConfig)
    frozen_dm_9m: FrozenDmRivalConfig = field(default_factory=FrozenDmRivalConfig)
    stockfish: StockfishRivalConfig = field(default_factory=StockfishRivalConfig)


@dataclass
class DmValueHeadConfig:
    model: str = "136M"
    bucket_values_source: str = "searchless_chess"


@dataclass
class StockfishLeafConfig:
    movetime_ms: int = 50
    cp_clip: int = 1000


@dataclass
class LeafEvaluatorConfig:
    mode: str = "dm_value_head"
    dm_value_head: DmValueHeadConfig = field(default_factory=DmValueHeadConfig)
    stockfish: StockfishLeafConfig = field(default_factory=StockfishLeafConfig)


@dataclass
class TrainingConfig:
    num_epochs: int = 200
    batch_size: int = 16
    steps_per_epoch: int = 256
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    use_wandb: bool = True
    wandb_project: str = "Chess-GRPO-Bot"


@dataclass
class EvalConfig:
    games: int = 64
    seed: int = 0
    max_plies: int = 400
    randomize_opening: bool = True
    opening_plies: int = 6
    ablation_no_think: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig
    reasoning: ReasoningConfig
    grpo: GRPOConfig
    rival: RivalConfig
    leaf_evaluator: LeafEvaluatorConfig
    training: TrainingConfig
    eval: EvalConfig
    stockfish: StockfishConfig
    dataset: ChessDatasetConfig


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIGS_DIR / path
    with open(path) as handle:
        return yaml.safe_load(handle)


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for key, value in overrides.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_dataclass_type(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin in (Optional,):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else annotation
    if origin is list:
        return annotation
    if origin is dict:
        return annotation
    if origin is tuple:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return data
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        value = data[field_info.name]
        nested_type = _resolve_dataclass_type(type_hints.get(field_info.name, field_info.type))
        if isinstance(value, dict) and is_dataclass(nested_type):
            kwargs[field_info.name] = _from_dict(nested_type, value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)


def dict_to_dataclass(cls: type[Any], data: dict[str, Any]) -> Any:
    if data is None:
        return None
    return _from_dict(cls, data)


def _legacy_to_reasoning_schema(raw: dict[str, Any]) -> dict[str, Any]:
    if {"model", "reasoning", "rival", "leaf_evaluator"}.issubset(raw):
        return raw

    pretrain = raw.get("pretrain", {})
    grpo = raw.get("grpo", {})
    stockfish = raw.get("stockfish", {})
    eval_cfg = raw.get("eval", {})

    base_checkpoint = pretrain.get("checkpoint_path") or "checkpoints/dm_port/9M.pt"
    translated = {
        "model": {
            "base_checkpoint": base_checkpoint,
            "freeze_body": False,
            "max_seq_len": 120,
            "value_head": {
                "enabled": True,
                "hidden_dim": 256,
                "coef": 0.5,
            },
        },
        "reasoning": {
            "min_think_tokens": 2,
            "max_think_tokens": grpo.get("trajectory_depth", 12),
            "allow_early_end_think": True,
            "legal_mask_think": True,
            "exclude_specials_from_gradient": False,
        },
        "grpo": {
            "k_samples_per_root": grpo.get("num_trajectories", 8),
            "lr": grpo.get("lr", 1e-6),
            "clip_ratio": grpo.get("clip_ratio", 0.2),
            "kl_coef": grpo.get("kl_coef", 0.001),
            "ppo_epochs": grpo.get("ppo_steps", 1),
            "rollout_temperature": grpo.get("rollout_temperature", 1.0),
            "move_sampling_temperature": grpo.get("move_sampling_temperature", 1.0),
            "eval_every_n_epochs": grpo.get("eval_every_n_epochs", 10),
        },
        "rival": {
            "mode": "self_play",
            "self_play": {
                "think_enabled": False,
                "temperature": 0.0,
            },
            "frozen_dm_9m": {
                "checkpoint_path": base_checkpoint,
            },
            "stockfish": {
                "movetime_ms": stockfish.get("movetime_ms", 50),
                "skill_level": stockfish.get("skill_level", 2),
            },
        },
        "leaf_evaluator": {
            "mode": "stockfish",
            "dm_value_head": {
                "model": "136M",
                "bucket_values_source": "searchless_chess",
            },
            "stockfish": {
                "movetime_ms": stockfish.get("movetime_ms", 50),
                "cp_clip": 1000,
            },
        },
        "training": raw.get("training", {}),
        "eval": {
            **eval_cfg,
            "ablation_no_think": eval_cfg.get("ablation_no_think", True),
        },
        "stockfish": stockfish,
        "dataset": raw.get("dataset", {}),
    }
    return translated


def load_experiment_config(
    config_path: str = "default.yaml",
    overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> ExperimentConfig:
    raw = load_yaml_file(config_path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    raw = _legacy_to_reasoning_schema(raw)
    return _from_dict(ExperimentConfig, raw)


def load_grpo_config(
    config_path: str = "default.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> GRPOConfig:
    experiment_overrides = {"grpo": overrides} if overrides else None
    return load_experiment_config(config_path, overrides=experiment_overrides).grpo


def load_eval_config(
    config_path: str = "default.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> EvalConfig:
    experiment_overrides = {"eval": overrides} if overrides else None
    return load_experiment_config(config_path, overrides=experiment_overrides).eval


def load_stockfish_config(
    config_path: str = "default.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> StockfishConfig:
    experiment_overrides = {"stockfish": overrides} if overrides else None
    return load_experiment_config(config_path, overrides=experiment_overrides).stockfish


def load_dataset_config(
    config_path: str = "default.yaml",
    overrides: Optional[dict[str, Any]] = None,
) -> ChessDatasetConfig:
    experiment_overrides = {"dataset": overrides} if overrides else None
    return load_experiment_config(config_path, overrides=experiment_overrides).dataset


def list_available_configs() -> list[str]:
    return sorted(path.name for path in CONFIGS_DIR.glob("*.yaml"))


def print_config_summary(config: ExperimentConfig) -> None:
    print("=" * 60)
    print("REASONING-GRPO CONFIGURATION")
    print("=" * 60)
    print(f"checkpoint: {config.model.base_checkpoint}")
    print(f"max_think_tokens: {config.reasoning.max_think_tokens}")
    print(f"k_samples_per_root: {config.grpo.k_samples_per_root}")
    print(f"leaf_evaluator: {config.leaf_evaluator.mode}")
