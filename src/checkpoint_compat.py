"""Compatibility helpers for loading legacy checkpoints."""

from collections.abc import Mapping
import importlib
import re
import sys
import types
from typing import Optional

import torch

import src


def register_legacy_checkpoint_aliases() -> None:
    """Register module aliases used by checkpoints saved before refactors."""
    sys.modules.setdefault("src.grpo_self_play", src)


_KNOWN_CONFIG_CLASS_SOURCES = {
    "PretrainConfig": "src.pretrain.pretrain",
    "DistillConfig": "src.distill.distill",
}

_RUNTIME_MAIN_MODULE_SUFFIXES = {
    "src.distill.distill": "src/distill/distill.py",
    "src.train_self_play": "src/train_self_play.py",
}


def _resolve_symbol_class(symbol_name: str, target_module: str) -> type:
    source_module_name = _KNOWN_CONFIG_CLASS_SOURCES.get(symbol_name)
    if source_module_name:
        try:
            source_module = importlib.import_module(source_module_name)
            source_cls = getattr(source_module, symbol_name, None)
            if source_cls is not None:
                return source_cls
        except Exception:
            pass

    # Fallback lightweight placeholder so unpickling can proceed.
    placeholder = type(symbol_name, (), {})
    placeholder.__module__ = target_module
    return placeholder


def _resolve_runtime_module(module_name: str) -> Optional[types.ModuleType]:
    module = sys.modules.get(module_name)
    if isinstance(module, types.ModuleType):
        return module

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "")
    suffix = _RUNTIME_MAIN_MODULE_SUFFIXES.get(module_name)
    if isinstance(main_file, str) and suffix and main_file.endswith(suffix):
        if isinstance(main_module, types.ModuleType):
            sys.modules[module_name] = main_module
            return main_module

    try:
        return importlib.import_module(module_name)
    except Exception:
        placeholder_module = types.ModuleType(module_name)
        sys.modules[module_name] = placeholder_module
        return placeholder_module


def _register_pickled_symbol_alias(symbol_name: str, module_name: str) -> None:
    module = _resolve_runtime_module(module_name)
    if module is None:
        return
    if hasattr(module, symbol_name):
        return
    setattr(module, symbol_name, _resolve_symbol_class(symbol_name, module_name))


def _parse_missing_pickled_symbol(error_message: str) -> Optional[tuple[str, str]]:
    patterns = [
        r"Can't get attribute '([^']+)' on <module '([^']+)'",
        r"module '([^']+)' has no attribute '([^']+)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_message)
        if not match:
            continue
        if pattern.startswith("Can't get attribute"):
            symbol_name, module_name = match.group(1), match.group(2)
        else:
            module_name, symbol_name = match.group(1), match.group(2)
        if symbol_name.endswith("Config"):
            return symbol_name, module_name
    return None


def _register_legacy_dataclass_aliases() -> None:
    """Register known legacy dataclass symbols from earlier checkpoint formats."""
    _register_pickled_symbol_alias("PretrainConfig", "src.distill.distill")
    _register_pickled_symbol_alias("DistillConfig", "src.train_self_play")


def load_checkpoint_with_compat(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
):
    """Load checkpoint with retries for known legacy pickle symbol mismatches."""
    register_legacy_checkpoint_aliases()
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)
    except AttributeError as exc:
        message = str(exc)
        parsed = _parse_missing_pickled_symbol(message)
        if parsed is None:
            raise
        symbol_name, module_name = parsed
        _register_pickled_symbol_alias(symbol_name, module_name)
        _register_legacy_dataclass_aliases()
        return torch.load(checkpoint_path, map_location=map_location, weights_only=weights_only)


def _get_cls_embedding_resize_mismatch(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    load_error: RuntimeError,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return (checkpoint_shape, model_shape) for CLS embedding resize mismatches."""
    if "size mismatch for embedding.weight" not in str(load_error):
        return None
    if getattr(model, "readout", None) != "cls":
        return None

    checkpoint_embedding = state_dict.get("embedding.weight")
    model_embedding = model.state_dict().get("embedding.weight")
    if not isinstance(checkpoint_embedding, torch.Tensor) or not isinstance(model_embedding, torch.Tensor):
        return None
    if checkpoint_embedding.ndim != 2 or model_embedding.ndim != 2:
        return None

    checkpoint_shape = tuple(int(d) for d in checkpoint_embedding.shape)
    model_shape = tuple(int(d) for d in model_embedding.shape)

    if checkpoint_shape == model_shape:
        return None
    if checkpoint_shape[1] != model_shape[1]:
        return None
    if model_shape[0] <= checkpoint_shape[0]:
        return None

    return checkpoint_shape, model_shape


def load_state_dict_with_checkpoint_compat(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    checkpoint_path: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load model weights and raise a clearer error for known legacy incompatibilities."""
    try:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        mismatch = _get_cls_embedding_resize_mismatch(model, state_dict, exc)
        if mismatch is None:
            raise

        checkpoint_shape, model_shape = mismatch
        cls_token_id = getattr(model, "cls_token_id", None)
        cls_suffix = f" Current cls_token_id={cls_token_id}." if cls_token_id is not None else ""
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}': embedding.weight shape mismatch "
            f"(checkpoint={checkpoint_shape}, model={model_shape}). "
            "This model uses transformer.readout='cls', which can expand embedding.weight "
            "(default cls_token_id=vocab_size). The checkpoint was likely trained without CLS readout. "
            "Use a checkpoint trained with the same readout, switch transformer.readout to 'mean' or 'last', "
            f"or set transformer.cls_token_id to an existing token id that does not grow embedding.weight.{cls_suffix}"
        ) from exc

    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
