"""Compatibility helpers for loading legacy checkpoints."""

from collections.abc import Mapping
import sys
from typing import Optional

import torch

import src


def register_legacy_checkpoint_aliases() -> None:
    """Register module aliases used by checkpoints saved before refactors."""
    sys.modules.setdefault("src.grpo_self_play", src)


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
