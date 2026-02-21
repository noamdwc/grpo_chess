"""Compatibility helpers for loading legacy checkpoints."""

import sys

import src


def register_legacy_checkpoint_aliases() -> None:
    """Register module aliases used by checkpoints saved before refactors."""
    sys.modules.setdefault("src.grpo_self_play", src)
