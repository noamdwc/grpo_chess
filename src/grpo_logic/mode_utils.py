"""Helpers for safely controlling module train/eval mode."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def temporary_eval(*modules: Any) -> Iterator[None]:
    """Temporarily set modules to eval mode and restore prior mode on exit.

    This disables stochastic layers (e.g., dropout) for the wrapped region
    while preserving the caller's original train/eval state.
    """
    managed = [module for module in modules if module is not None]
    original_training = [module.training for module in managed]

    try:
        for module in managed:
            module.eval()
        yield
    finally:
        for module, was_training in zip(managed, original_training):
            module.train(was_training)
