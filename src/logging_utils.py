"""Logging utilities for GRPO training.

Uses Python's standard logging module which WandB captures automatically
in the Logs tab of a run.
"""

import logging

_initialized_loggers = set()


def get_logger(name: str = "grpo_chess") -> logging.Logger:
    """Get a logger that appears in WandB Logs tab.

    Args:
        name: Logger name (default: "grpo_chess")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if name not in _initialized_loggers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
        _initialized_loggers.add(name)

    return logger
