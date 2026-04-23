"""Diagnostic for the wrapper-only Stockfish gap.

Run explicitly with:
`RUN_SLOW_STOCKFISH_EVAL=1 ~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_reasoning_wrapper_stockfish_eval.py -vv -s`
"""
from __future__ import annotations

import os
import random

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.chess.stockfish import resolve_stockfish_path
from src.configs.config_loader import load_experiment_config
from src.dm_port.stockfish_eval import (
    DEFAULT_DM_PORT_CHECKPOINT,
    DMPortEvaluator,
    load_dm_port_and_reasoning_models,
)
from src.train_self_play import ReasoningEvaluator

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _summary_line(name: str, results: dict) -> str:
    return (
        f"{name}: score={results['score']:.3f} "
        f"elo_diff={results['elo_diff_vs_stockfish_approx']:+.1f} "
        f"W/D/L={results['wins']}/{results['draws']}/{results['losses']}"
    )


def test_reasoning_wrapper_stockfish_eval():
    if os.environ.get("RUN_SLOW_STOCKFISH_EVAL") != "1":
        pytest.skip("Set RUN_SLOW_STOCKFISH_EVAL=1 to run this slow Stockfish integration diagnostic")

    if not DEFAULT_DM_PORT_CHECKPOINT.exists():
        pytest.skip(f"Missing DM-port checkpoint dependency: {DEFAULT_DM_PORT_CHECKPOINT}")

    cfg = load_experiment_config("default.yaml")
    try:
        resolved_stockfish = resolve_stockfish_path(cfg.stockfish.path)
    except FileNotFoundError as exc:
        pytest.skip(f"Missing Stockfish dependency: {exc}")

    eval_cfg = replace(cfg.eval, games=64, seed=0)
    stockfish_cfg = replace(cfg.stockfish, path=resolved_stockfish)

    dm_model, wrapped_model = load_dm_port_and_reasoning_models(
        DEFAULT_DM_PORT_CHECKPOINT,
        reasoning_max_seq_len=cfg.model.max_seq_len,
        value_head_hidden=cfg.model.value_head.hidden_dim,
    )
    baseline_evaluator = DMPortEvaluator(eval_cfg=eval_cfg, stockfish_cfg=stockfish_cfg)
    wrapped_evaluator = ReasoningEvaluator(eval_cfg=eval_cfg, stockfish_cfg=stockfish_cfg, think_tokens=0)

    _seed_everything(eval_cfg.seed)
    baseline_results, _, baseline_pgns = baseline_evaluator.single_evaluation(dm_model)
    _seed_everything(eval_cfg.seed)
    wrapped_results, _, wrapped_pgns = wrapped_evaluator.single_evaluation(wrapped_model)

    print(_summary_line("baseline", baseline_results))
    print(_summary_line("wrapped", wrapped_results))
    print(f"delta_score={baseline_results['score'] - wrapped_results['score']:+.3f}")

    assert baseline_results["games"] == 64
    assert wrapped_results["games"] == 64
    assert len(baseline_pgns) == 64
    assert len(wrapped_pgns) == 64
