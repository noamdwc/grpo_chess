"""Deterministic parity checks for bare DM inference vs wrapped zero-think inference."""
from __future__ import annotations

from pathlib import Path

import chess
import pytest
import torch

from src.dm_port.stockfish_eval import DEFAULT_DM_PORT_CHECKPOINT
from src.reasoning.model import ReasoningModel, ReasoningModelConfig
from tests.debug_zero_think_parity import (
    build_fixed_fen_corpus,
    compare_position,
    load_debug_models,
    run_fen_sweep,
    trace_first_divergence,
)


CKPT = Path(DEFAULT_DM_PORT_CHECKPOINT)
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="DM-port checkpoint is required for parity diagnostics")


def test_single_fen_comparison_localizes_zero_think_divergence():
    dm_model, reasoning_model = load_debug_models(checkpoint_path=CKPT)
    label = None
    fen = None
    for candidate_label, candidate_fen in build_fixed_fen_corpus():
        comparison = compare_position(dm_model, reasoning_model, chess.Board(candidate_fen), label=candidate_label)
        if comparison.wrapped.chosen_action_id != comparison.bare.chosen_action_id:
            label = candidate_label
            fen = candidate_fen
            break

    assert label is not None and fen is not None

    comparison = compare_position(dm_model, reasoning_model, chess.Board(fen), label=label)

    assert comparison.bare.fen == comparison.wrapped.fen == fen
    assert comparison.bare.side_to_move == comparison.wrapped.side_to_move
    assert comparison.bare.fen_token_ids == comparison.wrapped.fen_token_ids
    assert comparison.bare.legal_moves_uci == comparison.wrapped.legal_moves_uci
    assert comparison.bare.legal_action_ids == comparison.wrapped.legal_action_ids
    assert comparison.bare.legal_mask_active_ids == comparison.wrapped.legal_mask_active_ids
    assert comparison.bare.roundtrip_violations == []
    assert comparison.wrapped.roundtrip_violations == []
    assert comparison.wrapped.decoded_move_is_legal

    assert comparison.wrapped_dm_semantics.chosen_action_id == comparison.bare.chosen_action_id
    assert comparison.wrapped_dm_semantics.chosen_move_uci == comparison.bare.chosen_move_uci
    assert comparison.wrapped.chosen_action_id != comparison.bare.chosen_action_id


def test_fen_sweep_preserves_dm_body_parity_and_exposes_wrapper_move_mismatches():
    dm_model, reasoning_model = load_debug_models(checkpoint_path=CKPT)

    sweep = run_fen_sweep(dm_model, reasoning_model, checkpoint_path=CKPT)

    assert 20 <= len(sweep) <= 50
    wrapped_move_mismatches = 0
    for comparison in sweep:
        assert comparison.bare.legal_action_ids == comparison.wrapped.legal_action_ids, comparison.label
        assert comparison.bare.roundtrip_violations == [], comparison.label
        assert comparison.wrapped.roundtrip_violations == [], comparison.label
        assert comparison.wrapped.decoded_move_is_legal, comparison.label
        assert comparison.wrapped_dm_semantics.chosen_action_id == comparison.bare.chosen_action_id, comparison.label
        if comparison.wrapped.chosen_action_id != comparison.bare.chosen_action_id:
            wrapped_move_mismatches += 1

    assert wrapped_move_mismatches >= 1


def test_fixed_game_trace_stops_at_first_zero_think_divergence_without_prior_board_drift():
    dm_model, reasoning_model = load_debug_models(checkpoint_path=CKPT)

    trace = trace_first_divergence(dm_model, reasoning_model, checkpoint_path=CKPT, max_plies=20)

    assert trace.first_divergence is not None
    assert trace.first_divergence.ply_index >= 0
    assert trace.first_divergence.bare_move_is_legal
    assert trace.first_divergence.wrapped_move_is_legal
    assert trace.first_divergence.bare_move_uci != trace.first_divergence.wrapped_move_uci
    assert trace.first_divergence.bare_fen_after != trace.first_divergence.wrapped_fen_after


def test_raw_dm_checkpoint_does_not_determine_wrapped_policy_head_weights():
    torch.manual_seed(0)
    model_seed_0 = ReasoningModel(ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)).eval()
    torch.manual_seed(1)
    model_seed_1 = ReasoningModel(ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)).eval()

    assert not torch.equal(model_seed_0.policy_head.weight, model_seed_1.policy_head.weight)
