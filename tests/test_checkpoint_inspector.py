"""Checkpoint family inspection tests."""
from pathlib import Path

import pytest


def test_inspect_pt_dm_port_checkpoint_reports_action_value_family():
    from src.checkpoint_inspector import inspect_checkpoint

    ckpt = Path("checkpoints/dm_port/9M.pt")
    if not ckpt.exists():
        pytest.skip("DM-port checkpoint missing")

    info = inspect_checkpoint(ckpt)

    assert info.checkpoint_format == "torch_pt"
    assert info.family == "dm_action_value"
    assert info.input_vocab_size == 1968
    assert info.output_size == 128
    assert info.positional_length == 79


def test_inspect_upstream_checkpoint_families_reports_distinct_input_interfaces():
    from src.checkpoint_inspector import inspect_checkpoint

    bc = Path("searchless_chess/checkpoints/9M_behavioral_cloning")
    sv = Path("searchless_chess/checkpoints/9M_state_value")
    av = Path("searchless_chess/checkpoints/9M")
    if not (bc.exists() and sv.exists() and av.exists()):
        pytest.skip("Expected upstream checkpoints are not all available")

    bc_info = inspect_checkpoint(bc)
    sv_info = inspect_checkpoint(sv)
    av_info = inspect_checkpoint(av)

    assert bc_info.checkpoint_format == "orbax"
    assert bc_info.family == "dm_behavioral_cloning"
    assert bc_info.input_vocab_size == 31
    assert bc_info.output_size == 1968
    assert bc_info.positional_length == 78

    assert sv_info.checkpoint_format == "orbax"
    assert sv_info.family == "dm_state_value"
    assert sv_info.input_vocab_size == 31
    assert sv_info.output_size == 128
    assert sv_info.positional_length == 78

    assert av_info.checkpoint_format == "orbax"
    assert av_info.family == "dm_action_value"
    assert av_info.input_vocab_size == 1968
    assert av_info.output_size == 128
    assert av_info.positional_length == 79
