"""Tests for reasoning-model warmstart provenance (spec: 2026-04-22-warmstart-provenance-test-design.md)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.reasoning.warmstart_provenance import (
    BC_CONTRACT,
    BC_CONTRACT_LEGACY,
    DM_AV_CONTRACT,
    DM_AV_CONTRACT_LEGACY,
    WarmstartContract,
    canonical_action_vocab_fingerprint,
    canonical_fen_tokenizer_fingerprint,
)


def test_fingerprints_are_deterministic_and_disagree():
    fen_a = canonical_fen_tokenizer_fingerprint()
    fen_b = canonical_fen_tokenizer_fingerprint()
    act_a = canonical_action_vocab_fingerprint()
    act_b = canonical_action_vocab_fingerprint()
    assert fen_a == fen_b, "FEN fingerprint must be deterministic across calls"
    assert act_a == act_b, "Action fingerprint must be deterministic across calls"
    assert fen_a != act_a, "FEN and action fingerprints must differ"
    assert fen_a.startswith("sha256:")
    assert act_a.startswith("sha256:")


def _build_synthetic_bc_ckpt(path: Path, embedding_dim: int = 32, num_layers: int = 2) -> None:
    """Build a tiny BC-shaped torch .pt checkpoint for tests (bypasses JAX/Orbax).

    IMPORTANT: this helper stamps fingerprints itself. It is NOT a test of the
    real `convert_behavioral_cloning.convert()` function — that is covered by
    `test_bc_converter_stamps_fingerprints` below.
    """
    from src.dm_port.transformer import DMTransformer, DMTransformerConfig

    cfg = DMTransformerConfig(
        vocab_size=31,
        output_size=1968,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=4,
        max_sequence_length=78,
    )
    model = DMTransformer(cfg)
    sd = model.state_dict()
    meta = {
        "family": "dm_behavioral_cloning",
        "input_vocab_size": 31,
        "output_size": 1968,
        "positional_length": 78,
        "fen_tokenizer_fingerprint": canonical_fen_tokenizer_fingerprint(),
        "action_vocab_fingerprint": canonical_action_vocab_fingerprint(),
    }
    torch.save({"state_dict": sd, "config": cfg.__dict__, "meta": meta}, path)


def test_synthetic_bc_helper_stamps_fingerprints(tmp_path):
    """Locks the shape of the test HELPER's output — not the real converter.
    The real converter is tested by test_bc_converter_stamps_fingerprints."""
    p = tmp_path / "bc.pt"
    _build_synthetic_bc_ckpt(p)
    payload = torch.load(p, weights_only=False)
    assert "meta" in payload
    assert payload["meta"]["fen_tokenizer_fingerprint"] == canonical_fen_tokenizer_fingerprint()
    assert payload["meta"]["action_vocab_fingerprint"] == canonical_action_vocab_fingerprint()


def _fake_bc_params(embedding_dim: int = 256):
    """Build a fake Orbax 'params' tree matching the layout the BC converter reads."""
    import numpy as np

    def _rand(*shape):
        return np.random.RandomState(0).randn(*shape).astype(np.float32)

    params = {
        "Embed_0": {"embedding": {"value": _rand(31, embedding_dim)}},
        "Embed_1": {"embedding": {"value": _rand(78, embedding_dim)}},
        "LayerNorm_16": {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)},
        "Dense_24": {"kernel": {"value": _rand(embedding_dim, 1968)}, "bias": _rand(1968)},
    }
    for i in range(8):
        params[f"LayerNorm_{2 * i}"] = {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)}
        params[f"LayerNorm_{2 * i + 1}"] = {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)}
        params[f"MultiHeadDotProductAttention_{i}"] = {
            "query": {"kernel": _rand(embedding_dim, embedding_dim)},
            "key": {"kernel": _rand(embedding_dim, embedding_dim)},
            "value": {"kernel": _rand(embedding_dim, embedding_dim)},
            "out": {"kernel": _rand(embedding_dim, embedding_dim)},
        }
        params[f"Dense_{3 * i}"] = {"kernel": {"value": _rand(embedding_dim, embedding_dim * 4)}}
        params[f"Dense_{3 * i + 1}"] = {"kernel": {"value": _rand(embedding_dim, embedding_dim * 4)}}
        params[f"Dense_{3 * i + 2}"] = {"kernel": {"value": _rand(embedding_dim * 4, embedding_dim)}}
    return params


def test_bc_converter_stamps_fingerprints(tmp_path, monkeypatch):
    """End-to-end check that the real BC converter writes fingerprints."""
    from src.dm_port import convert_behavioral_cloning as cv

    class _FakeCheckpointer:
        def restore(self, _dir):
            return {"params": _fake_bc_params()}

    monkeypatch.setattr(cv.ocp, "StandardCheckpointer", lambda: _FakeCheckpointer())

    out = tmp_path / "bc_real.pt"
    cv.convert(checkpoint_dir="ignored", out=str(out))

    payload = torch.load(out, weights_only=False)
    assert "meta" in payload, "BC converter must write a 'meta' block into the .pt file"
    assert payload["meta"]["fen_tokenizer_fingerprint"] == canonical_fen_tokenizer_fingerprint()
    assert payload["meta"]["action_vocab_fingerprint"] == canonical_action_vocab_fingerprint()

    sidecar = json.loads(out.with_suffix(".meta.json").read_text())
    assert sidecar["fen_tokenizer_fingerprint"] == canonical_fen_tokenizer_fingerprint()
    assert sidecar["action_vocab_fingerprint"] == canonical_action_vocab_fingerprint()


def test_contract_constants_are_wellformed():
    assert DM_AV_CONTRACT.name == "dm_av_v1"
    assert DM_AV_CONTRACT.checkpoint_family == "dm_action_value"
    assert DM_AV_CONTRACT.require_fingerprints is True
    assert "value_head" in DM_AV_CONTRACT.allowed_new_modules
    assert "dm.output_linear" in DM_AV_CONTRACT.allowed_new_modules

    assert BC_CONTRACT.name == "bc_with_dm_av_move_input_init_v1"
    assert BC_CONTRACT.checkpoint_family == "dm_behavioral_cloning"
    assert BC_CONTRACT.require_fingerprints is True
    assert BC_CONTRACT.allowed_new_modules == ("value_head",)

    bc_rows = {(r["param"], r["start"], r["end"]) for r in BC_CONTRACT.allowed_new_rows}
    assert ("dm.token_embedding.weight", 1968, 1971) in bc_rows
    assert ("dm.output_linear.weight", 1968, 1971) in bc_rows
    assert ("dm.output_linear.bias", 1968, 1971) in bc_rows

    assert DM_AV_CONTRACT_LEGACY.require_fingerprints is False
    assert "missing_fingerprint" in DM_AV_CONTRACT_LEGACY.allowed_warning_tags
    assert BC_CONTRACT_LEGACY.require_fingerprints is False
    assert "missing_fingerprint" in BC_CONTRACT_LEGACY.allowed_warning_tags


def test_verify_marks_equal_and_unequal_spans():
    from src.reasoning.warmstart_provenance import ProvenanceReport

    src_tensor = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    tgt_tensor = torch.zeros(12, 2, dtype=torch.float32)
    tgt_tensor[:10] = src_tensor

    report = ProvenanceReport(
        contract_version="test_v1",
        checkpoint_family="dm_behavioral_cloning",
        sources={"bc": {"path": "synthetic://bc", "sha256": "na"}},
        params={
            "foo.weight": {
                "status": "partial_copy",
                "target_shape": [12, 2],
                "spans": [
                    {
                        "target_start": 0,
                        "target_end": 10,
                        "source": "bc",
                        "source_key": "foo.weight",
                        "source_start": 0,
                        "source_end": 10,
                        "semantic_role": "copied",
                    },
                    {
                        "target_start": 10,
                        "target_end": 12,
                        "source": "new_init",
                        "semantic_role": "new",
                    },
                ],
            }
        },
    )
    loaded_sources = {"bc": {"foo.weight": src_tensor}}
    target_state = {"foo.weight": tgt_tensor}
    report.verify(target_state=target_state, loaded_sources=loaded_sources)
    spans = report.params["foo.weight"]["spans"]
    assert spans[0]["verified_equal"] is True
    assert spans[1]["verified_not_equal_to_any_source"] is True

    tgt_tensor[0, 0] = 999.0
    report.verify(target_state=target_state, loaded_sources=loaded_sources)
    assert report.params["foo.weight"]["spans"][0]["verified_equal"] is False
