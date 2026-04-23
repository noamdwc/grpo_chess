# Warmstart Provenance Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a provenance inspector + pytest suite that verifies, for every parameter of a warmstarted `ReasoningModel`, where each weight came from — and fix the BC loader so its contract actually holds.

**Architecture:** Loader records provenance claims at assignment time into a structured `ProvenanceReport`. Test calls `verify()` (numeric tensor comparison) and `assert_contract()` (allowlist check against a frozen `WarmstartContract`). BC loader is fixed to pull move-as-input embedding rows `[31, 1968)` from a second DM-AV checkpoint; FEN rows `[0, 31)` still come from BC. Checkpoint converters stamp canonical FEN/action vocab fingerprints so a vocabulary swap between BC and DM-AV is caught at load time.

**Tech Stack:** PyTorch 2.x, pytest, `dataclasses`, `hashlib`. Runs via `~/miniconda3/envs/grpo_chess/bin/python`. All paths relative to `/Users/noamc/repos/grpo_chess`.

**Spec:** `docs/superpowers/specs/2026-04-22-warmstart-provenance-test-design.md`.

---

## File structure

- **Create** `src/reasoning/warmstart_provenance.py` — `ProvenanceReport`, `WarmstartContract`, contract constants, canonical fingerprint helpers.
- **Modify** `src/reasoning/model.py` — `ReasoningModelConfig.dm_av_embedding_source` field, BC loader fix, fingerprint verification, provenance emission, `from_checkpoint` classmethod.
- **Modify** `src/dm_port/convert_jax.py` — stamp fingerprints into the saved `torch.save` payload and sidecar JSON.
- **Modify** `src/dm_port/convert_behavioral_cloning.py` — same.
- **Create** `tests/test_warmstart_provenance.py` — seven pytest functions.
- **Create** `tests/conftest.py` additions (only if not already present) — `--bc-checkpoint` / `--dm-av-checkpoint` CLI options for the smoke test.
- **Create** `docs/reasoning_grpo/warmstart_provenance.md` — one-page operator note.
- **Modify** `docs/reasoning_grpo/failure_modes.md` — append an entry referencing the contract.

---

## Task 1: Canonical fingerprints module

**Files:**
- Create: `src/reasoning/warmstart_provenance.py`
- Test: `tests/test_warmstart_provenance.py`

Fingerprints are stable SHA-256 hashes computed over the sorted FEN character → ID mapping and the ordered UCI move list, read from the authoritative `searchless_chess` submodule. They identify vocabulary compatibility in one string.

- [ ] **Step 1: Write failing test for fingerprint helper**

Create `tests/test_warmstart_provenance.py`:

```python
"""Tests for reasoning-model warmstart provenance (spec: 2026-04-22-warmstart-provenance-test-design.md)."""
from __future__ import annotations

from src.reasoning.warmstart_provenance import (
    canonical_fen_tokenizer_fingerprint,
    canonical_action_vocab_fingerprint,
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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_fingerprints_are_deterministic_and_disagree -v
```

Expected: `ModuleNotFoundError: No module named 'src.reasoning.warmstart_provenance'`.

- [ ] **Step 3: Implement the helpers**

Create `src/reasoning/warmstart_provenance.py`:

```python
"""Warmstart provenance records and contract for ReasoningModel.

See docs/superpowers/specs/2026-04-22-warmstart-provenance-test-design.md for
the full contract. This module only defines data structures and pure
helpers. The loader in src/reasoning/model.py is what fills the report in.
"""
from __future__ import annotations

import hashlib


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_fen_tokenizer_fingerprint() -> str:
    """Hash of the canonical FEN character->ID mapping (source of truth:
    searchless_chess/src/tokenizer.py::_CHARACTERS)."""
    from src.searchless_chess_imports import tokenizer as _tok

    characters = list(_tok._CHARACTERS)
    payload = "\n".join(f"{i}\t{c}" for i, c in enumerate(characters)).encode("utf-8")
    return _sha256(payload)


def canonical_action_vocab_fingerprint() -> str:
    """Hash of the canonical UCI action vocabulary (source of truth:
    searchless_chess/src/utils.py::MOVE_TO_ACTION)."""
    from src.searchless_chess_imports import utils as _utils

    move_to_action = dict(_utils.MOVE_TO_ACTION)
    ordered = sorted(move_to_action.items(), key=lambda kv: kv[1])
    payload = "\n".join(f"{idx}\t{move}" for move, idx in ordered).encode("utf-8")
    return _sha256(payload)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_fingerprints_are_deterministic_and_disagree -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): canonical FEN and action vocab fingerprints"
```

---

## Task 2: Stamp fingerprints into BC and DM-AV converter outputs

**Files:**
- Modify: `src/dm_port/convert_behavioral_cloning.py`
- Modify: `src/dm_port/convert_jax.py`
- Test: `tests/test_warmstart_provenance.py`

The converters must write the canonical fingerprints into both the `torch.save` payload (under `meta`) and the sidecar `.meta.json`. Existing on-disk checkpoints do NOT have these — they're handled by the legacy contract variant later.

- [ ] **Step 1: Add the synthetic-checkpoint test helper (not a converter test)**

Append to `tests/test_warmstart_provenance.py`. Note clearly that this helper is for later tests — it stamps fingerprints itself and is NOT a validation of the real converters.

```python
import json
import subprocess
from pathlib import Path

import torch

from src.reasoning.warmstart_provenance import (
    canonical_fen_tokenizer_fingerprint,
    canonical_action_vocab_fingerprint,
)


def _build_synthetic_bc_ckpt(path: Path, embedding_dim: int = 32, num_layers: int = 2) -> None:
    """Build a tiny BC-shaped torch .pt checkpoint for tests (bypasses JAX/Orbax).

    IMPORTANT: this helper stamps fingerprints itself. It is NOT a test of the
    real `convert_behavioral_cloning.convert()` function — that is covered by
    `test_bc_converter_stamps_fingerprints` below.
    """
    from src.dm_port.transformer import DMTransformerConfig, DMTransformer

    cfg = DMTransformerConfig(
        vocab_size=31, output_size=1968, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=4, max_sequence_length=79,
    )
    model = DMTransformer(cfg)
    sd = model.state_dict()
    meta = {
        "family": "dm_behavioral_cloning",
        "input_vocab_size": 31,
        "output_size": 1968,
        "positional_length": 79,
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
```

- [ ] **Step 2: Run to confirm the helper test passes**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_synthetic_bc_helper_stamps_fingerprints -v
```

Expected: PASS. If it fails, the fingerprint functions regressed.

- [ ] **Step 2b: Write failing test for the *real* BC converter**

Append to `tests/test_warmstart_provenance.py`. This monkeypatches Orbax so the test can run without a real Orbax checkpoint on disk, and drives `convert_behavioral_cloning.convert()` end-to-end.

```python
def _fake_bc_params(embedding_dim: int = 256):
    """Build a fake Orbax 'params' tree matching the layout the BC converter reads."""
    import numpy as np
    def _rand(*shape):
        return np.random.RandomState(0).randn(*shape).astype(np.float32)
    params = {
        "Embed_0": {"embedding": {"value": _rand(31, embedding_dim)}},
        "Embed_1": {"embedding": {"value": _rand(79, embedding_dim)}},
        "LayerNorm_16": {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)},
        "Dense_24": {"kernel": {"value": _rand(embedding_dim, 1968)}, "bias": _rand(1968)},
    }
    for i in range(8):
        params[f"LayerNorm_{2 * i}"]     = {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)}
        params[f"LayerNorm_{2 * i + 1}"] = {"scale": _rand(embedding_dim), "bias": _rand(embedding_dim)}
        params[f"MultiHeadDotProductAttention_{i}"] = {
            "query": {"kernel": _rand(embedding_dim, embedding_dim)},
            "key":   {"kernel": _rand(embedding_dim, embedding_dim)},
            "value": {"kernel": _rand(embedding_dim, embedding_dim)},
            "out":   {"kernel": _rand(embedding_dim, embedding_dim)},
        }
        params[f"Dense_{3 * i}"]     = {"kernel": {"value": _rand(embedding_dim, embedding_dim * 4)}}
        params[f"Dense_{3 * i + 1}"] = {"kernel": {"value": _rand(embedding_dim, embedding_dim * 4)}}
        params[f"Dense_{3 * i + 2}"] = {"kernel": {"value": _rand(embedding_dim * 4, embedding_dim)}}
    return params


def test_bc_converter_stamps_fingerprints(tmp_path, monkeypatch):
    """End-to-end check that the real BC converter writes fingerprints."""
    from src.dm_port import convert_behavioral_cloning as cv

    class _FakeCheckpointer:
        def restore(self, _dir): return {"params": _fake_bc_params()}

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
```

- [ ] **Step 2c: Run to confirm it fails**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_bc_converter_stamps_fingerprints -v
```

Expected: FAIL — converter doesn't stamp fingerprints yet.

- [ ] **Step 3: Modify BC converter to stamp fingerprints**

In `src/dm_port/convert_behavioral_cloning.py`, replace the `torch.save` and `meta` block (currently at lines 82-89) with:

```python
from src.reasoning.warmstart_provenance import (
    canonical_fen_tokenizer_fingerprint,
    canonical_action_vocab_fingerprint,
)

# ... (inside convert(), after the state_dict is fully assembled) ...
meta = {
    "family": "dm_behavioral_cloning",
    "input_vocab_size": config.vocab_size,
    "output_size": config.output_size,
    "positional_length": config.max_sequence_length,
    "fen_tokenizer_fingerprint": canonical_fen_tokenizer_fingerprint(),
    "action_vocab_fingerprint": canonical_action_vocab_fingerprint(),
}
torch.save({"state_dict": state_dict, "config": config.__dict__, "meta": meta}, out_path)
out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
print(f"Wrote {out_path}")
```

Make the import a top-level import (per global rule: imports at top of file).

- [ ] **Step 4: Modify DM-AV converter to stamp fingerprints**

In `src/dm_port/convert_jax.py`, locate the `torch.save` call (around line 222) and the sidecar-JSON write that follows. Apply the same change: add the two fingerprints to the `meta` dict and include `meta` in the `torch.save` payload. Import at the top:

```python
from src.reasoning.warmstart_provenance import (
    canonical_fen_tokenizer_fingerprint,
    canonical_action_vocab_fingerprint,
)
```

And update the save block so the saved payload dict is `{"state_dict": state_dict, "config": config.__dict__, "meta": meta}` with the two fingerprint fields in `meta`.

- [ ] **Step 5: Syntax-check both converters**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m py_compile src/dm_port/convert_behavioral_cloning.py src/dm_port/convert_jax.py
```

Expected: no output (success).

- [ ] **Step 6: Run the test**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dm_port/convert_behavioral_cloning.py src/dm_port/convert_jax.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): stamp vocab fingerprints into BC and DM-AV converters"
```

---

## Task 3: ProvenanceReport, WarmstartContract, and contract constants

**Files:**
- Modify: `src/reasoning/warmstart_provenance.py`
- Test: `tests/test_warmstart_provenance.py`

Pure data structures. No loader wiring yet.

- [ ] **Step 1: Write failing test for contract constants**

Append:

```python
from src.reasoning.warmstart_provenance import (
    WarmstartContract,
    DM_AV_CONTRACT,
    BC_CONTRACT,
    DM_AV_CONTRACT_LEGACY,
    BC_CONTRACT_LEGACY,
)


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
    assert ("dm.output_linear.weight",   1968, 1971) in bc_rows
    assert ("dm.output_linear.bias",     1968, 1971) in bc_rows

    assert DM_AV_CONTRACT_LEGACY.require_fingerprints is False
    assert "missing_fingerprint" in DM_AV_CONTRACT_LEGACY.allowed_warning_tags
    assert BC_CONTRACT_LEGACY.require_fingerprints is False
    assert "missing_fingerprint" in BC_CONTRACT_LEGACY.allowed_warning_tags
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_contract_constants_are_wellformed -v
```

Expected: `ImportError` for the contract names.

- [ ] **Step 3: Implement the data structures and constants**

Append to `src/reasoning/warmstart_provenance.py`:

```python
import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WarmstartContract:
    name: str
    checkpoint_family: str
    allowed_new_modules: tuple[str, ...]
    allowed_new_rows: tuple[dict, ...]
    require_fingerprints: bool = True
    allowed_warning_tags: tuple[str, ...] = ()


DM_AV_CONTRACT = WarmstartContract(
    name="dm_av_v1",
    checkpoint_family="dm_action_value",
    allowed_new_modules=("value_head", "dm.output_linear"),
    allowed_new_rows=(
        {"param": "dm.token_embedding.weight", "start": 1968, "end": 1971},
    ),
    require_fingerprints=True,
    allowed_warning_tags=(),
)

BC_CONTRACT = WarmstartContract(
    name="bc_with_dm_av_move_input_init_v1",
    checkpoint_family="dm_behavioral_cloning",
    allowed_new_modules=("value_head",),
    allowed_new_rows=(
        {"param": "dm.token_embedding.weight", "start": 1968, "end": 1971},
        {"param": "dm.output_linear.weight",   "start": 1968, "end": 1971},
        {"param": "dm.output_linear.bias",     "start": 1968, "end": 1971},
    ),
    require_fingerprints=True,
    allowed_warning_tags=(),
)

DM_AV_CONTRACT_LEGACY = dataclasses.replace(
    DM_AV_CONTRACT,
    require_fingerprints=False,
    allowed_warning_tags=("missing_fingerprint",),
)

BC_CONTRACT_LEGACY = dataclasses.replace(
    BC_CONTRACT,
    require_fingerprints=False,
    allowed_warning_tags=("missing_fingerprint",),
)


@dataclass
class ProvenanceReport:
    contract_version: str
    checkpoint_family: str
    sources: dict[str, dict]
    warnings: list[dict] = field(default_factory=list)
    vocabularies: dict[str, Any] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_json(self) -> str:
        import json
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)
```

(We add `verify` and `assert_contract` in later tasks — keeping this task focused on the schema.)

- [ ] **Step 4: Run test to verify it passes**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): ProvenanceReport, WarmstartContract, contract constants"
```

---

## Task 4: Implement `ProvenanceReport.verify`

**Files:**
- Modify: `src/reasoning/warmstart_provenance.py`
- Test: `tests/test_warmstart_provenance.py`

`verify()` walks every param span, compares target tensor slice to the claimed source tensor slice with `torch.equal`, and writes `verified_equal: bool` into the span. For `new_init` entries, records `verified_not_equal_to_any_source` (weak sanity check).

- [ ] **Step 1: Write failing test for `verify()`**

Append:

```python
import torch


def test_verify_marks_equal_and_unequal_spans():
    from src.reasoning.warmstart_provenance import ProvenanceReport

    src_tensor = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    tgt_tensor = torch.zeros(12, 2, dtype=torch.float32)
    tgt_tensor[:10] = src_tensor
    # rows 10-12 are left at zero == "new_init"

    report = ProvenanceReport(
        contract_version="test_v1",
        checkpoint_family="dm_behavioral_cloning",
        sources={"bc": {"path": "synthetic://bc", "sha256": "na"}},
        params={
            "foo.weight": {
                "status": "partial_copy",
                "target_shape": [12, 2],
                "spans": [
                    {"target_start": 0, "target_end": 10, "source": "bc",
                     "source_key": "foo.weight", "source_start": 0, "source_end": 10,
                     "semantic_role": "copied"},
                    {"target_start": 10, "target_end": 12, "source": "new_init",
                     "semantic_role": "new"},
                ],
            }
        },
    )
    loaded_sources = {"bc": {"foo.weight": src_tensor}}
    target_state = {"foo.weight": tgt_tensor}
    report.verify(target_state=target_state, loaded_sources=loaded_sources)
    spans = report.params["foo.weight"]["spans"]
    assert spans[0]["verified_equal"] is True
    assert spans[1]["verified_not_equal_to_any_source"] is True  # zeros != arange slice

    # Break the copy, re-verify: first span should flip.
    tgt_tensor[0, 0] = 999.0
    report.verify(target_state=target_state, loaded_sources=loaded_sources)
    assert report.params["foo.weight"]["spans"][0]["verified_equal"] is False
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_verify_marks_equal_and_unequal_spans -v
```

Expected: `AttributeError: 'ProvenanceReport' object has no attribute 'verify'`.

- [ ] **Step 3: Implement `verify`**

Add method to `ProvenanceReport` in `src/reasoning/warmstart_provenance.py`:

```python
    def verify(self, target_state: dict, loaded_sources: dict[str, dict]) -> "ProvenanceReport":
        """Walk every span, compare claimed source slice to target slice.
        Mutates span dicts in place with `verified_equal` /
        `verified_not_equal_to_any_source`. Returns self for chaining."""
        import torch as _torch
        for param_name, pspec in self.params.items():
            tgt = target_state[param_name]
            if pspec["status"] == "new_init":
                pspec["verified_not_equal_to_any_source"] = _verify_new_init(tgt, loaded_sources)
                continue
            if pspec["status"] == "full_copy":
                src_key = pspec["source_key"]
                src = loaded_sources[pspec["source"]][src_key]
                pspec["verified_equal"] = bool(_torch.equal(tgt, src))
                continue
            for span in pspec.get("spans", []):
                ts, te = span["target_start"], span["target_end"]
                tgt_slice = tgt[ts:te]
                if span["source"] == "new_init":
                    span["verified_not_equal_to_any_source"] = _verify_new_init(tgt_slice, loaded_sources)
                    continue
                if span["source"] == "copy_of_last_pretrained_row":
                    src = loaded_sources[_find_source_with_key(loaded_sources, span["source_key"])][span["source_key"]]
                    ss, se = span["source_start"], span["source_end"]
                    expected = src[ss:se].expand_as(tgt_slice)
                    span["verified_equal"] = bool(_torch.equal(tgt_slice, expected))
                    continue
                src = loaded_sources[span["source"]][span["source_key"]]
                ss, se = span["source_start"], span["source_end"]
                span["verified_equal"] = bool(_torch.equal(tgt_slice, src[ss:se]))
        return self


def _find_source_with_key(loaded_sources: dict[str, dict], key: str) -> str:
    for name, sd in loaded_sources.items():
        if key in sd:
            return name
    raise KeyError(f"No loaded source provides key {key!r}")


def _verify_new_init(tensor, loaded_sources: dict[str, dict]) -> bool:
    """Weak sanity check: target must not bitwise-equal any source tensor of
    compatible shape. Not load-bearing — the real guarantee is the contract
    allowlist. See the spec's 'Warning and fingerprint semantics' section."""
    import torch as _torch
    for sd in loaded_sources.values():
        for src in sd.values():
            if src.shape == tensor.shape and _torch.equal(tensor, src):
                return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): ProvenanceReport.verify for numeric span checks"
```

---

## Task 5: Implement `ProvenanceReport.assert_contract`

**Files:**
- Modify: `src/reasoning/warmstart_provenance.py`
- Test: `tests/test_warmstart_provenance.py`

Raises `AssertionError` with a structured, readable message if: any span failed verification, any param is outside the contract's allowlist, checkpoint families mismatch, or a warning's tag is not in `allowed_warning_tags`.

- [ ] **Step 1: Write failing test**

Append:

```python
import pytest


def _minimal_ok_report_for_bc() -> "ProvenanceReport":
    from src.reasoning.warmstart_provenance import ProvenanceReport
    return ProvenanceReport(
        contract_version="bc_with_dm_av_move_input_init_v1",
        checkpoint_family="dm_behavioral_cloning",
        sources={"bc": {"path": "synthetic://bc", "sha256": "na"},
                 "dm_av": {"path": "synthetic://dm_av", "sha256": "na"}},
        params={
            "dm.token_embedding.weight": {
                "status": "partial_copy", "target_shape": [1971, 32],
                "spans": [
                    {"target_start": 0,    "target_end": 31,   "source": "bc",    "source_key": "token_embedding.weight", "source_start": 0, "source_end": 31, "semantic_role": "fen_input_embedding",    "verified_equal": True},
                    {"target_start": 31,   "target_end": 1968, "source": "dm_av", "source_key": "token_embedding.weight", "source_start": 31, "source_end": 1968, "semantic_role": "move_input_embedding", "verified_equal": True},
                    {"target_start": 1968, "target_end": 1971, "source": "new_init", "semantic_role": "reasoning_structural_tokens", "verified_not_equal_to_any_source": True},
                ],
            },
            "dm.output_linear.weight": {
                "status": "partial_copy", "target_shape": [1971, 32],
                "spans": [
                    {"target_start": 0,    "target_end": 1968, "source": "bc", "source_key": "output_linear.weight", "source_start": 0, "source_end": 1968, "semantic_role": "move_output_head", "verified_equal": True},
                    {"target_start": 1968, "target_end": 1971, "source": "new_init", "semantic_role": "reasoning_structural_output_rows", "verified_not_equal_to_any_source": True},
                ],
            },
            "dm.output_linear.bias": {
                "status": "partial_copy", "target_shape": [1971],
                "spans": [
                    {"target_start": 0,    "target_end": 1968, "source": "bc", "source_key": "output_linear.bias", "source_start": 0, "source_end": 1968, "semantic_role": "move_output_head_bias", "verified_equal": True},
                    {"target_start": 1968, "target_end": 1971, "source": "new_init", "semantic_role": "reasoning_structural_output_bias", "verified_not_equal_to_any_source": False},
                ],
            },
            "value_head.0.weight": {"status": "new_init", "target_shape": [256, 32], "semantic_role": "value_head", "verified_not_equal_to_any_source": True},
        },
        warnings=[],
    )


def test_assert_contract_accepts_valid_bc_report():
    from src.reasoning.warmstart_provenance import BC_CONTRACT
    report = _minimal_ok_report_for_bc()
    report.assert_contract(BC_CONTRACT)  # must not raise


def test_assert_contract_flags_unexpected_new_init_rows():
    from src.reasoning.warmstart_provenance import BC_CONTRACT
    report = _minimal_ok_report_for_bc()
    # Break: replace the middle span (DM-AV source) with new_init
    report.params["dm.token_embedding.weight"]["spans"][1] = {
        "target_start": 31, "target_end": 1968, "source": "new_init",
        "semantic_role": "move_input_embedding", "verified_not_equal_to_any_source": True,
    }
    with pytest.raises(AssertionError) as exc:
        report.assert_contract(BC_CONTRACT)
    msg = str(exc.value)
    assert "unexpected new-init rows in dm.token_embedding.weight" in msg
    assert "[31, 1968)" in msg


def test_assert_contract_rejects_family_mismatch():
    from src.reasoning.warmstart_provenance import DM_AV_CONTRACT
    report = _minimal_ok_report_for_bc()
    with pytest.raises(AssertionError) as exc:
        report.assert_contract(DM_AV_CONTRACT)
    assert "checkpoint_family mismatch" in str(exc.value)


def test_assert_contract_fails_on_unwhitelisted_warning():
    from src.reasoning.warmstart_provenance import BC_CONTRACT, BC_CONTRACT_LEGACY
    report = _minimal_ok_report_for_bc()
    report.warnings = [{"tag": "missing_fingerprint", "message": "no fp", "context": {}}]
    with pytest.raises(AssertionError) as exc:
        report.assert_contract(BC_CONTRACT)
    assert "missing_fingerprint" in str(exc.value)
    # But legacy variant whitelists it:
    report.assert_contract(BC_CONTRACT_LEGACY)  # must not raise
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: the four new tests fail on missing `assert_contract`.

- [ ] **Step 3: Implement `assert_contract`**

Add to `ProvenanceReport`:

```python
    def assert_contract(self, contract: "WarmstartContract") -> None:
        """See spec 'Warning and fingerprint semantics' and 'Per-path contracts'."""
        errors: list[str] = []
        if self.checkpoint_family != contract.checkpoint_family:
            errors.append(
                f"checkpoint_family mismatch: report={self.checkpoint_family!r} "
                f"contract={contract.checkpoint_family!r} ({contract.name})"
            )

        allowed_rows: dict[str, list[tuple[int, int]]] = {}
        for entry in contract.allowed_new_rows:
            allowed_rows.setdefault(entry["param"], []).append((entry["start"], entry["end"]))

        for pname, pspec in self.params.items():
            if pspec["status"] == "full_copy":
                if not pspec.get("verified_equal", False):
                    errors.append(f"verification failed (full_copy): {pname}")
                continue

            if pspec["status"] == "new_init":
                module = pname.rsplit(".", 1)[0]
                if not any(pname == m or module == m or module.startswith(m + ".") or pname.startswith(m + ".") for m in contract.allowed_new_modules):
                    # Also allow exact-module match when the contract names the module itself
                    top = pname.split(".", 1)[0]
                    if top not in contract.allowed_new_modules and module not in contract.allowed_new_modules and not any(pname.startswith(m + ".") for m in contract.allowed_new_modules):
                        errors.append(f"unexpected new_init parameter: {pname} "
                                      f"(not in allowed_new_modules={list(contract.allowed_new_modules)})")
                continue

            # partial_copy
            for span in pspec.get("spans", []):
                if span["source"] == "new_init":
                    ts, te = span["target_start"], span["target_end"]
                    param_allowlist = allowed_rows.get(pname, [])
                    if not any(ts >= a and te <= b for (a, b) in param_allowlist):
                        errors.append(
                            f"unexpected new-init rows in {pname}: [{ts}, {te})  "
                            f"(allowed: {param_allowlist}); semantic_role="
                            f"{span.get('semantic_role', '?')}"
                        )
                else:
                    if not span.get("verified_equal", False):
                        ts, te = span["target_start"], span["target_end"]
                        errors.append(f"verification failed: {pname} rows [{ts}, {te}) "
                                      f"claimed from {span['source']}/{span.get('source_key')}")

        for w in self.warnings:
            if w["tag"] not in contract.allowed_warning_tags:
                errors.append(f"unacceptable warning tag={w['tag']!r} message={w['message']!r}")

        if errors:
            header = (
                f"Warmstart contract violation "
                f"(contract={contract.name}, family={contract.checkpoint_family}):\n"
            )
            raise AssertionError(header + "\n".join(f"  - {e}" for e in errors))
```

- [ ] **Step 4: Run tests**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): ProvenanceReport.assert_contract with tagged warnings"
```

---

## Task 6: Add `dm_av_embedding_source` to `ReasoningModelConfig`

**Files:**
- Modify: `src/reasoning/model.py:28-34`
- Test: `tests/test_warmstart_provenance.py`

No loader behavior change yet — just the config field, plus a rejection test when BC is used without the field.

- [ ] **Step 1: Write failing test**

Append:

```python
def test_config_rejects_bc_without_dm_av_source(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    bc = tmp_path / "bc.pt"
    _build_synthetic_bc_ckpt(bc)
    cfg = ReasoningModelConfig(dm_checkpoint=str(bc), dm_av_embedding_source=None, max_seq_len=80)
    with pytest.raises(ValueError) as exc:
        ReasoningModel(cfg)
    assert "dm_av_embedding_source" in str(exc.value)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_config_rejects_bc_without_dm_av_source -v
```

Expected: either `TypeError: unexpected keyword argument` (field missing) or the BC loader proceeds silently (no rejection).

- [ ] **Step 3: Add the config field and validation**

In `src/reasoning/model.py`, replace the `ReasoningModelConfig` dataclass (lines 28-34) with:

```python
@dataclass
class ReasoningModelConfig:
    dm_checkpoint: str
    dm_av_embedding_source: str | None = None
    max_seq_len: int = 120
    value_head_hidden: int = 256
    freeze_body: bool = False
```

In `ReasoningModel.__init__`, immediately after the `inspect_checkpoint` call (line 49 in the current file), add:

```python
        if checkpoint_info.family == "dm_behavioral_cloning" and config.dm_av_embedding_source is None:
            raise ValueError(
                "BC warmstart requires ReasoningModelConfig.dm_av_embedding_source "
                "to point at a DM-AV checkpoint. Its move-as-input embedding rows "
                "[31, 1968) are used to populate the reasoning model's "
                "token_embedding rows [31, 1968) (BC's own embedding only supplies "
                "rows [0, 31))."
            )
```

- [ ] **Step 4: Run test**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_config_rejects_bc_without_dm_av_source -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reasoning/model.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): reject BC warmstart without dm_av_embedding_source"
```

---

## Task 7: BC loader fix — merge DM-AV move-input rows

**Files:**
- Modify: `src/reasoning/model.py` (`_load_and_extend_bc_weights`)
- Test: `tests/test_warmstart_provenance.py`

Load the DM-AV checkpoint, pull its `token_embedding.weight[31:1968]` into the reasoning model's rows `[31, 1968)`. FEN rows `[0, 31)` continue to come from BC. Assert DM-AV rows `[0, 31)` are NEVER read.

> **Namespace distinction (important — do not mix these up).** Two different parameter-name conventions appear in this task:
>
> - **Internal (inside `_load_and_extend_bc_weights`):** we work with `self.dm.state_dict()`, so keys are **without** the `dm.` prefix — e.g. `"token_embedding.weight"`, `"output_linear.weight"`, `"output_linear.bias"`, `"pos_embedding.weight"`, `"layers.0.attn.q_proj.weight"`.
> - **External (in the `ProvenanceReport` and `WarmstartContract` allowlists):** we use the full `ReasoningModel.state_dict()` keys, so they carry the `dm.` prefix — e.g. `"dm.token_embedding.weight"`, `"dm.output_linear.weight"`, `"dm.output_linear.bias"`.
>
> The loader code uses the internal names. The report builders in Task 9 use the external names. `verify()` in Task 4 reads target tensors via `target_state[param_name]` where `param_name` is the *external* name (passed from `model.state_dict()`), and reads source tensors from `loaded_sources[source][source_key]` where `source_key` is the *internal* name (as stored in the BC/DM-AV checkpoint's raw `state_dict`, which is flat — it has no `dm.` prefix because the checkpoint was saved from `DMTransformer`, not from `ReasoningModel`). Keep these two namespaces separate throughout.

- [ ] **Step 1: Write failing test that catches the BC bug**

Append:

```python
def _build_synthetic_dm_av_ckpt(path: Path, embedding_dim: int = 32, num_layers: int = 2,
                                fen_fingerprint: str | None = None,
                                action_fingerprint: str | None = None) -> None:
    from src.dm_port.transformer import DMTransformer, DMTransformerConfig
    cfg = DMTransformerConfig(
        vocab_size=1968, output_size=128, embedding_dim=embedding_dim,
        num_layers=num_layers, num_heads=4, max_sequence_length=79,
    )
    model = DMTransformer(cfg)
    sd = model.state_dict()
    # Distinguishable FEN rows: DM-AV rows [0,31) = 2.0 so the test can
    # prove we pull BC's FEN rows, not DM-AV's.
    sd["token_embedding.weight"][:31] = 2.0
    sd["token_embedding.weight"][31:1968] = torch.randn(1968 - 31, embedding_dim) * 0.1
    meta = {
        "family": "dm_action_value",
        "input_vocab_size": 1968, "output_size": 128, "positional_length": 79,
        "fen_tokenizer_fingerprint": fen_fingerprint or canonical_fen_tokenizer_fingerprint(),
        "action_vocab_fingerprint": action_fingerprint or canonical_action_vocab_fingerprint(),
    }
    torch.save({"state_dict": sd, "config": cfg.__dict__, "meta": meta}, path)


def test_bc_loader_merges_dm_av_move_input_rows(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    bc = tmp_path / "bc.pt"
    dm = tmp_path / "dm.pt"
    # BC rows [0,31) = 1.0 so we can verify BC wins for FEN rows.
    _build_synthetic_bc_ckpt(bc)
    bc_payload = torch.load(bc, weights_only=False)
    bc_payload["state_dict"]["token_embedding.weight"][:31] = 1.0
    torch.save(bc_payload, bc)
    _build_synthetic_dm_av_ckpt(dm)

    cfg = ReasoningModelConfig(
        dm_checkpoint=str(bc), dm_av_embedding_source=str(dm), max_seq_len=80,
    )
    model = ReasoningModel(cfg)
    tok = model.dm.token_embedding.weight.detach()

    assert torch.allclose(tok[:31], torch.ones_like(tok[:31])), \
        "FEN rows must come from BC (1.0), not DM-AV (2.0)"
    assert not torch.allclose(tok[:31], torch.full_like(tok[:31], 2.0))

    dm_sd = torch.load(dm, weights_only=False)["state_dict"]
    assert torch.equal(tok[31:1968], dm_sd["token_embedding.weight"][31:1968]), \
        "Move-input rows [31, 1968) must equal DM-AV rows [31, 1968)"
```

- [ ] **Step 2: Run test to confirm it fails on current main**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_bc_loader_merges_dm_av_move_input_rows -v
```

Expected: FAIL — current BC loader doesn't read DM-AV at all; rows `[31, 1968)` are random.

- [ ] **Step 3: Update the BC loader**

In `src/reasoning/model.py`, modify `_load_and_extend_bc_weights`. Before it's called, the caller must pass the DM-AV state dict. Update the `__init__` call site (line 62) from:

```python
            self._load_and_extend_bc_weights(ckpt["state_dict"])
```

to:

```python
            dm_av_ckpt = torch.load(config.dm_av_embedding_source, weights_only=False)
            self._load_and_extend_bc_weights(ckpt["state_dict"], dm_av_ckpt["state_dict"])
```

Then change `_load_and_extend_bc_weights` (starting at line 117):

```python
    def _load_and_extend_bc_weights(self, bc_state_dict: dict, dm_av_state_dict: dict) -> None:
        """Load BC body weights, pulling move-input embedding rows [31, 1968)
        from DM-AV. BC's own embedding only supplies rows [0, 31) (FEN tokens)."""
        own_state = self.dm.state_dict()
        dm_av_embed = dm_av_state_dict["token_embedding.weight"]
        if dm_av_embed.shape[0] != 1968:
            raise RuntimeError(
                f"dm_av_embedding_source token_embedding.weight must have 1968 rows, "
                f"got {dm_av_embed.shape[0]}"
            )
        if dm_av_embed.shape[1] != own_state["token_embedding.weight"].shape[1]:
            raise RuntimeError(
                "dm_av_embedding_source embedding_dim does not match reasoning model"
            )

        loaded_keys: set[str] = set()
        for key, value in bc_state_dict.items():
            if key == "token_embedding.weight":
                own_state[key][: value.shape[0]] = value          # BC rows [0, 31)
                own_state[key][31:1968] = dm_av_embed[31:1968]    # DM-AV action rows
                # DM-AV rows [0, 31) (its FEN embedding) are deliberately NOT pulled.
                loaded_keys.add(key)
            elif key == "pos_embedding.weight":
                own_state[key][: value.shape[0]] = value
                last = value[-1]
                for i in range(value.shape[0], own_state[key].shape[0]):
                    own_state[key][i] = last
                loaded_keys.add(key)
            elif key == "output_linear.weight":
                own_state[key][: value.shape[0]] = value
                loaded_keys.add(key)
            elif key == "output_linear.bias":
                own_state[key][: value.shape[0]] = value
                loaded_keys.add(key)
            else:
                if key not in own_state:
                    raise RuntimeError(f"Unexpected checkpoint key: {key}")
                if own_state[key].shape == value.shape:
                    own_state[key] = value
                    loaded_keys.add(key)
                else:
                    raise RuntimeError(
                        f"Shape mismatch loading {key}: own={own_state[key].shape} ckpt={value.shape}"
                    )
        expected = set(own_state) - {"output_linear.weight", "output_linear.bias", "token_embedding.weight", "pos_embedding.weight"}
        missing_keys = sorted(expected - loaded_keys)
        if missing_keys:
            preview = ", ".join(missing_keys[:5])
            raise RuntimeError(f"Missing checkpoint keys: {preview}")
        self.dm.load_state_dict(own_state)
```

- [ ] **Step 4: Run the regression test**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_bc_loader_merges_dm_av_move_input_rows -v
```

Expected: PASS.

- [ ] **Step 5: Run the full reasoning test suite**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py tests/test_zero_think_bc_parity.py -v
```

Expected: all PASS (or pre-existing xfails unchanged). Investigate any new failures before committing.

- [ ] **Step 6: Commit**

```bash
git add src/reasoning/model.py tests/test_warmstart_provenance.py
git commit -m "fix(warmstart): BC loader pulls move-input embedding rows from DM-AV

BC's token embedding only covers FEN tokens (31 rows). Previously,
rows [31, 1968) of the reasoning model's input embedding were left
at random init under BC warmstart — silently. Now pulled from DM-AV's
embedding rows [31, 1968) (which DM-AV trained by feeding move
indices as input tokens). DM-AV's own FEN rows [0, 31) are
deliberately NOT used; BC's FEN embedding is authoritative."
```

---

## Task 8: Fingerprint verification at load time

**Files:**
- Modify: `src/reasoning/model.py` (BC loader entry point and DM-AV loader entry point)
- Test: `tests/test_warmstart_provenance.py`

If BC or DM-AV carries a `meta.fen_tokenizer_fingerprint` or `meta.action_vocab_fingerprint` that mismatches the canonical fingerprint, raise `ValueError` at load time. If a fingerprint is absent, emit a warning record (the report collects these — implemented as a list on the `ReasoningModel` instance until Task 9 wires it into a full `ProvenanceReport`).

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_action_vocab_swap_detected(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    bc = tmp_path / "bc.pt"
    dm = tmp_path / "dm.pt"
    _build_synthetic_bc_ckpt(bc)
    _build_synthetic_dm_av_ckpt(dm, action_fingerprint="sha256:deadbeef")
    cfg = ReasoningModelConfig(dm_checkpoint=str(bc), dm_av_embedding_source=str(dm), max_seq_len=80)
    with pytest.raises(ValueError) as exc:
        ReasoningModel(cfg)
    msg = str(exc.value).lower()
    assert "action_vocab_fingerprint" in msg and "mismatch" in msg


def test_fen_vocab_swap_detected(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    bc = tmp_path / "bc.pt"
    dm = tmp_path / "dm.pt"
    _build_synthetic_bc_ckpt(bc)
    # Rewrite BC's fingerprint to something wrong:
    payload = torch.load(bc, weights_only=False)
    payload["meta"]["fen_tokenizer_fingerprint"] = "sha256:bad"
    torch.save(payload, bc)
    _build_synthetic_dm_av_ckpt(dm)
    cfg = ReasoningModelConfig(dm_checkpoint=str(bc), dm_av_embedding_source=str(dm), max_seq_len=80)
    with pytest.raises(ValueError) as exc:
        ReasoningModel(cfg)
    msg = str(exc.value).lower()
    assert "fen_tokenizer_fingerprint" in msg and "mismatch" in msg
```

- [ ] **Step 2: Run to confirm failure**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_action_vocab_swap_detected tests/test_warmstart_provenance.py::test_fen_vocab_swap_detected -v
```

Expected: FAIL (no fingerprint check exists yet).

- [ ] **Step 3: Add a fingerprint verification helper**

Add to `src/reasoning/warmstart_provenance.py`:

```python
def verify_checkpoint_fingerprints(
    label: str, ckpt: dict, warnings_out: list[dict]
) -> None:
    """Verify checkpoint's fingerprints against canonical.

    Mismatch -> raise ValueError (hard fatal).
    Missing  -> append a {'tag': 'missing_fingerprint', ...} warning to
                warnings_out.
    """
    meta = ckpt.get("meta", {}) or {}
    canonical_fen = canonical_fen_tokenizer_fingerprint()
    canonical_act = canonical_action_vocab_fingerprint()

    fen = meta.get("fen_tokenizer_fingerprint")
    act = meta.get("action_vocab_fingerprint")

    if fen is not None and fen != canonical_fen:
        raise ValueError(
            f"{label} fen_tokenizer_fingerprint mismatch: "
            f"checkpoint={fen} canonical={canonical_fen}. "
            "This means the checkpoint was trained against a different FEN "
            "tokenizer than the reasoning model expects — token IDs 0-30 "
            "would denote different characters. Do not merge."
        )
    if act is not None and act != canonical_act:
        raise ValueError(
            f"{label} action_vocab_fingerprint mismatch: "
            f"checkpoint={act} canonical={canonical_act}. "
            "This means the checkpoint was trained against a different UCI "
            "move ordering — action ID k would denote a different move. "
            "Do not merge."
        )
    if fen is None:
        warnings_out.append({
            "tag": "missing_fingerprint",
            "message": f"{label} checkpoint lacks fen_tokenizer_fingerprint",
            "context": {"label": label, "field": "fen_tokenizer_fingerprint"},
        })
    if act is None:
        warnings_out.append({
            "tag": "missing_fingerprint",
            "message": f"{label} checkpoint lacks action_vocab_fingerprint",
            "context": {"label": label, "field": "action_vocab_fingerprint"},
        })
```

- [ ] **Step 4: Wire into the loader**

In `src/reasoning/model.py`, add `self._warmstart_warnings: list[dict] = []` at the top of `__init__`. Then, right after the DM-AV checkpoint is loaded in the BC path:

```python
from src.reasoning.warmstart_provenance import verify_checkpoint_fingerprints
# (imports at top of file)

# ... inside __init__, BC branch ...
            dm_av_ckpt = torch.load(config.dm_av_embedding_source, weights_only=False)
            verify_checkpoint_fingerprints("bc", ckpt, self._warmstart_warnings)
            verify_checkpoint_fingerprints("dm_av", dm_av_ckpt, self._warmstart_warnings)
            self._load_and_extend_bc_weights(ckpt["state_dict"], dm_av_ckpt["state_dict"])
```

And for the DM-AV branch (line 60 in current file), right before `_load_and_extend_dm_weights`:

```python
        if checkpoint_info.family == "dm_action_value":
            verify_checkpoint_fingerprints("dm_av", ckpt, self._warmstart_warnings)
            self._load_and_extend_dm_weights(ckpt["state_dict"])
```

- [ ] **Step 5: Run all provenance tests**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reasoning/model.py src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): fingerprint verification at checkpoint load"
```

---

## Task 9: Loader emits a full `ProvenanceReport`; add `from_checkpoint`

**Files:**
- Modify: `src/reasoning/model.py`
- Modify: `src/reasoning/warmstart_provenance.py` (add a small helper to build the expected schema for each family)
- Test: `tests/test_warmstart_provenance.py`

The two `_load_and_extend_*` methods now build a `ProvenanceReport` alongside their assignments. `ReasoningModel.from_checkpoint(config)` exposes it.

- [ ] **Step 1: Write failing test for `from_checkpoint`**

Append:

```python
def test_dm_av_warmstart_provenance_synthetic(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.warmstart_provenance import DM_AV_CONTRACT, select_contract_for_family

    dm = tmp_path / "dm.pt"
    _build_synthetic_dm_av_ckpt(dm)
    cfg = ReasoningModelConfig(dm_checkpoint=str(dm), max_seq_len=80)

    model, report = ReasoningModel.from_checkpoint(cfg)

    # Guard against wrong-family detection:
    selected = select_contract_for_family(report.checkpoint_family)
    assert selected.name == "dm_av_v1"

    loaded_sources = {"dm_av": torch.load(dm, weights_only=False)["state_dict"]}
    report.verify(target_state=model.state_dict(), loaded_sources=loaded_sources)
    report.assert_contract(DM_AV_CONTRACT)


def test_bc_warmstart_provenance_synthetic(tmp_path):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.warmstart_provenance import BC_CONTRACT, select_contract_for_family

    bc = tmp_path / "bc.pt"
    dm = tmp_path / "dm.pt"
    _build_synthetic_bc_ckpt(bc)
    # Distinguishable FEN rows.
    bc_payload = torch.load(bc, weights_only=False)
    bc_payload["state_dict"]["token_embedding.weight"][:31] = 1.0
    torch.save(bc_payload, bc)
    _build_synthetic_dm_av_ckpt(dm)

    cfg = ReasoningModelConfig(dm_checkpoint=str(bc), dm_av_embedding_source=str(dm), max_seq_len=80)
    model, report = ReasoningModel.from_checkpoint(cfg)
    assert select_contract_for_family(report.checkpoint_family).name == "bc_with_dm_av_move_input_init_v1"

    loaded_sources = {
        "bc":    torch.load(bc, weights_only=False)["state_dict"],
        "dm_av": torch.load(dm, weights_only=False)["state_dict"],
    }
    report.verify(target_state=model.state_dict(), loaded_sources=loaded_sources)
    report.assert_contract(BC_CONTRACT)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_dm_av_warmstart_provenance_synthetic tests/test_warmstart_provenance.py::test_bc_warmstart_provenance_synthetic -v
```

Expected: `AttributeError: type object 'ReasoningModel' has no attribute 'from_checkpoint'`.

- [ ] **Step 3: Add schema builders**

Append to `src/reasoning/warmstart_provenance.py`:

```python
def select_contract_for_family(family: str) -> "WarmstartContract":
    if family == "dm_action_value":
        return DM_AV_CONTRACT
    if family == "dm_behavioral_cloning":
        return BC_CONTRACT
    raise ValueError(f"No default contract for checkpoint family {family!r}")


def build_dm_av_report(sources: dict[str, dict], warnings: list[dict],
                       max_seq_len: int, pos_len: int,
                       target_state: dict) -> "ProvenanceReport":
    """Claim provenance for the DM-AV warmstart path. `target_state` is
    `model.state_dict()` of the just-built reasoning model — used only so
    each param entry can carry its real `target_shape`. The loader must
    actually perform the copies described here; verify() checks."""
    def _shape(name: str) -> list[int]:
        return list(target_state[name].shape)

    report = ProvenanceReport(
        contract_version=DM_AV_CONTRACT.name,
        checkpoint_family="dm_action_value",
        sources=sources,
        warnings=list(warnings),
        vocabularies={
            "fen_tokenizer_fingerprint": canonical_fen_tokenizer_fingerprint(),
            "action_vocab_fingerprint":  canonical_action_vocab_fingerprint(),
            "num_actions": 1968, "fen_vocab_size": 31,
            "source": "searchless_chess/src/{tokenizer,utils}.py",
        },
        params={},
    )
    # Token embedding: DM-AV rows [0, 1968) copied in; [1968, 1971) new.
    report.params["dm.token_embedding.weight"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.token_embedding.weight"),
        "spans": [
            {"target_start": 0, "target_end": 1968,
             "source": "dm_av", "source_key": "token_embedding.weight",
             "source_start": 0, "source_end": 1968,
             "semantic_role": "full_pretrained_embedding",
             "reason": "DM-AV's 1968-row embedding covers both FEN tokens (0-30) and move-input tokens (31-1967)."},
            {"target_start": 1968, "target_end": 1971,
             "source": "new_init", "semantic_role": "reasoning_structural_tokens",
             "tokens": ["<think>", "</think>", "<move>"]},
        ],
    }
    # Pos embedding: DM-AV rows [0, pos_len) copied in; [pos_len, max_seq_len) = last row.
    report.params["dm.pos_embedding.weight"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.pos_embedding.weight"),
        "spans": [
            {"target_start": 0, "target_end": pos_len,
             "source": "dm_av", "source_key": "pos_embedding.weight",
             "source_start": 0, "source_end": pos_len,
             "semantic_role": "pretrained_positions"},
            {"target_start": pos_len, "target_end": max_seq_len,
             "source": "copy_of_last_pretrained_row",
             "source_key": "pos_embedding.weight",
             "source_start": pos_len - 1, "source_end": pos_len,
             "semantic_role": "extended_positions",
             "reason": "Last pretrained row copied to avoid cold-start blowup (FM 7.9)."},
        ],
    }
    # Output head fully new-init for DM-AV (128-bucket head incompatible with 1971-move head).
    report.params["dm.output_linear.weight"] = {
        "status": "new_init", "target_shape": _shape("dm.output_linear.weight"),
        "semantic_role": "value_head_unrelated",
        "reason": "DM-AV's 128-bucket head is incompatible with the 1971-move head.",
    }
    report.params["dm.output_linear.bias"] = {
        "status": "new_init", "target_shape": _shape("dm.output_linear.bias"),
        "semantic_role": "value_head_unrelated",
    }
    # Value head: always new.
    for k in ("value_head.0.weight", "value_head.0.bias", "value_head.2.weight", "value_head.2.bias"):
        report.params[k] = {
            "status": "new_init", "target_shape": _shape(k),
            "semantic_role": "value_head",
        }
    # Transformer blocks: full copies from DM-AV. Walk every remaining key.
    for k, t in target_state.items():
        if k in report.params:
            continue
        if not k.startswith("dm.layers.") and not k.startswith("dm.post_ln.") and not k.startswith("dm.pre_ln."):
            continue  # anything else should already be in report.params — protect against silent drops
        # Strip the leading "dm." to get the key inside the DM-AV checkpoint's flat state_dict.
        src_key = k[len("dm."):]
        report.params[k] = {
            "status": "full_copy", "target_shape": list(t.shape),
            "source": "dm_av", "source_key": src_key,
            "semantic_role": "transformer_block",
        }
    return report


def build_bc_report(sources: dict[str, dict], warnings: list[dict],
                    max_seq_len: int, pos_len: int,
                    target_state: dict) -> "ProvenanceReport":
    """See build_dm_av_report's docstring. `target_state` supplies real
    shapes so every param entry carries `target_shape`."""
    def _shape(name: str) -> list[int]:
        return list(target_state[name].shape)

    report = ProvenanceReport(
        contract_version=BC_CONTRACT.name,
        checkpoint_family="dm_behavioral_cloning",
        sources=sources,
        warnings=list(warnings),
        vocabularies={
            "fen_tokenizer_fingerprint": canonical_fen_tokenizer_fingerprint(),
            "action_vocab_fingerprint":  canonical_action_vocab_fingerprint(),
            "num_actions": 1968, "fen_vocab_size": 31,
            "source": "searchless_chess/src/{tokenizer,utils}.py",
        },
        params={},
    )
    report.params["dm.token_embedding.weight"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.token_embedding.weight"),
        "spans": [
            {"target_start": 0, "target_end": 31,
             "source": "bc", "source_key": "token_embedding.weight",
             "source_start": 0, "source_end": 31,
             "semantic_role": "fen_input_embedding",
             "reason": "BC provides pretrained FEN embeddings."},
            {"target_start": 31, "target_end": 1968,
             "source": "dm_av", "source_key": "token_embedding.weight",
             "source_start": 31, "source_end": 1968,
             "semantic_role": "move_input_embedding",
             "reason": "BC lacks move-as-input embeddings. DM-AV rows [0, 31) deliberately NOT used."},
            {"target_start": 1968, "target_end": 1971,
             "source": "new_init", "semantic_role": "reasoning_structural_tokens",
             "tokens": ["<think>", "</think>", "<move>"]},
        ],
    }
    report.params["dm.pos_embedding.weight"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.pos_embedding.weight"),
        "spans": [
            {"target_start": 0, "target_end": pos_len,
             "source": "bc", "source_key": "pos_embedding.weight",
             "source_start": 0, "source_end": pos_len,
             "semantic_role": "pretrained_positions"},
            {"target_start": pos_len, "target_end": max_seq_len,
             "source": "copy_of_last_pretrained_row",
             "source_key": "pos_embedding.weight",
             "source_start": pos_len - 1, "source_end": pos_len,
             "semantic_role": "extended_positions"},
        ],
    }
    report.params["dm.output_linear.weight"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.output_linear.weight"),
        "spans": [
            {"target_start": 0, "target_end": 1968,
             "source": "bc", "source_key": "output_linear.weight",
             "source_start": 0, "source_end": 1968,
             "semantic_role": "move_output_head"},
            {"target_start": 1968, "target_end": 1971,
             "source": "new_init", "semantic_role": "reasoning_structural_output_rows"},
        ],
    }
    report.params["dm.output_linear.bias"] = {
        "status": "partial_copy",
        "target_shape": _shape("dm.output_linear.bias"),
        "spans": [
            {"target_start": 0, "target_end": 1968,
             "source": "bc", "source_key": "output_linear.bias",
             "source_start": 0, "source_end": 1968,
             "semantic_role": "move_output_head_bias"},
            {"target_start": 1968, "target_end": 1971,
             "source": "new_init", "semantic_role": "reasoning_structural_output_bias"},
        ],
    }
    for k in ("value_head.0.weight", "value_head.0.bias", "value_head.2.weight", "value_head.2.bias"):
        report.params[k] = {
            "status": "new_init", "target_shape": _shape(k),
            "semantic_role": "value_head",
        }
    for k, t in target_state.items():
        if k in report.params:
            continue
        if not k.startswith("dm.layers.") and not k.startswith("dm.post_ln.") and not k.startswith("dm.pre_ln."):
            continue
        src_key = k[len("dm."):]
        report.params[k] = {
            "status": "full_copy", "target_shape": list(t.shape),
            "source": "bc", "source_key": src_key,
            "semantic_role": "transformer_block",
        }
    return report
```

- [ ] **Step 4: Add `from_checkpoint` to `ReasoningModel`**

In `src/reasoning/model.py`, add classmethod at the end of the class:

```python
    @classmethod
    def from_checkpoint(cls, config: ReasoningModelConfig) -> tuple["ReasoningModel", "ProvenanceReport"]:
        """Build a ReasoningModel and return its ProvenanceReport.

        The report is built from the detected checkpoint family. Callers
        should invoke report.verify() with the loaded source state dicts
        before assert_contract() — the loader's claims are not trusted
        until the test numerically verifies them."""
        from src.reasoning.warmstart_provenance import (
            build_dm_av_report, build_bc_report, ProvenanceReport,
        )
        model = cls(config)
        ckpt_info = inspect_checkpoint(config.dm_checkpoint)
        sources: dict[str, dict] = {}
        sources_primary_label = "dm_av" if ckpt_info.family == "dm_action_value" else "bc"
        sources[sources_primary_label] = {
            "path": str(config.dm_checkpoint),
            "sha256": _file_sha256(config.dm_checkpoint),
        }
        target_state = model.state_dict()
        if ckpt_info.family == "dm_behavioral_cloning":
            sources["dm_av"] = {
                "path": str(config.dm_av_embedding_source),
                "sha256": _file_sha256(config.dm_av_embedding_source),
            }
            report = build_bc_report(
                sources=sources, warnings=model._warmstart_warnings,
                max_seq_len=config.max_seq_len, pos_len=ckpt_info.positional_length,
                target_state=target_state,
            )
        elif ckpt_info.family == "dm_action_value":
            report = build_dm_av_report(
                sources=sources, warnings=model._warmstart_warnings,
                max_seq_len=config.max_seq_len, pos_len=ckpt_info.positional_length,
                target_state=target_state,
            )
        else:
            raise ValueError(f"Unsupported checkpoint family: {ckpt_info.family}")
        return model, report
```

And add `_file_sha256` helper near the top of `src/reasoning/model.py`:

```python
import hashlib as _hashlib

def _file_sha256(path: str) -> str:
    h = _hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()
```

- [ ] **Step 5: Run all tests**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/reasoning/model.py src/reasoning/warmstart_provenance.py tests/test_warmstart_provenance.py
git commit -m "feat(warmstart): ReasoningModel.from_checkpoint returns ProvenanceReport"
```

---

## Task 10: BC regression test (would catch the original bug)

**Files:**
- Test: `tests/test_warmstart_provenance.py`

The design's failure-message example names **"unexpected new-init rows in dm.token_embedding.weight"** with range `[31, 1968)`. That message fires only when the report *claims* those rows are new-init — which is exactly the shape of the original BC bug: the report honestly says "rows 31-1967 are new" and the contract says "no, only 1968-1970 may be new."

We reproduce that scenario by monkeypatching `build_bc_report` to emit a report that claims `new_init` for the middle span (matching what a pre-fix loader would truthfully produce), then call `assert_contract` and expect the design's exact failure wording.

- [ ] **Step 1: Write the regression test**

Append:

```python
def test_bc_regression_more_than_3_new_rows_fails(tmp_path, monkeypatch):
    """Would have caught the pre-2026-04-22 BC bug: the BC loader left
    token_embedding rows [31, 1968) at random init. If the report
    truthfully recorded that (status='new_init' for the middle span),
    assert_contract must refuse with 'unexpected new-init rows'."""
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning import warmstart_provenance as wp
    from src.reasoning.warmstart_provenance import BC_CONTRACT

    bc = tmp_path / "bc.pt"
    dm = tmp_path / "dm.pt"
    _build_synthetic_bc_ckpt(bc)
    _build_synthetic_dm_av_ckpt(dm)

    real_builder = wp.build_bc_report

    def _buggy_builder(*args, **kwargs):
        report = real_builder(*args, **kwargs)
        # Pre-fix loader semantics: rows [31, 1968) were actually new-init.
        # A truthful report captures that — which is what this regression
        # is designed to catch at assert_contract time.
        spans = report.params["dm.token_embedding.weight"]["spans"]
        spans[1] = {
            "target_start": 31, "target_end": 1968,
            "source": "new_init",
            "semantic_role": "move_input_embedding",
            "reason": "[simulated pre-fix BC bug: rows not sourced from DM-AV]",
        }
        return report

    monkeypatch.setattr(wp, "build_bc_report", _buggy_builder)

    cfg = ReasoningModelConfig(dm_checkpoint=str(bc), dm_av_embedding_source=str(dm), max_seq_len=80)
    model, report = ReasoningModel.from_checkpoint(cfg)
    loaded_sources = {
        "bc":    torch.load(bc, weights_only=False)["state_dict"],
        "dm_av": torch.load(dm, weights_only=False)["state_dict"],
    }
    report.verify(target_state=model.state_dict(), loaded_sources=loaded_sources)

    with pytest.raises(AssertionError) as exc:
        report.assert_contract(BC_CONTRACT)
    msg = str(exc.value)
    assert "unexpected new-init rows in dm.token_embedding.weight" in msg
    assert "[31, 1968)" in msg
    assert "semantic_role=move_input_embedding" in msg
```

- [ ] **Step 2: Run**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_bc_regression_more_than_3_new_rows_fails -v
```

Expected: PASS. The assertion message matches the design's example wording verbatim.

- [ ] **Step 3: Commit**

```bash
git add tests/test_warmstart_provenance.py
git commit -m "test(warmstart): regression guard for the pre-fix BC bug"
```

---

## Task 11: Real-checkpoint smoke test

**Files:**
- Create/Modify: `tests/conftest.py`
- Test: `tests/test_warmstart_provenance.py`

Paths come from env vars or pytest CLI flags — no hardcoded paths. Skips cleanly if either is missing.

- [ ] **Step 1: Add pytest CLI options**

Check if `tests/conftest.py` exists. If yes, append; if no, create it:

```python
import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--bc-checkpoint", action="store", default=None,
        help="Path to a BC checkpoint for warmstart smoke test.",
    )
    parser.addoption(
        "--dm-av-checkpoint", action="store", default=None,
        help="Path to a DM-AV checkpoint for warmstart smoke test.",
    )


@pytest.fixture
def bc_checkpoint_path(request) -> str | None:
    return request.config.getoption("--bc-checkpoint") or os.environ.get("GRPO_CHESS_BC_CHECKPOINT")


@pytest.fixture
def dm_av_checkpoint_path(request) -> str | None:
    return request.config.getoption("--dm-av-checkpoint") or os.environ.get("GRPO_CHESS_DM_AV_CHECKPOINT")
```

(If `conftest.py` already exists and defines `pytest_addoption`, merge these options into the existing function.)

- [ ] **Step 2: Write the smoke test**

Append to `tests/test_warmstart_provenance.py`:

```python
def test_real_checkpoint_smoke(bc_checkpoint_path, dm_av_checkpoint_path):
    import os
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.warmstart_provenance import (
        BC_CONTRACT_LEGACY, select_contract_for_family,
    )

    if not bc_checkpoint_path or not os.path.exists(bc_checkpoint_path):
        pytest.skip("BC checkpoint path not provided or missing")
    if not dm_av_checkpoint_path or not os.path.exists(dm_av_checkpoint_path):
        pytest.skip("DM-AV checkpoint path not provided or missing")

    cfg = ReasoningModelConfig(
        dm_checkpoint=bc_checkpoint_path,
        dm_av_embedding_source=dm_av_checkpoint_path,
        max_seq_len=120,
    )
    model, report = ReasoningModel.from_checkpoint(cfg)
    assert select_contract_for_family(report.checkpoint_family).name == "bc_with_dm_av_move_input_init_v1"

    loaded_sources = {
        "bc":    torch.load(bc_checkpoint_path, weights_only=False)["state_dict"],
        "dm_av": torch.load(dm_av_checkpoint_path, weights_only=False)["state_dict"],
    }
    report.verify(target_state=model.state_dict(), loaded_sources=loaded_sources)
    # Legacy variant whitelists `missing_fingerprint` for pre-stamp checkpoints.
    report.assert_contract(BC_CONTRACT_LEGACY)
```

- [ ] **Step 3: Run (should SKIP without paths)**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_real_checkpoint_smoke -v
```

Expected: SKIPPED with the reason string.

- [ ] **Step 4: (Optional, if real checkpoints are available) Run with paths**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_real_checkpoint_smoke -v \
  --bc-checkpoint=checkpoints/dm_port/9M_behavioral_cloning.pt \
  --dm-av-checkpoint=checkpoints/dm_port/9M_action_value.pt
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_warmstart_provenance.py
git commit -m "test(warmstart): real-checkpoint smoke test via env/CLI paths"
```

---

## Task 12: Operator documentation

**Files:**
- Create: `docs/reasoning_grpo/warmstart_provenance.md`
- Modify: `docs/reasoning_grpo/failure_modes.md`

One page on what the contract guards, how to run the tests, and how to read a failure.

- [ ] **Step 1: Write `docs/reasoning_grpo/warmstart_provenance.md`**

```markdown
# Warmstart provenance contract

Every `ReasoningModel` built via `ReasoningModel.from_checkpoint(config)`
returns a `ProvenanceReport` alongside the model. The report records,
per parameter and per embedding row, where weights came from — a
pretrained checkpoint (BC or DM-AV), a structural copy (last-row pos
extension), or new init. `assert_contract()` enforces the allowlist.

## Contracts

- **DM-AV warmstart** — `DM_AV_CONTRACT` (`dm_av_v1`). Token embedding
  fully from DM-AV, output head fully new-init (DM-AV's 128-bucket
  head is incompatible), `value_head` new-init.
- **BC warmstart** — `BC_CONTRACT` (`bc_with_dm_av_move_input_init_v1`).
  FEN rows `[0, 31)` from BC, move-input rows `[31, 1968)` from DM-AV,
  structural rows `[1968, 1971)` new. Requires both `dm_checkpoint` and
  `dm_av_embedding_source` in `ReasoningModelConfig`.
- **Legacy variants** — `DM_AV_CONTRACT_LEGACY` / `BC_CONTRACT_LEGACY`
  whitelist `missing_fingerprint` warnings for pre-stamp on-disk
  checkpoints. Used only by the real-checkpoint smoke test.

## Run the tests

```bash
# Synthetic tests (fast, always run):
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v

# Real checkpoints (skipped unless paths are given):
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_real_checkpoint_smoke -v \
  --bc-checkpoint=path/to/bc.pt --dm-av-checkpoint=path/to/dm_av.pt
# Or via env vars: GRPO_CHESS_BC_CHECKPOINT, GRPO_CHESS_DM_AV_CHECKPOINT.
```

## Reading a failure

Example:

```
Warmstart contract violation (contract=bc_with_dm_av_move_input_init_v1, family=dm_behavioral_cloning):
  - unexpected new-init rows in dm.token_embedding.weight: [31, 1968)
    (allowed: [(1968, 1971)]); semantic_role=move_input_embedding
```

This means the loader left a region of the input embedding random
that the contract expected to be sourced. Check that
`ReasoningModelConfig.dm_av_embedding_source` is set and that the
DM-AV checkpoint's embedding is full-sized.

## Row-span convention

Every span is half-open `[start, end)`. Schema fields
`target_start`/`target_end`/`source_start`/`source_end` are separate
integers (never 2-element arrays) so the JSON is unambiguous.

## Bumping the contract version

When the contract changes (e.g., a new structural token is added or
the source for a row range changes):

1. Update the relevant `build_*_report` function.
2. Bump the `name` on the affected contract (e.g., `_v1` → `_v2`).
3. Update tests that assert the contract name string.
4. Add a changelog entry under "Contract history" below.

## Contract history

- `bc_with_dm_av_move_input_init_v1` / `dm_av_v1` — initial version,
  2026-04-22. Fixes the pre-existing bug where BC warmstart left move-
  as-input embedding rows `[31, 1968)` at random init.
```

- [ ] **Step 2: Append an entry to `failure_modes.md`**

Add a section to `docs/reasoning_grpo/failure_modes.md` (append to the end):

```markdown
## BC warmstart silently random-inits move-input embedding

**Symptom.** With BC warmstart, `dm.token_embedding.weight` rows
`[31, 1968)` start at random init rather than inheriting DM's
move-input semantics. GRPO training on a BC warmstart shows a slow
initial climb that the DM-AV warmstart path avoids.

**How you'll notice.** `tests/test_warmstart_provenance.py` now fails
the BC regression test if rows `[31, 1968)` are not sourced from a
DM-AV embedding. `ReasoningModel.__init__` raises if
`dm_av_embedding_source` is missing on a BC config.

**Recovery.** Point `ReasoningModelConfig.dm_av_embedding_source` at a
DM-AV checkpoint (`checkpoints/dm_port/*_action_value.pt` or
equivalent). Re-run warmstart. If no DM-AV checkpoint is available in
your environment, this configuration cannot be used — only DM-AV
warmstart is supported in that case.

**History.** Discovered 2026-04-22. Contract
`bc_with_dm_av_move_input_init_v1` in
`src/reasoning/warmstart_provenance.py` codifies the fix.
```

- [ ] **Step 3: Commit**

```bash
git add docs/reasoning_grpo/warmstart_provenance.md docs/reasoning_grpo/failure_modes.md
git commit -m "docs(warmstart): operator note on provenance contract + failure mode"
```

---

## Task 13: Final smoke run

**Files:** none modified.

- [ ] **Step 1: Run the full reasoning-related test suite to confirm no regressions**

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py tests/test_zero_think_bc_parity.py tests/test_zero_think_dm_parity.py -v
```

Expected: all PASS. Any new failure in the BC/DM parity tests means the loader change broke the downstream forward path — investigate before calling the feature done.

- [ ] **Step 2: Check git log is clean and squash-ready**

```bash
git log --oneline main..HEAD
```

Expected: 13 focused commits (one per task 1-12 plus any step-level commits), each with a clear imperative subject.

---

## Spec-coverage self-check

Running against `docs/superpowers/specs/2026-04-22-warmstart-provenance-test-design.md`:

- **Goal 1 (provenance inspector):** Task 3 (schema) + Task 4 (`verify`) + Task 5 (`assert_contract`) + Task 9 (`from_checkpoint`).
- **Goal 2 (pytest suite verifying numerically):** Tasks 9, 10, 11.
- **Goal 3 (BC loader fix):** Task 7.
- **Goal 4 (regression test):** Task 10.
- **Per-path contracts (Section "Per-path contracts"):** Task 3 (constants) + Task 9 (build_*_report).
- **Tokenizer/action-vocab consistency (Section):** Task 1 (fingerprints) + Task 2 (converter stamps) + Task 8 (verification).
- **Warning/fingerprint semantics (Section):** Task 3 (`allowed_warning_tags`) + Task 5 (enforcement) + Task 11 (legacy variant in smoke test).
- **Failure-message format (Section "Failure-message format"):** Task 5 implementation produces structured messages matching the spec example.
- **Files touched list:** all six items from spec covered (Tasks 1, 2, 6-9, 11, 12).

Gaps identified: none.
