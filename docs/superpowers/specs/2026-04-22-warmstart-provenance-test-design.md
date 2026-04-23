# Warmstart Provenance Test — Design

**Date:** 2026-04-22
**Status:** Approved (ready for implementation plan)
**Contract version:** `bc_with_dm_av_move_input_init_v1`

## Background

`ReasoningModel` is built on top of the ported DM 9M transformer and warmstarted from either a DM action-value (DM-AV) checkpoint or a DM behavioral-cloning (BC) checkpoint. The reasoning vocabulary adds three structural tokens (`<think>`, `</think>`, `<move>`) on top of DM's 1968 action IDs, giving `TOTAL_VOCAB_SIZE = 1971`.

Investigation on 2026-04-22 found that the BC warmstart path in `src/reasoning/model.py::_load_and_extend_bc_weights` only populates rows `[0, 31)` of `token_embedding.weight` (BC's input vocab size is 31 — FEN tokens only). Rows `[31, 1968)` — which the reasoning model feeds as *input* tokens during think/move steps — are left at `nn.Embedding`'s random init. Effectively ~1937 move-as-input embedding rows start from scratch on every BC warmstart, silently. The loader's docstring claims only the 3 structural-token rows are new, which is true for DM-AV warmstart but false for BC.

This is exactly the kind of bug a provenance test should prevent from recurring. The design below fixes the BC loader *and* adds a structural regression guard.

## Goal

1. A provenance inspector that records, per parameter and per embedding row, where weights came from (source checkpoint + row range) or that they are new-init — with semantic labels explaining *why* each source is semantically correct.
2. A pytest suite that verifies claims numerically (not just names and shapes) for both DM-AV and BC warmstart paths.
3. A BC-loader fix so the correct contract (exactly 3 new embedding rows, `value_head` is the only new module) actually holds on both paths.
4. A regression test that would have caught the original BC bug — a clear, loud failure naming the offending row range.

## Non-goals

- Refactoring unrelated warmstart code.
- Training-time behavior changes (loss, sampler, attention mask). Untouched.
- Backfilling provenance for legacy models. Reasoning model only.

## Design

### Loader changes (`src/reasoning/model.py`)

**Config:**

```python
@dataclass
class ReasoningModelConfig:
    dm_checkpoint: str
    dm_av_embedding_source: str | None = None   # required iff dm_checkpoint is BC
    max_seq_len: int = 120
    value_head_hidden: int = 256
    freeze_body: bool = False
```

If the detected family is `dm_behavioral_cloning` and `dm_av_embedding_source is None`, raise with a clear message pointing at a canonical DM-AV checkpoint path.

**BC loader fix** (`_load_and_extend_bc_weights`):

1. Load BC state dict.
2. Load DM-AV state dict; verify `token_embedding.weight` is `[1968, D]` and `D` matches reasoning model's `embedding_dim`. Fail loud otherwise.
3. `dm.token_embedding.weight` is populated in three disjoint spans:
   - `[0, 31)` ← BC `token_embedding.weight[0:31]` — FEN semantics. BC is authoritative here.
   - `[31, 1968)` ← DM-AV `token_embedding.weight[31:1968]` — move-as-input (action) embeddings. **DM-AV rows `[0, 31)` are deliberately NOT used** (those are DM-AV's FEN rows; BC's FEN rows win).
   - `[1968, 1971)` ← left at default `nn.Embedding` init (structural tokens).
4. A loader-internal assertion rejects any DM-AV span with `source_key == "token_embedding.weight"` and `source_start < 31`, so a future refactor can't silently start pulling DM-AV's FEN rows.
5. All other keys: unchanged from today.

The DM-AV warmstart path is unchanged except for provenance emission.

**Provenance emission:** the loader builds a `ProvenanceReport` as it performs each assignment. A new classmethod `ReasoningModel.from_checkpoint(config) -> tuple[ReasoningModel, ProvenanceReport]` exposes the report. The existing `__init__` stays, for callers that don't care.

### Provenance utility (`src/reasoning/warmstart_provenance.py`)

```python
@dataclass
class ProvenanceReport:
    contract_version: str
    checkpoint_family: str
    sources: dict[str, dict]    # {name: {"path": str, "sha256": str}}
    warnings: list[str]         # TOP-LEVEL, not inside summary
    params: dict[str, dict]     # per-param spec (see schema below)
    summary: dict

    def to_json(self) -> str: ...

    def verify(self, model: nn.Module, loaded_sources: dict[str, dict]) -> "ProvenanceReport":
        """Walk every span, compare target slice to claimed source slice
        with torch.equal. For new_init entries, confirm the target tensor
        is not equal to any source tensor of compatible shape. Mutates
        `verified_equal` / `verified_not_equal_to_any_source` in place."""

    def assert_contract(self,
                        allowed_new_modules: list[str],
                        allowed_new_embedding_rows: list[dict]) -> None:
        """Raise AssertionError with a structured, human-readable message
        if: any verification failed, any param is outside the allowed
        new-init set, or `warnings` is non-empty."""
```

### Report schema

All row ranges are **half-open** `[start, end)`, encoded with separate `*_start` / `*_end` JSON fields (never a 2-element array) so the JSON is unambiguous.

```json
{
  "contract_version": "bc_with_dm_av_move_input_init_v1",
  "checkpoint_family": "dm_behavioral_cloning",
  "sources": {
    "bc":    {"path": "/abs/path/bc.pt",    "sha256": "..."},
    "dm_av": {"path": "/abs/path/dm_av.pt", "sha256": "..."}
  },
  "warnings": [],
  "params": {
    "dm.token_embedding.weight": {
      "status": "partial_copy",
      "target_shape": [1971, 256],
      "spans": [
        {
          "target_start": 0, "target_end": 31,
          "source": "bc",
          "source_key": "token_embedding.weight",
          "source_start": 0, "source_end": 31,
          "semantic_role": "fen_input_embedding",
          "reason": "BC provides pretrained FEN token embeddings (input_vocab=31).",
          "verified_equal": true
        },
        {
          "target_start": 31, "target_end": 1968,
          "source": "dm_av",
          "source_key": "token_embedding.weight",
          "source_start": 31, "source_end": 1968,
          "semantic_role": "move_input_embedding",
          "reason": "BC lacks move-as-input embeddings (input_vocab=31). Pulling ONLY rows [31, 1968) from DM-AV — those are the action-index embeddings DM-AV trained by feeding move indices as input tokens. DM-AV rows [0, 31) are deliberately NOT used; BC's FEN embedding is authoritative for that range.",
          "verified_equal": true
        },
        {
          "target_start": 1968, "target_end": 1971,
          "source": "new_init",
          "semantic_role": "reasoning_structural_tokens",
          "reason": "<think>, </think>, <move> are new vocabulary entries with no pretrained analogue.",
          "tokens": ["<think>", "</think>", "<move>"],
          "verified_not_equal_to_any_source": true
        }
      ]
    },
    "dm.pos_embedding.weight": {
      "status": "partial_copy",
      "target_shape": [120, 256],
      "spans": [
        {"target_start": 0,  "target_end": 79,  "source": "bc", "source_key": "pos_embedding.weight", "source_start": 0, "source_end": 79,  "semantic_role": "pretrained_positions", "verified_equal": true},
        {"target_start": 79, "target_end": 120, "source": "copy_of_last_pretrained_row", "source_key": "pos_embedding.weight", "source_start": 78, "source_end": 79, "semantic_role": "extended_positions", "reason": "Last pretrained row copied to avoid cold-start blowup (FM 7.9).", "verified_equal": true}
      ]
    },
    "dm.output_linear.weight": {
      "status": "partial_copy",
      "target_shape": [1971, 256],
      "spans": [
        {"target_start": 0,    "target_end": 1968, "source": "bc", "source_key": "output_linear.weight", "source_start": 0, "source_end": 1968, "semantic_role": "move_output_head", "verified_equal": true},
        {"target_start": 1968, "target_end": 1971, "source": "new_init", "semantic_role": "reasoning_structural_output_rows", "verified_not_equal_to_any_source": true}
      ]
    },
    "value_head.0.weight": {"status": "new_init", "target_shape": [256, 256], "semantic_role": "value_head", "reason": "Auxiliary value head (FM 7.5) has no pretrained source.", "verified_not_equal_to_any_source": true},
    "value_head.0.bias":   {"status": "new_init", "target_shape": [256],      "semantic_role": "value_head", "verified_not_equal_to_any_source": true},
    "value_head.2.weight": {"status": "new_init", "target_shape": [1, 256],   "semantic_role": "value_head", "verified_not_equal_to_any_source": true},
    "value_head.2.bias":   {"status": "new_init", "target_shape": [1],        "semantic_role": "value_head", "verified_not_equal_to_any_source": true},
    "dm.layers.0.attn.q_proj.weight": {"status": "full_copy", "target_shape": [256, 256], "source": "bc", "source_key": "layers.0.attn.q_proj.weight", "verified_equal": true}
  },
  "summary": {
    "allowed_new_modules":   ["value_head"],
    "unexpected_new_modules": [],
    "allowed_new_embedding_rows": [
      {"param": "dm.token_embedding.weight", "start": 1968, "end": 1971},
      {"param": "dm.output_linear.weight",   "start": 1968, "end": 1971}
    ]
  }
}
```

**Schema rules:**

- Every param has `target_shape` (including `full_copy` and `new_init`).
- Every span carries `semantic_role`; `reason` is required wherever the source choice is non-obvious.
- `verified_equal` / `verified_not_equal_to_any_source` are written by `ProvenanceReport.verify`, not by the loader. The loader *claims* provenance; the test *verifies* it by comparing tensors. Claims without verification don't satisfy the contract.
- `warnings` is top-level and populated by a post-build classifier that walks every parameter: anything outside the allowlist (non-`full_copy`, non-`partial_copy`, not in `allowed_new_modules`, not in `allowed_new_embedding_rows`) becomes a warning.

### Per-path contracts

The contract's allowlist differs by warmstart path, because the two checkpoints supply different things. `assert_contract` is called with path-specific arguments.

**DM-AV warmstart:**
- `allowed_new_modules`: `["value_head", "dm.output_linear"]` — DM-AV's 128-bucket output head is incompatible with our 1971-move head, so the entire `dm.output_linear` is new-init.
- `allowed_new_embedding_rows`: `[{"param": "dm.token_embedding.weight", "start": 1968, "end": 1971}]`.
- Expected: `dm.token_embedding.weight` is `partial_copy` (rows `[0, 1968)` from DM-AV, rows `[1968, 1971)` new). `dm.pos_embedding.weight` is `partial_copy` (rows `[0, 79)` copied, rows `[79, max_seq_len)` = copy of row 78). `dm.output_linear.{weight,bias}` are `new_init`. All transformer block params are `full_copy`. `value_head.*` are `new_init`.

**BC warmstart:**
- `allowed_new_modules`: `["value_head"]` — BC's 1968-row output head IS compatible (first 1968 rows copy in, only the 3 new structural rows are new-init).
- `allowed_new_embedding_rows`:
  - `{"param": "dm.token_embedding.weight", "start": 1968, "end": 1971}`
  - `{"param": "dm.output_linear.weight",   "start": 1968, "end": 1971}`
- Expected: `dm.token_embedding.weight` is `partial_copy` with three spans (BC `[0, 31)`, DM-AV `[31, 1968)`, new-init `[1968, 1971)`). `dm.output_linear.weight` and `.bias` are `partial_copy` (BC `[0, 1968)`, new-init `[1968, 1971)`). Pos embedding and transformer blocks as in DM-AV path. `value_head.*` are `new_init`.

The JSON example in the previous subsection shows the BC case. The DM-AV case differs only in (a) `dm.output_linear.*` being a single `new_init` entry instead of `partial_copy`, and (b) `dm.token_embedding.weight` having two spans instead of three (one `dm_av` copy span and one `new_init` span).

### Tokenizer / action-vocabulary consistency

Row-level value equality is necessary but not sufficient. Copying DM-AV's embedding row `k` into our input embedding row `k` is only correct if "token ID `k`" denotes the same chess concept across BC, DM-AV, and the reasoning model. Two swap modes are possible:

- **FEN vocabulary swap** (rows `[0, 31)`): BC and DM-AV were trained with different FEN character → ID mappings. Silent if we only compare row values after the copy, because both sides *are* ones vs. twos in the synthetic test — the test passes while production is wrong.
- **Action vocabulary swap** (rows `[0, 1968)` for DM-AV's move-input embedding, and `[0, 1968)` for BC's output head): DM-AV's action ID `k` denotes a different UCI move than BC's action ID `k`.

**Single source of truth.** The `searchless_chess` submodule holds the canonical FEN tokenizer (`searchless_chess/src/tokenizer.py`) and the canonical action list (`searchless_chess/src/utils.py::NUM_ACTIONS` and the UCI list it's derived from). Both BC and DM-AV JAX checkpoints were converted from models trained against this same canonical vocabulary — this is an invariant we want to verify, not assume.

**Recorded in the report.** Add a top-level `vocabularies` block:

```json
"vocabularies": {
  "fen_tokenizer_fingerprint": "sha256:...",     // hash of the sorted FEN char→ID map
  "action_vocab_fingerprint":   "sha256:...",     // hash of the UCI move list in ID order
  "num_actions": 1968,
  "fen_vocab_size": 31,
  "source": "searchless_chess/src/{tokenizer,utils}.py"
}
```

**Test additions** (in both `test_dm_av_warmstart_provenance_synthetic` and `test_bc_warmstart_provenance_synthetic`):

1. The synthetic checkpoints carry tokenizer-fingerprint metadata in their `ckpt["meta"]` dict. When the reasoning model loads them, the loader asserts:
   - BC `meta.fen_tokenizer_fingerprint` == canonical fingerprint.
   - DM-AV `meta.fen_tokenizer_fingerprint` == canonical fingerprint.
   - DM-AV `meta.action_vocab_fingerprint` == canonical fingerprint.
   - BC `meta.action_vocab_fingerprint` == canonical fingerprint.
   Mismatch raises `ValueError` with a message naming which side disagrees.

2. A dedicated test `test_action_vocab_swap_detected` builds a BC + DM-AV pair whose action fingerprints differ (simulating a swap), and asserts the loader refuses to merge them.

3. A dedicated test `test_fen_vocab_swap_detected` does the same for FEN fingerprints.

**Real-checkpoint path.** The converters (`dm_port/convert_jax.py`, `dm_port/convert_behavioral_cloning.py`) will be updated to stamp the canonical fingerprints into `ckpt["meta"]` at conversion time. For existing on-disk checkpoints that lack the stamp, the loader treats a missing fingerprint as `unknown` and emits a top-level warning (`"BC checkpoint lacks tokenizer fingerprint; cannot verify vocab consistency"`) rather than hard-failing. The real-checkpoint smoke test asserts the warning is present or absent as appropriate and does not hard-fail on legacy checkpoints.

**Why fingerprints, not the full vocab.** The full vocabulary is long and would bloat every checkpoint. A stable hash captures equality with one line in the report.

### Test layout (`tests/test_warmstart_provenance.py`)

Seven tests:

1. **`test_dm_av_warmstart_provenance_synthetic`** — build a tiny synthetic DM-AV checkpoint (2 layers, `embedding_dim=32`, `vocab=1968`, `output=128`, `pos=79`). Load via `ReasoningModel.from_checkpoint`. Call `report.verify(...)` then `report.assert_contract(...)`. Expect `value_head` as the sole new module and `[1968, 1971)` as the only new embedding rows.

2. **`test_bc_warmstart_provenance_synthetic`** — build a tiny synthetic BC checkpoint (`input_vocab=31`, `output=1968`, `pos=79`) **and** a tiny synthetic DM-AV checkpoint with the same `embedding_dim`. Crucially, populate BC rows `[0, 31)` and DM-AV rows `[0, 31)` with **distinguishable values** (e.g., ones vs. twos). Load reasoning model. Verify:
   - Target `token_embedding.weight[0:31]` equals BC (ones), *not* DM-AV (twos) — guards against accidentally pulling DM-AV's FEN rows.
   - Target `token_embedding.weight[31:1968]` equals DM-AV's `[31:1968]`.
   - Target `token_embedding.weight[1968:1971]` is not equal to any source row.

3. **`test_bc_regression_more_than_3_new_rows_fails`** — monkeypatch `_load_and_extend_bc_weights` to its pre-fix behavior (skip the DM-AV merge, leave `[31, 1968)` random). Run the inspector and call `assert_contract`. Expect `AssertionError` whose message contains:
   - the phrase `"unexpected new-init rows in dm.token_embedding.weight"`, and
   - the range `[31, 1968)`.

4. **`test_config_rejects_bc_without_dm_av_source`** — constructing a reasoning model from a BC checkpoint with `dm_av_embedding_source=None` raises `ValueError` with a message that names the missing field.

5. **`test_action_vocab_swap_detected`** — build BC + DM-AV synthetic pair with mismatched `action_vocab_fingerprint`. Expect `ValueError` naming the mismatch.

6. **`test_fen_vocab_swap_detected`** — same, but mismatched `fen_tokenizer_fingerprint`.

7. **`test_real_checkpoint_smoke`** — gated on `os.path.exists` for both the on-disk BC and DM-AV checkpoint paths. Loads them, runs inspector, asserts contract. `pytest.skip` cleanly if either file is missing.

### Failure-message format

`assert_contract` produces structured, readable messages:

```
Warmstart contract violation (contract_version=bc_with_dm_av_move_input_init_v1):

  dm.token_embedding.weight:
    allowed new-init rows: [1968, 1971)
    found new-init rows:   [31, 1968), [1968, 1971)
    UNEXPECTED: [31, 1968)  (1937 rows)
      semantic_role of offending range: move_input_embedding
      likely cause: BC checkpoint only supplies rows [0, 31); move-input
        rows were not sourced from DM-AV. Check that
        ReasoningModelConfig.dm_av_embedding_source is set.

  unexpected new modules: none
  failed verifications:   none
```

### Documentation

`docs/reasoning_grpo/warmstart_provenance.md` — one page covering:
- What the test guards (the contract).
- How to run: `~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v`.
- How to read a failure (link to failure-message format above).
- Half-open interval convention.
- When/how to bump `contract_version`.

## Files touched

- `src/reasoning/model.py` — config field, BC loader fix, `from_checkpoint` classmethod, provenance emission, tokenizer/action fingerprint verification.
- `src/reasoning/warmstart_provenance.py` — new utility, including `compute_canonical_fingerprints()` that reads `searchless_chess/src/{tokenizer,utils}.py` as the source of truth.
- `src/dm_port/convert_jax.py` — stamp `meta.fen_tokenizer_fingerprint` and `meta.action_vocab_fingerprint` into DM-AV conversion output.
- `src/dm_port/convert_behavioral_cloning.py` — same stamp for BC conversion output.
- `tests/test_warmstart_provenance.py` — new pytest file.
- `docs/reasoning_grpo/warmstart_provenance.md` — new note.
- `docs/reasoning_grpo/failure_modes.md` — append an entry referencing this contract.

## Out of scope

- Changing any other warmstart path.
- Adding provenance to the legacy `ChessTransformer` or distill/pretrain paths.
- Runtime emission of provenance in training jobs (it's a load-time/test-time artifact only).
