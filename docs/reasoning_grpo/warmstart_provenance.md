# Warmstart provenance contract

Every `ReasoningModel` built via `ReasoningModel.from_checkpoint(config)` returns a
`ProvenanceReport` alongside the model. The report records, per parameter and
per embedding row, where weights came from: a pretrained checkpoint (BC or
DM-AV), a structural copy (last-row positional extension), or new init.
`assert_contract()` enforces the allowlist.

## Contracts

- **DM-AV warmstart**: `DM_AV_CONTRACT` (`dm_av_v1`). Token embedding fully
  from DM-AV, output head fully new-init, `value_head` new-init.
- **BC warmstart**: `BC_CONTRACT` (`bc_with_dm_av_move_input_init_v1`). FEN
  rows `[0, 31)` from BC, move-input rows `[31, 1968)` from DM-AV, structural
  rows `[1968, 1971)` new. Requires both `dm_checkpoint` and
  `dm_av_embedding_source` in `ReasoningModelConfig`.
- **Legacy variants**: `DM_AV_CONTRACT_LEGACY` / `BC_CONTRACT_LEGACY`
  whitelist `missing_fingerprint` warnings for pre-stamp checkpoints. Used by
  the real-checkpoint smoke test.

## Run the tests

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py -v

~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_warmstart_provenance.py::test_real_checkpoint_smoke -v \
  --bc-checkpoint=path/to/bc.pt --dm-av-checkpoint=path/to/dm_av.pt
```

You can also provide `GRPO_CHESS_BC_CHECKPOINT` and
`GRPO_CHESS_DM_AV_CHECKPOINT` via the environment.

## Reading a failure

Example:

```text
Warmstart contract violation (contract=bc_with_dm_av_move_input_init_v1, family=dm_behavioral_cloning):
  - unexpected new-init rows in dm.token_embedding.weight: [31, 1968)
    (allowed: [(1968, 1971)]); semantic_role=move_input_embedding
```

This means the loader left a region of the input embedding random that the
contract expected to be sourced. Check that
`ReasoningModelConfig.dm_av_embedding_source` is set and that the DM-AV
checkpoint embedding is full-sized.

## Row-span convention

Every span is half-open `[start, end)`. The schema uses separate
`target_start`/`target_end`/`source_start`/`source_end` integers.

## Bumping the contract version

1. Update the relevant `build_*_report` function.
2. Bump the `name` on the affected contract.
3. Update tests that assert the contract name string.
4. Add a short history note here.

## Contract history

- `bc_with_dm_av_move_input_init_v1` / `dm_av_v1`: initial version,
  2026-04-22. Fixes the BC warmstart bug where rows `[31, 1968)` were left at
  random init.
