## Why

The wrapped zero-think inference path can catastrophically disagree with the bare DM port even when both are pointed at the same `checkpoints/dm_port/9M.pt` checkpoint. Stockfish match results showed the collapse, but they do not identify the first causal divergence or distinguish wrapper bugs from model weakness.

## What Changes

- Add deterministic parity diagnostics for a single fixed FEN, a fixed 24-FEN sweep, and a fixed full-game trace without Stockfish in the loop.
- Compare bare DM selection, wrapped zero-think selection, and wrapped-transformer-plus-frozen-DM-head selection on the same positions.
- Record intermediate artifacts needed to localize the first divergence: FEN tokens, legal move/action sets, top action scores, chosen action ids, decoded moves, and post-move FENs.
- Add pytest coverage that locks in the earliest discovered root cause.
- Write a concise debugging note summarizing the evidence and the remaining product decision for raw-DM zero-think evaluation.

## Capabilities

### New Capabilities
- `zero-think-dm-parity-diagnostics`: Deterministic tooling and tests that compare bare DM and wrapped zero-think behavior at the position and game-trace level.

### Modified Capabilities

## Impact

- Affected code: `tests/debug_zero_think_parity.py`, `tests/test_zero_think_dm_parity.py`, `research_docs/runs/2026-04-21_zero-think-wrapper-vs-dm-parity.md`
- Systems: DM-port checkpoint loading, wrapped zero-think debugging workflow, parity investigation workflow
- Dependencies: existing local DM checkpoint only; no new external dependencies
