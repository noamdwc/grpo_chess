# Zero-Think Wrapper vs Bare DM Parity

## What was compared

- Bare DM path: `src/dm_port/stockfish_eval.py`
- Wrapped zero-think path: `src/reasoning/real_play.py`
- Checkpoint: `checkpoints/dm_port/9M.pt`
- Deterministic harnesses:
  - single-position comparison
  - 24-FEN sweep
  - fixed full-game first-divergence trace

The new harness lives in `tests/debug_zero_think_parity.py`, with pytest coverage in `tests/test_zero_think_dm_parity.py`.

## Earliest causal divergence

The earliest break is **not** legal masking, move decode, or board drift.

For the first mismatching corpus position:

- Label: `after_e4`
- FEN: `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1`
- Bare DM choice: `c7c5` (`action_id=1439`)
- Wrapped zero-think choice: `b7b5` (`action_id=1412`)
- Wrapped-body + frozen DM head choice: `c7c5` (`action_id=1439`)

Invariant checks on that same position:

- FEN tokens match between bare and wrapped paths.
- Side to move matches.
- Legal move list matches.
- Legal action-id mask matches exactly.
- Move encode/decode round-trip violations: none.
- Wrapped decoded move is still legal.

Top actions on that position:

- Bare DM: `c7c5 0.472374`, `e7e6 0.465439`, `e7e5 0.462311`, `d7d5 0.455362`, `c7c6 0.452952`
- Wrapped zero-think: `b7b5 0.245455`, `d7d5 0.116637`, `f7f6 0.089716`, `e7e5 0.085636`, `a7a6 0.082786`
- Wrapped body + frozen DM head: identical to bare DM

## Root cause

`ReasoningModel` does **not** preserve the DM checkpoint’s trained 128-bucket return head for zero-think selection.

Instead:

1. `ReasoningModel.__init__` replaces DM’s output head with a new `TOTAL_VOCAB_SIZE` policy head.
2. That policy head is freshly initialized (`nn.init.trunc_normal_`) rather than loaded from `9M.pt`.
3. `select_zero_think_move()` runs `[FEN | <think> | <end_think> | <move>]` and reads that random action-token head directly.

So the wrapped zero-think path is not evaluating the same function as the bare DM path. When the checkpoint is the raw DM port, zero-think behavior depends on random policy-head initialization, not on the checkpoint’s trained move values.

Evidence:

- Two `ReasoningModel` instances built from the same raw DM checkpoint with different `torch.manual_seed(...)` values produce different `policy_head.weight` tensors.
- Reusing the wrapped transformer body with the frozen DM return head on `[FEN | move | 0]` reproduces bare DM choices across the full 24-FEN sweep.

## Sweep and trace results

- 24 fixed FENs checked.
- Bare DM vs wrapped zero-think move mismatches: 20 / 24.
- Wrapped-body + frozen DM head vs bare DM mismatches: 0 / 24.

Fixed trace from the standard initial position:

- Ply 0: both chose `e2e4`
- Ply 1 first divergence:
  - bare: `c7c5`
  - wrapped: `b7b5`
  - both moves legal
  - pre-divergence boards identical
  - post-move FENs diverge immediately after that choice

## Fix status

No production-path fix was applied in this change.

Reason:

- The harness narrowed the issue to a precise semantic mismatch in how the raw DM checkpoint is being consumed.
- A safe production fix needs a policy decision:
  - either forbid `select_zero_think_move()` for raw DM checkpoints,
  - or preserve/use the frozen DM return head for raw-checkpoint zero-think parity/debugging,
  - or route raw-DM evaluation through `DMPortEvaluator` only.

## Residual uncertainty

- This analysis proves the collapse for the raw DM checkpoint case.
- It does not by itself prove that post-GRPO checkpoints have no additional wrapper bug.
- If parity is still required for trained reasoning checkpoints, rerun the same harness against that checkpoint and compare:
  - legal mask parity
  - chosen-token parity
  - board progression parity

## Follow-up

A follow-up note documents the released checkpoint-family interface differences and the BC warmstart path:

- [`2026-04-22_bc-checkpoint-interface-and-warmstart.md`](</Users/noamc/repos/grpo_chess/research_docs/runs/2026-04-22_bc-checkpoint-interface-and-warmstart.md:1>)
