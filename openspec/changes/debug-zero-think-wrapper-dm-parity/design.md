## Context

The failure mode shows up when the bare DM checkpoint is wrapped in `ReasoningModel` and evaluated through the zero-think inference path. The core debugging need is to compare both paths on the same board state without Stockfish noise and to expose the earliest artifact that diverges.

This change spans multiple modules:

- `src/dm_port/stockfish_eval.py` semantics define the bare baseline
- `src/reasoning/real_play.py` semantics define wrapped zero-think behavior
- the DM checkpoint loader inside `src/reasoning/model.py` determines what information survives wrapping

The design therefore focuses on deterministic diagnostics rather than an immediate product-path behavior change.

## Goals / Non-Goals

**Goals:**

- Make the first divergence reproducible on a single fixed FEN.
- Sweep a diverse fixed corpus to distinguish systemic path mismatch from one-off positions.
- Trace one fixed game until the first divergent move while proving there is no earlier board drift.
- Separate three concerns cleanly: legal action set parity, move encoding parity, and selection/head semantics.

**Non-Goals:**

- Re-run Stockfish matches as the primary signal.
- Change trained reasoning-checkpoint inference semantics without a separate product decision.
- Introduce new runtime dependencies or refactor the evaluation stack broadly.

## Decisions

Use a test-local harness instead of production instrumentation.
Rationale: the debugging payload is detailed and position-by-position; keeping it in `tests/` avoids polluting runtime code paths.
Alternative considered: add temporary logging inside `real_play.py`. Rejected because it is less reproducible and couples diagnostics to live eval code.

Compare three paths, not only two.
Rationale: adding a wrapped-transformer-plus-frozen-DM-head path isolates whether the divergence comes from the wrapped transformer body or from prompt/head/selection semantics.
Alternative considered: compare only bare vs wrapped. Rejected because it would identify the symptom without localizing the cause.

Pin model construction with a manual seed inside the harness.
Rationale: the wrapped policy head is freshly initialized when loading a raw DM checkpoint, so the harness itself must force deterministic construction to avoid moving targets.
Alternative considered: leave initialization unconstrained. Rejected because the observed wrapped move can change across runs even with the same checkpoint.

Express the FEN sweep as a curated fixed corpus rather than random sampling.
Rationale: the goal is reproducible path parity coverage, including castling, en passant, promotion, and check handling.
Alternative considered: random board sampling. Rejected because it makes failures harder to compare across runs and weakens regression value.

## Risks / Trade-offs

[Harness proves the raw-DM misuse case but not every trained-checkpoint issue] -> Keep the harness checkpoint-agnostic so it can be rerun against later reasoning checkpoints.

[Test-local diagnostics may be overlooked during routine development] -> Back them with ordinary pytest coverage and a written run note in `research_docs/runs/`.

[No immediate production fix leaves the unsafe path available] -> Document the exact policy decision still needed: forbid raw-DM zero-think wrapping, or preserve/use the DM head for that case.

## Migration Plan

No runtime migration is required.

Adoption steps:

1. Run `pytest tests/test_zero_think_dm_parity.py -q` when investigating raw-DM wrapper parity.
2. Reuse the harness with different checkpoints if trained reasoning checkpoints need parity investigation.
3. Make any later production-path change in a follow-up change once the desired zero-think semantics are decided.

## Open Questions

- Should raw DM checkpoints be considered invalid inputs for `select_zero_think_move()`?
- If raw checkpoints must be supported, should the wrapped model retain and expose the frozen DM return head explicitly?
- For trained reasoning checkpoints, is the intended no-think baseline “trained policy head with zero think tokens” or “DM-style action-value evaluation through the wrapped body”?
