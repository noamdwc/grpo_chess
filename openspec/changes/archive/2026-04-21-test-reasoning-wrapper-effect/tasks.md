## 1. Setup and scaffolding

- [x] 1.1 Confirm DM-port 9M checkpoint location and Stockfish binary discovery logic used by existing tests (reuse, don't duplicate)
- [x] 1.2 Identify the exact evaluation path used by `StockfishEvalCallback` at step 0 (entry point in `evaluator.py` + `reasoning/real_play.py`) and the Stockfish skill/depth it uses

## 2. Helpers for two move-selection callables

- [x] 2.1 Build a callable that wraps the bare DM-port model for move selection, compatible with `evaluator.py`'s evaluation loop
- [x] 2.2 Build a callable that wraps the untrained reasoning model (loaded with the same DM-port weights) via the existing `reasoning/real_play.py` path — no bespoke sampler code
- [x] 2.3 Add a tiny loader that produces both models from one DM-port checkpoint, asserting transformer weights are shared / identical

## 3. Evaluation test

- [x] 3.1 Create `tests/test_reasoning_wrapper_stockfish_eval.py`
- [x] 3.2 Implement `pytest.skip` paths for missing Stockfish binary and missing DM-port checkpoint, each with a clear message
- [x] 3.3 Seed Python/NumPy/PyTorch and pin the Stockfish starting-position set so the test is deterministic
- [x] 3.4 Run 64 games vs. Stockfish for the baseline config and 64 games for the wrapped config, under identical settings
- [x] 3.5 Compute and print: per-config score, per-config elo_diff, W/D/L counts, and the score delta (baseline − wrapped)
- [x] 3.6 Mark the test as slow/integration so it isn't part of the default fast suite; document how to run it

## 4. Validation

- [x] 4.1 Run locally end-to-end; capture the baseline/wrapped/delta numbers in the PR description (or a short note in `research_docs/`)
- [x] 4.2 Re-run once with the same seed to confirm determinism (identical outcomes)
- [x] 4.3 Sanity-check that the baseline score roughly matches the DM-port expectation and that the wrapped score roughly matches the GRPO step-0 eval we've been observing
