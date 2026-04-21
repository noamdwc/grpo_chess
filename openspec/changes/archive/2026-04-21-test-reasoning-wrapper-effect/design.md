## Context

Fine-tuning the DeepMind-ported 9M checkpoint with Reasoning-GRPO shows a lower-than-expected step-0 Stockfish eval. Local JAX↔PyTorch parity (`scripts/compare_jax_pytorch_port.py`, `tests/test_dm_port_parity*.py`) passes, so the port itself is trustworthy. The remaining suspect is the reasoning wrapper (extended vocab, think/end-think/move tokens, segment+causal attention mask) which changes the model's effective forward pass even when the underlying transformer weights are unchanged. We need an empirical comparison against Stockfish to quantify this wrapping effect before touching training.

## Goals / Non-Goals

**Goals:**
- Measure, via 64 games vs. Stockfish, the play-strength gap between (a) base DM-port model and (b) same weights wrapped in the untrained reasoning model.
- Reuse the existing evaluator (`evaluator.py`, `chess/stockfish.py`, `StockfishEvalCallback` harness) so numbers are directly comparable to GRPO step-0 eval.
- Deterministic and skippable: fixed seed, graceful skip if Stockfish or the DM checkpoint is unavailable.

**Non-Goals:**
- Fixing the gap. This change only measures it.
- Training or fine-tuning either model.
- Full cross-Stockfish-skill-level sweeps — one skill level (matching the GRPO eval callback default) is enough to diagnose.
- Replacing the existing JAX↔PyTorch parity check.

## Decisions

- **Scope: 64 games per configuration.** Matches a reasonable trade-off between noise and runtime. Two configs → 128 games total. At the existing callback's skill level, this fits in a normal local test budget. Alternative (fewer games) rejected: too noisy to see a subtle wrapping-only gap.
- **Reuse `evaluator.py` / `real_play.py`.** Build two move-selection callables and run them through the same evaluation harness GRPO uses at step 0. Alternative (bespoke game loop) rejected: would make numbers non-comparable to training eval.
- **Wrapped config uses the reasoning model in inference mode with weights loaded from the DM port.** No GRPO updates, no training, no change to sampler logic. This is the exact same path GRPO uses at step 0.
- **Same weights source for both configs.** Load one DM-port checkpoint; the reasoning model consumes its transformer-weight subset. This guarantees any gap is attributable to wrapping, not to different initializations.
- **Seeding.** Fix PyTorch/NumPy/Python seeds and the Stockfish starting-position set so the test is reproducible. Report mean score and per-game results.
- **Test placement.** `tests/test_reasoning_wrapper_stockfish_eval.py`. Marked as a slow/integration test; skipped if Stockfish binary is missing (following the convention in existing Stockfish tests).

## Risks / Trade-offs

- **Runtime cost** → 128 games may be slow locally. Mitigation: run at the same Stockfish skill/depth already used by `StockfishEvalCallback`; allow reducing game count via env var / pytest param for smoke runs.
- **Noise at 64 games** → The effect may be within noise if small. Mitigation: also report per-outcome counts (W/D/L) and a simple confidence interval; treat the result as diagnostic, not a hypothesis test.
- **Move-selection path mismatch** → The wrapped-model callable must use exactly the same sampling as GRPO step 0 (greedy vs. temperature, think-token handling). Mitigation: route through `reasoning/real_play.py`, which already backs `StockfishEvalCallback`.
- **Checkpoint availability** → DM 9M checkpoint must be locally resolvable. Mitigation: `pytest.skip` with a clear message when missing, mirroring existing port tests.
