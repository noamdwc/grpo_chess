## Why

When fine-tuning the 9M DeepMind-ported checkpoint with Reasoning-GRPO, the initial (step-0) evaluation score against Stockfish is noticeably lower than expected. JAX↔PyTorch port parity tests already pass, so the drop is not from the port itself. We suspect the reasoning wrapper (think/move head, extended vocab, segment attention mask) degrades play strength even before any GRPO updates. We need a head-to-head evaluation against Stockfish that isolates and quantifies this wrapping-only effect, so we can decide whether to change the wrapper, the initialization, or our expectations for the starting eval.

## What Changes

- Add an evaluation test that plays **64 games against Stockfish** under the existing `StockfishEvalCallback` / `evaluator.py` harness for each of two configurations, using the **same** DM-port 9M weights:
  - **Baseline**: ported PyTorch DM model (no reasoning wrapper).
  - **Wrapped**: same weights loaded into the reasoning model (no training), playing via the think→move path used at step 0 of GRPO.
- Report win rate / score and `elo_diff` for each configuration and the delta between them, so the wrapper-only performance gap becomes a tracked, reviewable number.
- Keep runtime reasonable (64 games, fixed seed, low Stockfish skill level consistent with existing callback settings); skip gracefully if Stockfish or the DM checkpoint is unavailable.

## Capabilities

### New Capabilities
- `reasoning-wrapper-stockfish-eval`: diagnostic that plays 64 Stockfish games with the base DM-port model and 64 with the same weights through the untrained reasoning wrapper, and reports the score gap attributable to wrapping alone.

### Modified Capabilities
<!-- none -->

## Impact

- New test file under `tests/` (e.g., `tests/test_reasoning_wrapper_stockfish_eval.py`).
- Possibly a small evaluation helper (or script in `scripts/`) that loads DM-port weights into the reasoning model without training and exposes a move-selection callable compatible with `evaluator.py` / `real_play.py`.
- Reuses existing components: `dm_port/transformer.py`, `reasoning/{model,sampler,real_play}.py`, `evaluator.py`, `chess/stockfish.py`.
- No changes to training code, configs, or the Colab ladder.
