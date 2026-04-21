## Context

The PyTorch port of the DeepMind searchless-chess transformer lives in `src/dm_port/` (`transformer.py`, `convert_jax.py`, with layout notes in `convert_jax_layout.md`). The upstream JAX model and 9M checkpoint ship with the `searchless_chess` submodule at the repo root (`searchless_chess/src/transformer.py`, `searchless_chess/checkpoints/9M/6400000/`). The upstream 9M config is `num_layers=8, embedding_dim=256, num_heads=8, num_return_buckets=128, policy='action_value'`.

We fine-tune on top of the 9M checkpoint via the reasoning stack, so that checkpoint's port is the load-bearing dependency whose correctness we need to validate. Any numeric divergence between the JAX reference and our PyTorch port would silently corrupt every GRPO/reasoning result we produce.

The reasoning variant in `src/reasoning/model.py` was considered as a third variant but is out of scope: it replaces DM's 128-bucket head with a freshly-initialised `Linear → TOTAL_VOCAB_SIZE` policy head, extends the vocab and positional embedding, and uses a prefix-LM attention mask. Its outputs are not directly comparable to JAX/DM-port logits and it cannot play Stockfish games from an untrained head. A reasoning-stack parity check is a separate follow-up change.

## Goals / Non-Goals

**Goals:**
- Detect whether `src/dm_port/transformer.py` + `src/dm_port/convert_jax.py`, applied to the 9M checkpoint, reproduces the JAX reference within a tight numerical tolerance.
- Surface behavioral differences via a 64-game Stockfish match played by each variant, and compare the two variants' match results head-to-head.
- Produce an actionable report that, on mismatch, points at the largest divergences (per position and per game).
- Keep the diagnostic strictly local and opt-in — it must not run in the default `pytest` suite or in Colab/Lightning jobs.

**Non-Goals:**
- Fixing any bug the diagnostic finds.
- Re-implementing or "cleaning up" the port conversion code — the existing `convert_jax.py` is the subject under test.
- Training-time behavior. This is a forward-pass + eval-play parity check only.
- Multi-checkpoint sweeps (136M, 270M). 9M only.
- Reasoning-stack parity (separate change).

## Decisions

### D1. Use the existing `convert_jax.py` as-is
Any parity claim that doesn't exercise the actual port code is worthless. We import and call `src.dm_port.convert_jax.convert(...)` directly to produce the PyTorch state dict from the JAX 9M checkpoint. No inline re-implementation, no "simplified" conversion. The script writes its output to a temp file; the diagnostic then loads that file into a `DMTransformer`. The net effect is the exact production path.
*Alternative considered:* hand-write a second converter for the test. Rejected — it would validate the second converter, not the one we ship.

### D2. Target the 9M action-value model
The 9M action-value checkpoint at `searchless_chess/checkpoints/9M/6400000/` is the fine-tuning base. Config: `num_layers=8, embedding_dim=256, num_heads=8, num_return_buckets=128, seq_len=SEQUENCE_LENGTH+2, policy='action_value'`. Use it identically on both sides. The 9M behavioral-cloning / state-value variants are out of scope.

### D3. Forward-pass comparison — delegate to existing test
`tests/test_dm_port_parity.py` already runs JAX↔PT-port on 32 fixed FENs at float64 (`<1e-10`) and float32 (`<5e-4`). The diagnostic invokes those two test functions directly (`from tests.test_dm_port_parity import test_port_matches_jax_float64, test_port_matches_jax_float32`) as a **precondition gate** before any match. On failure, the script stops and reports which gate failed. No re-implementation.

### D4. 64-game Stockfish evaluation
Each of the two variants plays **64 games** against Stockfish and we compare their match results. Details:
- For the JAX variant, use the upstream `searchless_chess.src.engines.neural_engines` (`ActionValueEngine.play`) wired to play against Stockfish via `python-chess`. For the PyTorch variant, implement a thin `play()` adapter in the script that mirrors the upstream `analyse` → legal-move argmax exactly, backed by `DMTransformer`. No re-implementation of game logic — we use `python-chess` for legality and game loop.
- **Fixed Stockfish skill level, fixed node budget, fixed opening set.** All games start from a fixed list of 32 opening FENs drawn from the curated FEN file; each variant plays each opening twice (once as White, once as Black) against the same Stockfish for 64 games total. Stockfish: single thread, fixed hash size, fixed `nodes` budget (more reproducible than wall-clock time).
- Model move selection is greedy (argmax over the expected return across legal moves with `temperature ≈ 0` as in the upstream notebook). No sampling, no seeds needed for the model side. Stockfish non-determinism at a fixed node budget is expected to be negligible but we log its binary version.
- Reported per variant: W / D / L vs. Stockfish, score (W + 0.5·D), Elo-difference estimate.
- Reported pairwise (JAX vs PT): number of games with at least one move-level divergence, first-divergence ply distribution, cross-tabulation of per-game results across the 64 openings.

Pass criterion: all 64 games identical move-for-move between JAX and the PyTorch port. If forward-pass diffs are within tolerance but greedy play diverges, that itself is a finding worth surfacing.

### D5. Opening set for matches
Reuse the 32 curated FENs in `tests/test_dm_port_parity.FIXED_FENS` as the match opening set. Each opening is played twice (White, Black) → 64 games per variant. No separate positions file.

### D6. Placement and exclusion from default tests
Put the entry point under `scripts/` (`scripts/compare_jax_pytorch_port.py`) — not under `tests/`. It is a CLI diagnostic invoked explicitly. This makes "not in the default test suite" the default.
*Alternative considered:* put it in `tests/` with a `@pytest.mark.local_only` marker excluded via `pytest.ini`. Rejected — more configuration surface, and people routinely run `pytest tests/` without markers.

### D7. JAX dependency is opt-in
The script imports JAX and the `searchless_chess` submodule (both already available locally per the existing distill/teacher pipeline). If either is missing, fail loudly with a clear install hint. Do not add JAX to the project's default dependency set.

## Risks / Trade-offs

- **Numerical noise from JAX↔PyTorch matmul / layernorm ordering** → use float32 end-to-end on both sides, set deterministic flags, and pick a tolerance (1e-4) that tolerates fused-op noise but still catches real bugs. If the tolerance proves too tight in practice, raise it with a written justification rather than silently loosening.
- **Tokenizer / input-encoding drift** — if JAX and PyTorch see even slightly different token sequences, everything downstream is meaningless → share a single tokenization path (upstream `searchless_chess/src/tokenizer.py`, which the existing `distill/teacher.py` and `chess/dm_leaf_oracle.py` already use) and assert byte-equal input tensors before the forward pass.
- **Stockfish non-determinism across variants** → pin skill level, node budget (not time), threads=1, hash size; log the exact Stockfish binary version. If Stockfish still varies run-to-run at a fixed node budget, record it as a known caveat and still report the move-level divergence distribution, which is strictly model-driven.
- **64 games is low statistical power on its own** → we rely primarily on move-level comparison (deterministic, exact). Win/draw/loss aggregates are a secondary signal; a small W/D/L gap across 64 games is not a conclusion, but any move-level divergence is.
- **Checkpoint path / submodule not initialized** locally → detect and print a remediation command (`git submodule update --init`, `searchless_chess/checkpoints/download.sh`) instead of a stack trace.
