## Why

We suspect a bug in the JAX → PyTorch port of the DeepMind searchless-chess transformer (`src/dm_port/`). Our reasoning stack builds on this port, so if the port silently diverges from the original, every downstream GRPO/reasoning result is contaminated. We need a local, reproducible two-way comparison — original JAX vs. ported PyTorch — to confirm or rule out a port bug before investing in more training. We target the **9M** checkpoint because that is the model we actually fine-tune.

The reasoning variant (`src/reasoning/`) is deliberately out of scope here: it replaces DM's 128-bucket head with a randomly-initialised policy head over an extended vocabulary, so its outputs are not comparable to JAX/DM-port logits at construction time. Reasoning-stack parity is a separate follow-up if needed.

## What Changes

- Add a local-only diagnostic script (`scripts/compare_jax_pytorch_port.py`) that:
  1. **Reuses the existing forward-pass parity gate** (`tests/test_dm_port_parity.py`): runs the float64 (`<1e-10`) and float32 (`<5e-4`) JAX↔PT-port checks as a precondition. If these fail, the script stops — there is no point running matches on a broken port.
  2. **Runs a 64-game Stockfish match** per variant (JAX via `searchless_chess.src.engines.ENGINE_BUILDERS['9M']`, PyTorch via a thin greedy adapter over the converted DM port), under identical Stockfish configuration (fixed node budget, skill, threads=1, hash), identical opening set, identical colors.
  3. **Reports** per-variant W/D/L, score, Elo-diff, plus pairwise move-for-move comparison: games with any divergence, first-divergence ply distribution, per-game result cross-tabulation.
- The forward-pass comparison is NOT reimplemented here — we delegate to the existing test file so there is a single source of truth for the numeric gate.
- Script lives under `scripts/`, not `tests/`, so it is invoked explicitly and does not run during `pytest tests/`.
- Document how to run it locally, including prerequisites and the converted-checkpoint path.

Non-goals: no changes to training code, the port, the reasoning model, or the default test suite. If the comparison surfaces a bug, fixing it is a follow-up change.

## Capabilities

### New Capabilities
- `jax-pytorch-parity-check`: Local diagnostic that compares outputs of the JAX reference 9M model and the PyTorch port (via the existing `convert_jax.py`) on identical positions and in identical Stockfish matches, and reports agreement metrics.

### Modified Capabilities
<!-- none -->

## Impact

- New files under `scripts/` plus a short README / doc section.
- Adds a local dependency on JAX + the `searchless_chess` submodule being installed; this is opt-in and not required for existing workflows.
- No changes to `src/dm_port/` (including `convert_jax.py`), `src/reasoning/`, training pipeline, or the default test suite — the port code is the subject under test and must be exercised as-is.
- Runs locally only; not wired into Colab or Lightning.ai job submission.
