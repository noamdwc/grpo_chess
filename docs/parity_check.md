# JAX ↔ PyTorch DM-port parity diagnostic

Local-only script that verifies the `src/dm_port/` PyTorch port matches the
JAX reference 9M model on both forward-pass logits and greedy Stockfish play.

## Why it exists

`src/dm_port/` is the PyTorch port of the DeepMind searchless-chess 9M
transformer. The reasoning stack (`src/reasoning/`) inherits its body
weights from that port. If the port silently diverges from JAX, every
downstream GRPO/reasoning result is contaminated. This script is the
belt-and-braces check before trusting the port.

## Prerequisites

- `searchless_chess` submodule initialized: `git submodule update --init`
- 9M JAX checkpoint present under `searchless_chess/checkpoints/9M/6400000/`
  (run `bash searchless_chess/checkpoints/download.sh` if missing)
- Converted PyTorch checkpoint at `checkpoints/dm_port/9M.pt` — the script
  auto-produces it via `src.dm_port.convert_jax` if missing (pass
  `--no-auto-convert` to opt out)
- JAX installed in the `grpo_chess` conda env
- `stockfish` on `PATH` (`brew install stockfish` on macOS)

## How to run

```bash
~/miniconda3/envs/grpo_chess/bin/python scripts/compare_jax_pytorch_port.py \
    --report-path research_docs/runs/jax_pt_parity.md
```

Useful flags:

- `--num-games N` (default 64 — 32 openings × 2 colors)
- `--stockfish-nodes` (default 100_000; deterministic node budget)
- `--stockfish-skill` (default 20)
- `--skip-forward-gate` — skip the numeric precondition, for iterating on
  the match code only
- `--no-auto-convert` — fail instead of regenerating the PT checkpoint

## Expected runtime

- Forward-pass gate: ~30–60s (loads JAX, runs 32 FENs at float64 + float32)
- 64-game match per variant: depends on `--stockfish-nodes`. At 100k nodes
  on a modern laptop, expect ~10–20 minutes total for both variants.

## Interpreting the report

The report has four sections: configuration, forward-pass gate,
per-variant match, cross-variant divergence, verdict.

- **Forward-pass gate PASS** (float64 `<1e-10` AND float32 `<5e-4`) → the
  port is mathematically equivalent to JAX. Proceed to the match.
- **Forward-pass gate FAIL** → there is a real bug in the port. The script
  stops; do not trust the match. Debug order: LayerNorm eps, PE indexing,
  Q/K/V transpose in `convert_jax.py`, attention einsum.
- **Per-variant match** reports W/D/L vs Stockfish for each variant.
  Small deltas are expected (Stockfish is non-deterministic at identical
  node budgets in edge cases).
- **Cross-variant move-for-move**: this is the decisive signal. If the
  forward gate passes but matches diverge, it means float32 accumulation
  order is flipping argmax on near-tied moves. That is a real behavioral
  finding, distinct from a bug in the port.

## What to do on failure

1. **Forward-pass fail** — investigate the port. See
   `tests/test_dm_port_parity.py` header comment for the debug order.
2. **Match move-level divergence with forward-pass pass** — examine the
   `examples` in the report. If the divergence is systematic and early-ply,
   something is wrong beyond numeric drift. If it's late-game and only a
   handful of games, it's near-tie flips and likely benign.
3. **Don't silently relax thresholds.** The float64 gate is the real
   contract. Update the code, not the threshold.

## Not part of the default test suite

The script lives under `scripts/`, not `tests/`. Running
`~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/` does not
execute it. The related `tests/test_dm_port_parity.py` (forward-pass only)
does run with pytest but auto-skips if prerequisites are missing.
