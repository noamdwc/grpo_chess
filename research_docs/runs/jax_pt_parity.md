# JAX vs PyTorch DM-port parity report
_Generated: 2026-04-21T19:38:25_

## Configuration
- Stockfish: `Stockfish 17.1`
- Nodes/move: 100000
- Skill Level: 20
- Hash (MB): 16  |  Threads: 1
- Games per variant: 64
- PT checkpoint: `/Users/noamc/repos/grpo_chess/checkpoints/dm_port/9M.pt`
- JAX checkpoint: `/Users/noamc/repos/grpo_chess/searchless_chess/checkpoints/9M`

## Forward-pass gate
- float64 max_abs_diff = 3.517e-13  (threshold 1e-10) → PASS
- float32 max_abs_diff = 1.831e-04  (threshold 5e-4) → PASS

## Per-variant match results
- **JAX**: W/D/L = 3/20/41  score = 0.203  Elo Δ = -237  unfinished = 0
  - terminations: {'CHECKMATE': 44, 'THREEFOLD_REPETITION': 15, 'INSUFFICIENT_MATERIAL': 5}
- **PT port**: W/D/L = 3/20/41  score = 0.203  Elo Δ = -237  unfinished = 0
  - terminations: {'CHECKMATE': 44, 'THREEFOLD_REPETITION': 15, 'INSUFFICIENT_MATERIAL': 5}

## Cross-variant move-for-move
- Diverged games: **0 / 64**
- First-divergence ply histogram: `{}`
- Result cross-tab (jax|pt): `{'0-1|0-1': 18, '1-0|1-0': 26, '1/2-1/2|1/2-1/2': 20}`

## Verdict
**PASS**
