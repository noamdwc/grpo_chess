## 1. Prereq discovery

- [x] 1.1 Confirmed 9M config: `num_layers=8, embedding_dim=256, num_heads=8, num_return_buckets=128`, checkpoint at `searchless_chess/checkpoints/9M/6400000/`, step `6_400_000`.
- [x] 1.2 Confirmed `convert_jax.convert(args)` entry; output `.pt` is `{state_dict, config}`. Existing converted checkpoint: `checkpoints/dm_port/9M.pt`.
- [x] 1.3 Confirmed JAX play adapter: `ENGINE_BUILDERS['9M']()` → `ActionValueEngine` with `.play(board)`.
- [x] 1.4 Confirmed existing forward-pass gate in `tests/test_dm_port_parity.py` (float64 `<1e-10`, float32 `<5e-4`) — will call directly as precondition.
- [x] 1.5 Confirmed Stockfish binary at `/opt/homebrew/bin/stockfish`; reuse `src/chess/stockfish.py` and `src/eval_utils.py` for game loop.

## 2. Script scaffolding

- [x] 2.1 Create `scripts/compare_jax_pytorch_port.py` with CLI (argparse): `--num-games` (default 64), `--stockfish-nodes` (default 100_000), `--stockfish-skill` (default 20), `--report-path`, `--seed` (default 0), `--skip-forward-gate`.
- [x] 2.2 Add prereq checks at startup: JAX import, submodule present, 9M JAX checkpoint present, converted PT checkpoint present (offer `convert_jax.convert` if missing), Stockfish binary resolvable.

## 3. Forward-pass precondition

- [x] 3.1 Import and invoke `tests.test_dm_port_parity.test_port_matches_jax_float64` and `test_port_matches_jax_float32`. On AssertionError, record the failing gate + measured diff and exit non-zero.

## 4. Match setup

- [x] 4.1 Reuse `tests.test_dm_port_parity.FIXED_FENS` as the 32-opening set. Build the game plan: 32 openings × 2 colors (model=White, model=Black) → 64 games per variant.
- [x] 4.2 Build the JAX engine via `ENGINE_BUILDERS['9M']()`. Its `.play(board)` returns a move given greedy (temperature=None) behavior — the default builder sets no temperature → `np.argmax` (see `neural_engines.ActionValueEngine.play`).
- [x] 4.3 Build a PyTorch-port `play(board)` in the script that mirrors `ActionValueEngine.analyse` + argmax: tokenize state, append `legal_action` and dummy bucket, forward `DMTransformer`, take `[:, -1]` log_probs, compute expected return against `return_buckets_values`, apply repetition update, argmax over legal moves. No sampling.

## 5. Game loop

- [x] 5.1 Implement a `play_match(model_play_fn, stockfish_cfg, openings) -> list[GameRecord]` helper local to the script: python-chess board loop, model on one side, Stockfish on the other (via `chess.engine.SimpleEngine.popen_uci` at fixed `Limit(nodes=...)`, `setoption` Threads=1, Hash=16, Skill Level=skill).
- [x] 5.2 Each `GameRecord` captures: opening FEN, model color, PGN string, per-ply move list (model side only), terminal result, termination reason.
- [x] 5.3 Run the match twice: once with JAX model, once with PT port. Write Stockfish version + uci options to the report header.

## 6. Metrics & reporting

- [x] 6.1 Per variant: W / D / L / score / Elo-diff (use `src/eval_utils.py` helpers where applicable).
- [x] 6.2 Pairwise: for each game, diff the model-side move sequences; record `first_divergence_ply` (None if identical). Count games with any divergence. Bucket first-divergence plies (histogram). Build a cross-tabulation of per-game results (W/D/L × W/D/L).
- [x] 6.3 Render Markdown report to `--report-path` and stdout: header (dates, Stockfish version, config), forward-pass gate section, per-variant match section, cross-variant section, pass/fail verdict.
- [x] 6.4 Exit 0 only if forward-pass gate passed AND move-for-move agreement holds. Otherwise exit non-zero listing failing checks.

## 7. Exclusion from default test suite

- [x] 7.1 Confirm script is under `scripts/`, not `tests/`, and is not imported anywhere in `tests/`.
- [x] 7.2 Run `~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/ --collect-only -q | grep -i compare_jax_pytorch_port` → must print nothing.

## 8. Documentation

- [x] 8.1 Add `docs/parity_check.md`: prerequisites, how to run, expected runtime, how to interpret the report, what to do on failure.
- [x] 8.2 Cross-link from `CLAUDE.md` (one-line note under a local-only diagnostics mention).

## 9. Local dry run

- [x] 9.1 Smoke-run with `--num-games 2` to verify both engines play and report renders.
- [ ] 9.2 Full 64-game run; capture `--report-path`. _(run by user; takes ~10–20 min at default nodes)_
