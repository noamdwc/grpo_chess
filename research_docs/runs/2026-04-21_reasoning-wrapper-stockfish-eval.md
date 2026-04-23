# Reasoning Wrapper Stockfish Eval

Date: 2026-04-21

## Command

```bash
RUN_SLOW_STOCKFISH_EVAL=1 ~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_reasoning_wrapper_stockfish_eval.py -vv -s
```

## Setup

- Config source: `src/configs/default.yaml`
- Stockfish settings: skill level 2, `movetime_ms=50`, threads 1, hash 128
- Eval settings: 64 games, seed 0, randomized 6-ply openings from the seeded default evaluator config
- Shared checkpoint: `checkpoints/dm_port/9M.pt`
- Baseline path: bare DM port via `src/dm_port/stockfish_eval.py`
- Wrapped path: zero-think reasoning wrapper via `src/reasoning/real_play.py`

## Results

### Final-code run 1

- Baseline: score `0.945`, elo diff `+495.1`, W/D/L `57/7/0`
- Wrapped: score `0.008`, elo diff `-841.5`, W/D/L `0/1/63`
- Delta: `+0.938` baseline minus wrapped

### Final-code run 2

- Baseline: score `0.984`, elo diff `+719.7`, W/D/L `62/2/0`
- Wrapped: score `0.000`, elo diff `-2400.0`, W/D/L `0/0/64`
- Delta: `+0.984` baseline minus wrapped

## Takeaways

- The wrapper-only gap is very large and easy to reproduce qualitatively: the wrapped zero-think model collapses against Stockfish while the bare DM port dominates the same opponent.
- The run is not deterministic under the current time-limited Stockfish configuration. Same-seed reruns produced materially different aggregate results.
- The bare DM baseline is much stronger than the older `~0.625` / `+89 Elo` 9M teacher reference in `research_docs/2026-02-24_deepmind-136m-teacher-baseline.md`, so task 4.3 remains unresolved.

## Completion Note

This change is complete as a diagnostic. The purpose was to test and quantify the wrapper-only effect, not to fix the gap, force determinism, or reconcile the stronger-than-expected bare DM baseline. Those follow-up questions should be handled in a separate change.
