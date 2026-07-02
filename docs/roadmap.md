# Project Roadmap (2026-07)

Where the project goes from the `av-ppo-rebuild` pivot (see
`docs/av_ppo_rebuild_plan.md`). Design principles: get **one crisp positive
result** with honest baselines before anything else; keep the differentiating
question — *does test-time compute (think tokens) buy playing strength without
search?* — as the flagship; keep the `research_docs/` evidence trail intact.

Invariant: this remains a searchless chess project. No MCTS or tree search.

## Phase 1 — Finish the PPO rebuild (weeks 1–2)

Engineering, not research. Wire the committed pieces (`src/dm_port/av_adapter.py`,
`src/grpo_logic/ppo_loss.py`) into a runnable training path:

- `src/grpo_logic/ppo_model.py` — `AVPPOLightningModule`: rollout with the old
  policy, Stockfish scalar feedback on the post-action state, advantage =
  reward − critic value, a **real** `ppo_epochs` loop (the `lrreykii` fix),
  old-policy sync per rollout batch, and the optimization contract logged to
  WandB.
- `src/configs/ppo_av.yaml` + `load_av_ppo_config` (bypasses the legacy
  schema translation that silently degrades `ppo_epochs`).
- `src/train_av_ppo.py` entry point; `AVEvalPolicy` eval adapter feeding the
  existing `Evaluator`/`StockfishEvalCallback`, with `eval_stockfish_init/*`
  pinning the pre-RL baseline on every run.

**Done when:** tests green, local CPU smoke shows `ratio_first_epoch ≈ 1.0`
and low clip fraction at step 0, then one L4 Lightning smoke job.

## Phase 2 — Milestone A: the first positive result (weeks 2–6)

**Claim: PPO post-training with Stockfish feedback measurably improves a
strong supervised searchless-chess policy.**

- Pin the "before" number properly. The existing 9M baseline
  (`research_docs/2026-02-24_deepmind-136m-teacher-baseline.md`: 0.625 over 32
  games) is inside its own noise band. Re-evaluate at 2–3 Stockfish skill
  levels, several hundred games, repeated to measure eval noise directly.
- Train PPO on the 9M model on L4. Success criterion:
  `eval_stockfish/elo_diff` lift over the warmstart baseline that survives ≥3
  seeds (or clearly exceeds the measured eval noise).
- Use the observability recommendations from
  `research_docs/runs/2026-04-22_lrreykii-bc-run-learning-failure-analysis.md`
  (fixed-FEN move-agreement probes, paired eval configs).
- **Gate:** only after 9M lifts, optionally scale to 136M (A100). If 9M
  refuses to lift, diagnose with `docs/reasoning_grpo/failure_modes.md` before
  touching scale.

**Done when:** a run report in `research_docs/runs/` shows a before/after Elo
table with seed spread.

## Phase 3 — Differentiating science (weeks 6–10)

1. **Reward-source ablation** (cheap): identical PPO runs with
   `leaf_evaluator.mode: dm_value_head` vs `stockfish` — config-only once
   Phase 1 lands. Directly tests the proxy-mismatch hypothesis from the
   lrreykii diagnosis.
2. **Reintroduce reasoning (flagship):** think tokens on top of the *working*
   PPO loop, with the credit-assignment fix, measuring Elo as a function of
   think-token budget. A positive scaling curve = test-time compute without
   search; a flat one = a rigorous negative result. Both are publishable
   outcomes.

Each experiment gets a plan doc in `research_docs/experiments/` and a run
report; the flagship produces an Elo-vs-think-budget plot.

## Phase 4 — Packaging (weeks 10–12, light)

- 4–6 page arXiv-style report: DM base → port + distillation → why naive
  reasoning-GRPO failed → clean PPO lift → think-token result. Negative
  results included honestly.
- README overhaul: motivation, results table, 1–2 plots, pointers to the
  report and `research_docs/`.

## What NOT to do

- No MCTS/tree search.
- No 136M scale-up, new algorithms/domains, or reasoning reintroduction before
  Milestone A.
- No repo polish before there is a result to showcase.
