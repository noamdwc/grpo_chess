# Interview Presentation Speaker Notes

## Prework Artifacts (Requested First Steps)

### Top 10 files relied on
1. `README.md` — project scope, architecture summary, baseline table pointer.
2. `src/train_self_play.py` — GRPO pipeline entrypoint and callback wiring.
3. `src/grpo_logic/model.py` — GRPO module behavior and rollout temp usage.
4. `src/grpo_logic/loss.py` — PPO/GRPO loss terms and logged metrics semantics.
5. `src/grpo_logic/sampling.py` — trajectory sampling and per-step reward logic.
6. `src/chess/boards_dataset.py` — GRPO start-state generation and phase distribution.
7. `src/pretrain/pretrain_dataset.py` — pretrain data source and scale notes.
8. `src/distill/distill_dataset.py` — distill shard schema and split logic.
9. `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md` — GRPO failure metrics.
10. `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md` and `research_docs/2026-02-24_deepmind-136m-teacher-baseline.md` — key quality/baseline numbers.

### Key metrics and dataset stats summary
| Metric | Value | Source |
|---|---:|---|
| Teacher 136M score vs SF2 | 0.688 (12/20/0), +137 Elo | `research_docs/2026-02-24_deepmind-136m-teacher-baseline.md` |
| Teacher 9M score vs SF2 | 0.625 (8/24/0), +89 Elo | same |
| GRPO failure average score | ~0.03 | `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md` |
| GRPO failure peak score | 0.10 | same |
| GRPO clip fraction | ~0.46–0.52 | same |
| Distill collapse final eval score | 0.015625 | `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md` |
| Distill label sparsity | `k_mean=1.026`, entropy mean `0.018` | same |
| Pretrain raw corpus scale | 14M games, 7.3GB | `src/pretrain/pretrain_dataset.py` |
| Distill default sample budget | 2,000,000 | `src/configs/distill.yaml` |

### Draft slide titles (before full deck)
1. GRPO Chess: Searchless RL for Scalable Chess Policy Learning
2. Problem: Learn Strong Chess Without Tree Search
3. Why This Is Hard
4. High-Level Solution
5. Data: Sources, Splits, Schema, Scale
6. Modeling: Feature Extraction + Model
7. Training/Eval Design
8. Results
9. Error Analysis
10. Ablations
11. Production Plan
12. Lessons + Next Steps
13. Appendix

---

## Slide 1 — GRPO Chess

### Talk track (1-2 min)
I’d open by framing this as a searchless chess policy-learning project rather than a classical search-engine project. The objective is to train a compact transformer policy that can map board states directly to moves, then improve with a staged pipeline: supervised pretraining, teacher-student distillation, and GRPO self-play. The interview value here is the engineering and research discipline: consistent configs, reproducible run docs, and explicit failure analysis rather than only reporting best-case numbers.

### Evidence paths
- `README.md`
- `src/README.md`
- `AGENTS.md`

---

## Slide 2 — The Problem

### Talk track (1-2 min)
The prediction problem is policy learning over a fixed move action space, with legal-move masking at inference/training time. The practical target is to improve policy strength measured against Stockfish while preserving an efficient inference footprint. I’d emphasize that this matters when you need predictable serving complexity and can’t rely on expensive search at runtime.

### Evidence paths
- `src/models.py`
- `src/eval_utils.py`
- `README.md`

---

## Slide 3 — Why It Is Hard

### Talk track (1-2 min)
There are three core difficulties. First, rewards are generated via engine evaluation, which is noisy and expensive. Second, data distributions change across stages: human games for pretrain, teacher label distributions for distill, and on-policy self-play for GRPO. Third, the optimization dynamics can degrade quickly: one run analysis shows sustained high clip fraction and weak effective policy gradient signal, while another analysis attributes collapse to sparse distillation labels. So this is both an ML and systems-debugging problem.

### Evidence paths
- `src/grpo_logic/sampling.py`
- `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md`
- `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`

---

## Slide 4 — High-Level Solution

### Talk track (1-2 min)
This diagram shows the actual project flow implemented in code: pretrain a policy on high-ELO game positions, distill teacher guidance into a smaller/faster student representation, then run GRPO self-play with Stockfish-based reward/evaluation. Evaluation artifacts and research docs close the loop. I’d call out that Stockfish is used for reward/eval, but the policy learning itself remains searchless, matching the repo invariant.

### Evidence paths
- `src/train_self_play.py`
- `src/pretrain/pretrain.py`
- `src/distill/distill.py`
- `research_docs/README.md`

---

## Slide 5 — Data

### Talk track (1-2 min)
For pretrain, the code documents a 14M-game, 7.3GB corpus with filtering and hashed train/eval split. Distillation uses shard files containing tokenized boards, legal masks, and variable-length teacher distributions. GRPO data is generated on the fly from sampled chess positions with configured opening/middlegame/endgame proportions and optional quality filtering by centipawn bounds. This staged data design is a strength, but also where distribution mismatch and label-quality regressions can originate.

### Evidence paths
- `src/pretrain/pretrain_dataset.py`
- `src/distill/distill_dataset.py`
- `src/chess/boards_dataset.py`
- `src/configs/default.yaml`
- `src/configs/pretrain.yaml`
- `src/configs/distill.yaml`

---

## Slide 6 — Modeling

### Talk track (1-2 min)
The core model is an encoder-only transformer with configurable readout (`mean/last/cls`) and a policy head over 1968 actions. Inputs are tokenized board-state strings; outputs are logits masked by legal moves before sampling or greedy selection. The same family is reused across pretrain, distill, and GRPO, which reduces integration complexity and makes failures easier to attribute to data/optimization rather than architecture churn.

### Evidence paths
- `src/models.py`
- `src/configs/default.yaml`
- `src/configs/distill_labelsafe_sharp_v1.yaml`

---

## Slide 7 — Training & Evaluation Design

### Talk track (1-2 min)
GRPO sampling takes groups of trajectories per board, computes per-step reward deltas, and optimizes a PPO-style clipped objective with KL regularization. Evaluation protocol is explicit and reproducible: alternating colors, random opening plies, fixed game counts, and score computed as wins plus half-draws divided by games. I’d highlight that this helps avoid hand-wavy model-quality claims because the scoring function and config knobs are codified.

### Evidence paths
- `src/grpo_logic/sampling.py`
- `src/grpo_logic/loss.py`
- `src/eval_utils.py`
- `src/evaluator.py`
- `src/configs/default.yaml`

---

## Slide 8 — Results

### Talk track (1-2 min)
I’d present two result groups. First, teacher baselines against Stockfish skill 2: 136M at 0.688 score (+137 Elo) and 9M at 0.625 (+89 Elo), both with zero losses in the documented 32-game runs. Second, GRPO/distill outcomes in failure analyses: one GRPO run averages around 0.03 score with a noisy peak at 0.10, and a distill-collapse run reports final score 0.015625. This establishes both what “good” looks like and where current bottlenecks are.

### Evidence paths
- `research_docs/2026-02-24_deepmind-136m-teacher-baseline.md`
- `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md`
- `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`

---

## Slide 9 — Error Analysis

### Talk track (1-2 min)
The strongest error pattern in repo evidence is supervision collapse in distillation: most converted samples effectively carry one-move labels (`k_mean=1.026`), which undermines soft-policy distillation and correlates with low `val/top1_match` and weak Stockfish outcomes. On the GRPO side, failure analysis shows high clip fraction and an imbalanced loss budget. Together these suggest that upstream label quality and downstream policy optimization both need guardrails.

### Evidence paths
- `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`
- `research_docs/2026-02-20_distill-grpo-collapse-analysis.md`
- `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md`

### Missing evidence note
- TODO: missing in repo: confusion-matrix-like move-class error artifact and curated PGN error examples.

---

## Slide 10 — Ablations

### Talk track (1-2 min)
I’d be explicit that full controlled ablation evidence is incomplete in-repo. What we do have is strong single-run forensic evidence: e.g., a temperature mismatch bug analysis reports clip fraction 0.556 at step 0 in a clean-run context, implying clipping from artifact rather than learning dynamics. There are also documented recommendations and config deltas for distill quality-mode recovery. But I’d state clearly that a same-seed, same-budget ablation matrix is still needed.

### Evidence paths
- `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`
- `research_docs/2026-02-20_distill-label-safety-recovery.md`
- `research_docs/2026-02-22_distill-quality-mode-two-paths.md`

### Missing evidence note
- TODO: missing in repo: consolidated multi-seed ablation results table.

---

## Slide 11 — Productionization Plan

### Talk track (1-2 min)
For production, I’d keep the inference path minimal: model forward pass + legal mask + move policy. On monitoring, I’d reuse existing training/eval metrics taxonomy: score, W/D/L, clip fraction, reward stats, entropy, KL trends. On reliability, I’d apply quality gates before promotion and preserve lineage with config+checkpoint metadata. The existing trainer/config architecture already supports reproducible runs and periodic checkpoints, which maps cleanly to model promotion workflows.

### Evidence paths
- `src/train_self_play.py`
- `src/trainer.py`
- `src/evaluator.py`
- `src/configs/default.yaml`
- `src/configs/distill_labelsafe_quality.yaml`

---

## Slide 12 — Lessons + Next Steps

### Talk track (1-2 min)
The central lesson is that this project’s main risk is not only model capacity, but signal quality and optimization correctness. Distillation quality checks, rollout/probability consistency, and stable eval protocols are high-leverage. My immediate next steps would be to formalize dataset preflight thresholds and run a controlled post-fix ablation batch, then only advance checkpoints that clear fixed eval gates.

### Evidence paths
- `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`
- `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md`
- `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`

---

## Slide 13 — Appendix

### Talk track (1-2 min)
This slide is my credibility anchor: where every major claim came from and which artifacts back each chart. In an interview, I’d use this to answer “where did that number come from?” instantly. I’d also explicitly call out what is missing to avoid overstating maturity.

### Evidence paths
- `README.md`
- `src/models.py`
- `src/grpo_logic/loss.py`
- `src/grpo_logic/sampling.py`
- `src/pretrain/pretrain_dataset.py`
- `src/distill/distill_dataset.py`
- `research_docs/2026-02-24_deepmind-136m-teacher-baseline.md`
- `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`
- `research_docs/2026-02-06_loss-budget-and-monitor-analysis.md`
- `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`

### Missing evidence note
- TODO: missing in repo: final unified benchmark dashboard (all major runs, identical protocol, confidence intervals).
