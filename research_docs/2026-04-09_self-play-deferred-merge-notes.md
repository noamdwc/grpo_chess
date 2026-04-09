---
title: "self_play Branch Deferred Merge Notes"
date: "2026-04-09 00:00:00"
agent: "gpt-5"
git_commit: "2374b6ead25d4c5294ce44f118ce0bd44b8ead1d"
git_branch: "integration/unified"
uncommitted_changes: true
files_analyzed:
  - src/configs/default.yaml
  - src/eval_utils.py
  - src/evaluator.py
  - src/grpo_logic/loss.py
  - src/grpo_logic/model.py
  - tests/test_stockfish_eval_callback.py
  - research_docs/KNOWN_NOT_IMPLEMENTED_CHANGES.md
wandb_runs: []
tools_used:
  - "git log"
  - "git diff"
  - "rg"
  - "sed"
prompt: |
  write a documantion for the self_play changes, we will leave it out and add it after we mege to main and in a alter stage
---

# Executive Summary

The `self_play` branch contains a small set of GRPO-era experiments and evaluation changes that are intentionally not part of the current integration branch. We are leaving that branch out of the merge to `main` and will reconsider selected ideas only after `integration/unified` lands on `main` and the repo is stable again.

# Context

During branch consolidation, `self_play` was identified as an older divergent line with overlap in core GRPO files and experiment defaults. The goal of this note is to preserve the useful context from that branch without reintroducing stale behavior into the merged trunk.

# Methodology

- Compared `main..self_play` commit history against `integration/unified`
- Reviewed the substantive `self_play` commits:
  - `b5897d8` `Implement research recs 1-6: unfreeze layers, normalize advantages, increase lr/teacher forcing, step-based sync, multi-skill eval`
  - `41c7788` `Tune config for faster learning: higher temp, more PPO steps, larger batches`
  - `7f38a26` `Parallelize evaluate_policy_vs_stockfish with ThreadPoolExecutor`
  - `9962516` `add last reseach documents`
- Checked current code paths that would conflict with or supersede those changes

# Findings

## 1. `self_play` should remain out of the merge to `main`

The branch overlaps with current GRPO and eval code but does not represent the project's current trunk. The active integration branch already contains newer work around distillation, 9M warm-start GRPO, checkpoint compatibility, Colab/Lightning flows, and hardened evaluation behavior.

## 2. Most `self_play` changes are stale experiment choices, not safe merge candidates

`self_play` changes `src/configs/default.yaml` to more aggressive training defaults:

- `lr: 1e-6 -> 3e-6`
- `num_trajectories: 16 -> 32`
- `ppo_steps: 1 -> 4`
- `rollout_temperature: 1.3 -> 1.6`
- `teacher_forcing_prob: 0.1 -> 0.3`
- `teacher_forcing_depth: 4 -> 8`
- `eval.games: 64 -> 128`
- `pretrain.freeze_layers: 2 -> 0`

These are experiment settings, not structural fixes. They should be reintroduced only through a new experiment config after the `main` merge, not by changing project defaults.

## 3. One `self_play` idea is still potentially useful: parallel Stockfish evaluation

The strongest candidate for later adoption is the eval rewrite from `7f38a26`, which introduces:

- optional `num_workers` in `EvalConfig`
- deterministic per-game RNG for randomized openings
- parallel game execution with one Stockfish engine per worker
- explicit aggregation helpers and tests

This is not merged now because current eval code has newer protections that should not be overwritten:

- recursion-safe progress fallback in `src/eval_utils.py`
- hardened callback failure handling in `src/evaluator.py`

If we bring this back later, it should be re-implemented on top of current `integration/unified`, not cherry-picked directly from `self_play`.

## 4. Parts of `self_play` are already superseded

The per-step advantage standardization idea from `b5897d8` is already present in current code in `src/grpo_logic/loss.py`. There is no reason to port that commit as-is.

Likewise, the branch's multi-skill evaluation idea is partially covered by the existing `Evaluator.eval_ladder()` flow in `src/evaluator.py`, even though it is not currently wired into the main training loop as automatic logging.

## 5. Step-based old-policy sync is a design choice, not a merge cleanup

`b5897d8` replaces epoch-based old-policy sync with configurable step-based sync. Current code still syncs on epoch boundaries in `GRPOChessTransformer.on_train_epoch_start()`. This may still be worth testing later, but it changes GRPO behavior and should be treated as a new training experiment after the main merge.

## 6. The docs from `self_play` can be archived independently if desired

`9962516` only adds research notes. Those can be copied into the repo for historical context without reviving the rest of the branch behavior.

The following archived docs were imported for reference and marked as originating from an unmerged branch:

- `research_docs/2026-02-08_post-tempfix-run-analysis.md`
- `research_docs/2026-02-10_post-recs-run-nryk-analysis.md`
- `research_docs/grpo_chess_debug_research.md`
- `research_docs/postfix_grpo_run_review_2026-02-10.md`

# Recommendations

1. Merge `integration/unified` to `main` without taking any `self_play` commits.
2. Keep `self_play` as an archive/reference branch until after the `main` merge settles.
3. Revisit only these items later:
   - parallel evaluation from `7f38a26`
   - step-based old-policy sync from `b5897d8`
   - archival research docs from `9962516`
4. If any of those are revived, implement them as fresh changes on top of `main`, with targeted tests and explicit experiment configs where needed.

# Open Questions

- Is faster parallel Stockfish evaluation still worth the additional complexity now that the repo has broader training and distillation workflows?
- Should post-merge GRPO experiments live in separate config files rather than modifying `src/configs/default.yaml`?
- Do we want the `self_play` research notes preserved in `research_docs/`, even if none of the code is revived?
