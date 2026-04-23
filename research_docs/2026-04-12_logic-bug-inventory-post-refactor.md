# Logic Bug And Logic Error Inventory From `research_docs/`

Date: 2026-04-12

This file inventories **logic / correctness bugs and logic errors documented in `research_docs/`** and splits them into:

1. Bugs that are **still relevant after the reasoning/GRPO refactor**
2. Bugs that are **not relevant anymore after the refactor or later fixes**

Notes:
- This is intentionally **not** a list of hyperparameter issues, training-dynamics problems, eval-noise problems, or infra/ops failures.
- This inventory is derived from the existing `research_docs/` corpus. It does **not** include logic bugs discovered later unless they were already documented there.

## Still Relevant After Refactor

### 1. No temporal credit assignment: one scalar advantage is still broadcast across sequence positions
- Source: `research_docs/2026-01-09_grpo-learning-failure-analysis.md`
- Documented logic error:
  - A single trajectory/sample reward is reused for all moves/timesteps, so the learner cannot distinguish which specific decision was good or bad.
- Why it is still relevant:
  - The current reasoning loss still computes one advantage per sampled trajectory and expands it across all sequence positions before PPO loss.
- Current evidence:
  - `src/grpo_logic/reasoning_loss.py`
- Current status:
  - Still relevant.

### 2. Endgame position generation is still known-bad
- Source: `research_docs/2026-01-09_grpo-learning-failure-analysis.md`
- Documented bug:
  - `generate_endgame_position()` was called out as broken.
- Why it is still relevant:
  - Current code still contains the same TODO and approximate fallback implementation in `src/chess/boards_dataset.py`.
  - The function still does not generate principled random endgames; it just keeps playing random/capture-biased moves until material is low.
- Current status:
  - Still relevant.

### 3. DeepMind `.bag` conversion still produces sparse, low-information labels
- Source: `research_docs/2026-02-22_distill-quality-collapse-sparse-labels.md`
- Documented bug:
  - The converter assumes enough per-FEN move coverage to form useful soft targets, but the source data is mostly one move per FEN, so the resulting labels are nearly hard labels and low-information.
- Why it is still relevant:
  - The current conversion path in `src/distill/convert_deepmind_data.py` still groups by FEN and applies the same top-k / `min_win_prob` style processing.
  - I did not find a replacement quality-mode pipeline wired in that would make the converter safe for production-quality distillation by default.
- Current status:
  - Still relevant for the converter-based distillation path.

### 4. Eval still treats Stockfish failure / no reply as an effective draw
- Source: `research_docs/grpo_chess_debug_research.md`
- Documented logic error:
  - “policy returned None / illegal” or similar no-move edge cases should count as a loss, not a draw-like neutral outcome.
- Why it is still relevant:
  - In the current evaluator, if `stockfish.act(board)` returns `None`, the loop breaks and the game is later scored as `0.5` when there is no winner.
  - That preserves the same diagnostic distortion the research doc warned about.
- Current evidence:
  - `src/evaluator.py`
- Current status:
  - Still relevant.

## Not Relevant After Refactor / Later Fixes

### 1. Hardcoded rollout temperature of `1.0`
- Source: `research_docs/2026-01-19_loss-increase-and-lr-regression-analysis.md`
- Documented bug:
  - Rollout sampling temperature was hardcoded and not configurable.
- Why it is no longer relevant:
  - Current config supports `grpo.rollout_temperature` and `grpo.move_sampling_temperature`.
  - Current GRPO model code reads and uses those values.
- Current evidence:
  - `src/configs/default.yaml`
  - `src/configs/config_loader.py`
  - `src/grpo_logic/model.py`

### 2. Sampling/PPO temperature mismatch
- Source: `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`
- Documented bug:
  - Old log-probs were computed under rollout temperature while PPO recomputation used the default temperature, causing large clip fractions even when weights were unchanged.
- Why it is no longer relevant:
  - Current refactored GRPO path explicitly passes rollout and final-move temperatures through recomputation.
  - The refactored code no longer relies on the old self-play branch implementation described in that research doc.
- Current evidence:
  - `src/grpo_logic/model.py`

### 3. Distill loss accepted illegal teacher targets without guarding or renormalizing
- Source: `research_docs/2026-02-20_distill-grpo-collapse-analysis.md`
- Documented bug:
  - Distillation loss used teacher indices directly without validating them against the legal move mask.
- Why it is no longer relevant:
  - Current distill loss filters teacher targets by legality, renormalizes surviving probability mass, and errors out if no valid teacher mass remains.
- Current evidence:
  - `src/distill/distill.py`

### 4. DeepMind conversion path failed to validate move legality
- Source: `research_docs/2026-02-20_distill-grpo-collapse-analysis.md`
- Documented bug:
  - The converter mapped moves into the action space without verifying legality for the FEN.
- Why it is no longer relevant:
  - Current converter builds `legal_uci` from the board and discards moves not legal in that position before producing teacher indices.
- Current evidence:
  - `src/distill/convert_deepmind_data.py`

## Excluded From Both Lists

These appeared repeatedly in `research_docs/`, but I excluded them because they are not logic bugs:
- entropy collapse
- weak reward signal / noisy self-play reward
- stale-policy sync as a tuning/design concern
- excessive clipping as a symptom
- dead monitors / poor defaults
- missing eval logging / eval variance
- Lightning / Colab / branch / checkpoint bootstrap failures

## Summary

From the research-doc corpus, the logic bugs and logic errors that still matter in the current codebase are:
- no temporal credit assignment because one per-sample advantage is broadcast across sequence positions
- broken/approximate endgame generation
- sparse-label `.bag` conversion being unsuitable for quality distillation
- eval treating no-reply / failure cases as draw-like outcomes

The main logic bugs that were real in older codepaths but are no longer relevant are:
- hardcoded rollout temperature
- rollout/PPO temperature mismatch
- distill legality mismatch in the loss
- missing legality validation in the DeepMind converter
