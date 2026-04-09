---
title: "Distill Quality Collapse Root Cause: Sparse Labels from DeepMind .bag Conversion"
date: "2026-02-22 00:00:00 UTC"
agent: "codex-gpt-5"

git_commit: "4785f7c08752d35d6e6c3baf0af2e8d98bd92b72"
git_branch: "feature/lightning_training"
uncommitted_changes: true

files_analyzed:
  - src/distill/convert_deepmind_data.py
  - src/distill/distill.py
  - src/distill/distill_dataset.py
  - src/distill/generate_dataset.py
  - src/distill/teacher.py
  - src/configs/distill.yaml
  - scripts/lightning/run_distill_labelsafe.sh
  - searchless_chess/src/data_loader.py
  - searchless_chess/src/engines/neural_engines.py
  - research_docs/2026-02-20_distill-grpo-collapse-analysis.md

wandb_runs:
  - run_id: "t7sr0k6w"
    run_name: "distill-20260212-0822-na3l"
  - run_id: "mmv7cmyc"
    run_name: "distill-20260215-1539-k15l"
  - run_id: "1v8f943g"
    run_name: "distill-20260221-2145-6uct"
  - run_id: "2x0su5j2"
    run_name: "distill-20260222-0831-jzji"
  - run_id: "27bkfy9l"
    run_name: "chess-grpo-20260218-2255-d1xh"

tools_used:
  - Read
  - Grep
  - Bash
  - WandB MCP (list_runs, compare_runs, sample_metrics, get_run_summary)

prompt: |
  ok, use $research-insights , how can we fix the weak model quality?

tags:
  - distillation
  - data-quality
  - lightning
  - root-cause
---

# Distill Quality Collapse Root Cause: Sparse Labels from DeepMind .bag Conversion

## Executive Summary
The weak model quality is mainly a supervision-quality failure, not an optimizer stability failure. The current DeepMind `.bag` conversion path produces overwhelmingly sparse teacher labels (mostly 1 move per sample), which collapses policy distillation into noisy hard-label imitation.

**TL;DR:** The converter pipeline is giving low-information labels; use teacher-generated full-policy labels for quality runs, and keep `.bag` conversion only for fast smoke tests.

## Context

### Background
Recent distill runs on Lightning trained successfully but stayed around `val/top1_match ~0.04-0.05` and then failed badly against Stockfish skill 2 (`0W/2D/62L`).

### Research Question
Why did distillation quality collapse, and what concrete pipeline change can restore useful chess strength?

### Scope
Included:
- Distillation data generation/conversion code and loss path.
- Recent distill run metrics from WandB.
- Local probes of converted shard label structure.

Excluded:
- GRPO reward shaping redesign.
- Search-based methods (out of project scope).

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB runs | `t7sr0k6w`, `mmv7cmyc`, `1v8f943g`, `2x0su5j2` |
| Code files | `src/distill/*`, `scripts/lightning/run_distill_labelsafe.sh`, `searchless_chess/src/*` |
| Local artifacts | `data/distill_test/shard_*.pt`, `data/distill_test/raw_bags/action_value-00000-of-02148_data.bag` |

### Analysis Approach
1. Compared pre-collapse and post-collapse distill runs in WandB.
2. Traced label creation and loss assumptions in converter + distill trainer.
3. Probed raw bag/shard statistics to test whether labels are genuinely soft top-k distributions.

### Tools Used
- **Read/Grep**: line-level code inspection.
- **WandB MCP**: run-level and metric-level comparisons.
- **Bash + Python probes**: empirical label sparsity/duplication checks on local converted data.

## Findings

### Finding 1: Converted DeepMind labels are overwhelmingly single-move targets

**Evidence:**
- Converter groups records by FEN and filters records by `min_win_prob` before creating teacher soft labels (`src/distill/convert_deepmind_data.py:276`, `src/distill/convert_deepmind_data.py:289`, `src/distill/convert_deepmind_data.py:298`).
- Local probe on converted shards (`data/distill_test`, first 20 shards, 400,000 samples):
  - `k_mean = 1.026`
  - `k_hist = {(1): 389600, (2): 10229, (3): 168, (4): 3}`
  - Teacher entropy mean `0.018` (near-deterministic labels).
- Raw `.bag` probe (`2,000,000` records):
  - `1,995,736` unique FENs (dup rate `0.213%`)
  - max FEN occurrence `3`.

**Interpretation:**
This dataset is mostly one `(fen, move, win_prob)` per board. Converting it as if it contains dense per-FEN move sets does not produce meaningful top-k policy targets.

### Finding 2: Distill loss expects informative soft teacher targets; sparse labels break that assumption

**Evidence:**
- Distill objective uses teacher probability mass over `teacher_action_indices` (`src/distill/distill.py:218`, `src/distill/distill.py:219`).
- Top-1/Top-5 metrics compare student prediction against teacher top-1 (`src/distill/distill.py:225`, `src/distill/distill.py:236`).
- With near-singleton labels, this becomes hard-label imitation over a 1968-action space (`src/configs/distill.yaml:67`), not rich policy matching.

**Interpretation:**
Training signal becomes low-information/noisy, so model entropy stays high and move ranking quality stays weak.

### Finding 3: Metric regression aligns with converter-era pipeline, not just hyperparameter noise

**Metrics from WandB:**

| Run | Date | `val/top1_match` | `val/top5_match` | `val/entropy` | Notes |
|-----|------|------------------|------------------|---------------|-------|
| `t7sr0k6w` | 2026-02-12 | `0.1999` | `0.5950` | `2.8686` | Last strong distill baseline |
| `mmv7cmyc` | 2026-02-15 | `0.0465` | `0.2031` | `3.4477` | Collapse appears |
| `1v8f943g` | 2026-02-21 | `0.0386` | `0.1726` | `3.5674` | Label-safe era |
| `2x0su5j2` | 2026-02-22 | `0.0404` | `0.1716` | `3.5496` | Final eval score `0.015625` |

**Interpretation:**
The quality drop is systematic across runs after switching to converted-data training; this is not an isolated failed run.

### Finding 4: The high-quality path still exists in code and should be used for quality mode

**Evidence:**
- Teacher-generation path constructs per-position move distributions using teacher inference (`src/distill/generate_dataset.py:250`, `src/distill/generate_dataset.py:255`, `src/distill/teacher.py:352`, `src/distill/teacher.py:370`).
- This path naturally creates proper top-k soft labels per board.

**Interpretation:**
A two-mode pipeline is viable:
- `quality` mode: teacher-generated labels (slow, high signal).
- `fast` mode: converter labels (cheap, debug only).

## Analysis

### Root Cause Analysis
Primary root cause:
- **Supervision mismatch**: `.bag` conversion assumes many action-value rows per FEN, but data is mostly unique FEN rows, yielding sparse labels.

Secondary contributors:
- `min_win_prob` filtering (`src/distill/convert_deepmind_data.py:289`) further reduces already sparse label mass.
- Distillation is still run as policy soft-label matching, which is the wrong objective for this sparse target structure.

### Impact Assessment
- Distill models trained with sparse labels remain near-random in move ranking quality.
- GRPO warm-start from these checkpoints is fragile and quickly underperforms.
- Cloud spend increases while producing low-value checkpoints.

### Trade-offs
- Teacher-generation mode costs more compute/time but yields valid policy-distillation targets.
- Converter mode is cheaper/faster and still useful for infra smoke tests and CI sanity checks.

## Recommendations

### Immediate Actions (High Priority)
1. **Implement explicit two-mode distill pipeline**
- What: Add `DISTILL_MODE={quality|fast}` in Lightning runner.
- Why: Separate quality training from infra/debug runs.
- Where: `scripts/lightning/run_distill_labelsafe.sh`
- Estimated effort: Low

2. **Route `quality` mode to teacher-generated dataset path**
- What: Use `src/distill/generate_dataset.py` in quality mode (GPU machine), not `.bag` converter.
- Why: Restore dense soft policy labels.
- Where: `scripts/lightning/run_distill_labelsafe.sh`, `src/distill/generate_dataset.py`
- Estimated effort: Medium

3. **Add dataset preflight quality checks**
- What: Compute/validate `k_mean`, `k1_frac`, teacher entropy before training.
- Why: Fail fast on low-information labels.
- Where: `scripts/lightning/run_distill_labelsafe.sh` and/or `src/distill/distill.py`
- Estimated effort: Medium

4. **Keep quality gate strict in quality mode**
- What: Re-enable `quality_gate_min_val_top1 >= 0.08` in quality mode; keep relaxed in fast mode.
- Why: Prevent shipping weak checkpoints.
- Where: `src/distill/distill.py`, runner config generation.
- Estimated effort: Low

### Medium-Term Improvements
1. **Store and reuse teacher-generated shards in persistent backend**
- What: Cache quality-mode distill shards to persistent storage.
- Why: Avoid repeated expensive label generation.

2. **Add mode-specific dashboards**
- What: Track `dataset/k_mean`, `dataset/teacher_entropy_mean`, `distill_mode` in WandB.
- Why: Make silent data-quality regressions obvious.

### Long-Term Considerations
1. **Add objective switch for sparse-label datasets**
- What: If running converter data intentionally, use dedicated weighted-BC objective instead of top-k policy distill.
- Why: Align objective to data semantics.

## Open Questions
- [ ] What threshold values are best for dataset preflight (`k_mean`, entropy) for this model size?
- [ ] Should quality mode use `136M` or `270M` teacher by default on Lightning GPUs?
- [ ] Do we keep any `min_win_prob` filtering in quality mode, or disable entirely?

## Appendix

### A. Key Code References
- Converter FEN grouping + threshold filter: `src/distill/convert_deepmind_data.py:276`, `src/distill/convert_deepmind_data.py:289`, `src/distill/convert_deepmind_data.py:298`.
- Distill KD objective and metrics: `src/distill/distill.py:218`, `src/distill/distill.py:225`, `src/distill/distill.py:236`.
- Teacher-generated top-k soft labels: `src/distill/generate_dataset.py:250`, `src/distill/generate_dataset.py:255`, `src/distill/teacher.py:352`, `src/distill/teacher.py:370`.

### B. Related Documents
- `research_docs/2026-02-20_distill-grpo-collapse-analysis.md`

## → Handoff to /plan-experiment

**Recommended experiment direction:**
- Implement `DISTILL_MODE=quality|fast` and run quality mode with teacher-generated labels.
- Add preflight dataset-quality checks and fail quality runs when labels are sparse.
- Re-run distillation and then GRPO warm-start only from checkpoints passing quality gate.

**Key metrics to beat:**
- Distill: `val/top1_match > 0.10` (target `>0.15`), `val/top5_match > 0.35`.
- Distill eval: `eval_stockfish/score > 0.05` at skill 2, 64 games.
- Dataset quality: `k_mean > 3` and `k1_frac < 0.80` in quality mode.

**Relevant prior run(s):**
- `t7sr0k6w` (best recent distill baseline)
- `mmv7cmyc`, `1v8f943g`, `2x0su5j2` (collapsed label regime)
- `27bkfy9l` (GRPO failure from weak distilled checkpoint)
