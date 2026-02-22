## Experiment Plan: Distill Quality Mode with Two Data Paths

**Goal:** Restore distillation quality by separating production-quality label generation from fast debug conversion.

**Proposed Changes:**
- Add `DISTILL_MODE={quality|fast}` to the Lightning distill runner (`scripts/lightning/run_distill_labelsafe.sh`).
- Keep current DeepMind `.bag` converter path for `fast` mode.
- Add `quality` mode path that builds teacher-generated distill shards via `src.distill.generate_dataset`.
- Add dataset preflight checks (`k_mean`, `k1_frac`, teacher entropy) and fail quality runs if labels are too sparse.
- Re-enable strict quality gate defaults in quality mode while keeping relaxed defaults in fast mode.

**Mechanism:** The weak checkpoints are caused by sparse, low-information labels from the converter pipeline, not by optimizer instability alone. A mode split allows us to keep fast infra/debug runs cheap while forcing production runs to use high-signal teacher-generated soft labels. Dataset preflight thresholds prevent spending GPU time on low-quality shards, and quality-gate defaults ensure only useful checkpoints survive.

**Expected Result:** Quality-mode distillation should move back toward useful imitation quality (`val/top1_match` above 0.10, target above 0.15), and produce a checkpoint that scores materially better than `0.015625` against Stockfish skill 2.

---
### Config File: `src/configs/distill_labelsafe_quality.yaml`
Based on: `src/configs/distill_labelsafe_v3.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `dataset.data_dir` | `data/distill` | `data/distill_quality` | Avoid mixing fast and quality datasets |
| `generate.max_samples` | `2000000` | `300000` | First quality run sized for faster cycle time |
| `dataset.max_shards` | `8` | `8` | Match run-scale cap |
| `distill.quality_gate_min_val_top1` | `0.08` | `0.08` (unchanged) | Keep strict threshold for quality-mode acceptance |

### Success Criteria
- Distill quality mode passes preflight checks: `k_mean > 3`, `k1_frac < 0.80`.
- `val/top1_match >= 0.10` by end of run (target `>= 0.15`).
- `val/top5_match >= 0.35`.
- `eval_stockfish/score > 0.05` (skill 2, 64 games).

### Estimated Credits
~$10-20 (basis: A10G run including teacher-data generation + 10 distill epochs).

## → Next Step

Plan saved: `research_docs/experiments/2026-02-22_distill-quality-mode-two-paths.md`
Config: `src/configs/distill_labelsafe_quality.yaml`

Use `/code-implementation` with the plan doc above.
