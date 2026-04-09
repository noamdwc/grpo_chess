## Experiment Plan: Distill Sharp v1

**Goal:** Improve distilled chess strength by replacing weak near-uniform teacher targets with sharper supervision (zscore-normalized targets + hybrid hard/soft loss).

**Proposed Changes:**
- Reduce teacher label support from top-8 to top-4 (`generate.top_k: 4`).
- Lower teacher temperature (`generate.teacher_temperature: 0.7`).
- Enable per-position EV score normalization before softmax (`generate.teacher_target_score_norm: "zscore"`).
- Use hybrid distillation loss with hard top-1 emphasis:
  - `distill.distill_lambda_soft: 0.30`
  - `distill.distill_lambda_hard: 0.70`
  - `distill.distill_target_top1_mix_alpha: 0.15`
- Keep Stockfish eval every epoch and select/export best checkpoint by `eval_stockfish/score`.
- Run a smaller first-cycle data budget (`generate.max_samples: 200000`, `dataset.max_shards: 4`) for faster iteration.

**Mechanism:** The teacher outputs EV-derived preferences, not calibrated policy logits, and the previous top-k softmax labels were nearly uniform. `zscore` normalization amplifies per-position ranking signal when EV ranges are compressed, while lower `top_k` and temperature reduce target entropy further. The hard CE component restores directional learning on teacher top-1 decisions when soft targets remain diffuse, and mixed targets preserve some distributional information to stabilize optimization.

**Expected Result:** Compared with the latest quality baseline (`distill-quality-20260223-r8`), this run should increase teacher sharpness metrics (`teacher_top1_prob_mean`, `teacher_top1_minus_top2_mean`), improve early `val/top1_match`, and raise best `eval_stockfish/score` toward or above `0.10`.

---
### Config File: `src/configs/distill_labelsafe_sharp_v1.yaml`
Based on: `src/configs/distill_labelsafe_quality.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `generate.top_k` | `8` | `4` | Reduce label smoothing from broad top-k support |
| `generate.teacher_temperature` | `1.0` | `0.7` | Sharper soft labels |
| `generate.teacher_target_score_norm` | `plain` (implicit) | `zscore` | Normalize compressed EV scales per position |
| `generate.max_samples` | `300000` | `200000` | Faster iteration cycle for first sharp-target run |
| `dataset.max_shards` | `8` | `4` | Match reduced sample budget |
| `distill.distill_lambda_soft` | `1.0` | `0.30` | Keep soft supervision as auxiliary signal |
| `distill.distill_lambda_hard` | `0.0` | `0.70` | Strong top-1 directional learning |
| `distill.distill_target_top1_mix_alpha` | `0.0` | `0.15` | Mild soft-target sharpening without full collapse |
| `dataset.data_dir` | `data/distill_quality` | `data/distill_sharp_v1` | Keep data lineage separated |
| `distill.checkpoint_dir` | `checkpoints/distill_labelsafe` | `checkpoints/distill_labelsafe_sharp_v1` | Isolate checkpoint lineage |

### Success Criteria
- Preflight label quality:
  - `teacher_top1_prob_mean >= 0.22`
  - `teacher_top1_minus_top2_mean >= 0.05`
  - `teacher_entropy_mean <= 1.85`
- Training quality:
  - `val/top1_match >= 0.08` by epoch 3
- Chess quality:
  - best `eval_stockfish/score >= 0.10`
  - final exported checkpoint `eval_stockfish/score >= 0.08`

### Estimated Credits
~$8-15 (basis: L4 quality mode with 200k teacher samples, 4 shards, 10 distill epochs, and per-epoch Stockfish evaluation).
