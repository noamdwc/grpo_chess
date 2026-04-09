## Experiment Plan: Distill Label-Safety Recovery

**Goal:** Recover a usable distillation checkpoint by hardening label handling and retraining with conservative settings.

**Proposed Changes:**
- Add legality-aware filtering and per-sample renormalization in distillation loss (`src/distill/distill.py`).
- Add explicit legality filtering when converting DeepMind data (`src/distill/convert_deepmind_data.py`).
- Log label-quality diagnostics during distillation (`teacher_legal_fraction`, `teacher_valid_sample_fraction`).
- Train a new distillation run from `src/configs/distill_labelsafe.yaml`.

**Mechanism:** Invalid teacher labels can silently destroy the supervision signal, especially with top-k sparse targets. Filtering teacher indices by board legality and renormalizing remaining mass ensures the student only learns from valid moves. Conversion-time legality checks prevent corrupt samples from entering shards. Conservative optimization settings reduce catastrophic drift from pretrained initialization while the cleaned signal is learned.

**Expected Result:** Distill quality should improve substantially (`val/top1_match` rising above 0.10), producing a stronger checkpoint for the next GRPO warm-start cycle.

---
### Config File: `src/configs/distill_labelsafe.yaml`
Based on: `src/configs/distill.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `distill.lr` | `1e-4` | `3e-5` | Reduce destructive updates during warm-start distillation |
| `distill.batch_size` | `2048` | `1024` | Improve optimization stability and gradient signal quality |
| `distill.num_epochs` | `10` | `8` | Avoid over-training while validating corrected labels |
| `distill.warmup_steps` | `1000` | `2000` | Gentler optimizer ramp for pretrained init |
| `distill.max_grad_norm` | `1.0` | `0.5` | Additional stability against spike updates |
| `distill.num_workers` | `4` | `0` | Deterministic loading and easier debugging |
| `distill.pretrain_checkpoint` | unset | `/content/drive/MyDrive/data/grpo-chess/pretrain_checkpoints/pretrain-20260128-1215-16es-epoch=12-train/loss=2.3285.ckpt` | Consistent warm-start baseline used previously |
| `eval.games` | `32` | `64` | More stable checkpoint quality signal |

### Success Criteria
- `val/top1_match >= 0.10` by end of distill run.
- `val/top5_match >= 0.35` by end of distill run.
- No NaN in `train/loss`, `val/loss`, `train/kl_divergence`, `val/kl_divergence`.

### Estimated Credits
~$8-15 (basis: prior distill runs at ~11 hours on A10G with similar dataset size).

## → Next Step

Plan saved: `research_docs/experiments/2026-02-20_distill-label-safety-recovery.md`
Config: `src/configs/distill_labelsafe.yaml`

Use `/code-implementation` with the plan doc above.

## → Next Step: /run-experiment

Commit: `666db78185b3649aa695172e11b530f7bdbc0b55` (working tree has uncommitted changes)
Plan doc: `research_docs/experiments/2026-02-20_distill-label-safety-recovery.md`
Config: `src/configs/distill_labelsafe.yaml`
