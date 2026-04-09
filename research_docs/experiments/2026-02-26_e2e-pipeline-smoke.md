## Experiment Plan: E2E Pipeline Smoke (Pretrain -> Distill -> GRPO)

**Goal:** Verify that the full training chain runs end-to-end on current code (`pretrain -> distill warm-start -> GRPO warm-start`) with no startup/runtime integration breaks.

**Proposed Changes:**
- Add a small pretrain smoke config that trains for one short epoch and exports `pretrain_final.pt`.
- Add a small distill smoke config that warms from the pretrain checkpoint and exports `distill_final.pt`.
- Add a small GRPO smoke config that warms from the distill checkpoint and runs one short epoch.
- Keep `stockfish.path` Linux-compatible (`/usr/games/stockfish`) and disable WandB for this smoke run to reduce external failure modes.

**Mechanism:** Running all three stages in one Lightning job guarantees checkpoint handoff paths are local and deterministic. The reduced step/epoch sizes keep runtime/cost low while still executing every critical path: pretrain checkpoint export, distill checkpoint load/export, and GRPO warm-start load. This specifically validates the new checkpoint-loading behavior under real pipeline conditions.

**Expected Result:** One job completes all three commands successfully and writes:
- `checkpoints/pipeline_e2e_pretrain/pretrain_final.pt`
- `checkpoints/pipeline_e2e_distill/distill_final.pt`
- GRPO checkpoints under `checkpoints/pipeline_e2e_grpo/`

---
### Config File: `src/configs/pretrain_pipeline_e2e_smoke.yaml`
Based on: `src/configs/pretrain.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `pretrain.num_epochs` | 22 | 1 | Smoke runtime reduction |
| `pretrain.batch_size` | 4096 | 512 | Safer GPU memory for generic runner |
| `pretrain.checkpoint_dir` | `checkpoints/pretrain` | `checkpoints/pipeline_e2e_pretrain` | Stage-isolated artifacts |
| `pretrain.use_wandb` | true | false | Reduce external dependency in smoke |
| `pretrain.eval_every_n_epochs` | 1 | 1000 | Skip expensive eval in short run |
| `dataset.max_samples` | 5000000 | 20000 | Faster data pass for smoke |

### Config File: `src/configs/distill_pipeline_e2e_smoke.yaml`
Based on: `src/configs/distill_9m.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `distill.checkpoint_dir` | `checkpoints/distill_9m` | `checkpoints/pipeline_e2e_distill` | Stage-isolated artifacts |
| `distill.pretrain_checkpoint` | unset | `checkpoints/pipeline_e2e_pretrain/pretrain_final.pt` | Pretrain -> distill handoff |
| `distill.use_wandb` | false | false | Keep smoke deterministic |
| `distill.eval_every_n_epochs` | 1 | 1000 | Skip expensive eval in short run |
| `stockfish.path` | `/opt/homebrew/bin/stockfish` | `/usr/games/stockfish` | Lightning Linux compatibility |
| `dataset.data_dir` | `data/distill_9m` | `data/pipeline_e2e_distill` | Isolate generated shards |

### Config File: `src/configs/grpo_pipeline_e2e_smoke.yaml`
Based on: `src/configs/default.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `training.num_epochs` | 400 | 1 | Smoke runtime reduction |
| `training.steps_per_epoch` | 512 | 32 | Faster completion |
| `training.checkpoint_dir` | `checkpoints` | `checkpoints/pipeline_e2e_grpo` | Stage-isolated artifacts |
| `training.use_wandb` | true | false | Reduce external dependency in smoke |
| `pretrain.checkpoint_path` | null | `checkpoints/pipeline_e2e_distill/distill_final.pt` | Distill -> GRPO handoff |
| `grpo.eval_every_n_epochs` | 10 | 1000 | Skip expensive eval in short run |
| `dataset.max_steps` | 512 | 32 | Match steps per epoch |

### Success Criteria
- Pretrain stage exits 0 and writes `pretrain_final.pt`.
- Distill stage exits 0 and writes `distill_final.pt`.
- GRPO stage exits 0 and reaches at least one completed training epoch.
- No checkpoint load failures during distill/GRPO warm-start.

### Estimated Credits
~$3-$8 (basis: one short L4 GPU job with reduced epochs/steps and no WandB artifact uploads).
