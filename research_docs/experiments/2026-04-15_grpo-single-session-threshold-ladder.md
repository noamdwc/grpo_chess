## Experiment Plan: Single-Session GRPO Threshold Ladder

**Goal:** Find the first post-refactor LR/KL regime where PPO is no longer inert while remaining stable enough to justify spending the rest of a single 25-hour Colab A100 session on it.

**Proposed Changes:**
- Run three short Stage 1 probes for 40 epochs each with more aggressive `grpo.lr` and weaker `grpo.kl_coef`.
- Keep `grpo.clip_ratio`, `grpo.ppo_epochs`, `grpo.k_samples_per_root`, `training.batch_size`, `training.steps_per_epoch`, and Stage 1 rollout temperature fixed at baseline.
- Run one 100-epoch Stage 2 midpoint bracket selected from the best Stage 1 result.
- Use the remaining session time for one longer confirmation run starting from the Stage 2 winner, or from the best Stage 1 run if Stage 2 is inconclusive.
- Prioritize movement metrics first, stability second, and early eval signal third.

**Mechanism:** The baseline post-refactor run is stable but under-updating: `ratio` stays near `1.0`, `clip_fraction` stays near `0`, PPO/KL terms are tiny, and eval remains at floor. Increasing LR while reducing KL should move the policy farther from the old policy so the PPO trust region starts doing real work. The short probes are designed to quickly find the movement threshold; the bracket then narrows the usable region without widening into a full sweep.

**Expected Result:** At least one Stage 1 probe should show clearly non-inert PPO behavior by epochs 20-30 without obvious instability, and the final confirmation run should spend the remainder of the session in the best stable non-inert regime.

---
### Config Files
Based on: `src/configs/grpo_colab_main.yaml`

Stage 1:
- `src/configs/grpo_colab_probe_a1_lr3e6_kl5e4.yaml`
- `src/configs/grpo_colab_probe_a2_lr5e6_kl3e4.yaml`
- `src/configs/grpo_colab_probe_a3_lr8e6_kl1e4.yaml`

Stage 2:
- `src/configs/grpo_colab_bracket_b1_mid_low.yaml`
- `src/configs/grpo_colab_bracket_b1_mid_high.yaml`

### Config Diff
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `training.num_epochs` | `200` | `40` for Stage 1 probes | Cheap threshold probes that fit a single session |
| `grpo.lr` | `1e-6` | `3e-6`, `5e-6`, `8e-6` | Force visible policy movement beyond the current inert regime |
| `grpo.kl_coef` | `1e-3` | `5e-4`, `3e-4`, `1e-4` | Relax trust-region pressure so higher LR can matter |
| `training.num_epochs` | `200` | `100` for Stage 2 bracket | Enough runtime to validate a midpoint without spending the whole session |
| `grpo.lr` | `1e-6` | `4e-6` or `6.5e-6` in Stage 2 | Midpoint bracket between the best Stage 1 run and its adjacent probe |
| `grpo.kl_coef` | `1e-3` | `4e-4` or `2e-4` in Stage 2 | Midpoint KL pairing for the chosen bracket direction |

### Session Schedule
- Stage 1: three 40-epoch probes (`120` epochs total, about `5.6h`)
- Stage 2: one 100-epoch bracket (`100` epochs, about `4.7h`)
- Stage 3: final confirmation run using the remaining time (`~280-310` epochs, usually plan for `300`, about `14.0h`)
- Total planned load: `~520` epochs (`~24.3h` at the measured `~2.80 min/epoch`)

### Stage 1 Table
| Run | Epochs | LR | KL Coef | Purpose |
|-----|--------|----|---------|---------|
| `A1` | `40` | `3e-6` | `5e-4` | Lower aggressive edge |
| `A2` | `40` | `5e-6` | `3e-4` | Center threshold probe |
| `A3` | `40` | `8e-6` | `1e-4` | Upper high-risk probe |

### Stage 2 Neighbor Rule
- If `A1` wins, bracket upward toward `A2` using midpoint `B1` low: `lr=4e-6`, `kl_coef=4e-4`
- If `A2` wins and `A1` was still inert, bracket upward toward `A3` using midpoint `B1` high: `lr=6.5e-6`, `kl_coef=2e-4`
- If `A2` wins and `A3` looked unstable, bracket downward toward `A1` using midpoint `B1` low: `lr=4e-6`, `kl_coef=4e-4`
- If `A3` wins, bracket downward toward `A2` using midpoint `B1` high: `lr=6.5e-6`, `kl_coef=2e-4`

### Stop / Promote Rubric
- Dead-run rule: stop a Stage 1 probe around epochs 20-30 if `ratio` is still pinned near `1.0`, `clip_fraction` is still pinned near `0`, and PPO/KL terms remain in the same tiny regime as the inert baseline.
- Instability stop rule: stop immediately if `clip_fraction` jumps and stays very high, `ratio` swings hard, PPO/KL/loss terms spike and remain elevated, or reward/eval degrade together with unstable optimization signals.
- Promotion rule: rank runs by movement metrics first, stability second, and eval third. A run with clearly non-inert PPO and acceptable stability beats a run with slightly better eval but still inert updates.
- Stage 3 fallback rule: if Stage 2 is inconclusive or not clearly better, start Stage 3 from the best Stage 1 run rather than spending the remaining session on an ambiguous bracket result.

### Success Criteria
- At least one Stage 1 probe shows clearly non-inert PPO behavior by epochs 20-30.
- The promoted Stage 2 or Stage 1 winner remains stable for a longer continuation.
- Early eval improvement is a bonus signal, not the main success condition for the short probes.

### Estimated Credits
~0 Lightning credits. This plan is explicitly scoped for a single Colab A100 40GB session with a 25-hour runtime cap.
