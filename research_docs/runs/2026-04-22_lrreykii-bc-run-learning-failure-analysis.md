---
title: "Why WandB Run lrreykii Fails To Learn"
date: "2026-04-22 09:52:43 UTC"
agent: "gpt-5"
git_commit: "125fe99c38b0a325df8b65019fc0ab7bc21f7321"
git_branch: "search_refactor"
uncommitted_changes: true
files_analyzed:
  - chess_model_run_git.ipynb
  - src/configs/grpo_colab_main.yaml
  - src/configs/config_loader.py
  - src/grpo_logic/model.py
  - src/grpo_logic/reasoning_loss.py
  - src/reasoning/model.py
  - src/reasoning/real_play.py
  - src/reasoning/sampler.py
  - src/train_self_play.py
  - src/trainer.py
  - src/evaluator.py
  - tests/test_zero_think_dm_parity.py
  - tests/test_reasoning_wrapper_stockfish_eval.py
  - research_docs/runs/2026-04-21_zero-think-wrapper-vs-dm-parity.md
  - research_docs/runs/2026-04-22_bc-checkpoint-interface-and-warmstart.md
wandb_runs:
  - run_id: "lrreykii"
    run_name: "chess-grpo-20260421-2157-ftjn"
tools_used:
  - Read
  - Grep
  - WandB MCP (compare_runs, sample_metrics, get_run_summary, get_plots, list_runs)
prompt: |
  $research-insights analyze the run lrreykii in wandb - try to figure out why it fails to learn
tags:
  - grpo
  - reasoning
  - behavioral-cloning
  - training
  - debugging
---

# Why WandB Run `lrreykii` Fails To Learn

## Executive Summary

Run `lrreykii` was not a raw-DM warmstart failure. The Colab notebook launched it from `9M_behavioral_cloning.pt`, and `ReasoningModel` preserves the BC action head when loading that checkpoint family. The more likely explanation is a combination of weak effective optimization, proxy-objective mismatch, and thin credit assignment: the run is making tiny, stable PPO updates on a final-move-only reward signal tied to a DM value-head oracle, while evaluation is against Stockfish through a different inference path.

**TL;DR:** `lrreykii` appears to be failing because the reasoning GRPO loop is under-optimizing a weak proxy objective rather than because it is exploding; the strongest concrete bug is that `ppo_epochs: 4` in the Colab config is ignored by the training code.

## Context

### Background

The user asked for an analysis of WandB run `lrreykii` (`chess-grpo-20260421-2157-ftjn`) to determine why it fails to learn.

During discussion, the user clarified two important pieces of context:

1. The run started from `9M_behavioral_cloning.pt`, not from raw `9M.pt`.
2. The run was launched from `chess_model_run_git.ipynb`, which rewrites `src/configs/grpo_colab_main.yaml` into a runtime config before calling `src.train_self_play.train`.

### Research Question

Why does BC-warmstarted reasoning-GRPO run `lrreykii` fail to show improvement against Stockfish?

### Scope

Included:

- the run metadata and surfaced WandB metrics for `lrreykii`
- the Colab notebook launch path
- the reasoning-GRPO loss and training loop
- the evaluation path used during fit-start and periodic eval
- prior April 2026 notes on wrapper parity and BC warmstart semantics

Excluded:

- code changes
- fresh experiment execution
- broad comparison against many historical runs beyond the directly relevant April 2026 notes

## Methodology

### Data Sources

| Source | Details |
|--------|---------|
| WandB Runs | `lrreykii` / `chess-grpo-20260421-2157-ftjn` |
| Code Files | Notebook launch path, config loader, reasoning model, GRPO loop, evaluation code |
| Prior docs | April 21-22 wrapper-parity and BC-warmstart notes |

### Analysis Approach

1. Verified git state and checked recent `research_docs/` titles for relevant prior work.
2. Queried WandB for `lrreykii` summary and training metrics.
3. Verified the actual launch path in `chess_model_run_git.ipynb`.
4. Read the reasoning model warmstart code to determine whether the BC head was preserved.
5. Read the training loop, loss, sampler, and evaluation code to identify mismatches between the configured intent and actual optimization behavior.

### Tools Used

- **Read**: Examined source files and notebook lines directly.
- **Grep**: Located the notebook checkpoint path, config usage, and `ppo_epochs` references.
- **WandB MCP**: Queried run state and surfaced the available metrics for `lrreykii`.

## Findings

### Finding 1: `lrreykii` was a BC warmstart, not a raw-DM random-head run

**Evidence:**

In [`chess_model_run_git.ipynb`](</Users/noamc/repos/grpo_chess/chess_model_run_git.ipynb:39>):

```python
CHECKPOINT_FAMILY = "bc_pt"
BASE_CHECKPOINT_PATH = "/content/drive/MyDrive/data/grpo-chess/base/9M_behavioral_cloning.pt"
```

In the same notebook at [lines 201-202](</Users/noamc/repos/grpo_chess/chess_model_run_git.ipynb:201>):

```python
else:
    print("Using Drive-hosted converted BC checkpoint:", base_checkpoint)
```

The notebook then rewrites the runtime config to point `model.base_checkpoint` at that BC checkpoint at [line 271](</Users/noamc/repos/grpo_chess/chess_model_run_git.ipynb:271>) and launches GRPO at [lines 319-322](</Users/noamc/repos/grpo_chess/chess_model_run_git.ipynb:319>).

In [`src/reasoning/model.py`](</Users/noamc/repos/grpo_chess/src/reasoning/model.py:117>), BC-family warmstart explicitly preserves the pretrained action head:

```python
elif key == "output_linear.weight":
    own_state[key][: value.shape[0]] = value
elif key == "output_linear.bias":
    own_state[key][: value.shape[0]] = value
```

This behavior is also documented in [`research_docs/runs/2026-04-22_bc-checkpoint-interface-and-warmstart.md`](</Users/noamc/repos/grpo_chess/research_docs/runs/2026-04-22_bc-checkpoint-interface-and-warmstart.md:42>).

**Interpretation:**

The earlier raw-DM wrapper-parity failure is relevant background, but it is not the primary explanation for `lrreykii`. This run began from the converted behavioral-cloning checkpoint with the action classifier head preserved, so it should at least start from a sensible move prior.

### Finding 2: the run is stable, but the effective PPO update is much smaller than the config claims

**Evidence:**

WandB surfaced the following step metrics for `lrreykii`:

| Run | Metric | Value |
|-----|--------|-------|
| `lrreykii` | `train_total_loss_step` | `0.025195281952619553` |
| `lrreykii` | `train/clip_fraction_step` | `0.015625` |
| `lrreykii` | `train/ratio_step` | `0.9998854398727416` |
| `lrreykii` | `train/kl_divergence_step` | `0.002269981661811471` |
| `lrreykii` | `train/reward_mean_step` | `0.13236325979232788` |
| `lrreykii` | `train/reward_spread_within_group_step` | `0.2874335050582886` |
| `lrreykii` | `train/advantage_std_step` | `0.9354143738746644` |
| `lrreykii` | `train/L_value_mean_step` | `0.3783547878265381` |
| `lrreykii` | `eval_stockfish_init/score` | `0` |
| `lrreykii` | `eval_stockfish/score` | `0` |

In [`src/configs/grpo_colab_main.yaml`](</Users/noamc/repos/grpo_chess/src/configs/grpo_colab_main.yaml:20>), the intended config is:

```yaml
grpo:
  k_samples_per_root: 8
  lr: 1.0e-5
  clip_ratio: 0.20
  kl_coef: 0.001
  ppo_epochs: 4
```

But `ppo_epochs` is not used anywhere in the reasoning GRPO loop. `rg` only finds it in configs and the dataclass definition, not in the training implementation. The actual `training_step` in [`src/grpo_logic/model.py`](</Users/noamc/repos/grpo_chess/src/grpo_logic/model.py:135>) samples once, computes one loss, and returns it; there is no loop over `self.cfg.grpo.ppo_epochs`.

**Interpretation:**

This is the strongest concrete implementation-level explanation for “fails to learn.” The run is not exploding; it is making tiny, conservative updates. The config advertises `ppo_epochs: 4`, but the code effectively behaves like `ppo_epochs: 1`. That means materially less optimization per rollout than the notebook/config imply.

### Finding 3: the reward and value targets optimize a proxy that is only loosely connected to Stockfish win rate

**Evidence:**

In [`src/configs/grpo_colab_main.yaml`](</Users/noamc/repos/grpo_chess/src/configs/grpo_colab_main.yaml:41>), the default leaf evaluator is:

```yaml
leaf_evaluator:
  mode: "dm_value_head"
```

In [`src/grpo_logic/model.py`](</Users/noamc/repos/grpo_chess/src/grpo_logic/model.py:173>), the reward for each sampled trajectory is computed from the post-rival leaf:

```python
post_fen, _ = self._post_fen_after_rival(...)
rewards.append(
    evaluate_leaf(post_fen, pov_is_white=pov_is_white, mode=self.cfg.leaf_evaluator.mode)
)
```

The auxiliary value target is then computed from the imagined leaf FEN at [lines 181-187](</Users/noamc/repos/grpo_chess/src/grpo_logic/model.py:181>):

```python
v_targets.append(
    evaluate_leaf(
        result.imagined_leaf_fens[sample_idx],
        pov_is_white=pov_is_white,
        mode=self.cfg.leaf_evaluator.mode,
    )
)
```

Evaluation, however, is against Stockfish via [`src/train_self_play.py`](</Users/noamc/repos/grpo_chess/src/train_self_play.py:78>) and [`src/evaluator.py`](</Users/noamc/repos/grpo_chess/src/evaluator.py:98>), not against the DM leaf oracle.

**Interpretation:**

The model is being trained to improve a one-ply self-play proxy scored by the DM value head, then judged by periodic games against Stockfish. Positive `train/reward_mean_step` does not guarantee better Stockfish results. `lrreykii` can therefore look “healthy” under the training proxy while remaining flat at `eval_stockfish/score = 0`.

### Finding 4: the policy gradient signal is thin and only attached to the final move token

**Evidence:**

The reasoning loss only optimizes the final move position. In [`src/grpo_logic/reasoning_loss.py`](</Users/noamc/repos/grpo_chess/src/grpo_logic/reasoning_loss.py:30>):

```python
mask[batch_idx, result.final_move_positions] = True
return mask & result.sampled_mask
```

The per-sample group advantage is scalar at [lines 39-50](</Users/noamc/repos/grpo_chess/src/grpo_logic/reasoning_loss.py:39>), then broadcast across positions at [line 90](</Users/noamc/repos/grpo_chess/src/grpo_logic/reasoning_loss.py:90>):

```python
advantages = adv_per_sample[:, None].expand(num_samples, seq_len)
```

But PPO is applied only through the final-move mask at [lines 92-98](</Users/noamc/repos/grpo_chess/src/grpo_logic/reasoning_loss.py:92>).

**Interpretation:**

This setup gives no direct policy-gradient supervision to the sampled think tokens and only a single scalar relative reward to the committed move. That is a weak learning signal for improving reasoning behavior, especially if the model already has a decent BC move prior and the updates are conservative.

### Finding 5: train-time and eval-time semantics are split enough to obscure true progress

**Evidence:**

During training, rollouts are sampled from `old_policy_model` in [`src/grpo_logic/model.py`](</Users/noamc/repos/grpo_chess/src/grpo_logic/model.py:141>) and self-play rivals reply through zero-think selection in [`src/reasoning/real_play.py`](</Users/noamc/repos/grpo_chess/src/reasoning/real_play.py:69>).

At evaluation time, the callback deliberately uses different inference modes. In [`src/train_self_play.py`](</Users/noamc/repos/grpo_chess/src/train_self_play.py:53>), fit-start uses a zero-think baseline evaluator, while periodic eval uses `think_tokens=cfg.reasoning.max_think_tokens` at [lines 89-103](</Users/noamc/repos/grpo_chess/src/train_self_play.py:89>).

Prior evidence also shows the wrapper path itself can materially alter strength. [`research_docs/runs/2026-04-21_zero-think-wrapper-vs-dm-parity.md`](</Users/noamc/repos/grpo_chess/research_docs/runs/2026-04-21_zero-think-wrapper-vs-dm-parity.md:45>) documents that the wrapped zero-think policy can be much weaker than the bare DM path for raw-DM checkpoints.

**Interpretation:**

Even though `lrreykii` is BC-warmstarted, the system is still training, baselining, and periodically evaluating under different inference regimes. That makes it harder to know whether the model is truly learning chess, overfitting to rollout mechanics, or simply failing to transfer the proxy improvements into the Stockfish evaluation path.

## Analysis

### Root Cause Analysis

The evidence supports the following ranked explanation:

1. **Under-optimization bug**: `ppo_epochs: 4` is configured but ignored, so the run is receiving substantially less policy optimization than intended.
2. **Proxy mismatch**: the run optimizes a DM-value-head leaf reward under self-play, but success is judged by Stockfish games.
3. **Thin credit assignment**: only the final move token receives PPO pressure, while think tokens get no direct policy signal.
4. **Inference-regime mismatch**: training and evaluation are not measuring exactly the same behavior.

What the evidence does **not** support:

- classic PPO instability
- obvious clip-fraction collapse
- the specific raw-DM random-head warmstart bug as the main explanation for this run

### Impact Assessment

The practical consequence is that `lrreykii` can consume significant compute while looking superficially healthy in training metrics and still fail to move the actual target metric. This is worse than a crash because the run can silently produce misleadingly “clean” curves.

### Trade-offs

- Using the DM value-head oracle is cheaper and less noisy than per-sample Stockfish rewards, but it may not align tightly enough with game outcomes.
- Conservative PPO updates reduce collapse risk, but when coupled with ignored `ppo_epochs` they may be too weak to improve a BC prior.
- Final-move-only optimization simplifies the objective, but it undermines the central premise of training a reasoning wrapper.

## Recommendations

### Immediate Actions (High Priority)

1. **Make `ppo_epochs` real or remove it**
   - What: implement repeated optimization over the same sampled rollout batch, or delete the config field to avoid false expectations.
   - Why: the current code/config mismatch is concrete and directly reduces effective learning.
   - Where: [`src/grpo_logic/model.py`](</Users/noamc/repos/grpo_chess/src/grpo_logic/model.py:135>), [`src/configs/grpo_colab_main.yaml`](</Users/noamc/repos/grpo_chess/src/configs/grpo_colab_main.yaml:20>)
   - Estimated effort: Medium

2. **Log the actual optimization contract into WandB**
   - What: log `ppo_epochs`, checkpoint family, leaf evaluator mode, and think-token settings as config or summary fields.
   - Why: `lrreykii` was initially easy to misdiagnose without the BC-warmstart context.
   - Where: notebook runtime config generation and trainer initialization
   - Estimated effort: Low

3. **Run paired evals under both zero-think and think-enabled settings**
   - What: log both eval modes periodically for the same checkpoint.
   - Why: this will distinguish “model got better” from “wrapper/eval semantics changed.”
   - Where: [`src/train_self_play.py`](</Users/noamc/repos/grpo_chess/src/train_self_play.py:53>)
   - Estimated effort: Low

### Medium-Term Improvements

1. **Test reward alignment by comparing DM-value-head and Stockfish leaf rewards**
   - What: run a controlled ablation with identical training settings and only the leaf evaluator changed.
   - Why: this will quantify whether the proxy objective is the bottleneck.
   - Estimated effort: Medium

2. **Increase observability on reasoning behavior**
   - What: log think-token lengths, end-think timing, top-move concentration on a fixed FEN set, and agreement between zero-think and think-enabled moves.
   - Why: flat Stockfish score alone does not reveal whether the model is changing in a useful or harmful direction.
   - Estimated effort: Medium

3. **Revisit final-move-only credit assignment**
   - What: decide whether think tokens should receive explicit learning signal or whether the wrapper should be simplified to pure move selection.
   - Why: current training does not directly supervise the “reasoning” portion of the sequence.
   - Estimated effort: High

### Long-Term Considerations

1. **Tighten the training/eval contract**
   - What: train on an objective that more directly matches the metric you care about.
   - Why: a good proxy is useful only if it predicts downstream chess strength.

2. **Treat the reasoning wrapper as a separate model-design problem**
   - What: validate whether extra think tokens actually improve chess when the move prior is already BC-trained.
   - Why: if the wrapper adds complexity without stronger play, the project should simplify that path rather than overfit to it.

## Open Questions

- [ ] How many optimization steps had `lrreykii` actually completed when WandB was queried, given that the run still showed state `running`?
- [ ] Do later points in the run ever improve `eval_stockfish/score`, or does it remain flat throughout?
- [ ] How different are the zero-think and think-enabled policies for the BC-initialized checkpoint family?
- [ ] Would a Stockfish-scored leaf evaluator materially outperform the current DM-value-head proxy despite higher noise/cost?

## Appendix

### A. Additional WandB Notes

Some WandB endpoints timed out during analysis, so this document relies on the metrics that successfully surfaced through `compare_runs`. The missing history does not affect the main code-level finding that `ppo_epochs` is configured but unused.

### B. Related Documents

- [Zero-think wrapper vs bare DM parity](./2026-04-21_zero-think-wrapper-vs-dm-parity.md)
- [BC checkpoint interface and warmstart](./2026-04-22_bc-checkpoint-interface-and-warmstart.md)
