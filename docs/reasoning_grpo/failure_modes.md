# Reasoning-GRPO failure modes

This file is prose. No registries, no code-generated content. When a new
failure mode appears in a run, add a new H2 section here using the template
below and leave a comment at the code site explaining the same thing in its
own words. Duplication between code comments and this file is intentional
and a developer reading the code at 3am should not have to leave the file.

## 7.1 DM port parity break (Phase-0 gate)

### Symptom
The PyTorch DM port no longer matches the JAX model on identical inputs, so every downstream result is suspect.

### How you'll notice
`tests/test_dm_port_parity.py` fails and reports `max_abs_diff` above the accepted threshold.

### Recovery
Do not train. Check layer-norm epsilon, positional embedding shape and extension, residual wiring, embedding mapping, and attention-mask construction before touching anything higher-level.

### History
Task 5 established the current parity gate. Any future break here is a regression against that baseline.

## 7.2 Reasoning collapse

### Symptom
The model learns to emit `<end_think>` as early as it is allowed, so the reasoning span collapses into a near-empty trace.

### How you'll notice
`train/think_length_mean` falls toward `min_think_tokens`, and `train/frac_min_length_samples` trends toward 1 if that metric is logged.

### Recovery
Raise `reasoning.min_think_tokens`, keep `<end_think>` masked for the first N steps, and only consider teacher-forced thinking after simpler fixes fail.

### History
No run is recorded yet in this repo version, but the failure mode is expected enough that the sampler and logging are designed around it.

## 7.3 Reasoning runaway

### Symptom
Almost every sample uses the full thinking budget, but the extra thinking does not improve the final move reward.

### How you'll notice
`train/frac_max_length_samples` stays near 1 while `L_value` remains high and `R_final_move` does not improve.

### Recovery
Lower `reasoning.max_think_tokens`, raise `model.value_head.coef`, and verify special-token masking is not hiding the model's incentive to stop.

### History
No concrete run is pinned yet; this entry exists because the design makes long traces cheap enough that the policy may learn to stall.

## 7.4 Auxiliary-value / policy-gradient tug-of-war

### Symptom
The auxiliary value head improves while move quality stalls, or the reverse happens and the two objectives fight each other.

### How you'll notice
Plot `train/L_value_mean` beside `train/reward_mean`, and compare head/body gradient norms if you need to confirm imbalance.

### Recovery
Tune `model.value_head.coef` first. If the conflict remains, split optimizers or otherwise decouple the value and policy updates.

### History
This is a foreseen optimization conflict rather than a logged incident in the current branch.

## 7.5 Thinking dishonesty (imagined-line cheating)

### Symptom
The model invents optimistic imagined lines and inflates `v_hat` without earning better real-play outcomes.

### How you'll notice
`train/v_hat_minus_R_move_mean` stays large and positive instead of shrinking toward zero.

### Recovery
Keep the auxiliary value target active, increase `model.value_head.coef` if needed, and consider `rival.mode = frozen_dm_9m` if self-play rivaling is too easy to exploit.

### History
The current design added the oracle value target specifically to guard against this failure before it shows up in a run.

## 7.6 Clip-fraction blowup

### Symptom
PPO updates are heavily clipped, especially immediately after rollout or after syncing the old policy.

### How you'll notice
`train/clip_fraction` spikes above roughly 0.3 in WandB.

### Recovery
First check for rollout/recompute temperature mismatch. Then inspect old-policy sync cadence and any other source of stale-reference KL drift.

### History
This mirrors the earlier temperature-mismatch bug documented in `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`.

## 7.7 GRPO group collapse

### Symptom
All K samples in a group receive the same reward, so the normalized advantage becomes zero and PPO stops learning from that group.

### How you'll notice
`train/reward_spread_within_group` approaches zero.

### Recovery
Raise `grpo.rollout_temperature`, verify dropout is enabled where intended, and ensure per-sample randomness is not accidentally shared across the K rollouts.

### History
No run is pinned yet; this is a standard GRPO failure mode and is tracked proactively.

## 7.8 Reasoning doesn't help (the null result)

### Symptom
The model can emit reasoning traces, but playing with thinking enabled is no better than disabling thinking.

### How you'll notice
`eval_stockfish/think_delta` or the direct `think_delta` evaluator output stays near zero after meaningful training.

### Recovery
Treat this as a design failure, not a tuning nibble. Revisit the brainstorm: teacher-forced thinking bootstrap, better leaf evaluation, different imagined-line semantics, or moving to fuller self-play.

### History
This is the main hypothesis test for the project, so every evaluation run should be read with this failure mode in mind.

## 7.9 PE cold-start blowup

### Symptom
Early training is unstable, the thinking span looks like garbage, and the newly added positional rows behave like an uninitialized region.

### How you'll notice
`train/loss` never settles and sampled reasoning traces stay nonsensical, especially at positions beyond the original DM context window.

### Recovery
Start with the current mitigation: copy the last DM positional row into all extended rows. If that is still not enough, lower LR on PE rows or briefly freeze the body and train only the new rows and heads.

### History
The neighbor-copy initialization in `src/reasoning/model.py` was added explicitly to avoid this failure from the start.
