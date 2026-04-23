---
title: "Reasoning-GRPO Refactor: Fine-Tuning DeepMind Searchless Chess with LLM-Style Reasoning Traces"
date: "2026-04-09"
status: "draft — awaiting implementation plan"
supersedes:
  - "src/models.py (ChessTransformer)"
  - "src/models_9m.py"
  - "src/pretrain/ (from-scratch pretraining pipeline)"
  - "src/grpo_logic/sampling.py (per-step rollout sampler)"
---

# Reasoning-GRPO Refactor

## 1. Context and motivation

The current codebase trains a custom `ChessTransformer` from scratch, then GRPO-fine-tunes it on Stockfish-reward trajectories of fixed depth. After extended iteration we have stable training and moderate Stockfish-relative improvements, but two observations push us toward a bigger change:

1. **We already have access to DeepMind's searchless chess checkpoints** (9M, 136M, 270M) and they are strictly stronger than anything we can train from scratch in-project. Continuing to train our own base architecture is duplicating effort.
2. **The current "search" at eval time is cosmetic.** `TrajectorySearcher` samples a few rollouts and picks the one with the best leaf, but the network was trained to output good one-step moves, so running the search barely changes the result versus taking the network's one-step argmax. The model does not actually "think" — the search adds no capability at inference time that wasn't already in the weights.

This refactor changes three things at once:

1. **Base model.** Replace the from-scratch `ChessTransformer` with a PyTorch port of the DeepMind searchless transformer. The DM model becomes the starting point of fine-tuning, not a frozen teacher used for distillation.
2. **Inference-time reasoning.** Fine-tune the same model to be an autoregressive reasoner: for every move it plays, it first generates a *reasoning trace* — a sequence of imagined chess moves played out on an imagined board — and then emits the actually-played move conditioned on that trace. This matches the structural role reasoning plays in o1/R1-style LLMs: removing the reasoning step at inference must measurably degrade move quality, or the design has failed its thesis.
3. **Training objective.** Replace fixed-depth GRPO over per-step rollouts with GRPO over sampled reasoning sequences, rewarded on the quality of the final played move against a real opponent, plus an auxiliary value-prediction loss on the imagined leaf trained against a frozen oracle.

The rest of this document specifies what exactly to build.

## 2. Goals and non-goals

### Goals

- Port the DeepMind 9M (and later 136M) searchless transformer to PyTorch with verified parity against the JAX reference.
- Extend the same ported model with a minimal set of adjustments (PE extension, vocabulary extension, new output head, prefix-LM attention on the thinking tail) so it can act as an autoregressive reasoning model.
- Fine-tune the full model with GRPO on sampled reasoning traces, rewarded by the quality of the final played move.
- Ensure the reasoning is **load-bearing at inference**: `eval_stockfish/think_delta = score_with_think − score_no_think` is the project's north-star metric.
- Constrain the reasoning to legal chess lines via an imagined-board legal mask at every decoder step inside `<think>`.
- Train both colors inside the imagined line: the same model generates both sides' moves, with turn alternation enforced by the imagined board.
- Document every known failure mode, metric, and safety check in prose, next to the code and in one central docs file.

### Non-goals (explicitly deferred)

- **Separate encoder + decoder architecture.** Evaluated during brainstorming and rejected in favor of a single fine-tuned model. See §10.
- **MCTS / tree search / AlphaZero-style training.** Evaluated and rejected — MCTS and the LLM-reasoning analogy are structurally incompatible, and MCTS + GRPO is not a coherent pair. See §10.
- **Teacher-forced thinking / SFT bootstrap of the `<think>` span.** Cheap add-on if the model fails to discover a useful reasoning protocol on its own. Not in this spec.
- **Full-game self-play episodes with game-outcome rewards.** Phase-2 extension. Not in this spec.
- **270M fine-tuning.** Cost is not justified over 136M given current eval noise.
- **Multi-PV / tree-of-thoughts reasoning.** A single linear imagined line per sample. Branching is deferred.
- **Porting the DM value head.** The frozen DM 136M value head stays in JAX as the leaf oracle.
- **Adaptive KL, entropy floors, entropy coefficient.** The project already removed these and stabilized. Do not reintroduce preemptively.

## 3. Architecture

### 3.1 Single-model design

One PyTorch model, instantiated once, fine-tuned end to end. It is the DeepMind searchless transformer with three minimal changes, all of which fall out directly from the requirement "use it autoregressively over a longer mixed-vocabulary sequence."

**The model state:**

- DM's transformer body (self-attention blocks, FFNs, LayerNorms), loaded from the JAX checkpoint via the weight converter.
- DM's existing token embedding matrix (FEN characters + 1968 action tokens), loaded from the checkpoint, plus **3 new rows** for `<think>`, `<end_think>`, `<move>`. The 3 new rows are initialized from scratch.
- An extended **learned positional encoding** matrix. DM's PE has exactly 79 slots; we extend it to `max_seq_len = 77 (FEN) + 1 (<think>) + max_think_tokens + 1 (<end_think>) + 1 (<move>) + 1 (m_final)`. The thinking line is linear — one move token per step regardless of which color is to move — so the budget is `max_think_tokens`, not `2 × max_think_tokens`. The configured `max_seq_len` in §5 includes a safety pad. New rows are initialized by copying the last DM-trained PE row (position 78), not random — this avoids a cold-start blowup where random PE dominates attention for hundreds of steps.
- A new **output head**: one linear layer projecting the final hidden state to logits over the full vocabulary (1968 actions + 3 specials, with tokens we'd never generate masked during sampling). Replaces DM's 128-bucket head entirely. Initialized from scratch.
- A new **auxiliary value head**: a 2-layer MLP on the hidden state at the `<end_think>` position, outputting a scalar in roughly `[-1, 1]`. Used for the auxiliary regression loss described in §4.

**Attention pattern (prefix-LM):**

The self-attention mask is constructed each forward pass and combines padding masking and positional constraints in one pass:

- **FEN positions (0–76)** attend bidirectionally to each other and to later tokens. This preserves DM's bidirectional FEN understanding — the part of DM that's actually doing heavy lifting is untouched.
- **Thinking / move positions (77+)** attend bidirectionally to all FEN positions (they need the board context) and **causally** to themselves and prior thinking/move positions.

The causal constraint on the thinking tail is necessary for the PPO recompute step to produce log-probs that match sampling-time conditioning. See §4 for the derivation. The mask is built in the same code path as padding, so adding the causal constraint is essentially free in compute — it's an `AND` of two boolean masks.

### 3.2 What "thinking" is, concretely

An inference pass on a root position `P` produces:

```
[FEN(P)] <think> m1 m2 m3 … m_T <end_think> <move> m_final
```

- `m1…m_T` are **chess moves** drawn from DM's existing 1968-action vocabulary — the same tokens DM was already trained with as inputs.
- The decoder steps inside `<think>` maintain an **imagined board**, initialized as a copy of `P`. After each sampled move, the imagined board is advanced with that move via `python-chess`. The next step's **legal mask** is computed from the imagined board's current side-to-move. Sides alternate naturally — one side per token — so both colors' moves come from the same model.
- When `<end_think>` is emitted (or when `max_think_tokens` is hit and `<end_think>` is forced), the imagined board is discarded.
- `<move>` is emitted as a fixed structural token.
- `m_final` is sampled with the legal mask computed from the **real root** `P`, not the imagined leaf.
- The real board is then advanced with `m_final`.

At inference time, the "reasoning" is directly visible in the sequence and directly influences `m_final` because the model sees the whole sequence as context when predicting `m_final`. This is what makes the reasoning load-bearing: ablating the `<think>` span and jumping straight from `[FEN]` to `<move>` produces a different (worse) `m_final`, which is the `think_delta` we measure.

### 3.3 Module layout

**New modules (all under `src/`):**

- **`dm_port/transformer.py`** — PyTorch reimplementation of the DM transformer (the body — attention blocks, FFNs, LNs, embedding lookups, PE). Mirrors `searchless_chess/src/transformer.py` layer-for-layer. Does not include DM's action-value head.
- **`dm_port/convert_jax.py`** — One-time weight conversion. Loads DM params from `searchless_chess/checkpoints/<9M|136M>/`, maps each JAX parameter to the corresponding PyTorch `state_dict` key, saves as `checkpoints/dm_port/<9M|136M>.pt`. Writes a side-car JSON with parameter-count checksums.
- **`tests/test_dm_port_parity.py`** — **Phase-0 gate.** On a fixed 32-FEN batch, runs the JAX DM encoder and the PyTorch port through equivalent inputs, asserts `max_abs_diff < 1e-2` in bfloat16, `< 1e-4` in float32. Nothing else merges until this is clean.
- **`reasoning_model.py`** — Wrapper over `dm_port.transformer` that adds: the 3 extra vocabulary rows, the extended PE, the new token-prediction head, the auxiliary value head, and the `build_attention_mask` helper that implements prefix-LM + padding in one pass. Loads the ported DM weights at construction time and leaves everything else trainable. Exposes:
  - `sample(root_fens, k, max_think_tokens)` — batched rollout. Returns token sequences, per-token `log_prob_old`, imagined-leaf FENs, `m_final` indices, and the bookkeeping masks the loss needs.
  - `logprob_sequence(root_fens, token_sequences)` — PPO recompute. Runs one forward pass per sequence with the prefix-LM mask and reads off per-position log-probs.
  - `value_at_end_think(root_fens, token_sequences)` — auxiliary value head output at the `<end_think>` position of each sample.
- **`rollout/reasoning_sampler.py`** — The rollout loop. For a batch of `B` roots × `K` samples, autoregressively generates `<think> m1 m2 … <end_think> <move> m_final` while maintaining per-sample imagined boards and computing legal masks. Calls `reasoning_model.sample()`.
- **`rollout/real_play.py`** — Takes each sample's `m_final`, pushes it on the real root board, has the self-play rival (same model, `<think>`-disabled forward pass with argmax over root moves) respond, and returns the post-response FEN for leaf evaluation. Decoupled from the reasoning sampler.
- **`grpo_logic/reasoning_loss.py`** — Computes group-normalized advantages, the PPO surrogate over sampled tokens, the KL term against the frozen old policy, and the auxiliary `L_value = MSE(v̂, v_frozen_leaf)`. Returns the total loss plus a dict of logged scalars. Reuses primitives from the existing `grpo_logic/loss.py` where possible.

**Rewritten modules:**

- **`grpo_logic/model.py`** — Lightning module. Same skeleton; `training_step` rewired to the reasoning path (see §4).
- **`chess/rewards.py`** — Single function `evaluate_leaf(fen, turn) -> float` in `[-1, 1]`. Internally dispatches to Stockfish shallow eval or the frozen JAX DM 136M value head based on `leaf_evaluator.mode`. Interface stable; backend configurable.

**Deleted modules:**

- `src/models.py` (`ChessTransformer`).
- `src/models_9m.py`.
- `src/grpo_logic/sampling.py` (replaced by `rollout/reasoning_sampler.py`).
- `src/grpo_self_play/` (already empty).
- `src/pretrain/` (no longer needed — DM is the starting point).
- The `transformer:` and `pretrain:` blocks in `configs/default.yaml`.

**Unchanged:**

- `trainer.py`, `evaluator.py`, `eval_utils.py`, `config_loader.py`.
- `chess/boards_dataset.py`, `chess/stockfish.py`, `chess/chess_logic.py`, `chess/policy_player.py`.
- `StockfishEvalCallback`, WandB logging infrastructure.
- `distill/` — off the critical path, left as-is.

## 4. Training loop and data flow

One full training step, end to end:

1. **Root sampling.** `boards_dataset.py` yields a batch of `B` root FENs, phase-distributed and quality-filtered. No change from today.

2. **Reasoning rollout.** For each root, the sampler generates `K` parallel samples. For each sample, starting from position `<think>`:
   - Maintain an imagined board (initialized as a copy of the root).
   - At each step inside `<think>`: forward-pass the model on the sequence so far, read logits at the last position, compute the legal mask from the imagined board's current side-to-move (plus `<end_think>` if `step >= min_think_tokens`), softmax + sample at `rollout_temperature`, record `log_prob_old` for the sampled token, advance the imagined board.
   - When `<end_think>` is sampled (or forced at `max_think_tokens`), freeze the imagined board's final FEN as the imagined leaf.
   - Emit `<move>` as a fixed structural token, one more forward pass.
   - At the final move step: compute the legal mask from the **real root** (not the imagined leaf), sample `m_final`, record `log_prob_old`.
   - Outputs per sample: full token sequence, per-token `log_prob_old`, imagined-leaf FEN, `m_final` index, masks indicating which positions contribute to which loss component.

3. **Real play for reward scoring.** For each sample, push `m_final` on the real root board, have the self-play rival (same model, `<think>`-disabled forward pass, argmax over root moves for determinism) respond, produce the post-response FEN.

4. **Leaf evaluation — main reward.** The post-response FEN is scored by the frozen leaf evaluator → `R_final_move ∈ [-1, 1]`. Terminal positions get game-result rewards (±1 / 0).

5. **Leaf evaluation — auxiliary target.** The imagined-leaf FEN is scored by the **same** frozen evaluator → `v_frozen`. This is the regression target for the auxiliary value head.

6. **Group-normalized advantages.** Reshape `R_final_move` to `[B, K]`, compute `adv = (r − r.mean(dim=1)) / (r.std(dim=1) + eps)`, reshape back to `[B·K]`, broadcast across time so every token in a sample receives its sample's advantage.

7. **PPO recompute and surrogate.** One forward pass per sequence with the prefix-LM attention mask. Read `log_prob_new` at every sampled token position. Compute:
   ```
   ratio     = exp(log_prob_new − log_prob_old)
   unclipped = ratio * adv
   clipped   = clamp(ratio, 1-ε, 1+ε) * adv
   ppo_loss  = -min(unclipped, clipped)   # masked
   kl        = (log_prob_old − log_prob_new)   # masked
   ```
   **Correctness note.** The prefix-LM causal mask on the thinking tail is required here: without it, a single-pass recompute would compute `log_prob_new` conditioned on tokens that came *after* the token being predicted in the sampled sequence, breaking the PPO ratio. With prefix-LM, the single-pass recompute is equivalent to running `T` separate prefix-by-prefix forward passes, at roughly `T`× less compute.

8. **Auxiliary value loss.** `L_value = MSE(v̂, v_frozen)` computed at exactly the `<end_think>` position of each sample — nowhere else.

9. **Total loss.** `L = ppo_loss.mean() + kl_coef · kl.mean() + value_coef · L_value.mean()`.

10. **Metrics** logged to WandB. Full list in §6.

11. **Old-policy sync** once per epoch, unchanged. Copies all trainable weights (embedding rows, PE rows, transformer body, output head, value head).

**Temperature invariant.** The value of `rollout_temperature` used when sampling in step 2 **must** be the same value used when recomputing log-probs in step 7. This is the exact bug fixed in `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`; the port must not reintroduce it. See §7 failure mode "Clip-fraction blowup" for the history.

## 5. Configuration

Starting values. All tunable.

```yaml
model:
  base_checkpoint: "checkpoints/dm_port/9M.pt"   # 136M.pt after parity
  freeze_body: false                             # full fine-tune; true → head + PE only
  max_seq_len: 120                               # FEN (77) + think budget + specials
  value_head:
    enabled: true
    hidden_dim: 256
    coef: 0.5                                    # weight of L_value in total loss

reasoning:
  min_think_tokens: 2
  max_think_tokens: 12
  allow_early_end_think: true                    # decoder may emit <end_think> any time after min
  legal_mask_think: true                         # enforce legal chess moves inside <think>
  exclude_specials_from_gradient: false          # start false; ablate later

grpo:
  k_samples_per_root: 8                          # group size for GRPO
  lr: 1e-6                                       # DM-finetune range; will need tuning
  clip_ratio: 0.20
  kl_coef: 0.001
  ppo_epochs: 1
  rollout_temperature: 1.0
  move_sampling_temperature: 1.0                 # temperature for the m_final token specifically

rival:
  mode: "self_play"                              # "self_play" | "frozen_dm_9m" | "stockfish"
  self_play:
    think_enabled: false                         # rival does not think when computing R_final_move
    temperature: 0.0                             # argmax for determinism
  frozen_dm_9m:
    checkpoint_path: "checkpoints/dm_port/9M.pt"
  stockfish:
    movetime_ms: 50
    skill_level: 2

leaf_evaluator:
  mode: "dm_value_head"                          # "dm_value_head" | "stockfish"
  dm_value_head:
    model: "136M"                                # stays in JAX — no port needed for the oracle
    bucket_values_source: "searchless_chess"
  stockfish:
    movetime_ms: 50
    cp_clip: 1000

training:
  num_epochs: 200
  batch_size: 16                                 # roots per step; effective batch = 16 * k_samples_per_root
  steps_per_epoch: 256
  old_policy_sync: "per_epoch"
  checkpoint_every_n_epochs: 5
  keep_n_checkpoints: 3
  use_wandb: true
  wandb_project: "Chess-GRPO-Bot"

eval:
  games: 64
  seed: 0
  max_plies: 400
  randomize_opening: true
  opening_plies: 6
  ablation_no_think: true                        # eval also runs a variant with <think> skipped

dataset:
  # unchanged: phase distribution, quality filter, etc.
```

### Compute budget (9M phase)

- One sampled sequence ≈ `max_think_tokens + 3` forward steps ≈ 15.
- Per training step:
  - Sampling: `16 roots × 8 samples × 15 steps` ≈ 1.9k forward passes.
  - PPO recompute: one forward pass per sequence = `16 × 8 = 128` forward passes.
  - Real-play rival: one forward pass per sample = 128.
  - Leaf evaluator: one JAX call per sample (or two when also scoring the imagined leaf for `L_value`) = up to 256.
- At 9M this fits comfortably on a single L4/A100 with training steps measured in seconds.

### Scaling to 136M

Per-token FLOPs grow ~15×. Start at `batch_size=8`, `k_samples=6` and scale from there.

## 6. Metrics

All metrics logged under `train/` or `eval_stockfish/`.

**Policy / loss:**
- `train/loss`, `train/ppo_loss`, `train/kl_divergence`, `train/clip_fraction`, `train/ratio`.
- `train/advantage_mean`, `train/advantage_std`.
- `train/reward_mean` — mean of `R_final_move`.

**Reasoning path:**
- `train/think_length_mean`, `train/think_length_p50`, `train/think_length_p90`.
- `train/frac_min_length_samples`, `train/frac_max_length_samples`.
- `train/pad_fraction`.
- `train/leaf_value_mean` — mean `v_frozen` across imagined leaves.
- `train/v_hat_mean`, `train/L_value_mean` — auxiliary head health.
- `train/v_hat_minus_R_move_mean` — optimism of imagined lines relative to real play. Should trend to zero.
- `train/reward_spread_within_group` — per-group std of `R_final_move`. If this → 0, GRPO has no signal.

**North-star eval metric:**
- `eval_stockfish/score_with_think`.
- `eval_stockfish/score_no_think`.
- `eval_stockfish/think_delta = score_with_think − score_no_think` — **the headline metric**. If it stays at zero, the refactor has failed its thesis.

**Observability aids (non-scalar):**
- Every ~500 training steps, dump 10 generated sample sequences as readable text to WandB. Text inspection catches pathologies scalars won't — e.g., the model repeating the same 3 moves, one-sided cheating, hallucinated pieces.
- Every eval epoch, log the top-10 most common thinking prefixes.
- Every ~500 steps, log a `v̂` vs `v_frozen` scatter plot.

## 7. Known failure modes

Each mode below is also documented in `docs/reasoning_grpo/failure_modes.md` with its own H2 section in prose: **Symptom / How you'll notice / Recovery / History**. When new failure modes are discovered in live runs, they are added to that file as new H2 entries — no code changes, no test updates, just writing. See §9 for the documentation rule.

### 7.1 DM port parity break (Phase-0 gate)
- **Symptom.** PyTorch port produces different transformer outputs than JAX on identical inputs.
- **How you'll notice.** `dm_port/test_parity.py` fails with `max_abs_diff` above threshold.
- **Recovery.** Do not train. Most likely culprits: LN epsilon, positional-encoding shape, residual scaling, embedding load mapping, attention-mask construction.

### 7.2 Reasoning collapse
- **Symptom.** Model emits `<end_think>` immediately; thinking span degenerates.
- **How you'll notice.** `train/think_length_mean` drops toward `min_think_tokens`; `train/frac_min_length_samples` → 1.
- **Recovery.** Raise `min_think_tokens`; mask `<end_think>` for the first N steps; as a last resort, enable teacher-forced thinking (§11).

### 7.3 Reasoning runaway
- **Symptom.** Nearly every sample hits `max_think_tokens`. `L_value` stays high, `R_final_move` doesn't improve.
- **How you'll notice.** `train/frac_max_length_samples` stays near 1.
- **Recovery.** Lower `max_think_tokens`; raise `value_coef`; check whether special-token masking is hiding a length signal.

### 7.4 Auxiliary-value / policy-gradient tug-of-war
- **Symptom.** `L_value` decreases but `R_final_move` stagnates, or vice versa.
- **How you'll notice.** Plot `L_value` and `R_final_move` together; grad-norm ratios between heads go out of balance.
- **Recovery.** Tune `value_coef`. If conflict persists, decouple optimizers.

### 7.5 Thinking dishonesty (imagined-line cheating)
- **Symptom.** Model imagines lines where it plays well and the imagined opponent plays badly, inflating `v̂` without corresponding improvement in real play.
- **How you'll notice.** `train/v_hat_minus_R_move_mean` stays large and positive.
- **Recovery.** The two safeguards in the design — MSE regression of `v̂` against the frozen oracle, and GRPO penalty on bad `R_final_move` — should self-correct. If they don't: raise `value_coef`, then consider switching `rival.mode` to `frozen_dm_9m` to reduce the reward-hacking incentive.

### 7.6 Clip-fraction blowup
- **Symptom.** `train/clip_fraction` > ~0.3, especially at step 0 or right after old-policy sync.
- **How you'll notice.** `train/clip_fraction` spike in WandB.
- **Recovery.** First suspect: temperature mismatch between rollout and recompute (see §4 invariant). This is the repeat of the 55%-clip-fraction bug documented in `research_docs/2026-02-07_clean-run-temperature-mismatch-bug.md`. Second suspect: KL drift from old-policy sync cadence.

### 7.7 GRPO group collapse
- **Symptom.** All `K` samples in a group get the same reward. Advantages are zero; nothing learns.
- **How you'll notice.** `train/reward_spread_within_group` → 0.
- **Recovery.** Raise `rollout_temperature`; verify dropout is on; verify the K samples use different random seeds.

### 7.8 Reasoning doesn't help (the null result)
- **Symptom.** After meaningful training, `eval_stockfish/score_with_think ≈ score_no_think`.
- **How you'll notice.** `eval_stockfish/think_delta` stays near zero.
- **Recovery.** Design-level failure, not tuning. Reopen the brainstorm. Candidates: teacher-forced thinking bootstrap, reward shaping on imagined lines, different leaf evaluator, jump to Phase 2 (full-game self-play), reconsider the imagined-line semantics.

### 7.9 PE cold-start blowup
- **Symptom.** Early-training loss is nonsense; thinking tokens look random even after many steps; gradients at PE positions 80+ are huge.
- **How you'll notice.** `train/loss` trajectory never settles; inspecting sample sequences shows garbage in the thinking span.
- **Recovery.** The design mitigates this by initializing new PE rows from the last DM-trained row (position 78), not random. If this still blows up, try: lower LR on PE rows only for first N steps; or warmup-freeze the body and train PE + head only for the first few hundred steps.

## 8. Testing strategy

KISS. Tests check behavior, not meta-consistency between docs and code. Duplication between test intent and code comment is fine.

**Phase-0 gate (must pass before anything else merges):**
- `test_dm_port_parity.py` — ported model matches JAX on a fixed 32-FEN batch within the §7.1 thresholds.

**Invariants:**
- `test_temperature_consistency.py` — asserts rollout sampling temperature equals PPO recompute temperature in a small smoke training step.
- `test_prefix_lm_mask.py` — constructs an attention mask for a synthetic sequence, asserts: FEN positions attend bidirectionally to everything; thinking positions attend only to FEN + prior thinking positions; padding positions are fully masked.

**Rollout sampler unit tests:**
- `test_reasoning_sampler_legal_masks.py` — sampler never produces illegal moves on the imagined board, for both white-to-move and black-to-move roots, including for opponent-turn tokens inside the line.
- `test_reasoning_sampler_imagined_board_reset.py` — `<move>` token mask is computed from the real root, not the imagined leaf.
- `test_reasoning_sampler_termination.py` — sampler handles mate, stalemate, and insufficient-material terminations inside the imagined line.
- `test_reasoning_sampler_shapes.py` — for assorted `(B, K, max_think_tokens)` combinations, output tensor shapes match contract.

**Loss unit tests:**
- `test_reasoning_loss_group_normalization.py` — known rewards in, expected group-normalized advantages out.
- `test_reasoning_loss_value_head_position.py` — `L_value` is computed at exactly the `<end_think>` position, nowhere else.
- `test_reasoning_loss_mask_coverage.py` — pad and special-token masks behave as configured.

**Integration / smoke:**
- `test_e2e_training_step_smoke.py` — one full `training_step` at `B=2, K=2, max_think_tokens=3`. Finite loss, gradients flow into embedding (including new rows), PE (including new rows), body, output head, value head. All expected WandB keys present.
- `test_think_ablation_wiring.py` — sampler called with thinking disabled emits only `<move> m_final`, correctly passes through loss.

**Not doing in Phase 1:** property-based tests on the sampler, fuzz tests on weird FENs, adversarial test positions. Worth doing eventually, not now.

## 9. Documentation rule

**Knowledge home.** `docs/reasoning_grpo/failure_modes.md` is the single file where known failure modes live in long form. Each failure mode is one H2 section with four fixed subsections: **Symptom**, **How you'll notice**, **Recovery**, **History**. Prose only. No registries, no data structures, no generated content.

**Inline code comments.** At every metric log site and every safety check, a block comment restates the relevant symptom and recovery **in its own words**, in full. Duplication with the docs file is intentional — a developer reading the code never has to leave the file to understand what a metric means or how to respond to a trip.

**Error messages.** When a safety check trips, the error message is a self-contained paragraph in plain English: what was observed, what the symptom means, what to try first, and a pointer to `docs/reasoning_grpo/failure_modes.md`. No central helper. These messages are read at 3am when a run is stuck; a self-contained paragraph is worth more than a tidy abstraction.

**When contributing.** If you add a safety check, a metric that exists because a past run went wrong, or handle a new failure observed in a run: write it up in `docs/reasoning_grpo/failure_modes.md` as a new H2 section using the template above. And leave a comment at the code site that explains the same thing in its own words. Duplication is intentional.

This rule is also pinned in `CLAUDE.md` under a "Troubleshooting" section so it stays visible.

## 10. Alternatives considered and rejected

**Approach 1: GRPO with PV-leaf rewards (search at training only).** Sample K rollouts, score leaves with a value function, GRPO. Rejected because the reasoning trace is only a training-time artifact; at inference, the one-step argmax encodes everything the search found during training, so removing the search at inference costs almost nothing. The reasoning is not load-bearing at inference, which fails the core thesis.

**Approach 2: PPO with a learned critic.** Viable but adds a second head with its own failure modes (value blowups, policy-critic divergence). The repo's recent history shows GRPO's simplicity has been the thing that actually ran stably.

**Approach 3: Expert iteration / search-guided SFT.** Use search to produce better labels, SFT the student on them, repeat. Stable but sidesteps the premise — reasoning is cosmetic at inference, same as Approach 1. Kept in mind as an emergency backup.

**Path β: AlphaZero-style MCTS + policy distillation.** MCTS is a valid chess algorithm but structurally not how LLMs reason. LLM reasoning is autoregressive sampling of a single linear trace; MCTS is tree search with PUCT and backup. The two are not analogs. MCTS + GRPO is not a coherent pair: MCTS-generated traces are off-policy w.r.t. the network's prior, and the PPO ratio becomes ill-defined.

**Separate encoder + decoder architecture.** The original version of Approach 4 had a separate small decoder cross-attending into the DM encoder. Rejected in favor of the single fine-tuned model because: (a) fewer parameters; (b) reasoning sees board representations directly rather than through a cross-attention bottleneck; (c) matches how real LLMs work; (d) simpler code. The three changes (PE extension, vocab extension, new output head) and the prefix-LM attention mask are small and fall out naturally from the single-model approach.

**Sequential prefix-by-prefix recompute.** Evaluated as an alternative to the prefix-LM causal mask. Would avoid any attention-mask change but cost `~T+3` forward passes per sample during recompute. Rejected because the padding mask already has to be constructed during every forward, and adding a causal constraint to it is essentially free (same tensor op), whereas the O(T) recompute multiplier compounds over training.

**Porting the DM value head.** The frozen DM 136M value head stays in JAX as the leaf oracle. Porting it would add work for no benefit — it's frozen, so JAX inference in `distill/teacher.py`-style fashion is sufficient.

## 11. Deferred / future work

- **Teacher-forced thinking (SFT bootstrap for `<think>`).** During early training, substitute the model's sampled thinking tokens with a reference line (Stockfish PV at shallow depth or DM 136M's best line) and train the head to match via cross-entropy. Analogous to R1's cold-start SFT before the pure-RL phase. A cheap add-on if the model fails to discover a useful reasoning protocol on its own.
- **Full-game self-play episodes.** Roll out entire games with reasoning at every move, use game outcomes as the reward. Phase-2 extension once leaf-reward GRPO is working.
- **136M scaling run.** After 9M end-to-end pipeline shows improvement, reuse the same code for 136M via a config change plus bigger-GPU config.
- **Decoupled optimizers** for the policy head vs value head, if §7.4 proves hard to tune away.
- **Multi-PV / tree-of-thoughts reasoning.** The model emits multiple candidate lines per sample with a structural separator token. Richer reasoning, more design work.
- **Backtracking inside `<think>` (`<undo>` token).** Add one more special token that pops one ply off the imagined board when sampled, letting the model recover from a bad first move in a line instead of wasting the rest of the budget. Deferred for Phase 1 because (a) the MVP question is "does `think_delta > 0` at all?" and adding backtrack muddies that measurement, (b) in LLMs backtracking emerged from pretraining corpora full of human self-correction and we don't have that corpus here, so the model would have to discover `<undo>` from RL alone — a much harder exploration problem, (c) the linear line already degrades gracefully since imagined moves don't commit to the real board, so backtracking is an optimization over a design that already works, and (d) it introduces new failure modes (e.g., `<undo>` spam that stalls the budget) on top of a refactor that already changes a lot. Revisit alongside multi-PV once linear reasoning is validated.
- **PE cold-start mitigations beyond neighbor-copy init.** Per-parameter LR schedules or warmup-freezes if §7.9 is not solved by the current init scheme.
- **Property-based sampler tests** (Hypothesis), fuzz tests on weird FENs, adversarial positions.

## 12. Open questions

- **Should the `m_final` token's gradient be down-weighted relative to thinking tokens?** Currently all tokens in a sample receive the same advantage. If `m_final`'s gradient dominates (because it's tied most directly to the reward), the thinking tokens may not receive enough signal. Measure in the first training runs.
- **Should the rival ever think?** Phase-1 default is `think_enabled: false` on the rival for compute and determinism, but a reasoning rival might make `R_final_move` more informative.
- **How aggressively to schedule `max_think_tokens`?** Start small and grow is the default plan, but the schedule shape is not fixed in this spec.
- **Does `exclude_specials_from_gradient = true` help?** Unknown. Start `false`, measure, flip if useful.
- **Should new PE rows be initialized from position 78 (last trained) or from an average of the last few trained rows?** Unclear; the latter is slightly more robust but marginal.

---

*End of design document. Next step: implementation plan via the `writing-plans` skill.*
