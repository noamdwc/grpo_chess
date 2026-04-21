# BC Checkpoint Interface and Warmstart

## Summary

This change documents and implements support for the released DeepMind 9M behavioral-cloning checkpoint as a distinct checkpoint family.

The key outcome is that the released upstream checkpoint families do **not** share the same input interface:

- `dm_action_value` uses input vocab `1968`, output size `128`, positional length `79`
- `dm_behavioral_cloning` uses input vocab `31`, output size `1968`, positional length `78`
- `dm_state_value` uses input vocab `31`, output size `128`, positional length `78`

This means the behavioral-cloning checkpoint cannot be treated as a drop-in replacement for the existing DM action-value port.

## What changed

### 1. Checkpoint inspection utility

Added [`src/checkpoint_inspector.py`](</Users/noamc/repos/grpo_chess/src/checkpoint_inspector.py:1>) to classify checkpoints and report:

- checkpoint format
- family
- input vocab size
- output size
- positional length
- checkpoint step when available

Examples:

```bash
~/miniconda3/envs/grpo_chess/bin/python -m src.checkpoint_inspector checkpoints/dm_port/9M.pt
~/miniconda3/envs/grpo_chess/bin/python -m src.checkpoint_inspector searchless_chess/checkpoints/9M_behavioral_cloning
~/miniconda3/envs/grpo_chess/bin/python -m src.checkpoint_inspector searchless_chess/checkpoints/9M_state_value
```

Observed outputs on local checkpoints:

- `checkpoints/dm_port/9M.pt` → `dm_action_value`, `1968 -> 128`, `79`
- `searchless_chess/checkpoints/9M_behavioral_cloning` → `dm_behavioral_cloning`, `31 -> 1968`, `78`
- `searchless_chess/checkpoints/9M_state_value` → `dm_state_value`, `31 -> 128`, `78`

### 2. BC warmstart support in `ReasoningModel`

Updated [`src/reasoning/model.py`](</Users/noamc/repos/grpo_chess/src/reasoning/model.py:1>) so model construction now distinguishes checkpoint families.

Supported warmstart families:

- `dm_action_value`
- `dm_behavioral_cloning`

For `dm_behavioral_cloning`, the model now:

- preserves the pretrained 31-row FEN embedding
- preserves the pretrained 1968-row action classifier head
- extends positional embeddings from 78 to the requested max sequence length by copying the last row
- leaves reasoning-specific rows initialized for the unified reasoning token space

Unsupported families now fail fast instead of being silently misused.

### 3. BC checkpoint conversion tool

Added [`src/dm_port/convert_behavioral_cloning.py`](</Users/noamc/repos/grpo_chess/src/dm_port/convert_behavioral_cloning.py:1>) to convert the released upstream Orbax BC checkpoint into a PyTorch `.pt` file compatible with the updated `ReasoningModel`.

Usage:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
~/miniconda3/envs/grpo_chess/bin/python -m src.dm_port.convert_behavioral_cloning \
  --checkpoint-dir searchless_chess/checkpoints/9M_behavioral_cloning \
  --out checkpoints/dm_port/9M_behavioral_cloning.pt
```

The `8` host devices are required because the released BC Orbax checkpoint is saved with 8-way sharding.

## Why this was necessary

The earlier wrapper-parity failure was initially investigated against the raw action-value `9M` checkpoint. During follow-up, it became clear that:

1. behavioral-cloning is the correct family if we want a pretrained action classifier
2. the released BC checkpoint uses a different input interface from the action-value checkpoint
3. the old `ReasoningModel` always threw away the pretrained output head anyway

So even with the correct checkpoint family, parity remained broken until the model explicitly supported BC-family warmstart semantics.

## Tests

Added/updated:

- [`tests/test_checkpoint_inspector.py`](</Users/noamc/repos/grpo_chess/tests/test_checkpoint_inspector.py:1>)
- [`tests/test_reasoning_model_forward.py`](</Users/noamc/repos/grpo_chess/tests/test_reasoning_model_forward.py:1>)

Verified with:

```bash
~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/test_checkpoint_inspector.py tests/test_reasoning_model_forward.py -q
```

Result:

- `10 passed`

## Remaining limitation

`ReasoningModel` does **not** directly consume the raw Orbax BC checkpoint at runtime.

The supported path is:

1. inspect the checkpoint with `src.checkpoint_inspector`
2. convert the BC checkpoint with `src.dm_port.convert_behavioral_cloning`
3. pass the converted `.pt` file into `ReasoningModel`

This keeps ordinary runtime model construction free of Orbax/XLA restore requirements.
