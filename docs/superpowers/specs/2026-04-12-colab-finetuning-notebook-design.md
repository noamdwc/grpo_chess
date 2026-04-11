---
title: "Main Colab Notebook for Reasoning-GRPO Fine-Tuning"
date: "2026-04-12"
status: "draft — awaiting implementation plan"
supersedes:
  - "grpo_9m_colab.ipynb"
  - "full_pipeline_notebook.ipynb"
  - "README.md notebook entry guidance"
---

# Main Colab Notebook for Reasoning-GRPO Fine-Tuning

## 1. Context and motivation

The branch's notebook story is currently out of sync with the code:

- `grpo_9m_colab.ipynb` still calls `src.train_self_play.train(..., dataloader_kwargs=...)`, but the refactored `train()` entry point no longer accepts that argument.
- `full_pipeline_notebook.ipynb` still imports deleted pretrain modules and assumes the older pretrain → distill → GRPO pipeline exists on this branch.
- `README.md` still points to `chess_model_run_git.ipynb`, which is not present in the repository.

At the same time, the branch now has a clear primary training path: the reasoning-enabled DeepMind-model fine-tuning flow exposed through `src.train_self_play`. The main Colab notebook should reflect that path directly instead of preserving legacy stages that no longer match the refactored project.

This design replaces the notebook entry point for the project with a single Colab workflow for:

1. mounting Google Drive,
2. preparing a Drive-local base checkpoint,
3. generating a Colab runtime config aligned with the refactored schema, and
4. launching reasoning-GRPO fine-tuning through the current `src.train_self_play.train()` API.

## 2. Goals and non-goals

### Goals

- Make one notebook the canonical Colab entry point for the project.
- Align the notebook with the current reasoning-GRPO fine-tuning stack on this branch.
- Support Google Drive as the persistent root for checkpoints, caches, configs, and run outputs.
- Include checkpoint preparation steps for Drive-local base checkpoints.
- Support three user-facing modes: `smoke`, `full`, and `resume`.
- Keep the notebook operational across reruns in the same Colab runtime without manual cleanup.
- Make resume behavior explicit and bounded by what the current code actually supports.

### Non-goals

- Reintroducing the deleted pretraining pipeline into the notebook.
- Reintroducing the legacy full pipeline notebook structure as a first-class workflow.
- Supporting external checkpoint download sources in the first version.
- Implementing true PyTorch Lightning trainer-state resume if the current training entry point does not support it.
- Creating multiple primary notebooks for different training paths.

## 3. Recommended approach

Three approaches were considered:

1. Replace the project's main notebook with a GRPO-only reasoning fine-tuning notebook that includes Drive-based checkpoint preparation.
2. Keep a hybrid notebook with GRPO as the primary path but leave partial or legacy pretrain/distill sections in place.
3. Split the workflow across multiple notebooks, such as one for setup and one for training.

The recommended approach is **Approach 1**.

It matches the current branch reality:

- the stable training entry point is `src.train_self_play`,
- the refactor is centered on reasoning-enabled GRPO fine-tuning of the DeepMind model, and
- the old notebook content is already stale enough to mislead users.

Replacing the main notebook is preferable to patching the old notebooks because it gives the project one clear story instead of a mix of working and non-working Colab paths.

## 4. Notebook scope

The notebook is the main Colab interface for the project's current fine-tuning path:

- start from a Google-Drive-local DeepMind-compatible checkpoint,
- run the reasoning-enabled model path introduced by the refactor,
- fine-tune that model with GRPO via `src.train_self_play.train(config_path=...)`, and
- write runtime configs and outputs back to Google Drive.

The notebook is intentionally **not** a general project orchestrator. It does not manage pretraining or distillation stages. Those flows are no longer the right top-level story for this branch.

## 5. Runtime shape

The notebook should be organized into five high-level sections.

### 5.1 Environment setup

This section:

- mounts Google Drive,
- clones the repository into the Colab runtime,
- installs Python dependencies,
- configures cache and artifact paths,
- ensures the runtime can be reused without fragile manual cleanup,
- validates that the notebook is operating against the checked-out repository copy.

The setup logic should be Colab-native and robust to reruns.

### 5.2 Run configuration

This section exposes a small number of notebook parameters:

- `MODE`: `smoke`, `full`, or `resume`
- `DRIVE_ROOT`
- `BASE_CHECKPOINT_PATH`
- `RESUME_CHECKPOINT_OR_DIR`
- `RUN_NAME`
- `USE_WANDB`

The parameter cell should remain small and human-editable. Anything derived from those inputs belongs in helper cells, not in the user-facing parameter block.

### 5.3 Checkpoint preparation

This section handles Drive-local checkpoints only.

For `smoke` and `full`:

- validate that the base checkpoint exists in Google Drive,
- normalize its path into the runtime config,
- ensure all output and cache directories under Drive exist.

For `resume`:

- validate the user-provided resume target,
- normalize directory-vs-file input if needed,
- map the chosen checkpoint into the config strategy described below.

No external download logic is included in the initial design.

### 5.4 Training launch

This section:

- selects a committed base YAML suited for reasoning-GRPO,
- applies mode-specific overrides,
- writes a runtime YAML for Colab execution,
- imports the current training entry point,
- launches `src.train_self_play.train(config_path=runtime_config_name)`.

The notebook must not call deleted modules or pass stale notebook-only arguments such as `dataloader_kwargs`.

### 5.5 Verification and handoff

This section:

- prints the resolved runtime config path,
- prints the resolved checkpoint and output directories,
- verifies that expected run artifacts were created,
- shows the exact paths the user should supply to continue a run later.

This closes the loop cleanly for Colab usage, where path visibility matters more than on local development.

## 6. Components

The implementation should be broken into the following notebook-level components.

### 6.1 Main notebook file

One notebook becomes the canonical project notebook and supersedes the current stale fine-tuning notebook. The README should refer to this notebook as the primary Colab entry point.

### 6.2 Parameter cell

The parameter cell contains only the primary knobs the user is expected to edit:

- mode,
- Drive root,
- checkpoint inputs,
- run name,
- WandB toggle.

This cell is the notebook's public interface.

### 6.3 Drive-backed path resolver

This component derives stable Drive paths for:

- repository-side artifacts,
- caches,
- generated runtime configs,
- checkpoint directories,
- run output directories.

All persistent state should be rooted under `DRIVE_ROOT`, not scattered across ad hoc Colab paths.

### 6.4 Runtime config generator

This component reads a committed YAML config from `src/configs/`, applies Colab- and mode-specific overrides, and writes a generated runtime YAML for the current run.

This is preferred over embedding long inline YAML blobs into notebook cells because:

- defaults stay versioned in the repository,
- diffs remain understandable,
- refactors to the config schema stay localized to the config layer.

### 6.5 Training launcher

This component imports the current refactored training entry point and launches it without relying on deleted or legacy APIs.

### 6.6 Final report cell

This component verifies and prints the final run state for the user, including exact Drive paths for checkpoints and runtime config files.

## 7. Data flow

### 7.1 `smoke` and `full`

For `smoke` and `full`, the flow is:

1. user selects a Drive-local base checkpoint,
2. notebook validates that checkpoint,
3. notebook selects a committed base GRPO config,
4. notebook applies Colab-specific and mode-specific overrides,
5. notebook writes a runtime config under Drive,
6. training runs and writes outputs back to Drive.

The difference between `smoke` and `full` is in the override set, not in the notebook structure.

### 7.2 `resume`

For `resume`, the flow is:

1. user selects an existing Drive-local run checkpoint or run directory,
2. notebook validates and normalizes that input,
3. notebook generates a runtime config that points training at the resumed model state,
4. training continues as a fresh launch from that checkpoint-derived state.

The notebook UI should accept either a checkpoint path or a run directory if that materially improves usability, but it should normalize the input into the exact form the code supports.

## 8. Resume semantics

The current codebase does **not** expose true trainer-state resume through `src.train_self_play`.

At present:

- `src.train_self_play.train()` calls `trainer.fit(module, dataloader)`,
- no `ckpt_path` is passed to `trainer.fit(...)`,
- the notebook therefore cannot honestly claim exact Lightning-state continuation.

For the first version of this notebook, `resume` means:

- use a prior saved checkpoint as the new base model state for further reasoning-GRPO fine-tuning,
- preserve the run's functional continuation from model weights,
- do not promise restoration of optimizer state, callback state, or exact epoch/step continuation unless implementation work explicitly adds that support.

This constraint should be visible in notebook text so the user understands what kind of resume they are getting.

## 9. Error handling and constraints

The notebook should fail fast on the most common Colab and path-related issues:

- Google Drive is not mounted,
- `DRIVE_ROOT` is invalid,
- the base checkpoint path is missing for `smoke` or `full`,
- the resume target is missing or invalid for `resume`,
- the source config file is missing,
- a generated path escapes the intended Drive root,
- the code import path is stale or incompatible with the notebook cells.

Validation should happen before launching training, not after a long setup sequence.

The notebook should also import `src.train_self_play.train` before launch so that stale notebook/code mismatches are caught early.

## 10. Testing and validation strategy

Validation should happen in two layers.

### 10.1 In-notebook validation

Before training:

- import the training entry point,
- load the chosen runtime YAML through the current config loader,
- print the resolved `model.base_checkpoint`,
- print the resolved `training.checkpoint_dir`,
- print the resolved mode,
- verify that referenced paths exist or are creatable.

After training:

- verify that the run output directory exists,
- verify that at least one checkpoint artifact was written,
- verify that the generated runtime YAML exists in Drive next to the run outputs or in a clearly related Drive location.

### 10.2 Repo-side verification

After implementation, the minimal repository-level verification should confirm that:

- the main notebook no longer imports deleted pretrain modules,
- the notebook no longer passes `dataloader_kwargs` into `src.train_self_play.train`,
- the README points to the new main notebook,
- the notebook structurally targets the reasoning-GRPO fine-tuning path rather than the deleted legacy pipeline.

## 11. Implementation boundaries

This design intentionally stays focused on the notebook entry point and related documentation updates.

Expected touched areas:

- the main notebook file,
- possibly the existing stale notebook files if they are removed or demoted,
- `README.md` notebook guidance,
- any committed Colab-oriented GRPO config template needed to support the new workflow.

Not in scope unless implementation reveals a hard blocker:

- broad trainer refactors,
- restoring the deleted pretrain pipeline,
- introducing external checkpoint download integrations,
- unrelated cleanup of non-notebook code paths.

## 12. Success criteria

This design is successful when:

- the project has one clear, main Colab notebook,
- that notebook matches the reasoning-GRPO refactor on this branch,
- a user can run `smoke`, `full`, or `resume` using Google Drive-backed paths,
- the notebook prepares a Drive-local checkpoint and writes Drive-local outputs,
- the notebook does not reference deleted pretrain code or stale GRPO APIs,
- the README points users to the correct notebook.
