# Lightning Cloud Jobs Runbook

This document explains how to run GRPO/Distill/Pretrain jobs for this project on Lightning.ai Teamspace.

## 1) Prerequisites

- Lightning access configured locally for MCP:
  - `mcp_lightning_server/config.py` uses:
    - `LIGHTNING_STUDIO_NAME`
    - `LIGHTNING_TEAMSPACE`
    - `LIGHTNING_USER`
- MCP server configured in `.mcp.json` under `"lightning"`.
- Repo branch with your changes pushed to GitHub (jobs clone from remote).
- Teamspace secret for W&B:
  - `WANDB_KEY` (recommended)
  - Runtime should map `WANDB_KEY -> WANDB_API_KEY`.

## 2) Machine Selection

Use GPU for training whenever possible.

- `L4`: good default for most training jobs.
- `A100` / `H100`: larger or faster runs, higher cost.
- `CPU-4`: debugging/startup checks only.

## 3) Standard Job Lifecycle (MCP)

1. Check credits:
   - `mcp__lightning__get_credits`
2. Check recent jobs:
   - `mcp__lightning__list_jobs`
3. Submit:
   - `mcp__lightning__submit_job`
4. Monitor status:
   - `mcp__lightning__list_jobs`
5. Pull logs when job is terminal:
   - `mcp__lightning__get_logs`
6. Stop bad runs:
   - `mcp__lightning__cancel_job`

## 4) Submission Templates

### GRPO

Command:

```bash
python -m src.train_self_play --config grpo_9m_posttrain.yaml
```

Submit with:
- `machine="L4"`
- unique job name, for example: `grpo-9m-posttrain-YYYYMMDD-hhmm`

### Distill

Command:

```bash
python -m src.distill.distill --config distill_labelsafe.yaml
```

Submit with:
- `machine="L4"`
- `WANDB_KEY` secret present in Teamspace

Recommended cloud entrypoint:

```bash
bash scripts/lightning/run_distill_labelsafe.sh
```

Useful env vars for that script:
- `PINNED_COMMIT=<sha>` to enforce commit pinning.
- `PRETRAIN_CKPT_DRIVE_URL=<drive_url>` for first-time checkpoint bootstrap.
- `DISTILL_MAX_SHARDS=6` (default) to bound memory.

Important config-path rule:
- For loaders using `load_yaml_file`, pass config as filename relative to `src/configs`, for example `distill_labelsafe.yaml`.
- Do not pass `src/configs/distill_labelsafe.yaml` or it can become `src/configs/src/configs/...`.

### Pretrain

Command:

```bash
python -m src.pretrain.pretrain --config pretrain.yaml
```

Submit with:
- `machine="L4"` or larger

## 5) Persistent Artifacts Pattern (Recommended)

Use Teamspace artifacts path to avoid repeated downloads/conversion on retries:

- Root: `/teamspace/studios/this_studio/artifacts`
- Suggested distill cache:
  - `/teamspace/studios/this_studio/artifacts/distill_labelsafe_v2/distill_data`
- Suggested pretrain checkpoint cache:
  - `/teamspace/studios/this_studio/artifacts/pretrain.ckpt`

Before downloading/converting, check if files already exist and reuse them.

## 6) W&B in Teamspace

At job start:

```bash
if [ -n "${WANDB_KEY:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="$WANDB_KEY"
fi
```

If no key is available, either:
- set `use_wandb: false` for that run, or
- fail fast with a clear message.

## 7) Distill Stability Notes

For distill runs in this repo, use these safeguards:

- Reuse cached distill shards from artifacts.
- Limit dataset size for memory safety (for example, prune to a bounded shard count).
- Ensure checkpoint backward compatibility alias is applied for older checkpoints:
  - `src.grpo_self_play` alias registration.
- Keep `num_workers: 0` when debugging data-loader failures.

## 8) Common Failure Modes

- `ModuleNotFoundError: src.grpo_self_play`
  - Add/check legacy checkpoint alias before `torch.load`.
- `No API key configured` (wandb)
  - Ensure Teamspace secret `WANDB_KEY` exists and is exported to `WANDB_API_KEY`.
  - Distill now supports `auto_disable_wandb_if_missing_key: true`.
- `No shard files found in data/distill`
  - Generate or copy distill shards into configured `dataset.data_dir`.
- OOM / process killed during distill dataset load
  - Reduce shard count / sample count; use bounded loading strategy.
- Distill run ends with `ValueError: Out of range float values are not JSON compliant: inf`
  - Caused by non-finite metrics reaching W&B artifact metadata.
  - Distill now guards non-finite batches and disables model artifact uploads by default (`log_model_artifacts: false`).
- Config file not found with doubled path
  - Pass config filename relative to `src/configs` (not prefixed path).
- `FileNotFoundError: Stockfish binary not found`
  - Lightning base images may not include Stockfish.
  - Distill cloud runner (`scripts/lightning/run_distill_labelsafe.sh`) now installs Stockfish automatically when missing (apt-based images).
  - Distill runner sets `require_stockfish_eval: true` and fails fast if Stockfish eval cannot be enabled.
  - To override binary location manually, set `STOCKFISH_PATH` or `stockfish.path`.

## 9) Suggested Run Naming

Use predictable names:

- `grpo-<config-stem>-<YYYYMMDD>-<hhmm>`
- `distill-<config-stem>-<YYYYMMDD>-<hhmm>`
- `pretrain-<config-stem>-<YYYYMMDD>-<hhmm>`

This makes `list_jobs`, log pulls, and run reports much easier.

## 10) New Session + `$experiment-cycle` (Lightning Quickstart)

Use this when you open a fresh chat/session and want the full cycle to run in Lightning without losing critical context.

### What persists vs what does not

- Persists across sessions:
  - repo files and commits
  - this runbook and `scripts/lightning/run_distill_labelsafe.sh`
  - Teamspace secrets (for example `WANDB_KEY`)
- Does not persist:
  - previous chat context (why the last run failed, chosen entry point, preferred job naming)

### Minimum context to provide at session start

Give the assistant these items up front:

- Skill + intent:
  - “Use `$experiment-cycle` and run on Lightning.”
- Branch/commit target:
  - branch name and optional pinned commit SHA
- Entry point:
  - `research` / `plan` / `implement` / `run`
- Artifacts if resuming:
  - plan doc path, config path, prior run report, WandB run ID
- Runtime constraints:
  - machine (`L4` recommended), budget/credit constraints, shard limits

### Copy-paste kickoff prompt (new session)

```text
Use $experiment-cycle. Run on Lightning only.
Start from: run
Branch: feature/lightning_training
Config: src/configs/distill_labelsafe.yaml
Use machine: L4
Use script: scripts/lightning/run_distill_labelsafe.sh
Pin commit: <optional_sha>
Use WANDB_KEY from Teamspace secret.
If resuming, use:
- plan doc: <path>
- prior run report: <path>
- baseline wandb run: <run_id>
```

### Lightning-specific expectations for distill runs

- Preferred cloud entrypoint:
  - `scripts/lightning/run_distill_labelsafe.sh`
- The script now:
  - auto-installs Stockfish if missing (apt-based environments)
  - exports `STOCKFISH_PATH` from discovered binary
  - sets `require_stockfish_eval: true` in generated config
  - fails fast if Stockfish eval cannot be enabled
- Result:
  - Stockfish metrics are expected on successful runs (`eval_stockfish/*` in logs/W&B)

### Resume protocol after interruption/failure

In a new session, ask to:

- pull latest branch
- list recent Lightning jobs
- fetch logs for latest failed job
- continue from the correct cycle step with artifact paths

Example:

```text
Use $experiment-cycle.
Pull latest, inspect latest failed Lightning job logs, fix, push, and resubmit.
Resume from cycle step: run.
Plan doc: research_docs/experiments/<file>.md
```

### Verification checklist (before submitting)

- `feature/lightning_training` (or target branch) is pushed
- config path is correct (filename relative to `src/configs` when required)
- Teamspace secret `WANDB_KEY` exists
- job command pins expected commit when reproducibility matters
- persistent artifact paths are used (`/teamspace/studios/this_studio/artifacts/...`)
