#!/usr/bin/env bash
set -euo pipefail

# Canonical Lightning distill entrypoint.
# Expected to run from repository root after checkout/submodule init.

if [[ -n "${PINNED_COMMIT:-}" ]]; then
  CURRENT_COMMIT="$(git rev-parse HEAD)"
  if ! PINNED_FULL_COMMIT="$(git rev-parse --verify "${PINNED_COMMIT}^{commit}" 2>/dev/null)"; then
    echo "ERROR: PINNED_COMMIT=${PINNED_COMMIT} does not resolve to a commit in this checkout." >&2
    exit 1
  fi
  if [[ "${CURRENT_COMMIT}" != "${PINNED_FULL_COMMIT}" ]]; then
    echo "ERROR: checked-out commit ${CURRENT_COMMIT} does not match PINNED_COMMIT=${PINNED_COMMIT}" >&2
    exit 1
  fi
fi

if [[ -n "${WANDB_KEY:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_KEY}"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  USE_WANDB=true
else
  USE_WANDB=false
fi

DISTILL_MODE="${DISTILL_MODE:-fast}"
if [[ "${DISTILL_MODE}" != "fast" && "${DISTILL_MODE}" != "quality" ]]; then
  echo "ERROR: DISTILL_MODE must be 'fast' or 'quality' (got '${DISTILL_MODE}')." >&2
  exit 1
fi

PERSIST_NAMESPACE="${PERSIST_NAMESPACE:-distill_labelsafe_v2}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/distill_labelsafe.lightning.yaml}"
PRETRAIN_CKPT_DRIVE_URL="${PRETRAIN_CKPT_DRIVE_URL:-}"
DEEPMIND_NUM_SHARDS="${DEEPMIND_NUM_SHARDS:-8}"
DEEPMIND_MIN_WIN_PROB="${DEEPMIND_MIN_WIN_PROB:-0.55}"
DEEPMIND_MAX_SAMPLES="${DEEPMIND_MAX_SAMPLES:-300000}"
DISTILL_MAX_SHARDS="${DISTILL_MAX_SHARDS:-8}"
STOCKFISH_INSTALL_ON_MISSING="${STOCKFISH_INSTALL_ON_MISSING:-1}"
PERSIST_BACKEND="${PERSIST_BACKEND:-lightning}"
GDRIVE_FOLDER_ID="${GDRIVE_FOLDER_ID:-}"
GDRIVE_PERSIST_REQUIRED="${GDRIVE_PERSIST_REQUIRED:-0}"
GDRIVE_SYNC_SHARDS="${GDRIVE_SYNC_SHARDS:-1}"
GDRIVE_DOWNLOAD_MAX_SHARDS="${GDRIVE_DOWNLOAD_MAX_SHARDS:-${DISTILL_MAX_SHARDS}}"
GDRIVE_UPLOAD_MAX_SHARDS="${GDRIVE_UPLOAD_MAX_SHARDS:-${DISTILL_MAX_SHARDS}}"
GDRIVE_SYNC_SCRIPT="${GDRIVE_SYNC_SCRIPT:-scripts/lightning/gdrive_sync.py}"
DISTILL_OUTPUT_DIR="${DISTILL_OUTPUT_DIR:-checkpoints/distill_labelsafe}"
DISTILL_FORCE_REBUILD_DATA="${DISTILL_FORCE_REBUILD_DATA:-0}"
DISTILL_QUALITY_GATE_MIN_VAL_TOP1="${DISTILL_QUALITY_GATE_MIN_VAL_TOP1:-}"
DISTILL_QUALITY_GATE_EPOCH="${DISTILL_QUALITY_GATE_EPOCH:-3}"
DISTILL_LAMBDA_SOFT="${DISTILL_LAMBDA_SOFT:-}"
DISTILL_LAMBDA_HARD="${DISTILL_LAMBDA_HARD:-}"
DISTILL_TARGET_TOP1_MIX_ALPHA="${DISTILL_TARGET_TOP1_MIX_ALPHA:-}"
AUTO_USE_S3_CONNECTION_CACHE="${AUTO_USE_S3_CONNECTION_CACHE:-1}"
AUTO_USE_EFS_CONNECTION_CACHE="${AUTO_USE_EFS_CONNECTION_CACHE:-1}"
REQUIRE_PERSISTENT_CACHE="${REQUIRE_PERSISTENT_CACHE:-0}"
LIGHTNING_SHARED_CACHE_DIR="${LIGHTNING_SHARED_CACHE_DIR:-/teamspace/uploads/grpo_chess_artifacts}"

TEACHER_MODEL="${TEACHER_MODEL:-136M}"
TEACHER_CHECKPOINT_DIR="${TEACHER_CHECKPOINT_DIR:-searchless_chess/checkpoints}"
TEACHER_CHECKPOINT_STEP="${TEACHER_CHECKPOINT_STEP:-6400000}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-512}"
TEACHER_TOP_K="${TEACHER_TOP_K:-8}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
TEACHER_TARGET_SCORE_NORM="${TEACHER_TARGET_SCORE_NORM:-}"
TEACHER_HF_CACHE_DIR="${TEACHER_HF_CACHE_DIR:-}"
TEACHER_SHARD_SIZE="${TEACHER_SHARD_SIZE:-50000}"
TEACHER_MIN_ELO="${TEACHER_MIN_ELO:-1800}"
TEACHER_MAX_SAMPLES="${TEACHER_MAX_SAMPLES:-300000}"
TEACHER_SKIP_FIRST_N_MOVES="${TEACHER_SKIP_FIRST_N_MOVES:-5}"
TEACHER_SKIP_LAST_N_MOVES="${TEACHER_SKIP_LAST_N_MOVES:-5}"
TEACHER_SAMPLE_POSITIONS_PER_GAME="${TEACHER_SAMPLE_POSITIONS_PER_GAME:-3}"
TEACHER_PROCESS_BATCH_SIZE="${TEACHER_PROCESS_BATCH_SIZE:-256}"
TEACHER_NUM_WORKERS="${TEACHER_NUM_WORKERS:-4}"
DISTILL_TORCH_RUNTIME_NUMPY="${DISTILL_TORCH_RUNTIME_NUMPY:-1.26.4}"
DISABLE_LIGHTNING_SITECUSTOMIZE="${DISABLE_LIGHTNING_SITECUSTOMIZE:-1}"
RUNTIME_SITECUSTOMIZE_DIR="${RUNTIME_SITECUSTOMIZE_DIR:-/tmp/grpo_chess_sitecustomize}"
QUALITY_REPAIR_PANDAS_ABI="${QUALITY_REPAIR_PANDAS_ABI:-1}"
JAX_USE_CUDA="${JAX_USE_CUDA:-auto}"
JAX_CUDA_REQUIRED="${JAX_CUDA_REQUIRED:-auto}"

DISTILL_PREFLIGHT_MIN_K_MEAN="${DISTILL_PREFLIGHT_MIN_K_MEAN:-}"
DISTILL_PREFLIGHT_MAX_K1_FRAC="${DISTILL_PREFLIGHT_MAX_K1_FRAC:-}"
DISTILL_PREFLIGHT_MIN_ENTROPY="${DISTILL_PREFLIGHT_MIN_ENTROPY:-}"
DISTILL_PREFLIGHT_MAX_TEACHER_ENTROPY="${DISTILL_PREFLIGHT_MAX_TEACHER_ENTROPY:-}"
DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_PROB="${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_PROB:-}"
DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_MARGIN="${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_MARGIN:-}"
DISTILL_PREFLIGHT_MAX_TEACHER_EFFECTIVE_K="${DISTILL_PREFLIGHT_MAX_TEACHER_EFFECTIVE_K:-}"
DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS="${DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS:-}"
DISTILL_PREFLIGHT_MAX_SHARDS="${DISTILL_PREFLIGHT_MAX_SHARDS:-${DISTILL_MAX_SHARDS}}"
DISTILL_PREFLIGHT_MAX_SAMPLES="${DISTILL_PREFLIGHT_MAX_SAMPLES:-200000}"

if [[ -z "${DISTILL_QUALITY_GATE_MIN_VAL_TOP1}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_QUALITY_GATE_MIN_VAL_TOP1="0.08"
  else
    DISTILL_QUALITY_GATE_MIN_VAL_TOP1="0.0"
  fi
fi

if [[ -z "${TEACHER_TARGET_SCORE_NORM}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    TEACHER_TARGET_SCORE_NORM="zscore"
  else
    TEACHER_TARGET_SCORE_NORM="plain"
  fi
fi

if [[ -z "${DISTILL_LAMBDA_SOFT}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_LAMBDA_SOFT="0.30"
  else
    DISTILL_LAMBDA_SOFT="1.0"
  fi
fi

if [[ -z "${DISTILL_LAMBDA_HARD}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_LAMBDA_HARD="0.70"
  else
    DISTILL_LAMBDA_HARD="0.0"
  fi
fi

if [[ -z "${DISTILL_TARGET_TOP1_MIX_ALPHA}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_TARGET_TOP1_MIX_ALPHA="0.15"
  else
    DISTILL_TARGET_TOP1_MIX_ALPHA="0.0"
  fi
fi

if [[ -z "${DISTILL_PREFLIGHT_MIN_K_MEAN}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MIN_K_MEAN="3.0"
  else
    DISTILL_PREFLIGHT_MIN_K_MEAN="0.0"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MAX_K1_FRAC}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MAX_K1_FRAC="0.80"
  else
    DISTILL_PREFLIGHT_MAX_K1_FRAC="1.0"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MIN_ENTROPY}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MIN_ENTROPY="0.70"
  else
    DISTILL_PREFLIGHT_MIN_ENTROPY="0.0"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MAX_TEACHER_ENTROPY}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MAX_TEACHER_ENTROPY="1.90"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_PROB}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_PROB="0.20"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_MARGIN}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_MARGIN="0.05"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_MAX_TEACHER_EFFECTIVE_K}" ]]; then
  if [[ "${DISTILL_MODE}" == "quality" ]]; then
    DISTILL_PREFLIGHT_MAX_TEACHER_EFFECTIVE_K="5.5"
  fi
fi
if [[ -z "${DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS}" ]]; then
  DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS="0"
fi

if [[ "${DISTILL_MODE}" == "quality" ]]; then
  DISTILL_DATASET_TAG_DEFAULT="teacher_data.v1:model=${TEACHER_MODEL},step=${TEACHER_CHECKPOINT_STEP},top_k=${TEACHER_TOP_K},temperature=${TEACHER_TEMPERATURE},score_norm=${TEACHER_TARGET_SCORE_NORM},min_elo=${TEACHER_MIN_ELO},max_samples=${TEACHER_MAX_SAMPLES},shard_size=${TEACHER_SHARD_SIZE},sample_positions_per_game=${TEACHER_SAMPLE_POSITIONS_PER_GAME}"
  DISTILL_REMOTE_DATA_DIR_DEFAULT="distill_data_quality"
else
  DISTILL_DATASET_TAG_DEFAULT="deepmind_data.v1:num_shards=${DEEPMIND_NUM_SHARDS},min_win_prob=${DEEPMIND_MIN_WIN_PROB},top_k=8,temperature=1.0,shard_size=50000,max_samples=${DEEPMIND_MAX_SAMPLES}"
  DISTILL_REMOTE_DATA_DIR_DEFAULT="distill_data"
fi

PERSIST_DATA_REMOTE_DIR="${PERSIST_DATA_REMOTE_DIR:-${DISTILL_REMOTE_DATA_DIR_DEFAULT}}"
DISTILL_DATASET_TAG="${DISTILL_DATASET_TAG:-${DISTILL_DATASET_TAG_DEFAULT}}"
export DISTILL_MODE DISTILL_DATASET_TAG DEEPMIND_NUM_SHARDS DEEPMIND_MIN_WIN_PROB DEEPMIND_MAX_SAMPLES
export TEACHER_MODEL TEACHER_CHECKPOINT_DIR TEACHER_CHECKPOINT_STEP TEACHER_BATCH_SIZE TEACHER_TOP_K TEACHER_TEMPERATURE
export TEACHER_HF_CACHE_DIR TEACHER_SHARD_SIZE TEACHER_MIN_ELO TEACHER_MAX_SAMPLES
export TEACHER_SKIP_FIRST_N_MOVES TEACHER_SKIP_LAST_N_MOVES TEACHER_SAMPLE_POSITIONS_PER_GAME
export TEACHER_PROCESS_BATCH_SIZE TEACHER_NUM_WORKERS
export TEACHER_TARGET_SCORE_NORM DISTILL_LAMBDA_SOFT DISTILL_LAMBDA_HARD DISTILL_TARGET_TOP1_MIX_ALPHA

has_nvidia_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi -L >/dev/null 2>&1
}

if [[ "${JAX_CUDA_REQUIRED}" == "auto" ]]; then
  if has_nvidia_gpu; then
    JAX_CUDA_REQUIRED="1"
  else
    JAX_CUDA_REQUIRED="0"
  fi
fi
export JAX_USE_CUDA JAX_CUDA_REQUIRED

# Lightning images may ship a global sitecustomize that eagerly imports
# Lightning/PyTorch modules on every python startup. That can break unrelated
# setup steps when temporary dependency versions (for teacher/JAX) are active.
if [[ "${DISABLE_LIGHTNING_SITECUSTOMIZE}" == "1" ]]; then
  mkdir -p "${RUNTIME_SITECUSTOMIZE_DIR}"
  cat > "${RUNTIME_SITECUSTOMIZE_DIR}/sitecustomize.py" <<'PY'
# Intentionally empty: override environment-provided sitecustomize side effects.
PY
  export PYTHONPATH="${RUNTIME_SITECUSTOMIZE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

choose_artifact_root() {
  if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
    printf '%s\n' "${ARTIFACT_ROOT}"
    return 0
  fi

  local conn_dir=""
  if [[ "${AUTO_USE_S3_CONNECTION_CACHE}" == "1" ]] && [[ -d "/teamspace/s3_connections" ]]; then
    for conn_dir in /teamspace/s3_connections/*; do
      [[ -d "${conn_dir}" ]] || continue
      if mkdir -p "${conn_dir}/grpo_chess_artifacts" 2>/dev/null; then
        printf '%s\n' "${conn_dir}/grpo_chess_artifacts"
        return 0
      fi
    done
  fi

  if [[ "${AUTO_USE_EFS_CONNECTION_CACHE}" == "1" ]] && [[ -d "/teamspace/efs_connections" ]]; then
    for conn_dir in /teamspace/efs_connections/*; do
      [[ -d "${conn_dir}" ]] || continue
      if mkdir -p "${conn_dir}/grpo_chess_artifacts" 2>/dev/null; then
        printf '%s\n' "${conn_dir}/grpo_chess_artifacts"
        return 0
      fi
    done
  fi

  # Prefer a shared Teamspace jobs path before falling back to job-local storage.
  if mkdir -p "${LIGHTNING_SHARED_CACHE_DIR}" 2>/dev/null; then
    printf '%s\n' "${LIGHTNING_SHARED_CACHE_DIR}"
    return 0
  fi

  printf '%s\n' "/teamspace/studios/this_studio/artifacts"
}

ARTIFACT_ROOT="$(choose_artifact_root)"
PERSIST_ROOT="${PERSIST_ROOT:-${ARTIFACT_ROOT}/${PERSIST_NAMESPACE}}"
DATA_DIR="${DATA_DIR:-${PERSIST_ROOT}/${PERSIST_DATA_REMOTE_DIR}}"
CKPT_PATH="${CKPT_PATH:-${PERSIST_ROOT}/pretrain.ckpt}"
DATASET_META_PATH="${DATA_DIR}/_build_meta.json"
CONVERSION_STATS_PATH="${DATA_DIR}/conversion_stats.json"
DATASET_QUALITY_STATS_PATH="${DATA_DIR}/dataset_quality_stats.json"
export DATA_DIR

if [[ "${PERSIST_BACKEND}" == "lightning" ]] && [[ "${ARTIFACT_ROOT}" == /teamspace/studios/this_studio/* ]]; then
  echo "Warning: ARTIFACT_ROOT=${ARTIFACT_ROOT} is job-local in Lightning Jobs and is not shared across jobs."
  echo "Warning: Cache reuse across job submissions will not work with this path."
  echo "Warning: Set ARTIFACT_ROOT to a persistent path (for example /teamspace/s3_connections/<mount>/grpo_chess_artifacts or /teamspace/efs_connections/<mount>/grpo_chess_artifacts)."
  if [[ "${REQUIRE_PERSISTENT_CACHE}" == "1" ]]; then
    echo "ERROR: REQUIRE_PERSISTENT_CACHE=1 but no persistent writable ARTIFACT_ROOT is configured." >&2
    exit 1
  fi
fi

mkdir -p "${PERSIST_ROOT}" "${DATA_DIR}"
echo "Distill mode: ${DISTILL_MODE}"
echo "Dataset tag: ${DISTILL_DATASET_TAG}"
echo "Dataset dir: ${DATA_DIR}"
echo "Teacher JAX: JAX_USE_CUDA=${JAX_USE_CUDA} JAX_CUDA_REQUIRED=${JAX_CUDA_REQUIRED}"

run_optional_sync() {
  if "$@"; then
    return 0
  fi
  if [[ "${GDRIVE_PERSIST_REQUIRED}" == "1" ]]; then
    echo "ERROR: required Google Drive sync command failed: $*" >&2
    exit 1
  fi
  echo "Warning: Google Drive sync command failed (continuing): $*" >&2
  return 0
}

ensure_drive_sync_deps() {
  python -m pip install --quiet "google-api-python-client>=2.0.0" "google-auth>=2.0.0" gdown
}

is_drive_enabled() {
  [[ "${PERSIST_BACKEND}" == "gdrive" ]] && [[ -n "${GDRIVE_FOLDER_ID}" ]]
}

if [[ "${PERSIST_BACKEND}" == "gdrive" ]] && [[ -z "${GDRIVE_FOLDER_ID}" ]]; then
  if [[ "${GDRIVE_PERSIST_REQUIRED}" == "1" ]]; then
    echo "ERROR: PERSIST_BACKEND=gdrive but GDRIVE_FOLDER_ID is not set." >&2
    exit 1
  fi
  echo "Warning: PERSIST_BACKEND=gdrive but GDRIVE_FOLDER_ID is empty; falling back to local-only persistence."
fi

if is_drive_enabled; then
  ensure_drive_sync_deps
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/pretrain.ckpt" \
    --local-path "${CKPT_PATH}" \
    --optional

  if [[ "${GDRIVE_SYNC_SHARDS}" == "1" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-glob \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --remote-dir "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}" \
      --local-dir "${DATA_DIR}" \
      --glob "shard_*.pt" \
      --max-files "${GDRIVE_DOWNLOAD_MAX_SHARDS}" \
      --optional
  fi
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/_build_meta.json" \
    --local-path "${DATASET_META_PATH}" \
    --optional
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/conversion_stats.json" \
    --local-path "${CONVERSION_STATS_PATH}" \
    --optional
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/dataset_quality_stats.json" \
    --local-path "${DATASET_QUALITY_STATS_PATH}" \
    --optional
fi

read_dataset_meta_tag() {
  local metadata_path="$1"
  python - "${metadata_path}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

tag = payload.get("dataset_tag", "")
if isinstance(tag, str):
    print(tag)
PY
}

write_dataset_meta() {
  local metadata_path="$1"
  python - "${metadata_path}" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "dataset_tag": os.environ["DISTILL_DATASET_TAG"],
    "distill_mode": os.environ["DISTILL_MODE"],
    "data_dir": os.environ["DATA_DIR"],
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
if os.environ["DISTILL_MODE"] == "fast":
    payload.update(
        {
            "num_shards": int(os.environ["DEEPMIND_NUM_SHARDS"]),
            "min_win_prob": float(os.environ["DEEPMIND_MIN_WIN_PROB"]),
            "max_samples": int(os.environ["DEEPMIND_MAX_SAMPLES"]),
            "top_k": 8,
            "temperature": 1.0,
            "shard_size": 50000,
        }
    )
else:
    payload.update(
        {
            "teacher_model": os.environ["TEACHER_MODEL"],
            "teacher_checkpoint_dir": os.environ["TEACHER_CHECKPOINT_DIR"],
            "teacher_checkpoint_step": int(os.environ["TEACHER_CHECKPOINT_STEP"]),
            "teacher_batch_size": int(os.environ["TEACHER_BATCH_SIZE"]),
            "top_k": int(os.environ["TEACHER_TOP_K"]),
            "teacher_temperature": float(os.environ["TEACHER_TEMPERATURE"]),
            "teacher_target_score_norm": os.environ["TEACHER_TARGET_SCORE_NORM"],
            "teacher_shard_size": int(os.environ["TEACHER_SHARD_SIZE"]),
            "teacher_min_elo": int(os.environ["TEACHER_MIN_ELO"]),
            "teacher_max_samples": int(os.environ["TEACHER_MAX_SAMPLES"]),
            "teacher_skip_first_n_moves": int(os.environ["TEACHER_SKIP_FIRST_N_MOVES"]),
            "teacher_skip_last_n_moves": int(os.environ["TEACHER_SKIP_LAST_N_MOVES"]),
            "teacher_sample_positions_per_game": int(os.environ["TEACHER_SAMPLE_POSITIONS_PER_GAME"]),
            "teacher_process_batch_size": int(os.environ["TEACHER_PROCESS_BATCH_SIZE"]),
            "teacher_num_workers": int(os.environ["TEACHER_NUM_WORKERS"]),
        }
    )

path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote dataset metadata: {path}")
PY
}

run_dataset_preflight() {
  local data_dir="$1"
  local stats_path="$2"
  python - "${data_dir}" "${stats_path}" "${DISTILL_MODE}" "${DISTILL_PREFLIGHT_MAX_SHARDS}" "${DISTILL_PREFLIGHT_MAX_SAMPLES}" "${DISTILL_PREFLIGHT_MIN_K_MEAN}" "${DISTILL_PREFLIGHT_MAX_K1_FRAC}" "${DISTILL_PREFLIGHT_MIN_ENTROPY}" "${DISTILL_PREFLIGHT_MAX_TEACHER_ENTROPY}" "${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_PROB}" "${DISTILL_PREFLIGHT_MIN_TEACHER_TOP1_MARGIN}" "${DISTILL_PREFLIGHT_MAX_TEACHER_EFFECTIVE_K}" "${DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS}" <<'PY'
import collections
import json
import math
import pathlib
import sys

import torch

data_dir = pathlib.Path(sys.argv[1])
stats_path = pathlib.Path(sys.argv[2])
mode = sys.argv[3]
max_shards = int(float(sys.argv[4]))
max_samples = int(float(sys.argv[5]))
min_k_mean = float(sys.argv[6])
max_k1_frac = float(sys.argv[7])
min_entropy = float(sys.argv[8])

def _parse_optional_float(raw: str):
    raw = (raw or "").strip()
    if raw == "" or raw.lower() in {"none", "null", "nan"}:
        return None
    return float(raw)

def _parse_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

max_teacher_entropy = _parse_optional_float(sys.argv[9])
min_teacher_top1_prob = _parse_optional_float(sys.argv[10])
min_teacher_top1_margin = _parse_optional_float(sys.argv[11])
max_teacher_effective_k = _parse_optional_float(sys.argv[12])
fail_on_flat_labels = _parse_bool(sys.argv[13])

shard_paths = sorted(data_dir.glob("shard_*.pt"))
if max_shards > 0:
    shard_paths = shard_paths[:max_shards]
if not shard_paths:
    raise SystemExit(f"ERROR: no shard_*.pt files found in {data_dir} for dataset preflight.")

total_samples = 0
total_k = 0.0
k1_count = 0
total_entropy = 0.0
total_top1_prob = 0.0
total_top1_margin = 0.0
total_effective_k = 0.0
k_hist = collections.Counter()

for shard_path in shard_paths:
    shard = torch.load(shard_path, weights_only=False)
    teacher_probs = shard.get("teacher_probs")
    if teacher_probs is None:
        continue

    for raw_probs in teacher_probs:
        if max_samples > 0 and total_samples >= max_samples:
            break

        if isinstance(raw_probs, torch.Tensor):
            probs = raw_probs.detach().cpu().float().flatten()
        else:
            probs = torch.tensor(raw_probs, dtype=torch.float32).flatten()

        k = int(probs.numel())
        if k <= 0:
            continue

        total_samples += 1
        total_k += k
        k_hist[str(k)] += 1
        if k == 1:
            k1_count += 1

        denom = float(probs.sum().item())
        if not math.isfinite(denom) or denom <= 0:
            probs = torch.full((k,), 1.0 / k, dtype=torch.float32)
        else:
            probs = probs / denom

        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum()

        entropy = float(-(probs * probs.log()).sum().item())
        top_probs = torch.topk(probs, k=min(2, k)).values
        top1_prob = float(top_probs[0].item())
        top2_prob = float(top_probs[1].item()) if top_probs.numel() > 1 else 0.0

        total_entropy += entropy
        total_top1_prob += top1_prob
        total_top1_margin += (top1_prob - top2_prob)
        total_effective_k += math.exp(entropy)

    if max_samples > 0 and total_samples >= max_samples:
        break

if total_samples == 0:
    raise SystemExit(f"ERROR: dataset preflight found 0 valid samples in {data_dir}.")

k_mean = total_k / total_samples
k1_frac = k1_count / total_samples
entropy_mean = total_entropy / total_samples
top1_prob_mean = total_top1_prob / total_samples
top1_margin_mean = total_top1_margin / total_samples
effective_k_mean = total_effective_k / total_samples
k_mode = int(max(k_hist.items(), key=lambda kv: kv[1])[0]) if k_hist else 0
max_entropy_ln_k_mode = math.log(k_mode) if k_mode > 0 else 0.0

stats = {
    "distill_mode": mode,
    "data_dir": str(data_dir),
    "shards_scanned": len(shard_paths),
    "samples_scanned": total_samples,
    "sample_cap": max_samples,
    "k_mean": k_mean,
    "k1_frac": k1_frac,
    "teacher_entropy_mean": entropy_mean,
    "teacher_top1_prob_mean": top1_prob_mean,
    "teacher_top1_minus_top2_mean": top1_margin_mean,
    "teacher_effective_k_mean": effective_k_mean,
    "inferred_top_k_mode": k_mode,
    "max_entropy_ln_k_mode": max_entropy_ln_k_mode,
    "k_histogram": dict(sorted(k_hist.items(), key=lambda kv: int(kv[0]))),
    "thresholds": {
        "min_k_mean": min_k_mean,
        "max_k1_frac": max_k1_frac,
        "min_teacher_entropy": min_entropy,
        "max_teacher_entropy": max_teacher_entropy,
        "min_teacher_top1_prob": min_teacher_top1_prob,
        "min_teacher_top1_margin": min_teacher_top1_margin,
        "max_teacher_effective_k": max_teacher_effective_k,
        "fail_on_flat_labels": fail_on_flat_labels,
    },
}

stats_path.parent.mkdir(parents=True, exist_ok=True)
stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote dataset preflight stats: {stats_path}")
print(json.dumps(stats, indent=2, sort_keys=True))

if mode == "quality":
    failures = []
    if k_mean < min_k_mean:
        failures.append(f"k_mean={k_mean:.4f} < required {min_k_mean:.4f}")
    if k1_frac > max_k1_frac:
        failures.append(f"k1_frac={k1_frac:.4f} > allowed {max_k1_frac:.4f}")
    if entropy_mean < min_entropy:
        failures.append(f"teacher_entropy_mean={entropy_mean:.4f} < required {min_entropy:.4f}")
    if max_teacher_entropy is not None and entropy_mean > max_teacher_entropy:
        failures.append(f"teacher_entropy_mean={entropy_mean:.4f} > allowed {max_teacher_entropy:.4f}")
    if min_teacher_top1_prob is not None and top1_prob_mean < min_teacher_top1_prob:
        failures.append(f"teacher_top1_prob_mean={top1_prob_mean:.4f} < required {min_teacher_top1_prob:.4f}")
    if min_teacher_top1_margin is not None and top1_margin_mean < min_teacher_top1_margin:
        failures.append(f"teacher_top1_minus_top2_mean={top1_margin_mean:.4f} < required {min_teacher_top1_margin:.4f}")
    if max_teacher_effective_k is not None and effective_k_mean > max_teacher_effective_k:
        failures.append(f"teacher_effective_k_mean={effective_k_mean:.4f} > allowed {max_teacher_effective_k:.4f}")

    if failures:
        print("WARNING: Distill dataset preflight detected weak/flat teacher signal.", file=sys.stderr)
        print(
            f"WARNING: inferred_top_k_mode={k_mode}, max_entropy_ln(k)={max_entropy_ln_k_mode:.4f}, "
            f"teacher_entropy_mean={entropy_mean:.4f}, teacher_top1_prob_mean={top1_prob_mean:.4f}, "
            f"teacher_top1_minus_top2_mean={top1_margin_mean:.4f}, teacher_effective_k_mean={effective_k_mean:.4f}",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"WARNING: {failure}", file=sys.stderr)
        print(
            "WARNING: Suggested fixes: lower TEACHER_TEMPERATURE, reduce TEACHER_TOP_K, use a stronger teacher model, "
            "or increase quality data/sample budget.",
            file=sys.stderr,
        )
        if fail_on_flat_labels:
            print(
                "WARNING: DISTILL_PREFLIGHT_FAIL_ON_FLAT_LABELS=1 is set, "
                "but this pipeline is configured to warn-only and continue.",
                file=sys.stderr,
            )
PY
}

restore_distill_runtime_numpy() {
  local numpy_version="$1"
  echo "Restoring runtime NumPy compatibility for distill: ${numpy_version}"
  python -m pip install --quiet "numpy==${numpy_version}"
  python - <<'PY'
import numpy as np
print(f"Runtime NumPy version: {np.__version__}")
PY
}

ensure_quality_generation_stack() {
  local repair_requested="$1"
  if python - <<'PY'
import numpy as np
import pandas as pd
import pyarrow as pa
print(f"Quality generation stack OK: numpy={np.__version__}, pandas={pd.__version__}, pyarrow={pa.__version__}")
PY
  then
    return 0
  fi

  if [[ "${repair_requested}" != "1" ]]; then
    echo "ERROR: quality generation stack check failed and QUALITY_REPAIR_PANDAS_ABI=0." >&2
    return 1
  fi

  echo "Repairing quality-generation pandas/pyarrow binary compatibility..."
  python -m pip install --quiet --upgrade --force-reinstall pandas pyarrow
  python - <<'PY'
import numpy as np
import pandas as pd
import pyarrow as pa
print(f"Quality generation stack repaired: numpy={np.__version__}, pandas={pd.__version__}, pyarrow={pa.__version__}")
PY
}

find_stockfish_binary() {
  local candidate=""
  local candidates=()
  if [[ -n "${STOCKFISH_PATH:-}" ]]; then
    candidates+=("${STOCKFISH_PATH}")
  fi
  if command -v stockfish >/dev/null 2>&1; then
    candidates+=("$(command -v stockfish)")
  fi
  candidates+=("/usr/games/stockfish" "/usr/bin/stockfish" "/usr/local/bin/stockfish" "/opt/homebrew/bin/stockfish")

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

install_stockfish_with_apt() {
  local -a apt_prefix=()
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: Stockfish not found and apt-get is unavailable on this machine." >&2
    return 1
  fi
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_prefix=(sudo)
    else
      echo "ERROR: Stockfish not found and sudo is unavailable for apt-get install." >&2
      return 1
    fi
  fi

  echo "Stockfish binary not found. Installing via apt-get..."
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" apt-get install -y --no-install-recommends stockfish
}

if STOCKFISH_BIN="$(find_stockfish_binary)"; then
  echo "Using Stockfish binary: ${STOCKFISH_BIN}"
else
  if [[ "${STOCKFISH_INSTALL_ON_MISSING}" == "1" ]]; then
    install_stockfish_with_apt
    STOCKFISH_BIN="$(find_stockfish_binary)" || {
      echo "ERROR: Stockfish install completed but binary is still not discoverable." >&2
      exit 1
    }
    echo "Installed Stockfish binary: ${STOCKFISH_BIN}"
  else
    echo "ERROR: Stockfish binary not found and STOCKFISH_INSTALL_ON_MISSING=0." >&2
    exit 1
  fi
fi
export STOCKFISH_PATH="${STOCKFISH_BIN}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  if [[ -z "${PRETRAIN_CKPT_DRIVE_URL}" ]]; then
    echo "ERROR: checkpoint missing at ${CKPT_PATH} and PRETRAIN_CKPT_DRIVE_URL is empty." >&2
    exit 1
  fi
  python -m pip install --quiet gdown
  gdown --fuzzy "${PRETRAIN_CKPT_DRIVE_URL}" -O "${CKPT_PATH}"
fi
ls -lh "${CKPT_PATH}"

if is_drive_enabled; then
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${CKPT_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/pretrain.ckpt"
fi

cat > "${CONFIG_PATH}" <<YAML
generate:
  teacher_model: "${TEACHER_MODEL}"
  checkpoint_dir: "${TEACHER_CHECKPOINT_DIR}"
  checkpoint_step: ${TEACHER_CHECKPOINT_STEP}
  teacher_batch_size: ${TEACHER_BATCH_SIZE}
  process_batch_size: ${TEACHER_PROCESS_BATCH_SIZE}
  num_workers: ${TEACHER_NUM_WORKERS}
  top_k: ${TEACHER_TOP_K}
  teacher_temperature: ${TEACHER_TEMPERATURE}
  teacher_target_score_norm: "${TEACHER_TARGET_SCORE_NORM}"
  hf_cache_dir: "${TEACHER_HF_CACHE_DIR}"
  output_dir: "${DATA_DIR}"
  shard_size: ${TEACHER_SHARD_SIZE}
  min_elo: ${TEACHER_MIN_ELO}
  max_samples: ${TEACHER_MAX_SAMPLES}
  skip_first_n_moves: ${TEACHER_SKIP_FIRST_N_MOVES}
  skip_last_n_moves: ${TEACHER_SKIP_LAST_N_MOVES}
  sample_positions_per_game: ${TEACHER_SAMPLE_POSITIONS_PER_GAME}

deepmind_data:
  num_shards: ${DEEPMIND_NUM_SHARDS}
  top_k: 8
  temperature: 1.0
  min_win_prob: ${DEEPMIND_MIN_WIN_PROB}
  output_dir: "${DATA_DIR}"
  shard_size: 50000
  max_samples: ${DEEPMIND_MAX_SAMPLES}

distill:
  lr: 0.00006
  batch_size: 1024
  num_epochs: 10
  warmup_steps: 1000
  weight_decay: 0.01
  max_grad_norm: 0.5
  checkpoint_dir: "checkpoints/distill_labelsafe"
  pretrain_checkpoint: "${CKPT_PATH}"
  use_wandb: ${USE_WANDB}
  wandb_project: "chess-grpo-pretrain"
  log_model_artifacts: false
  auto_disable_wandb_if_missing_key: true
  fail_on_nonfinite: false
  max_nonfinite_batches: 50
  require_stockfish_eval: true
  quality_gate_min_val_top1: ${DISTILL_QUALITY_GATE_MIN_VAL_TOP1}
  quality_gate_epoch: ${DISTILL_QUALITY_GATE_EPOCH}
  distill_lambda_soft: ${DISTILL_LAMBDA_SOFT}
  distill_lambda_hard: ${DISTILL_LAMBDA_HARD}
  distill_target_top1_mix_alpha: ${DISTILL_TARGET_TOP1_MIX_ALPHA}
  num_workers: 4
  val_check_interval: 0.1
  eval_every_n_epochs: 1

eval:
  games: 64
  seed: 0
  max_plies: 400
  randomize_opening: true
  opening_plies: 6

stockfish:
  path: "${STOCKFISH_BIN}"
  skill_level: 2
  movetime_ms: 50

policy:
  greedy: true

dataset:
  data_dir: "${DATA_DIR}"
  top_k: 8
  eval_fraction: 0.05
  max_shards: ${DISTILL_MAX_SHARDS}

transformer:
  vocab_size: 300
  embed_dim: 256
  num_layers: 4
  num_heads: 8
  action_dim: 1968
YAML

NEEDS_DATASET_REBUILD=0
DATASET_REBUILD_REASON=""

if [[ "${DISTILL_FORCE_REBUILD_DATA}" == "1" ]]; then
  NEEDS_DATASET_REBUILD=1
  DATASET_REBUILD_REASON="DISTILL_FORCE_REBUILD_DATA=1"
elif ! ls "${DATA_DIR}"/shard_*.pt >/dev/null 2>&1; then
  NEEDS_DATASET_REBUILD=1
  DATASET_REBUILD_REASON="no shard files found"
else
  CURRENT_DATASET_TAG="$(read_dataset_meta_tag "${DATASET_META_PATH}" || true)"
  if [[ -z "${CURRENT_DATASET_TAG}" ]]; then
    NEEDS_DATASET_REBUILD=1
    DATASET_REBUILD_REASON="missing dataset metadata tag"
  elif [[ "${CURRENT_DATASET_TAG}" != "${DISTILL_DATASET_TAG}" ]]; then
    NEEDS_DATASET_REBUILD=1
    DATASET_REBUILD_REASON="dataset tag mismatch (have='${CURRENT_DATASET_TAG}', want='${DISTILL_DATASET_TAG}')"
  fi
fi

if [[ "${NEEDS_DATASET_REBUILD}" == "1" ]]; then
  echo "Rebuilding distill dataset (${DATASET_REBUILD_REASON})"
  rm -f "${DATA_DIR}"/shard_*.pt "${DATASET_META_PATH}" "${CONVERSION_STATS_PATH}" "${DATASET_QUALITY_STATS_PATH}"
  if [[ "${DISTILL_MODE}" == "fast" ]]; then
    python -m src.distill.convert_deepmind_data \
      --config "${CONFIG_PATH}" \
      --num_shards "${DEEPMIND_NUM_SHARDS}" \
      --min_win_prob "${DEEPMIND_MIN_WIN_PROB}" \
      --max_samples "${DEEPMIND_MAX_SAMPLES}"
  else
    bash scripts/setup_distill_deps.sh --checkpoint "${TEACHER_MODEL}"
    ensure_quality_generation_stack "${QUALITY_REPAIR_PANDAS_ABI}"
    generate_args=(
      -m src.distill.generate_dataset
      --config "${CONFIG_PATH}"
      --max_samples "${TEACHER_MAX_SAMPLES}"
      --teacher_model "${TEACHER_MODEL}"
      --teacher_batch_size "${TEACHER_BATCH_SIZE}"
      --batch_size "${TEACHER_PROCESS_BATCH_SIZE}"
      --num_workers "${TEACHER_NUM_WORKERS}"
      --output_dir "${DATA_DIR}"
    )
    if [[ -n "${TEACHER_HF_CACHE_DIR}" ]]; then
      generate_args+=(--hf_cache_dir "${TEACHER_HF_CACHE_DIR}")
    fi
    python "${generate_args[@]}"
  fi
  write_dataset_meta "${DATASET_META_PATH}"
else
  echo "Reusing existing distill shards in ${DATA_DIR} (dataset tag matches)"
fi

run_dataset_preflight "${DATA_DIR}" "${DATASET_QUALITY_STATS_PATH}"

if is_drive_enabled; then
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${CONFIG_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/configs/distill_labelsafe.lightning.yaml"

  if [[ "${GDRIVE_SYNC_SHARDS}" == "1" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-glob \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --local-dir "${DATA_DIR}" \
      --remote-dir "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}" \
      --glob "shard_*.pt" \
      --max-files "${GDRIVE_UPLOAD_MAX_SHARDS}"
  fi
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${DATASET_META_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/_build_meta.json"
  if [[ -f "${CONVERSION_STATS_PATH}" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --local-path "${CONVERSION_STATS_PATH}" \
      --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/conversion_stats.json"
  fi
  if [[ -f "${DATASET_QUALITY_STATS_PATH}" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --local-path "${DATASET_QUALITY_STATS_PATH}" \
      --remote-path "${PERSIST_NAMESPACE}/${PERSIST_DATA_REMOTE_DIR}/dataset_quality_stats.json"
  fi
fi

if [[ "${DISTILL_MODE}" == "quality" ]]; then
  restore_distill_runtime_numpy "${DISTILL_TORCH_RUNTIME_NUMPY}"
fi

python -m src.distill.distill --config "${CONFIG_PATH}"

FINAL_CKPT_PATH="${DISTILL_OUTPUT_DIR}/distill_final.pt"
if [[ -f "${FINAL_CKPT_PATH}" ]] && is_drive_enabled; then
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${FINAL_CKPT_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/checkpoints/distill_final.pt"
fi
