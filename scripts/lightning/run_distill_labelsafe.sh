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

PERSIST_NAMESPACE="${PERSIST_NAMESPACE:-distill_labelsafe_v2}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/distill_labelsafe.lightning.yaml}"
PRETRAIN_CKPT_DRIVE_URL="${PRETRAIN_CKPT_DRIVE_URL:-}"
DEEPMIND_NUM_SHARDS="${DEEPMIND_NUM_SHARDS:-8}"
DEEPMIND_MIN_WIN_PROB="${DEEPMIND_MIN_WIN_PROB:-0.55}"
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
DISTILL_QUALITY_GATE_MIN_VAL_TOP1="${DISTILL_QUALITY_GATE_MIN_VAL_TOP1:-0.0}"
DISTILL_QUALITY_GATE_EPOCH="${DISTILL_QUALITY_GATE_EPOCH:-3}"
DEEPMIND_MAX_SAMPLES="${DEEPMIND_MAX_SAMPLES:-300000}"
AUTO_USE_S3_CONNECTION_CACHE="${AUTO_USE_S3_CONNECTION_CACHE:-1}"
REQUIRE_PERSISTENT_CACHE="${REQUIRE_PERSISTENT_CACHE:-0}"
DISTILL_DATASET_TAG_DEFAULT="deepmind_data.v1:num_shards=${DEEPMIND_NUM_SHARDS},min_win_prob=${DEEPMIND_MIN_WIN_PROB},top_k=8,temperature=1.0,shard_size=50000,max_samples=${DEEPMIND_MAX_SAMPLES}"
DISTILL_DATASET_TAG="${DISTILL_DATASET_TAG:-${DISTILL_DATASET_TAG_DEFAULT}}"
export DISTILL_DATASET_TAG DEEPMIND_NUM_SHARDS DEEPMIND_MIN_WIN_PROB DEEPMIND_MAX_SAMPLES

choose_artifact_root() {
  if [[ -n "${ARTIFACT_ROOT:-}" ]]; then
    printf '%s\n' "${ARTIFACT_ROOT}"
    return 0
  fi

  if [[ "${AUTO_USE_S3_CONNECTION_CACHE}" == "1" ]] && [[ -d "/teamspace/s3_connections" ]]; then
    local conn_dir=""
    for conn_dir in /teamspace/s3_connections/*; do
      [[ -d "${conn_dir}" ]] || continue
      if mkdir -p "${conn_dir}/grpo_chess_artifacts" 2>/dev/null; then
        printf '%s\n' "${conn_dir}/grpo_chess_artifacts"
        return 0
      fi
    done
  fi

  printf '%s\n' "/teamspace/studios/this_studio/artifacts"
}

ARTIFACT_ROOT="$(choose_artifact_root)"
PERSIST_ROOT="${PERSIST_ROOT:-${ARTIFACT_ROOT}/${PERSIST_NAMESPACE}}"
DATA_DIR="${DATA_DIR:-${PERSIST_ROOT}/distill_data}"
CKPT_PATH="${CKPT_PATH:-${PERSIST_ROOT}/pretrain.ckpt}"
DATASET_META_PATH="${DATA_DIR}/_build_meta.json"
CONVERSION_STATS_PATH="${DATA_DIR}/conversion_stats.json"
export DATA_DIR

if [[ "${PERSIST_BACKEND}" == "lightning" ]] && [[ "${ARTIFACT_ROOT}" == /teamspace/studios/this_studio/* ]]; then
  echo "Warning: ARTIFACT_ROOT=${ARTIFACT_ROOT} is job-local in Lightning Jobs and is not shared across jobs."
  echo "Warning: Cache reuse across job submissions will not work with this path."
  echo "Warning: Set ARTIFACT_ROOT to a writable mounted path (for example /teamspace/s3_connections/<mount>/grpo_chess_artifacts)."
  if [[ "${REQUIRE_PERSISTENT_CACHE}" == "1" ]]; then
    echo "ERROR: REQUIRE_PERSISTENT_CACHE=1 but no persistent writable ARTIFACT_ROOT is configured." >&2
    exit 1
  fi
fi

mkdir -p "${PERSIST_ROOT}" "${DATA_DIR}"

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
      --remote-dir "${PERSIST_NAMESPACE}/distill_data" \
      --local-dir "${DATA_DIR}" \
      --glob "shard_*.pt" \
      --max-files "${GDRIVE_DOWNLOAD_MAX_SHARDS}" \
      --optional
  fi
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/distill_data/_build_meta.json" \
    --local-path "${DATASET_META_PATH}" \
    --optional
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" download-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --remote-path "${PERSIST_NAMESPACE}/distill_data/conversion_stats.json" \
    --local-path "${CONVERSION_STATS_PATH}" \
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
    "num_shards": int(os.environ["DEEPMIND_NUM_SHARDS"]),
    "min_win_prob": float(os.environ["DEEPMIND_MIN_WIN_PROB"]),
    "max_samples": int(os.environ["DEEPMIND_MAX_SAMPLES"]),
    "top_k": 8,
    "temperature": 1.0,
    "shard_size": 50000,
    "data_dir": os.environ["DATA_DIR"],
    "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote dataset metadata: {path}")
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
  teacher_model: "136M"
  checkpoint_dir: "searchless_chess/checkpoints"
  checkpoint_step: 6400000
  teacher_batch_size: 512
  top_k: 8
  teacher_temperature: 1.0
  output_dir: "${DATA_DIR}"
  shard_size: 50000
  min_elo: 1800
  max_samples: 2000000
  skip_first_n_moves: 5
  skip_last_n_moves: 5
  sample_positions_per_game: 3

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
  rm -f "${DATA_DIR}"/shard_*.pt "${DATASET_META_PATH}" "${CONVERSION_STATS_PATH}"
  python -m src.distill.convert_deepmind_data \
    --config "${CONFIG_PATH}" \
    --num_shards "${DEEPMIND_NUM_SHARDS}" \
    --min_win_prob "${DEEPMIND_MIN_WIN_PROB}" \
    --max_samples "${DEEPMIND_MAX_SAMPLES}"
  write_dataset_meta "${DATASET_META_PATH}"
else
  echo "Reusing existing distill shards in ${DATA_DIR} (dataset tag matches)"
fi

if is_drive_enabled; then
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${CONFIG_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/configs/distill_labelsafe.lightning.yaml"

  if [[ "${GDRIVE_SYNC_SHARDS}" == "1" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-glob \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --local-dir "${DATA_DIR}" \
      --remote-dir "${PERSIST_NAMESPACE}/distill_data" \
      --glob "shard_*.pt" \
      --max-files "${GDRIVE_UPLOAD_MAX_SHARDS}"
  fi
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${DATASET_META_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/distill_data/_build_meta.json"
  if [[ -f "${CONVERSION_STATS_PATH}" ]]; then
    run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
      --folder-id "${GDRIVE_FOLDER_ID}" \
      --local-path "${CONVERSION_STATS_PATH}" \
      --remote-path "${PERSIST_NAMESPACE}/distill_data/conversion_stats.json"
  fi
fi

python -m src.distill.distill --config "${CONFIG_PATH}"

FINAL_CKPT_PATH="${DISTILL_OUTPUT_DIR}/distill_final.pt"
if [[ -f "${FINAL_CKPT_PATH}" ]] && is_drive_enabled; then
  run_optional_sync python "${GDRIVE_SYNC_SCRIPT}" upload-file \
    --folder-id "${GDRIVE_FOLDER_ID}" \
    --local-path "${FINAL_CKPT_PATH}" \
    --remote-path "${PERSIST_NAMESPACE}/checkpoints/distill_final.pt"
fi
