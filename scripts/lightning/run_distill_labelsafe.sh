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

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/teamspace/studios/this_studio/artifacts}"
PERSIST_ROOT="${PERSIST_ROOT:-${ARTIFACT_ROOT}/distill_labelsafe_v2}"
DATA_DIR="${DATA_DIR:-${PERSIST_ROOT}/distill_data}"
CKPT_PATH="${CKPT_PATH:-${ARTIFACT_ROOT}/pretrain.ckpt}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/distill_labelsafe.lightning.yaml}"
PRETRAIN_CKPT_DRIVE_URL="${PRETRAIN_CKPT_DRIVE_URL:-}"
DEEPMIND_NUM_SHARDS="${DEEPMIND_NUM_SHARDS:-1}"
DEEPMIND_MIN_WIN_PROB="${DEEPMIND_MIN_WIN_PROB:-0.95}"
DISTILL_MAX_SHARDS="${DISTILL_MAX_SHARDS:-6}"
STOCKFISH_INSTALL_ON_MISSING="${STOCKFISH_INSTALL_ON_MISSING:-1}"

mkdir -p "${PERSIST_ROOT}" "${DATA_DIR}"

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
  python -m pip install gdown
  gdown --fuzzy "${PRETRAIN_CKPT_DRIVE_URL}" -O "${CKPT_PATH}"
fi
ls -lh "${CKPT_PATH}"

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

distill:
  lr: 0.00003
  batch_size: 1024
  num_epochs: 8
  warmup_steps: 2000
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
  num_workers: 0
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

if ls "${DATA_DIR}"/shard_*.pt >/dev/null 2>&1; then
  echo "Reusing existing distill shards in ${DATA_DIR}"
else
  python -m src.distill.convert_deepmind_data --config "${CONFIG_PATH}" --num_shards "${DEEPMIND_NUM_SHARDS}" --min_win_prob "${DEEPMIND_MIN_WIN_PROB}"
fi

python -m src.distill.distill --config "${CONFIG_PATH}"
