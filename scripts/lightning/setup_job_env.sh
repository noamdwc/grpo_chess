#!/usr/bin/env bash
# setup_job_env.sh — Standard Lightning.ai job environment setup for grpo_chess.
#
# IMPORTANT: Must be SOURCED (not executed as a subshell) so that PYTHONPATH
# and other exports persist in the calling shell:
#
#   source scripts/lightning/setup_job_env.sh
#
# Do NOT run as: bash scripts/lightning/setup_job_env.sh  (exports are lost)
#
# Controls (env vars):
#   REPO_BRANCH           Branch to checkout (default: feature/lightning_training)
#   REPO_URL              Git remote (default: https://github.com/noamdwc/grpo_chess.git)
#   REPO_DIR              Where to clone (default: /teamspace/studios/this_studio/repo)
#   INSTALL_JAX           1|0 — install JAX + dm-haiku ecosystem (default: 1)
#   INSTALL_STOCKFISH     1|0 — apt-get install stockfish if missing (default: 1)
#   DOWNLOAD_9M_CKPT      1|0 — download 9M JAX checkpoint (default: 0)
#   RESTORE_NUMPY_VERSION numpy version to pin AFTER JAX install, e.g. "1.26.4"
#                         Leave unset to keep JAX's numpy (2.x). Set when JAX is
#                         only needed for a pre-processing step before PyTorch training,
#                         since torchmetrics/matplotlib are compiled against numpy 1.x.
#   WANDB_KEY             WandB API key (propagated to WANDB_API_KEY automatically)

set -euo pipefail

# ── 0. Propagate WandB key ─────────────────────────────────────────────────────
if [[ -n "${WANDB_KEY:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${WANDB_KEY}"
  echo "[setup] Propagated WANDB_KEY → WANDB_API_KEY"
fi

# ── 1. Fix Lightning sitecustomize (numpy 2.x ABI issue) ──────────────────────
# Lightning images ship a global sitecustomize.py that eagerly imports
# Lightning/PyTorch on every Python startup; this can break JAX installs.
_SC_FIX_DIR="/tmp/grpo_chess_sc_fix"
mkdir -p "${_SC_FIX_DIR}"
printf '# intentionally empty — overrides Lightning sitecustomize\n' > "${_SC_FIX_DIR}/sitecustomize.py"
export PYTHONPATH="${_SC_FIX_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "[setup] sitecustomize fix applied (PYTHONPATH prefix: ${_SC_FIX_DIR})"

# ── 2. Clone / update repo ─────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/noamdwc/grpo_chess.git}"
REPO_BRANCH="${REPO_BRANCH:-feature/lightning_training}"
REPO_DIR="${REPO_DIR:-/teamspace/studios/this_studio/repo}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  echo "[setup] Cloning ${REPO_URL} → ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin
git checkout "${REPO_BRANCH}"
git reset --hard "origin/${REPO_BRANCH}"
echo "[setup] Repo at $(git rev-parse --short HEAD) on ${REPO_BRANCH}"

# ── 3. Init submodules (searchless_chess) ─────────────────────────────────────
git submodule update --init --recursive
echo "[setup] Submodules updated"

# ── 4. Install Python deps ─────────────────────────────────────────────────────
echo "[setup] Installing core Python packages..."
pip install -q \
  torch torchvision pytorch-lightning \
  wandb chess stockfish \
  pyyaml numpy

INSTALL_JAX="${INSTALL_JAX:-1}"
if [[ "${INSTALL_JAX}" == "1" ]]; then
  echo "[setup] Installing JAX ecosystem..."
  # Detect GPU
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "[setup]   NVIDIA GPU detected — installing jax[cuda12]"
    pip install -q \
      "jax[cuda12]==0.8.2" \
      "dm-haiku==0.0.16" \
      "optax==0.2.7" \
      "orbax-checkpoint==0.11.32" \
      "grain==0.2.15" \
      "jaxtyping==0.3.4" \
      "chex==0.1.91" \
      "ml-collections" \
      "tensorflow-cpu"
  else
    echo "[setup]   No GPU — installing CPU JAX"
    pip install -q \
      "jax==0.8.2" "jaxlib==0.8.2" \
      "dm-haiku==0.0.16" \
      "optax==0.2.7" \
      "orbax-checkpoint==0.11.32" \
      "grain==0.2.15" \
      "jaxtyping==0.3.4" \
      "chex==0.1.91" \
      "ml-collections" \
      "tensorflow-cpu"
  fi
  echo "[setup] JAX ecosystem installed"
fi

# ── 5. Stockfish ───────────────────────────────────────────────────────────────
INSTALL_STOCKFISH="${INSTALL_STOCKFISH:-1}"
_find_stockfish() {
  for _p in "${STOCKFISH_PATH:-}" "$(command -v stockfish 2>/dev/null || true)" \
            /usr/games/stockfish /usr/bin/stockfish /usr/local/bin/stockfish; do
    [[ -x "${_p:-}" ]] && printf '%s\n' "${_p}" && return 0
  done
  return 1
}

if _STOCKFISH_BIN="$(_find_stockfish)"; then
  echo "[setup] Stockfish: ${_STOCKFISH_BIN}"
  export STOCKFISH_PATH="${_STOCKFISH_BIN}"
elif [[ "${INSTALL_STOCKFISH}" == "1" ]]; then
  echo "[setup] Installing Stockfish via apt-get..."
  apt-get update -qq && apt-get install -y --no-install-recommends stockfish
  export STOCKFISH_PATH="$(command -v stockfish)"
  echo "[setup] Stockfish installed: ${STOCKFISH_PATH}"
else
  echo "[setup] WARNING: Stockfish not found and INSTALL_STOCKFISH=0"
fi

# ── 6. Download 9M JAX checkpoint (optional) ───────────────────────────────────
DOWNLOAD_9M_CKPT="${DOWNLOAD_9M_CKPT:-0}"
_9M_CKPT_DIR="${REPO_DIR}/searchless_chess/checkpoints"
if [[ "${DOWNLOAD_9M_CKPT}" == "1" ]]; then
  if [[ -d "${_9M_CKPT_DIR}/9M" ]]; then
    echo "[setup] 9M checkpoint already present at ${_9M_CKPT_DIR}/9M"
  else
    echo "[setup] Downloading 9M checkpoint..."
    mkdir -p "${_9M_CKPT_DIR}"
    wget -q https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip \
        -O "${_9M_CKPT_DIR}/9M.zip"
    cd "${_9M_CKPT_DIR}" && unzip -q 9M.zip && rm 9M.zip
    cd "${REPO_DIR}"
    echo "[setup] 9M checkpoint ready at ${_9M_CKPT_DIR}/9M"
  fi
fi

# ── 7. Optionally restore numpy for PyTorch ecosystem ─────────────────────────
# JAX 0.8.2 upgrades numpy to 2.x, but torchmetrics/matplotlib need numpy 1.x.
# Set RESTORE_NUMPY_VERSION=1.26.4 when JAX is only used for a pre-processing step.
RESTORE_NUMPY_VERSION="${RESTORE_NUMPY_VERSION:-}"
if [[ -n "${RESTORE_NUMPY_VERSION}" ]]; then
  echo "[setup] Restoring numpy==${RESTORE_NUMPY_VERSION} for PyTorch compatibility..."
  pip install -q "numpy==${RESTORE_NUMPY_VERSION}"
  python -c "import numpy; print(f'[setup] numpy restored to {numpy.__version__}')"
fi

echo "[setup] Environment setup complete. Working directory: ${REPO_DIR}"
