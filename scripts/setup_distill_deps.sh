#!/usr/bin/env bash
# Setup dependencies for the distillation pipeline (DeepMind teacher model).
#
# Works on both Colab and local macOS. See scripts/DISTILL_DEPS.md for details.
#
# Usage:
#   bash scripts/setup_distill_deps.sh                    # install deps + download 270M
#   bash scripts/setup_distill_deps.sh --checkpoint 9M    # use 9M instead
#   bash scripts/setup_distill_deps.sh --skip-checkpoint  # deps only, no download
#
# Environment variables:
#   COLAB=1  — set this in Colab notebooks before running
#   PYTHON   — path to python binary (default: auto-detect)

set -euo pipefail

# ── Detect environment ────────────────────────────────────────────────────────

IS_COLAB="${COLAB:-0}"

if [ -n "${PYTHON:-}" ]; then
    : # use explicit PYTHON
elif [ "$IS_COLAB" = "1" ]; then
    PYTHON="python"
elif [ -x "$HOME/miniconda3/envs/grpo_chess/bin/python" ]; then
    PYTHON="$HOME/miniconda3/envs/grpo_chess/bin/python"
else
    PYTHON="python3"
fi

PIP="$PYTHON -m pip"
PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

# ── Parse arguments ───────────────────────────────────────────────────────────

MODEL="270M"
SKIP_CKPT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) MODEL="$2"; shift 2 ;;
        --skip-checkpoint) SKIP_CKPT=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Resolve paths ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CKPT_DIR="$REPO_DIR/searchless_chess/checkpoints"

echo "============================================"
echo "  Distillation dependency setup"
echo "  Environment: $( [ "$IS_COLAB" = "1" ] && echo 'Colab' || echo 'Local' )"
echo "  Python:      $PYTHON ($PY_VERSION)"
echo "  Model:       $MODEL"
echo "  Repo:        $REPO_DIR"
echo "============================================"
echo

# ── Step 1: Install Python packages ──────────────────────────────────────────

echo ">>> [1/4] Installing JAX ecosystem packages..."

$PIP install --quiet \
    jax \
    dm-haiku \
    chex \
    optax \
    orbax-checkpoint \
    grain \
    jaxtyping

# apache_beam: installs cleanly on Python <=3.12 (Colab) but broken on 3.14+
# because pyarrow<19 can't build from source. On 3.13+ we skip it and rely on
# the stub in generate_dataset.py (see scripts/DISTILL_DEPS.md).
if [ "$PY_MINOR" -lt 13 ]; then
    echo "    Installing apache-beam (Python $PY_VERSION)..."
    $PIP install --quiet apache-beam
else
    echo "    Skipping apache-beam (Python $PY_VERSION — will use built-in stub)"
fi

echo "    Done."

# ── Step 2: Verify imports ───────────────────────────────────────────────────

echo ">>> [2/4] Verifying imports..."

$PYTHON -c "
import sys, jax, haiku, chex, optax, orbax.checkpoint
print(f'  jax={jax.__version__}  haiku={haiku.__version__}  optax={optax.__version__}')
try:
    from apache_beam import coders
    print('  apache_beam: installed')
except ImportError:
    print(f'  apache_beam: stub (Python {sys.version_info.major}.{sys.version_info.minor})')
"

echo "    Imports OK."

# ── Step 3: Verify module loading ────────────────────────────────────────────

echo ">>> [3/4] Verifying searchless_chess module loading..."

cd "$REPO_DIR"
$PYTHON -c "from src.distill.generate_dataset import build_teacher_engine; print('  build_teacher_engine: OK')"

echo "    Module loading OK."

# ── Step 4: Download checkpoint ──────────────────────────────────────────────

if $SKIP_CKPT; then
    echo ">>> [4/4] Skipping checkpoint download (--skip-checkpoint)."
elif [ -d "$CKPT_DIR/$MODEL" ]; then
    echo ">>> [4/4] Checkpoint already exists at $CKPT_DIR/$MODEL — skipping."
else
    ZIP_URL="https://storage.googleapis.com/searchless_chess/checkpoints/${MODEL}.zip"
    ZIP_PATH="$CKPT_DIR/${MODEL}.zip"

    echo ">>> [4/4] Downloading $MODEL checkpoint..."
    echo "    URL: $ZIP_URL"

    mkdir -p "$CKPT_DIR"

    if command -v wget &>/dev/null; then
        wget -q --show-progress "$ZIP_URL" -O "$ZIP_PATH"
    else
        curl -L --progress-bar "$ZIP_URL" -o "$ZIP_PATH"
    fi

    echo "    Extracting..."
    unzip -o -q "$ZIP_PATH" -d "$CKPT_DIR"
    rm "$ZIP_PATH"

    echo "    Checkpoint ready at $CKPT_DIR/$MODEL"
fi

echo
echo "============================================"
echo "  Setup complete!"
echo "============================================"
