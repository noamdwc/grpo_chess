# Distillation Dependencies

The distillation pipeline loads DeepMind's searchless chess JAX model as a teacher.
This requires a JAX/Haiku stack that has non-trivial compatibility issues across environments.

## Quick Start

```bash
# Local (macOS)
bash scripts/setup_distill_deps.sh

# Colab notebook cell
import os
os.environ["COLAB"] = "1"
!bash scripts/setup_distill_deps.sh
```

## Required Packages

| Package | Purpose | Colab | Local (3.14) |
|---------|---------|-------|--------------|
| `jax` / `jaxlib` | DeepMind model runtime | Pre-installed (0.7.2 + CUDA) | Pinned 0.8.2 |
| `dm-haiku` | DeepMind model architecture | Installed | Pinned 0.0.16 |
| `chex` | JAX testing/assertions | Installed | Pinned 0.1.91 |
| `optax` | Optimizer (transitive dep of `training_utils.py`) | Installed | Pinned 0.2.7 |
| `orbax-checkpoint` | Checkpoint loading | Installed | Pinned 0.11.32 |
| `grain` | Data loading (transitive dep of `constants.py`) | Installed | Pinned 0.2.15 |
| `apache-beam` | Data pipeline coders (transitive dep of `constants.py`) | Installed | **Skipped** (stub) |

### Pinning strategy

- **Colab**: JAX/jaxlib are pre-installed with matching CUDA drivers — we don't
  touch them. Only the extras (`dm-haiku`, `chex`, etc.) are installed unpinned
  so they resolve against Colab's JAX.
- **Local**: All versions pinned to tested-working set (2026-02-10). No CUDA needed.

## The `apache_beam` Problem

### Root cause

DeepMind's `searchless_chess/src/constants.py` imports `apache_beam.coders` at
module level. These coders are only used for data pipeline serialization, never
for model inference. But because the import is at module level, loading *any*
searchless_chess module that transitively imports `constants.py` (which is all
of them — `transformer.py`, `training_utils.py`, `neural_engines.py`) will fail
if `apache_beam` can't be imported.

### Why it fails on Python 3.14

`apache_beam` requires `pyarrow<19`. On Python 3.14, `pyarrow<19` has no
pre-built wheel and fails to build from source because CMake can't find the
Arrow C++ library:

```
CMake Error at CMakeLists.txt:274 (find_package):
  Could not find a package configuration file provided by "Arrow"
```

Installing `apache-beam --no-deps` partially works but then fails at runtime
because `apache_beam.coders` pulls in a deep chain of internal imports that
require `grpc`, `proto`, and other packages — each with their own build issues
on 3.14.

### Solution: `_patch_searchless_chess_compat()`

`src/distill/generate_dataset.py` calls `_patch_searchless_chess_compat()` before
loading any searchless_chess modules. This function handles two issues:

**1. `apache_beam` stub** — Registers a minimal `apache_beam` module in
`sys.modules` with dummy `StrUtf8Coder`, `BigIntegerCoder`, `FloatCoder`, and
`TupleCoder` classes. This is safe because:

- The coders are only used in `constants.CODERS` dict for Beam pipeline
  serialization. Our code never reads `constants.CODERS`.
- The stub is only activated when the real `apache_beam` can't be imported.
  On Colab (Python 3.10-3.12) the real package loads normally.

**2. `jax.sharding.PositionalSharding` shim** — `training_utils.py` uses
`jax.sharding.PositionalSharding` as a type annotation on its `replicate()`
function. This attribute was removed/moved in newer JAX versions (missing on
both JAX 0.8.x locally and older Colab JAX). Since Python evaluates function
annotations at definition time (no `from __future__ import annotations` in that
file), the module fails to load. We add a dummy class at that path since
`replicate()` is never called by our inference code.

### The dependency chain that fails

```
build_teacher_engine()
  → _load_sc_module("transformer")
    → transformer.py: from searchless_chess.src import constants
      → constants.py: from apache_beam import coders          ← FAILS on 3.14
      → constants.py: from grain import python as pygrain     ← needs grain
      → constants.py: import haiku as hk                      ← needs dm-haiku
  → _load_sc_module("training_utils")
    → training_utils.py: jax.sharding.PositionalSharding      ← MISSING in JAX 0.8.x and Colab
    → training_utils.py: import optax                         ← needs optax
    → training_utils.py: import orbax.checkpoint              ← needs orbax
```

## Checkpoint Download

Checkpoints are hosted at `storage.googleapis.com/searchless_chess/checkpoints/`.
Available models:

| Model | Size (zip) | Parameters |
|-------|-----------|------------|
| `9M` | ~65 MB | 9M |
| `136M` | ~1 GB | 136M |
| `270M` | ~1.9 GB | 270M |

The setup script downloads to `searchless_chess/checkpoints/<MODEL>/` and
expects the checkpoint at step 6,400,000. The script skips download if the
directory already exists.
