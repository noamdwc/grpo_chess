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

| Package | Purpose | Installs on Colab? | Installs on Python 3.14? |
|---------|---------|-------------------|------------------------|
| `jax` | DeepMind model runtime | Yes | Yes |
| `dm-haiku` | DeepMind model architecture | Yes | Yes |
| `chex` | JAX testing/assertions | Yes | Yes |
| `optax` | Optimizer (transitive dep of `training_utils.py`) | Yes | Yes |
| `orbax-checkpoint` | Checkpoint loading | Yes | Yes |
| `grain` | Data loading (transitive dep of `constants.py`) | Yes | Yes |
| `apache-beam` | Data pipeline coders (transitive dep of `constants.py`) | Yes | **No** |

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

### Solution: `_ensure_apache_beam_stub()`

`src/distill/generate_dataset.py` registers a minimal `apache_beam` stub in
`sys.modules` before loading any searchless_chess modules. The stub provides
dummy `StrUtf8Coder`, `BigIntegerCoder`, `FloatCoder`, and `TupleCoder` classes
that accept arbitrary arguments. This is safe because:

1. The coders are only used in `constants.CODERS` dict, which is for data
   pipeline serialization (Beam workers).
2. Our code never reads `constants.CODERS` — we only use `constants.Predictor`
   (a simple dataclass) and `constants.Sequences` (a type alias).
3. The stub is only activated when the real `apache_beam` can't be imported.
   On Colab (Python 3.10-3.12) the real package loads normally.

### The dependency chain that fails

```
build_teacher_engine()
  → _load_sc_module("transformer")
    → transformer.py: from searchless_chess.src import constants
      → constants.py: from apache_beam import coders          ← FAILS on 3.14
      → constants.py: from grain import python as pygrain     ← needs grain
      → constants.py: import haiku as hk                      ← needs dm-haiku
  → _load_sc_module("training_utils")
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
