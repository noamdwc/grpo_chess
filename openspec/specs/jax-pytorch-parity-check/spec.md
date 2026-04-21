### Requirement: Two-variant parity diagnostic
The system SHALL provide a local CLI diagnostic that runs the existing forward-pass parity gate and then plays a 64-game Stockfish match for each of two variants built from the 9M DeepMind checkpoint: (1) the original JAX model from the `searchless_chess` submodule, and (2) the PyTorch port in `src/dm_port/` with weights produced by the existing `src/dm_port/convert_jax.py`.

#### Scenario: Runs end-to-end on a prepared machine
- **WHEN** a user with JAX installed, the `searchless_chess` submodule + 9M checkpoint available, the converted DM port checkpoint present, and Stockfish resolvable runs the diagnostic
- **THEN** the script runs the forward-pass gate, then the 64-game match per variant, and writes a report summarizing results
- **AND** the script exits non-zero if the forward-pass gate fails or any game produces move-level divergence

#### Scenario: Missing prerequisites produce actionable errors
- **WHEN** JAX, the submodule, the JAX 9M checkpoint, the converted PT checkpoint, or Stockfish is missing
- **THEN** the script exits with a message naming the missing prerequisite and the command to fix it

### Requirement: Forward-pass precondition delegates to existing test
The diagnostic SHALL execute the existing `tests/test_dm_port_parity.py::test_port_matches_jax_float64` and `tests/test_dm_port_parity.py::test_port_matches_jax_float32` as its forward-pass precondition. It MUST NOT reimplement the numeric comparison.

#### Scenario: Gate failure halts the diagnostic
- **WHEN** either the float64 (`<1e-10`) or float32 (`<5e-4`) gate fails
- **THEN** the diagnostic reports the failing gate with the measured `max_abs_diff` and exits non-zero without running the Stockfish matches

#### Scenario: Gate pass proceeds to matches
- **WHEN** both gates pass
- **THEN** the diagnostic records the measured `max_abs_diff` for each dtype in the report and proceeds to the 64-game matches

### Requirement: Port code under test is used unmodified
The diagnostic SHALL load PyTorch weights from the checkpoint produced by `src/dm_port/convert_jax.py` (expected at `checkpoints/dm_port/9M.pt`). If that file is missing, the diagnostic SHALL offer to invoke `src.dm_port.convert_jax` to produce it, and otherwise SHALL NOT use any alternative conversion path.

#### Scenario: Converted checkpoint is used
- **WHEN** the PyTorch variant is loaded
- **THEN** weights come from `checkpoints/dm_port/9M.pt` (produced by `src/dm_port/convert_jax.convert`)

#### Scenario: Missing converted checkpoint is offered a fix
- **WHEN** `checkpoints/dm_port/9M.pt` is missing
- **THEN** the script either calls `src.dm_port.convert_jax.convert` to produce it in-place, or exits with the exact command to run

### Requirement: 64-game Stockfish evaluation per variant
The diagnostic SHALL play 64 games (32 openings × 2 colors) against Stockfish for each variant, using identical Stockfish configuration, identical starting positions and colors, and greedy model move selection, and SHALL compare results across variants.

#### Scenario: Shared match configuration
- **WHEN** the match phase runs
- **THEN** both variants play the same 64 games in the same order, against a Stockfish configured with fixed skill level, fixed node budget, single thread, and fixed hash size
- **AND** the Stockfish binary version and configuration are recorded in the report

#### Scenario: Per-variant match results are reported
- **WHEN** both matches complete
- **THEN** the report records wins, draws, losses, score (W + 0.5·D), and Elo-difference estimate per variant

#### Scenario: Cross-variant comparison is reported
- **WHEN** both matches complete
- **THEN** the report records the number of games with at least one move-level divergence, the distribution of first-divergence plies, and a cross-tabulation of per-game results

#### Scenario: Move-for-move pass criterion
- **WHEN** the match phase completes
- **THEN** the diagnostic flags as fail any case where JAX and the PyTorch port produced different moves in any game

### Requirement: Excluded from default test suite
The diagnostic SHALL live under `scripts/` and SHALL NOT be collected or executed by `pytest tests/`.

#### Scenario: Default pytest run ignores the diagnostic
- **WHEN** a developer runs `~/miniconda3/envs/grpo_chess/bin/python -m pytest tests/`
- **THEN** the diagnostic script is not collected or executed
