## ADDED Requirements

### Requirement: Baseline and wrapped models share weights
The diagnostic SHALL load a single DeepMind-ported 9M checkpoint and use it for both the baseline (bare DM-port model) and the wrapped (reasoning-model, untrained) configurations, so any measured gap is attributable only to the reasoning wrapper.

#### Scenario: Same weights drive both configurations
- **WHEN** the test builds its two evaluation configurations
- **THEN** both configurations derive their transformer parameters from the same DM-port checkpoint, with no training, fine-tuning, or reinitialization in between

### Requirement: Evaluate each configuration with 64 games against Stockfish
The diagnostic SHALL play exactly 64 games against Stockfish for each of the two configurations, using the same Stockfish settings (skill/depth) that `StockfishEvalCallback` uses by default, and a fixed random seed.

#### Scenario: Each configuration plays 64 games
- **WHEN** the test runs to completion
- **THEN** the baseline configuration has played 64 games and the wrapped configuration has played 64 games against Stockfish under identical settings and seed

#### Scenario: Determinism
- **WHEN** the test is run twice with the same seed and the same Stockfish binary
- **THEN** both runs report the same per-game outcomes and aggregate score for each configuration

### Requirement: Reuse existing evaluation harness
The diagnostic SHALL route move selection through the existing `evaluator.py` / `reasoning/real_play.py` path so that results are directly comparable to the Stockfish score reported at step 0 of a GRPO run.

#### Scenario: Harness reuse
- **WHEN** the wrapped configuration selects moves during evaluation
- **THEN** it uses the same think→move path used by `StockfishEvalCallback` (no bespoke sampling code)

### Requirement: Report baseline, wrapped, and delta metrics
The diagnostic SHALL report, for each configuration, at least the aggregate score (win rate) and `elo_diff` as produced by the existing evaluator, plus the delta between the two configurations.

#### Scenario: Metrics are reported
- **WHEN** the test finishes
- **THEN** it emits (via logging or test output) baseline score, wrapped score, and `baseline_score - wrapped_score`, along with per-outcome counts (W/D/L) for each configuration

### Requirement: Graceful skip when dependencies are unavailable
The diagnostic SHALL skip (not fail) if the Stockfish binary or the DM-port checkpoint is unavailable in the local environment, with a message identifying the missing dependency.

#### Scenario: Stockfish missing
- **WHEN** the Stockfish binary cannot be located
- **THEN** the test is skipped with a message naming Stockfish as the missing dependency

#### Scenario: Checkpoint missing
- **WHEN** the DM-port 9M checkpoint cannot be located
- **THEN** the test is skipped with a message naming the checkpoint as the missing dependency
