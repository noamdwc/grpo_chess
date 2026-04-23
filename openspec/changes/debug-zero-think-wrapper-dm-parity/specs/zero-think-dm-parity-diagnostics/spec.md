## ADDED Requirements

### Requirement: Single-position zero-think parity comparison
The repository SHALL provide a deterministic harness that compares the bare DM path and the wrapped zero-think path on the same fixed FEN and records the intermediate artifacts needed to localize the first divergence.

#### Scenario: Compare one fixed FEN
- **WHEN** the single-position parity harness runs on a fixed FEN with the raw DM checkpoint
- **THEN** it MUST report the board FEN, side to move, encoded FEN tokens, legal move list, legal action ids, top action summaries, chosen action ids, decoded moves, and decoded-move legality for both paths

### Requirement: Multi-FEN parity sweep
The repository SHALL provide a deterministic fixed-position corpus that exercises zero-think parity across opening, midgame, castling, en passant, promotion, and check-related positions.

#### Scenario: Sweep the fixed corpus
- **WHEN** the parity sweep runs on the fixed corpus
- **THEN** it MUST compare bare and wrapped choices position-by-position and MUST preserve enough information to identify whether divergence comes from legal-action mismatch, move decoding mismatch, or move-selection semantics

### Requirement: First-divergence game trace
The repository SHALL provide a deterministic full-game trace that advances both paths from the same starting position and stops at the first divergent move.

#### Scenario: Stop at the first divergence
- **WHEN** the full-game tracer runs from a fixed start FEN
- **THEN** it MUST log the pre-move FEN, both selected moves, both chosen action ids, move legality, and both post-move FENs for each ply until the first divergence is found

### Requirement: Raw-DM zero-think root-cause evidence
The repository SHALL preserve an automated diagnostic that distinguishes wrapped-transformer-body parity from wrapped zero-think selection semantics when the input checkpoint is the raw DM port checkpoint.

#### Scenario: Compare wrapped body with frozen DM head
- **WHEN** the diagnostic reuses the wrapped transformer body with the frozen DM return head on bare DM-style inputs
- **THEN** the diagnostic MUST be able to show whether the wrapped transformer body still matches the bare DM path independently of the wrapped zero-think policy head
