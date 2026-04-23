## 1. Deterministic Diagnostics

- [ ] 1.1 Add a reusable parity harness that compares bare DM, wrapped zero-think, and wrapped-transformer-plus-frozen-DM-head behavior on a single position
- [ ] 1.2 Build a fixed 20-50 position corpus covering openings, castling, en passant, promotions, checks, and endgames
- [ ] 1.3 Add a first-divergence full-game tracer that advances both paths from the same starting FEN and stops on the first mismatching move

## 2. Automated Verification

- [ ] 2.1 Add pytest coverage that locks in legal-action parity, move round-trip parity, and wrapped-body parity against the frozen DM head
- [ ] 2.2 Add a regression test proving that raw DM checkpoints do not determine the wrapped action-token policy head weights

## 3. Documentation

- [ ] 3.1 Write a concise run note summarizing the first causal divergence, the supporting evidence, and the remaining product decision for raw-DM zero-think evaluation
