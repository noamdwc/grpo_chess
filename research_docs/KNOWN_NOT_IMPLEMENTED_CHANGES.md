# Known Not Implemented Changes (KNI)

This file tracks **specific changes or recommendations** that the user has explicitly decided **NOT** to implement (at least for now).  
It is used to prevent future agents from repeatedly proposing unwanted changes.

## Format

Each entry should follow this structure:

- **ID**: `KNI-00X` (short, stable identifier)
- **Source**: Where the change came from (e.g. research doc + section, external suggestion, audit ID)
- **Change description**: Concrete, code-level description of what would be changed
- **Scope**: Files/modules that would be affected
- **Reason NOT to implement**: User’s reasoning (complexity, conflicts with goals, negative results, etc.)
- **Status**: `"not_planned"`, `"reconsider_later"`, or `"rejected"`
- **Last reviewed**: Date and git commit hash when this decision was last revisited

Example template for future entries:

- ID: KNI-001  
  Source: TODO (e.g. `2026-01-19_loss-increase-and-lr-regression-analysis.md`, "Future Research Notes")  
  Change description: TODO  
  Scope: TODO  
  Reason NOT to implement: TODO  
  Status: not_planned \| reconsider_later \| rejected  
  Last reviewed: YYYY-MM-DD, COMMIT_SHA

## Current Entries

- ID: KNI-001
  Source: `research_docs/2026-04-09_self-play-deferred-merge-notes.md`
  Change description: Do not merge the `self_play` branch into the current integration path for `main`. Leave its GRPO defaults, step-sync behavior, parallel eval implementation, and docs out of the branch consolidation. Reconsider only selected ideas later as fresh changes on top of `main`.
  Scope: `src/configs/default.yaml`, `src/eval_utils.py`, `src/evaluator.py`, `src/grpo_logic/model.py`, `src/grpo_logic/loss.py`, `tests/test_eval_utils.py`, `research_docs/*`
  Reason NOT to implement: `self_play` is an older divergent experiment branch with overlap in core training and evaluation code. Merging it now would mix stale experiment defaults and branch-specific behavior into the unified trunk. The user wants it documented and deferred until after the merge to `main`.
  Status: reconsider_later
  Last reviewed: 2026-04-09, 2374b6ead25d4c5294ce44f118ce0bd44b8ead1d
