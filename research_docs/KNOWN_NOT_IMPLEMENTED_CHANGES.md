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

As of this initial audit, there are no explicit user-decided "do not implement" changes to record here yet.  
Future audits and decisions should add `KNI-*` entries to this section.

