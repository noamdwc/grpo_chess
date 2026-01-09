---
# =============================================================================
# METADATA (Required - Fill in ALL fields)
# =============================================================================
title: "Your Analysis Title Here"
date: "YYYY-MM-DD HH:MM:SS UTC"
agent: "model-id (e.g., claude-opus-4-5-20251101)"

# Git State - Critical for reproducibility
git_commit: "full-40-character-commit-hash"
git_branch: "branch-name"
uncommitted_changes: false  # true if you used files not yet committed

# Files Analyzed - List all files you read or referenced
files_analyzed:
  - src/example/file.py
  - src/another/module.py

# WandB Runs Referenced - List run IDs and names
wandb_runs:
  - run_id: "abc123xy"
    run_name: "chess-grpo-YYYYMMDD-HHMM-xxxx"
  - run_id: "def456yz"
    run_name: "another-run-name"

# Tools Used - What capabilities did you use?
tools_used:
  - Read
  - Grep
  - Glob
  - WandB MCP (list_runs, get_run_metrics, etc.)
  - WebSearch
  - Task (subagent exploration)

# Original Prompt - The exact prompt given to the agent
prompt: |
  Paste the original user prompt here.
  This helps future agents understand the context
  and scope of the analysis.

# Tags for categorization (optional but helpful)
tags:
  - training
  - rewards
  - debugging
---

# [Title]

## Executive Summary

<!-- 2-3 sentences summarizing the key findings. A reader should understand the main takeaway without reading further. -->

**TL;DR:** [Your main finding in one sentence]

## Context

### Background
<!-- What situation or problem prompted this analysis? -->

### Research Question
<!-- What specific question(s) were you trying to answer? -->

### Scope
<!-- What was included/excluded from this analysis? -->

## Methodology

### Data Sources
<!-- What data did you analyze? Be specific. -->

| Source | Details |
|--------|---------|
| WandB Runs | [List specific run IDs] |
| Code Files | [List files with commit context] |
| External | [Any external resources] |

### Analysis Approach
<!-- Step-by-step description of your analysis process -->

1. First, I...
2. Then, I examined...
3. Finally, I...

### Tools Used
<!-- Which tools did you use and how? -->

- **Read**: Examined source files at specific line numbers
- **WandB MCP**: Retrieved metrics from runs [list IDs]
- **Grep**: Searched for patterns like `[pattern]`

## Findings

### Finding 1: [Title]

**Evidence:**

In `src/path/to/file.py:XX-YY`:
```python
# Relevant code snippet
def example():
    pass
```

**Metrics from WandB:**

| Run | Metric | Value |
|-----|--------|-------|
| `run_id` | metric_name | value |

**Interpretation:**
<!-- What does this mean? -->

### Finding 2: [Title]

<!-- Repeat structure for each finding -->

## Analysis

### Root Cause Analysis
<!-- If debugging, what are the root causes? -->

### Impact Assessment
<!-- What is the impact of these findings? -->

### Trade-offs
<!-- What trade-offs exist between different approaches? -->

## Recommendations

### Immediate Actions (High Priority)

1. **[Action Name]**
   - What: [Description]
   - Why: [Justification]
   - Where: `src/path/to/file.py:XX`
   - Estimated effort: [Low/Medium/High]

2. **[Action Name]**
   - ...

### Medium-Term Improvements

1. **[Action Name]**
   - ...

### Long-Term Considerations

1. **[Action Name]**
   - ...

## Open Questions

<!-- What remains unknown or needs further investigation? -->

- [ ] Question 1: [Description]
- [ ] Question 2: [Description]

## Appendix

### A. Full Code References

<!-- Optional: Include longer code snippets here -->

### B. Additional Metrics

<!-- Optional: Include detailed metric tables -->

### C. Related Documents

<!-- Link to related research_docs files -->

- [Previous analysis](./YYYY-MM-DD_related-topic.md)

---

<!--
HUMAN REVIEW SECTION (Add below if a human reviews/updates this document)

## Human Review

**Reviewer:** [Name]
**Date:** YYYY-MM-DD
**Changes:**
- [What was updated]

**Additional Notes:**
- [Any corrections or additions]
-->
