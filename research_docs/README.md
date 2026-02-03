# Research Documentation Directory

This directory contains structured research insights, analysis documents, and findings generated during the development of the GRPO Chess project. It is designed to be used by both **AI agents** and **human researchers**.

## Purpose

- **Knowledge Persistence**: Capture insights that would otherwise be lost between agent sessions
- **Reproducibility**: Document exactly what code state, tools, and data were used for each analysis
- **Collaboration**: Enable future agents and humans to build on previous findings
- **Audit Trail**: Track the evolution of understanding about the project

## Quick Start for Agents

1. **Before writing**: Copy `TEMPLATE.md` to a new file with naming convention: `YYYY-MM-DD_<short-description>.md`
2. **Fill in metadata**: Complete all sections in the YAML frontmatter
3. **Document your work**: Follow the template structure
4. **Be specific**: Include file paths with line numbers, exact metric values, and concrete evidence

## File Naming Convention

```
YYYY-MM-DD_<short-kebab-case-description>.md
```

Examples:
- `2026-01-09_grpo-learning-failure-analysis.md`
- `2026-01-10_reward-signal-improvements.md`
- `2026-01-12_dataset-quality-investigation.md`

## Directory Structure

```
research_docs/
├── README.md           # This file
├── TEMPLATE.md         # Template for new documents
└── *.md                # Research documents (dated)
```

## Document Requirements

Each document MUST include:

### 1. YAML Frontmatter (Machine-Readable Metadata)

```yaml
---
title: "Document Title"
date: "YYYY-MM-DD HH:MM:SS"
agent: "model-name (e.g., claude-opus-4-5-20251101)"
git_commit: "full-commit-hash"
git_branch: "branch-name"
uncommitted_changes: true/false
files_analyzed:
  - path/to/file1.py
  - path/to/file2.py
wandb_runs:
  - run_id: "abc123"
    name: "run-name"
tools_used:
  - "WandB MCP"
  - "Grep"
  - "Read"
prompt: |
  The original prompt given to the agent
---
```

### 2. Executive Summary
A 2-3 sentence TL;DR of the key findings.

### 3. Context
What question was being investigated and why.

### 4. Methodology
What tools, files, and data sources were used.

### 5. Findings
Detailed findings with evidence (code snippets, metrics, file references).

### 6. Recommendations
Actionable next steps based on findings.

### 7. Open Questions
Unresolved issues for future investigation.

## Evidence Standards

When referencing code:
```markdown
**In `src/module/file.py:42-58`:**
```python
def function_name():
    # relevant code
```
```

When referencing metrics:
```markdown
| Run ID | Metric | Value | Interpretation |
|--------|--------|-------|----------------|
| abc123 | loss   | 0.05  | Converged      |
```

When referencing WandB runs:
```markdown
**Run `chess-grpo-20260109` (ID: `xyz789`):**
- Epochs: 180
- Final loss: 0.017
- Stockfish score: 1.5%
```

## For AI Agents

### Agent Prompts

For complex research or implementation tasks, see the specialized agent prompts:
- **Research tasks**: `.claude/agents/research-insights.md`
- **Code changes**: `.claude/agents/code-implementation.md`

These prompts define workflows with discussion checkpoints before finalizing work.

### Reading Existing Documents

**Do NOT read all prior documents.** Most content becomes stale quickly.

Instead:
1. List recent files: `ls -t research_docs/*.md | head -5`
2. Check titles - only read if directly relevant to your current task
3. Most tasks won't require reading prior docs - skip if not clearly relevant

### Writing New Documents

1. **Check code state first:**
   ```bash
   git rev-parse HEAD          # Get commit hash
   git status --porcelain      # Check for uncommitted changes
   git branch --show-current   # Get branch name
   ```

2. **Create from template:**
   ```bash
   cp research_docs/TEMPLATE.md research_docs/YYYY-MM-DD_your-topic.md
   ```

3. **Fill in ALL metadata fields** - this is critical for reproducibility

4. **Use specific references:**
   - File paths with line numbers: `src/grpo_self_play/loss.py:42`
   - Exact metric values: `mean_clip_fraction: 0.116`
   - WandB run IDs: `xyqjy01q`

### Quality Checklist

Before saving, verify:
- [ ] All YAML frontmatter fields are filled
- [ ] Git commit hash is correct and complete
- [ ] All referenced files exist in the codebase
- [ ] WandB run IDs are valid
- [ ] Code snippets include file paths and line numbers
- [ ] Recommendations are actionable
- [ ] Document is self-contained (reader doesn't need prior context)

## For Human Reviewers

### Verifying Agent Analysis

1. **Check code state**: Use `git checkout <commit>` to see the exact code analyzed
2. **Verify WandB data**: Cross-reference run IDs in the WandB dashboard
3. **Test recommendations**: Agent suggestions should be testable

### Updating Documents

If you update a document after an agent created it:
1. Add a "Human Review" section at the end
2. Note the date and what was changed
3. Keep the original agent analysis intact

## Reproducibility Reference

Based on the [ML Reproducibility Checklist](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf), each document should enable:

1. **Code Reproducibility**: Exact commit hash to checkout
2. **Data Reproducibility**: WandB run IDs for metrics and configs
3. **Environment Context**: Tools and dependencies used
4. **Analysis Reproducibility**: Step-by-step methodology

## Related Resources

- [Reproducibility in ML Research](https://arxiv.org/html/2406.14325v1)
- [Neptune.ai: How to Solve Reproducibility in ML](https://neptune.ai/blog/how-to-solve-reproducibility-in-ml)
- [Princeton Reproducibility Research](https://reproducible.cs.princeton.edu/)
