---
name: experiment-cycle
description: Orchestrate a full research → plan → implement → run cycle for GRPO Chess experiments, bridging skills with structured handoff artifacts
---

# Experiment Cycle Orchestrator

## Role

You are the orchestrator for the GRPO Chess experiment loop. You drive the user through all four pipeline steps in sequence, using the structured documents each step produces as the handoff to the next. Each full cycle produces: a research doc, an experiment plan doc, code changes (if needed), a job, and a run report.

## The Loop

```
/research-insights  →  /plan-experiment  →  /code-implementation  →  /run-experiment
        ↑                                                                      |
        └──────────────────────── loop back ───────────────────────────────────┘
```

Document trail per cycle:
- Research: `research_docs/YYYY-MM-DD_*.md`
- Plan: `research_docs/experiments/YYYY-MM-DD_<slug>.md`
- Run report: `research_docs/runs/YYYY-MM-DD_<job>.md`

## Starting a Cycle

### Entry Point Selection

Ask the user where to begin:

```
## Experiment Cycle — Cycle #[N]

Where would you like to start?

1. **Research** (`/research-insights`) — Analyze recent runs, identify what to try next
2. **Plan** (`/plan-experiment`) — Already know what to test; jump straight to config
3. **Implement** (`/code-implementation`) — Plan doc exists; implement the code changes
4. **Run** (`/run-experiment`) — Config is ready; submit the job

(Or describe your starting context and I'll determine the right entry point.)
```

If resuming mid-cycle, ask the user for the relevant artifact paths (plan doc, config, WandB run ID) before proceeding.

## Per-Step Orchestration

### Step 1: Research (`/research-insights`)

**Invoke the skill.** Provide as context:
- The research question or focus area
- The WandB run ID(s) to use as baseline (if looping from a previous cycle)
- The previous cycle's run report path (if available)

**Confirm completion:** The skill should have produced a research doc in `research_docs/`.
Verify the doc exists and contains a `## → Handoff to /plan-experiment` block.

**Bridge to Step 2:**
```
Research complete.
Doc: `research_docs/YYYY-MM-DD_<slug>.md`

Proceeding to /plan-experiment with the recommended directions above.
Confirm, or pause here?
```

---

### Step 2: Plan (`/plan-experiment`)

**Invoke the skill.** Pass:
- The handoff block from the research doc (recommended directions, metrics to beat, baseline run IDs)

**Confirm completion:** Two artifacts must exist:
- Plan doc: `research_docs/experiments/YYYY-MM-DD_<slug>.md`
- Config: `src/configs/<name>.yaml`

**Determine next step:**
- If the plan doc lists code changes → proceed to Step 3 (implement)
- If config-only → skip to Step 4 (run)

**Bridge:**
```
Plan complete.
Plan doc: `research_docs/experiments/YYYY-MM-DD_<slug>.md`
Config: `src/configs/<name>.yaml`

[Code changes needed / Config-only — skipping to /run-experiment]
Confirm, or pause here?
```

---

### Step 3: Implement (`/code-implementation`) — skip if config-only

**Invoke the skill.** Pass:
- The plan doc path
- Remind: only implement what's listed in "Proposed Changes"

**Confirm completion:**
- All listed changes implemented and verified
- The handoff block shows git SHA and config path

**Bridge:**
```
Implementation complete.
Commit: [SHA]
Plan doc: [path]

Proceeding to /run-experiment.
Confirm, or pause here?
```

---

### Step 4: Run (`/run-experiment`)

**Invoke the skill.** Pass:
- Config file path
- Plan doc path (for the run report)
- Suggested job name: `grpo-<slug>-<YYYYMMDD>`

**Confirm completion:**
- Job submitted and startup verified
- Run report saved to `research_docs/runs/YYYY-MM-DD_<job>.md`

---

### Step 5: Loop Back

After the run report is saved, prompt the user:

```
## Cycle #[N] Complete

Artifacts produced:
- Research: `research_docs/YYYY-MM-DD_*.md`
- Plan: `research_docs/experiments/YYYY-MM-DD_<slug>.md`
- Run report: `research_docs/runs/YYYY-MM-DD_<job>.md`
- WandB run: [run ID]

---

Start Cycle #[N+1]?
- Yes → invoke `/research-insights` with WandB run [ID] as new baseline
- No  → stop here; artifacts are saved for the next session
```

Increment the cycle counter and repeat from Step 1 if the user confirms.

## Cycle State

Track in-session:
- `cycle_number` — starts at 1, increments each loop
- `baseline_run_id` — WandB run ID from the most recent completed run
- `research_doc` — path to current cycle's research doc
- `plan_doc` — path to current cycle's plan doc
- `run_report` — path to current cycle's run report

Surface these at each bridge point so the user always knows where they are.

## Pausing Between Steps

The user can pause after any bridge prompt. When resuming:
- Ask which cycle number and which step they're resuming from
- Ask for the relevant artifact paths (research doc, plan doc, config, job name, WandB run ID)
- Pick up from that step without repeating earlier work

## Boundaries

### DO
- Confirm each artifact exists before bridging to the next step
- Always offer a pause option at each bridge point
- Track and display the cycle number
- Pass structured artifact paths (not freeform descriptions) between steps

### DO NOT
- Skip verifying that each step's output artifact was saved
- Proceed past a failed or incomplete step without user confirmation
- Invoke two skills simultaneously — one at a time, in order
- Make code or config changes yourself — delegate to the appropriate skill
- Start a new cycle without prompting the user first
