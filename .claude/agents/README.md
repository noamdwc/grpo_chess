# Agent Prompts

This directory contains specialized agent prompts for the GRPO Chess project.

## Available Agents

| Agent | File | Purpose |
|-------|------|---------|
| Research Insights | `research-insights.md` | Analyze codebase, WandB runs, and write research documents |
| Code Implementation | `code-implementation.md` | Implement recommendations from research documents |

## Usage

### With Claude Code

Copy the prompt content and provide it as context when starting a conversation:

```bash
# Start Claude Code with agent prompt
claude "$(cat .claude/agents/research-insights.md)"
```

Or reference the file in your conversation:
```
Please follow the agent prompt in .claude/agents/research-insights.md
```

### Workflow

1. **Research Phase**: Use `research-insights.md` agent to investigate issues
2. **Discussion**: Agent presents findings, user provides feedback
3. **Documentation**: Agent writes final document to `research_docs/`
4. **Implementation Phase**: Use `code-implementation.md` agent to implement fixes
5. **Review**: User reviews changes, provides feedback
6. **Iteration**: Repeat as needed

## Agent Design Principles

Based on [GitHub's AGENTS.md best practices](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/):

1. **Clear boundaries**: Define what the agent should and shouldn't do
2. **Specific tools**: List which tools the agent should use
3. **Output format**: Specify expected outputs
4. **Workflow steps**: Break down the process into clear phases
5. **Discussion checkpoints**: Include human-in-the-loop validation

## Creating New Agents

Use this template structure:
```markdown
# Agent Name

## Role
[One-line description]

## Context
[What the agent needs to know about the project]

## Tools Available
[List of tools the agent can use]

## Workflow
[Step-by-step process]

## Output Requirements
[What the agent should produce]

## Boundaries
[What the agent should NOT do]
```
