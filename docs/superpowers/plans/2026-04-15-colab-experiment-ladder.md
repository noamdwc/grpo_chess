# Colab Experiment Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automated Colab notebook that runs the approved GRPO threshold ladder and logs its promotion decisions.

**Architecture:** Keep training entrypoints unchanged. Add a small pure-Python ladder helper module for ranking and stage selection, then create a new notebook copied from the existing Colab runner that patches runtime configs, launches experiments sequentially, reads WandB/local summaries, and selects Stage 2 and Stage 3 automatically.

**Tech Stack:** Python, pytest, JSON notebook format, Google Colab, WandB, existing `src.train_self_play` runner

---
