#!/usr/bin/env python3
"""List recent W&B runs for the configured GRPO project."""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_wandb_server.tools import list_runs


async def main() -> None:
    """Print recent runs and selected summary metrics."""
    print("Recent W&B runs from the Chess-GRPO-Bot project")
    try:
        result = await list_runs(limit=10)
        data = json.loads(result)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    if "error" in data:
        print(f"Error: {data['error']}")
        print("Authenticate with W&B before using this helper.")
        return

    if not data:
        print("No runs found.")
        return

    print(f"Found {len(data)} recent runs")
    for run in data:
        run_id = run.get("id", "N/A")
        name = run.get("name", "N/A")
        state = run.get("state", "N/A")
        created = run.get("created_at", "N/A")
        print(f"- {run_id}: {name} [{state}] {created}")
        summary = run.get("summary_metrics", {})
        for key in ("train_total_loss", "eval_stockfish/score", "eval_stockfish/elo_diff"):
            value = summary.get(key)
            if isinstance(value, (int, float)):
                print(f"  {key}={value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
