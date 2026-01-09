#!/usr/bin/env python3
"""List recent WandB runs from Chess-GRPO-Bot project."""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mcp_wandb_server.tools import list_runs


async def main():
    """List recent runs."""
    print("\n📊 Recent WandB Runs from Chess-GRPO-Bot Project\n")
    
    try:
        result = await list_runs(limit=10)
        data = json.loads(result)
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
            print("\nMake sure you're authenticated:")
            print("  wandb login")
            return
        
        if not data:
            print("No runs found.")
            return
        
        print(f"Found {len(data)} recent runs:\n")
        print(f"{'Run ID':<25} {'Name':<35} {'State':<12} {'Created':<20}")
        print("-" * 95)
        
        for run in data:
            run_id = run.get("id", "N/A")[:24]
            name = run.get("name", "N/A")[:34]
            state = run.get("state", "N/A")
            created = run.get("created_at", "N/A")[:19] if run.get("created_at") else "N/A"
            
            print(f"{run_id:<25} {name:<35} {state:<12} {created:<20}")
            
            # Show key metrics if available
            summary = run.get("summary_metrics", {})
            if summary:
                key_metrics = ["train_total_loss", "eval_stockfish/score", "eval_stockfish/elo_diff"]
                metrics = [
                    f"{k}={v:.4f}" 
                    for k, v in summary.items() 
                    if k in key_metrics and isinstance(v, (int, float))
                ]
                if metrics:
                    print(f"  📈 Metrics: {', '.join(metrics)}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())




