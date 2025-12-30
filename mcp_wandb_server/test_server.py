"""Test script for WandB MCP Server."""
import asyncio
import json
import sys
from typing import Any

from .config import get_config, set_config, WandBConfig
from .tools import (
    list_runs,
    get_run_metrics,
    get_run_summary,
    get_plots,
    compare_runs
)
from .resources import list_resources, read_resource


async def test_list_runs():
    """Test list_runs tool."""
    print("Testing list_runs...")
    try:
        result = await list_runs(limit=5)
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        print(f"  SUCCESS: Found {len(data)} runs")
        if data:
            print(f"  First run: {data[0].get('name', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_get_run_summary():
    """Test get_run_summary tool."""
    print("\nTesting get_run_summary...")
    try:
        # First, get a run ID
        runs_result = await list_runs(limit=1)
        runs_data = json.loads(runs_result)
        
        if "error" in runs_data or not runs_data:
            print("  SKIP: No runs available to test")
            return True
        
        run_id = runs_data[0].get("id") or runs_data[0].get("name")
        if not run_id:
            print("  SKIP: Could not get run ID")
            return True
        
        result = await get_run_summary(run_id)
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        print(f"  SUCCESS: Got summary for run {data.get('name', run_id)}")
        print(f"  State: {data.get('state', 'N/A')}")
        print(f"  Metrics: {len(data.get('summary_metrics', {}))} summary metrics")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_get_run_metrics():
    """Test get_run_metrics tool."""
    print("\nTesting get_run_metrics...")
    try:
        # First, get a run ID
        runs_result = await list_runs(limit=1)
        runs_data = json.loads(runs_result)
        
        if "error" in runs_data or not runs_data:
            print("  SKIP: No runs available to test")
            return True
        
        run_id = runs_data[0].get("id") or runs_data[0].get("name")
        if not run_id:
            print("  SKIP: Could not get run ID")
            return True
        
        # Test with specific metrics
        result = await get_run_metrics(
            run_id=run_id,
            metric_keys=["train_total_loss", "eval_stockfish/score"]
        )
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        metrics = data.get("metrics", {})
        print(f"  SUCCESS: Retrieved {len(metrics)} metrics")
        for key in metrics:
            print(f"    {key}: {len(metrics[key])} data points")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_get_plots():
    """Test get_plots tool."""
    print("\nTesting get_plots...")
    try:
        # First, get a run ID
        runs_result = await list_runs(limit=1)
        runs_data = json.loads(runs_result)
        
        if "error" in runs_data or not runs_data:
            print("  SKIP: No runs available to test")
            return True
        
        run_id = runs_data[0].get("id") or runs_data[0].get("name")
        if not run_id:
            print("  SKIP: Could not get run ID")
            return True
        
        result = await get_plots(run_id=run_id)
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        plots = data.get("plots", [])
        print(f"  SUCCESS: Found {len(plots)} plots/images")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_compare_runs():
    """Test compare_runs tool."""
    print("\nTesting compare_runs...")
    try:
        # Get multiple runs
        runs_result = await list_runs(limit=2)
        runs_data = json.loads(runs_result)
        
        if "error" in runs_data or len(runs_data) < 2:
            print("  SKIP: Need at least 2 runs to compare")
            return True
        
        run_ids = [
            run.get("id") or run.get("name")
            for run in runs_data[:2]
            if run.get("id") or run.get("name")
        ]
        
        if len(run_ids) < 2:
            print("  SKIP: Could not get 2 run IDs")
            return True
        
        result = await compare_runs(
            run_ids=run_ids,
            metric_keys=["train_total_loss", "eval_stockfish/score"]
        )
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        print(f"  SUCCESS: Compared {len(data.get('runs', []))} runs")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_resources():
    """Test MCP resources."""
    print("\nTesting resources...")
    try:
        # Test list_resources
        resources = list_resources()
        print(f"  Available resources: {len(resources)}")
        for resource in resources:
            print(f"    - {resource.uri}: {resource.name}")
        
        # Test read_resource for recent runs
        content = await read_resource("wandb://runs/recent")
        if content and len(content) > 0:
            print(f"  SUCCESS: Read recent runs resource")
            return True
        else:
            print(f"  ERROR: Could not read recent runs resource")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("WandB MCP Server Test Suite")
    print("=" * 60)
    
    # Check configuration
    config = get_config()
    print(f"\nConfiguration:")
    print(f"  Project: {config.project}")
    print(f"  Entity: {config.entity or 'default'}")
    print(f"  API Key: {'Set' if config.api_key else 'Not set (using wandb config)'}")
    print()
    
    # Run tests
    tests = [
        ("List Runs", test_list_runs),
        ("Get Run Summary", test_get_run_summary),
        ("Get Run Metrics", test_get_run_metrics),
        ("Get Plots", test_get_plots),
        ("Compare Runs", test_compare_runs),
        ("Resources", test_resources),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  FATAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

