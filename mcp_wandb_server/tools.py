"""MCP tool implementations for WandB API."""
from typing import Any, Dict, List, Optional
from wandb.apis.public import Api, Run

from .config import get_config
from .utils import (
    format_metric_history,
    format_run_summary,
    format_runs_list,
    format_plot_info,
    format_comparison_table,
    format_json_response
)


def get_wandb_api() -> Api:
    """Get initialized WandB API client."""
    config = get_config()
    return Api(**config.get_api_kwargs())


def find_run(api: Api, run_id: str) -> Optional[Run]:
    """
    Find a run by ID or name.
    
    Args:
        api: WandB API client
        run_id: Run ID or name
        
    Returns:
        Run object or None if not found
    """
    config = get_config()
    try:
        # Try as run ID first
        run = api.run(f"{config.entity or api.default_entity}/{config.project}/{run_id}")
        return run
    except Exception:
        # Try to find by name
        try:
            runs = api.runs(f"{config.entity or api.default_entity}/{config.project}", {"display_name": run_id})
            if runs:
                return runs[0]
        except Exception:
            pass
    return None


async def list_runs(
    limit: int = 10,
    state: Optional[str] = None
) -> str:
    """
    List recent runs in the Chess-GRPO-Bot project.
    
    Args:
        limit: Number of runs to return (default: 10)
        state: Filter by run state (running, finished, crashed, etc.)
        
    Returns:
        JSON string with list of runs
    """
    try:
        api = get_wandb_api()
        config = get_config()
        
        filters = {}
        if state:
            filters["state"] = state
        
        runs = api.runs(
            f"{config.entity or api.default_entity}/{config.project}",
            filters=filters,
            per_page=limit
        )
        
        formatted_runs = format_runs_list(list(runs))
        return format_json_response(formatted_runs)
    except Exception as e:
        return format_json_response({"error": str(e)})


async def get_run_metrics(
    run_id: str,
    metric_keys: Optional[List[str]] = None
) -> str:
    """
    Retrieve metrics for a specific run.
    
    Args:
        run_id: WandB run ID or name
        metric_keys: Specific metrics to retrieve (if None, returns all metrics)
        
    Returns:
        JSON string with metric time-series data
    """
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)
        
        if not run:
            return format_json_response({"error": f"Run '{run_id}' not found"})
        
        # Get full history
        history = run.history()
        
        # Format metrics
        metrics = format_metric_history(history)
        
        # Filter by metric_keys if provided
        if metric_keys:
            metrics = {k: v for k, v in metrics.items() if k in metric_keys}
        
        result = {
            "run_id": run.id,
            "run_name": run.name,
            "metrics": metrics
        }
        
        return format_json_response(result)
    except Exception as e:
        return format_json_response({"error": str(e)})


async def get_run_summary(run_id: str) -> str:
    """
    Get summary statistics for a run (best/worst values, final values).
    
    Args:
        run_id: WandB run ID or name
        
    Returns:
        JSON string with run summary
    """
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)
        
        if not run:
            return format_json_response({"error": f"Run '{run_id}' not found"})
        
        summary = format_run_summary(run)
        
        # Add best/worst/final values for key metrics
        history = run.history()
        
        # Handle pandas DataFrame
        if hasattr(history, 'empty'):
            if not history.empty:
                history = history.to_dict('records')
            else:
                history = []
        
        if history:
            metric_keys = [k for k in history[0].keys() if not k.startswith("_")]
            summary["metric_statistics"] = {}
            
            for key in metric_keys:
                values = [row[key] for row in history if key in row and row[key] is not None and isinstance(row[key], (int, float))]
                if values:
                    summary["metric_statistics"][key] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "final": values[-1] if values else None
                    }
        
        return format_json_response(summary)
    except Exception as e:
        return format_json_response({"error": str(e)})


async def get_plots(
    run_id: str,
    plot_type: Optional[str] = None,
    include_image_data: bool = True
) -> str:
    """
    Retrieve plot/image data from wandb runs with base64-encoded image data.
    
    Args:
        run_id: WandB run ID or name
        plot_type: Filter by plot type (e.g., "image", "other")
        include_image_data: If True, download and include base64-encoded image data
        
    Returns:
        JSON string with list of plot metadata and base64-encoded image data
    """
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)
        
        if not run:
            return format_json_response({"error": f"Run '{run_id}' not found"})
        
        # Get files from the run
        files = run.files()
        
        # Filter image files
        image_files = [f for f in files if f.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))]
        
        # Format plots with image data
        plots = [format_plot_info(f, include_image_data=include_image_data) for f in image_files]
        
        # Filter by plot_type if provided
        if plot_type:
            plots = [p for p in plots if p["type"] == plot_type]
        
        result = {
            "run_id": run.id,
            "run_name": run.name,
            "plots": plots,
            "note": "Images are base64-encoded in the 'image_data' field and as data URIs in 'data_uri' field" if include_image_data else "Image data not included (metadata only)"
        }
        
        return format_json_response(result)
    except Exception as e:
        return format_json_response({"error": str(e)})


async def compare_runs(
    run_ids: List[str],
    metric_keys: List[str]
) -> str:
    """
    Compare metrics across multiple runs.
    
    Args:
        run_ids: List of run IDs or names to compare
        metric_keys: Metrics to compare
        
    Returns:
        JSON string with comparison table
    """
    try:
        api = get_wandb_api()
        runs = []
        
        for run_id in run_ids:
            run = find_run(api, run_id)
            if run:
                runs.append(run)
            else:
                return format_json_response({"error": f"Run '{run_id}' not found"})
        
        comparison = format_comparison_table(runs, metric_keys)
        return format_json_response(comparison)
    except Exception as e:
        return format_json_response({"error": str(e)})

