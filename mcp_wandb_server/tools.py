"""MCP tool implementations for WandB API."""
from typing import Any, Dict, List, Optional, Union
import base64
from wandb.apis.public import Api, Run
from mcp.types import TextContent, ImageContent

from .config import get_config
from .utils import (
    format_metric_history,
    format_run_summary,
    format_runs_list,
    format_plot_info,
    format_comparison_table,
    format_json_response,
    downscale_image_bytes,
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
    List recent runs in the configured WandB project.

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
    include_image_data: bool = True,
    max_dimension: int = 800,
) -> List[Union[TextContent, ImageContent]]:
    """
    Retrieve plot/image data from wandb runs as native ImageContent.

    Args:
        run_id: WandB run ID or name
        plot_type: Filter by plot type (e.g., "image", "other")
        include_image_data: If True, download images
        max_dimension: Max pixel dimension for downscaling (0 to disable)

    Returns:
        List of TextContent (metadata) and ImageContent (images)
    """
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)

        if not run:
            return [TextContent(type="text", text=format_json_response({"error": f"Run '{run_id}' not found"}))]

        files = run.files()
        image_files = [f for f in files if f.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))]

        all_metadata = []
        image_contents: List[ImageContent] = []

        for f in image_files:
            metadata, raw_bytes, mime_type = format_plot_info(f, include_image_data=include_image_data)

            if plot_type and metadata["type"] != plot_type:
                continue

            all_metadata.append(metadata)

            if raw_bytes and mime_type:
                if max_dimension > 0:
                    raw_bytes = downscale_image_bytes(raw_bytes, max_dimension)
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
                image_contents.append(
                    ImageContent(type="image", data=b64, mimeType=mime_type)
                )

        result = {
            "run_id": run.id,
            "run_name": run.name,
            "plots": all_metadata,
        }
        content: List[Union[TextContent, ImageContent]] = [
            TextContent(type="text", text=format_json_response(result))
        ]
        content.extend(image_contents)
        return content
    except Exception as e:
        return [TextContent(type="text", text=format_json_response({"error": str(e)}))]


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

