"""MCP resource implementations for WandB data."""
from typing import Any, Dict, Optional
import re
from mcp.types import Resource, TextContent

from .config import get_config
from .tools import get_wandb_api, find_run
from .utils import format_runs_list, format_run_summary, format_metric_history, format_json_response


def list_resources() -> list[Resource]:
    """
    List available MCP resources.

    Returns:
        List of Resource objects
    """
    config = get_config()
    resources = [
        Resource(
            uri="wandb://runs/recent",
            name="Recent Runs Summary",
            description=f"Summary of the 5 most recent runs in the {config.project} project",
            mimeType="application/json"
        )
    ]
    
    # Note: Dynamic resources (run-specific) are handled in read_resource
    return resources


async def read_resource(uri: str) -> list[TextContent]:
    """
    Read a resource by URI.
    
    Supported URIs:
    - wandb://runs/recent - Recent runs summary
    - wandb://runs/{run_id}/summary - Run summary
    - wandb://runs/{run_id}/metrics/{metric_key} - Metric history
    
    Args:
        uri: Resource URI
        
    Returns:
        List of content items
    """
    try:
        # Parse URI
        if uri == "wandb://runs/recent":
            return await _get_recent_runs_resource()
        
        # Match wandb://runs/{run_id}/summary
        match = re.match(r"wandb://runs/([^/]+)/summary", uri)
        if match:
            run_id = match.group(1)
            return await _get_run_summary_resource(run_id)
        
        # Match wandb://runs/{run_id}/metrics/{metric_key}
        match = re.match(r"wandb://runs/([^/]+)/metrics/(.+)", uri)
        if match:
            run_id = match.group(1)
            metric_key = match.group(2)
            return await _get_metric_history_resource(run_id, metric_key)
        
        # Unknown URI
        return [TextContent(
            type="text",
            text=f"Unknown resource URI: {uri}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error reading resource {uri}: {str(e)}"
        )]


async def _get_recent_runs_resource() -> list[TextContent]:
    """Get recent runs summary resource."""
    try:
        api = get_wandb_api()
        config = get_config()
        
        runs = api.runs(
            f"{config.entity or api.default_entity}/{config.project}",
            per_page=5
        )
        
        formatted_runs = format_runs_list(list(runs))
        content = format_json_response({
            "project": config.project,
            "recent_runs": formatted_runs
        })
        
        return [TextContent(
            type="text",
            text=content
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error fetching recent runs: {str(e)}"
        )]


async def _get_run_summary_resource(run_id: str) -> list[TextContent]:
    """Get run summary resource."""
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)
        
        if not run:
            return [TextContent(
                type="text",
                text=f"Run '{run_id}' not found"
            )]
        
        summary = format_run_summary(run)
        
        # Add summary metrics
        if run.summary:
            summary["summary_metrics"] = dict(run.summary)
        
        content = format_json_response(summary)
        
        return [TextContent(
            type="text",
            text=content
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error fetching run summary: {str(e)}"
        )]


async def _get_metric_history_resource(run_id: str, metric_key: str) -> list[TextContent]:
    """Get metric history resource."""
    try:
        api = get_wandb_api()
        run = find_run(api, run_id)
        
        if not run:
            return [TextContent(
                type="text",
                text=f"Run '{run_id}' not found"
            )]
        
        history = run.history()
        metrics = format_metric_history(history)
        
        if metric_key not in metrics:
            return [TextContent(
                type="text",
                text=f"Metric '{metric_key}' not found in run '{run_id}'"
            )]
        
        metric_data = {
            "run_id": run.id,
            "run_name": run.name,
            "metric_key": metric_key,
            "data": metrics[metric_key]
        }
        
        content = format_json_response(metric_data)
        
        return [TextContent(
            type="text",
            text=content
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error fetching metric history: {str(e)}"
        )]

