"""Main MCP server implementation for WandB integration."""
import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .config import get_config
from .tools import (
    list_runs,
    get_run_metrics,
    get_run_summary,
    get_plots,
    compare_runs
)
from .resources import list_resources, read_resource


# Create MCP server instance
server = Server("wandb-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="list_runs",
            description="List recent runs in the Chess-GRPO-Bot project",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of runs to return (default: 10)",
                        "default": 10
                    },
                    "state": {
                        "type": "string",
                        "description": "Filter by run state (running, finished, crashed, etc.)",
                        "enum": ["running", "finished", "crashed", "killed", "failed"]
                    }
                }
            }
        ),
        Tool(
            name="get_run_metrics",
            description="Retrieve metrics for a specific run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "WandB run ID or name"
                    },
                    "metric_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to retrieve (e.g., ['train_total_loss', 'eval_stockfish/score']). If not provided, returns all metrics."
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="get_run_summary",
            description="Get summary statistics for a run (best/worst values, final values, hyperparameters)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "WandB run ID or name"
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="get_plots",
            description="Retrieve plot/image data from wandb runs with base64-encoded image data for direct viewing",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "WandB run ID or name"
                    },
                    "plot_type": {
                        "type": "string",
                        "description": "Filter by plot type (e.g., 'image', 'other')",
                        "enum": ["image", "other"]
                    },
                    "include_image_data": {
                        "type": "boolean",
                        "description": "If true, download and include base64-encoded image data (default: true)",
                        "default": True
                    }
                },
                "required": ["run_id"]
            }
        ),
        Tool(
            name="compare_runs",
            description="Compare metrics across multiple runs",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of run IDs or names to compare"
                    },
                    "metric_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to compare (e.g., ['train_total_loss', 'eval_stockfish/score'])"
                    }
                },
                "required": ["run_ids", "metric_keys"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "list_runs":
            result = await list_runs(
                limit=arguments.get("limit", 10),
                state=arguments.get("state")
            )
        elif name == "get_run_metrics":
            result = await get_run_metrics(
                run_id=arguments["run_id"],
                metric_keys=arguments.get("metric_keys")
            )
        elif name == "get_run_summary":
            result = await get_run_summary(
                run_id=arguments["run_id"]
            )
        elif name == "get_plots":
            result = await get_plots(
                run_id=arguments["run_id"],
                plot_type=arguments.get("plot_type"),
                include_image_data=arguments.get("include_image_data", True)
            )
        elif name == "compare_runs":
            result = await compare_runs(
                run_ids=arguments["run_ids"],
                metric_keys=arguments["metric_keys"]
            )
        else:
            result = f'{{"error": "Unknown tool: {name}"}}'
        
        return [TextContent(type="text", text=result)]
    except Exception as e:
        error_msg = f'{{"error": "{str(e)}"}}'
        return [TextContent(type="text", text=error_msg)]


@server.list_resources()
async def handle_list_resources() -> list:
    """List available MCP resources."""
    return list_resources()


@server.read_resource()
async def handle_read_resource(uri: str) -> list[TextContent]:
    """Handle resource read requests."""
    return await read_resource(uri)


async def main():
    """Main entry point for the MCP server."""
    # Initialize wandb config (can be customized via environment variables)
    config = get_config()
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

