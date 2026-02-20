"""MCP server for Lightning.ai Studio job management."""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools import submit_job, list_jobs, cancel_job, get_logs, get_credits


server = Server("lightning-mcp-server")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="submit_job",
            description="Run a shell command as a batch job on Lightning.ai Studio",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run (e.g. 'python -m src.train_self_play --config grpo_9m_posttrain.yaml')"},
                    "name": {"type": "string", "description": "Unique name for the job"},
                    "machine": {"type": "string", "description": "Machine type (default: CPU-4)", "default": "CPU-4"},
                },
                "required": ["command", "name"],
            },
        ),
        Tool(
            name="list_jobs",
            description="List recent jobs and their status on Lightning.ai Studio",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of jobs to return (default: 10)", "default": 10},
                    "state": {"type": "string", "description": "Filter by job state (e.g. running, succeeded, failed)"},
                },
            },
        ),
        Tool(
            name="cancel_job",
            description="Stop a running job on Lightning.ai Studio",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Name of the job to cancel"},
                },
                "required": ["job_name"],
            },
        ),
        Tool(
            name="get_logs",
            description="Retrieve stdout/stderr logs from a Lightning.ai job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Name of the job"},
                },
                "required": ["job_name"],
            },
        ),
        Tool(
            name="get_credits",
            description="Check remaining Lightning.ai credits and total spent",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "submit_job":
            result = await submit_job(
                command=arguments["command"],
                name=arguments["name"],
                machine=arguments.get("machine", "CPU-4"),
            )
        elif name == "list_jobs":
            result = await list_jobs(
                limit=arguments.get("limit", 10),
                state=arguments.get("state"),
            )
        elif name == "cancel_job":
            result = await cancel_job(job_name=arguments["job_name"])
        elif name == "get_logs":
            result = await get_logs(job_name=arguments["job_name"])
        elif name == "get_credits":
            result = await get_credits()
        else:
            result = f'{{"error": "Unknown tool: {name}"}}'
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f'{{"error": "{str(e)}"}}')]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
