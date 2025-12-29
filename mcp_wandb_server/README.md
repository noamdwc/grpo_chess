# WandB MCP Server

A Model Context Protocol (MCP) server that provides coding agents with access to Weights & Biases metrics, plots, and run data from the Chess-GRPO-Bot project.

## Overview

This MCP server enables coding agents to query wandb data including:
- Training metrics (loss, rewards, KL divergence, etc.)
- Evaluation results (Stockfish scores, Elo differences, win rates)
- Run summaries and hyperparameters
- Plot and image data with base64-encoded images for direct viewing
- Run comparisons

## Installation

1. Install dependencies:
```bash
pip install -r requirents.txt
```

2. Set up WandB authentication:
```bash
# Option 1: Set environment variable
export WANDB_API_KEY=your_api_key_here

# Option 2: Use wandb login (recommended)
wandb login
```

## Configuration

The server is configured to use the "Chess-GRPO-Bot" project by default. You can customize this by modifying `mcp_wandb_server/config.py` or setting environment variables.

### Environment Variables

- `WANDB_API_KEY`: Your WandB API key (optional if using `wandb login`)

### Project Settings

Default project: `Chess-GRPO-Bot`

To change the project, modify the `WandBConfig` class in `config.py` or create a custom configuration.

## Running the Server

### Standalone Mode

Run the server as a standalone process:

```bash
python -m mcp_wandb_server.server
```

The server communicates via stdio (standard input/output) following the MCP protocol.

### MCP Client Configuration

#### Quick Setup (Recommended)

Use the provided setup script to automatically configure Cursor:

```bash
python setup_cursor_mcp.py
```

This script will:
- Detect your project root automatically
- Create the Cursor MCP configuration file
- Handle API key setup (from environment or prompt)
- Merge with existing MCP configurations

#### Manual Setup for Cursor

If you prefer to set it up manually, add to your MCP settings file:

**Windows:**
```
%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json
```

**Mac:**
```
~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
```

**Linux:**
```
~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
```

**Configuration:**
```json
{
  "mcpServers": {
    "wandb": {
      "command": "python",
      "args": ["-m", "mcp_wandb_server.server"],
      "cwd": "/path/to/grpo_chess",
      "env": {
        "WANDB_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

**Note:** Replace `/path/to/grpo_chess` with your actual project path. On Windows, use double backslashes: `C:\\Users\\user\\source\\repos\\grpo_chess`

#### Version-Controlled Configuration

A template configuration file (`.cursor-mcp-config.json.example`) is included in the repository. This allows you to:
- Version control the configuration structure
- Share setup instructions with your team
- Keep API keys out of the repository (use environment variables)

To use the template:
1. Copy `.cursor-mcp-config.json.example` to your Cursor config location
2. Replace `${PROJECT_ROOT}` with your actual project path
3. Replace `${WANDB_API_KEY}` with your API key or use environment variables

#### For Other MCP Clients

The server uses stdio transport. Configure your client to:
- Command: `python`
- Args: `["-m", "mcp_wandb_server.server"]`
- Transport: stdio

## Available Tools

### `list_runs`
List recent runs in the Chess-GRPO-Bot project.

**Parameters:**
- `limit` (optional): Number of runs to return (default: 10)
- `state` (optional): Filter by run state (running, finished, crashed, etc.)

**Example:**
```json
{
  "limit": 5,
  "state": "finished"
}
```

### `get_run_metrics`
Retrieve metrics for a specific run.

**Parameters:**
- `run_id` (required): WandB run ID or name
- `metric_keys` (optional): Specific metrics to retrieve

**Example:**
```json
{
  "run_id": "chess-grpo-20240101-1200-abcd",
  "metric_keys": ["train_total_loss", "eval_stockfish/score"]
}
```

### `get_run_summary`
Get summary statistics for a run including best/worst values, final values, and hyperparameters.

**Parameters:**
- `run_id` (required): WandB run ID or name

**Example:**
```json
{
  "run_id": "chess-grpo-20240101-1200-abcd"
}
```

### `get_plots`
Retrieve plot/image data from wandb runs with base64-encoded image data for direct viewing by agents.

**Parameters:**
- `run_id` (required): WandB run ID or name
- `plot_type` (optional): Filter by plot type (image, other)
- `include_image_data` (optional): If true, download and include base64-encoded image data (default: true)

**Returns:**
- Plot metadata (name, size, URL)
- Base64-encoded image data in `image_data` field
- Data URI in `data_uri` field (ready for embedding: `data:image/png;base64,...`)
- MIME type in `mime_type` field

**Example:**
```json
{
  "run_id": "chess-grpo-20240101-1200-abcd",
  "plot_type": "image",
  "include_image_data": true
}
```

**Note:** When `include_image_data` is true, agents can directly view and analyze the plot images. The images are provided as base64-encoded strings and data URIs that can be embedded or displayed.

### `compare_runs`
Compare metrics across multiple runs.

**Parameters:**
- `run_ids` (required): List of run IDs or names to compare
- `metric_keys` (required): Metrics to compare

**Example:**
```json
{
  "run_ids": ["run1", "run2", "run3"],
  "metric_keys": ["train_total_loss", "eval_stockfish/score", "eval_stockfish/elo_diff"]
}
```

## Available Resources

Resources provide read-only access to wandb data via URIs:

### `wandb://runs/recent`
Summary of the 5 most recent runs in JSON format.

### `wandb://runs/{run_id}/summary`
Full summary of a specific run including hyperparameters and metrics.

### `wandb://runs/{run_id}/metrics/{metric_key}`
Time-series data for a specific metric in a run.

**Example URIs:**
- `wandb://runs/recent`
- `wandb://runs/chess-grpo-20240101-1200-abcd/summary`
- `wandb://runs/chess-grpo-20240101-1200-abcd/metrics/train_total_loss`

## Key Metrics

The server prioritizes these metrics from the Chess-GRPO-Bot project:

### Training Metrics
- `train_total_loss` - Total training loss
- `train/avg_reward` - Average reward per trajectory group
- `train/reward_std` - Reward standard deviation
- `ppo_loss` - PPO loss component
- `mean_kl_divergence` - Mean KL divergence between old and new policies
- `mean_ratio` - Mean importance sampling ratio
- `mean_clip_fraction` - Fraction of clipped updates

### Evaluation Metrics
- `eval_stockfish/score` - Win rate against Stockfish (0-1)
- `eval_stockfish/elo_diff` - Approximate Elo difference vs Stockfish
- `eval_stockfish/wins` - Number of wins
- `eval_stockfish/draws` - Number of draws
- `eval_stockfish/losses` - Number of losses
- `eval_stockfish/games` - Total games played

### Trajectory Metrics
- `avg_trajectory_length` - Average trajectory length
- `pad_fraction` - Fraction of padded steps

## Usage Examples

### Example 1: Get Recent Runs
```python
# Agent calls: list_runs(limit=5)
# Returns: List of 5 most recent runs with metadata
```

### Example 2: Get Training Progress
```python
# Agent calls: get_run_metrics(
#   run_id="chess-grpo-20240101-1200-abcd",
#   metric_keys=["train_total_loss", "train/avg_reward"]
# )
# Returns: Time-series data for specified metrics
```

### Example 3: Compare Multiple Runs
```python
# Agent calls: compare_runs(
#   run_ids=["run1", "run2", "run3"],
#   metric_keys=["eval_stockfish/score", "eval_stockfish/elo_diff"]
# )
# Returns: Comparison table with metrics side-by-side
```

## Testing

Run the test script to verify the server works correctly:

```bash
python mcp_wandb_server/test_server.py
```

## Troubleshooting

### Authentication Issues
- Ensure `WANDB_API_KEY` is set or you've run `wandb login`
- Verify your API key has access to the "Chess-GRPO-Bot" project

### Run Not Found
- Check that the run ID or name is correct
- Ensure the run exists in the "Chess-GRPO-Bot" project
- Verify your entity/team has access to the project

### Connection Issues
- Ensure the server is running and accessible via stdio
- Check MCP client configuration
- Verify Python path is correct in client config

## Project Structure

```
mcp_wandb_server/
├── __init__.py       # Package initialization
├── server.py         # Main MCP server implementation
├── config.py         # Configuration and wandb setup
├── tools.py          # MCP tool implementations
├── resources.py      # MCP resource implementations
├── utils.py          # Helper functions for data formatting
├── test_server.py    # Test script
└── README.md         # This file
```

## License

Part of the grpo_chess project.

