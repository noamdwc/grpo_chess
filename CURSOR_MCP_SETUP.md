# Quick Setup Guide: Cursor MCP Server

This guide will help you set up the WandB MCP server for use with Cursor in under 2 minutes.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirents.txt
   ```

2. **Authenticate with WandB:**
   ```bash
   wandb login
   ```
   Or set environment variable:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

## Setup Steps

### Option 1: Automated Setup (Recommended)

Run the setup script:

```bash
python setup_cursor_mcp.py
```

The script will:
- ✅ Detect your project root automatically
- ✅ Create the Cursor MCP configuration file
- ✅ Handle API key setup
- ✅ Merge with existing MCP configurations

**Then restart Cursor** and you're done!

### Option 2: Manual Setup

1. **Find your Cursor MCP config file location:**

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

2. **Create/edit the config file** with this content:

   ```json
   {
     "mcpServers": {
       "wandb": {
         "command": "python",
         "args": ["-m", "mcp_wandb_server.server"],
         "cwd": "C:\\Users\\user\\source\\repos\\grpo_chess",
         "env": {
           "WANDB_API_KEY": "your_api_key_here"
         }
       }
     }
   }
   ```

   **Important:** Replace the `cwd` path with your actual project path. On Windows, use double backslashes (`\\`).

3. **Restart Cursor**

## Verify It's Working

After restarting Cursor, test it by asking:

- "What are the recent wandb runs?"
- "Show me the training metrics from the latest run"
- "Get the plots from the most recent run"

If the agent can access wandb data, you're all set! 🎉

## Troubleshooting

### "Module not found" error
- Make sure dependencies are installed: `pip install -r requirents.txt`
- Verify Python path in config points to the correct environment

### "Authentication failed" error
- Run `wandb login` or set `WANDB_API_KEY` environment variable
- Verify your API key is correct

### Server not connecting
- Check that the `cwd` path in config is correct
- Verify the path uses correct path separators for your OS
- Restart Cursor completely (not just reload window)

## Version Control

The configuration template (`.cursor-mcp-config.json.example`) is version-controlled, but the actual config file (with API keys) should NOT be committed. It's already in `.gitignore`.

For team members:
1. Run `python setup_cursor_mcp.py`
2. Or copy `.cursor-mcp-config.json.example` and customize it

## More Information

See `mcp_wandb_server/README.md` for detailed documentation on available tools and resources.

