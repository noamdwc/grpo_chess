#!/usr/bin/env python3
"""Update MCP config with full Python path."""
import json
import sys
from pathlib import Path

def update_mcp_config():
    """Update MCP config to use full Python path."""
    config_path = Path.home() / "Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
    
    # Read existing config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}
    
    # Update wandb server config with full Python path
    python_path = "/Users/noamc/miniconda3/envs/grpo_chess/bin/python"
    project_root = "/Users/noamc/repos/grpo_chess"
    
    config["mcpServers"]["wandb"] = {
        "command": python_path,
        "args": ["-m", "mcp_wandb_server.server"],
        "cwd": project_root
    }
    
    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Updated MCP config at: {config_path}")
    print(f"  Python: {python_path}")
    print(f"  CWD: {project_root}")
    print(f"\n⚠️  Please restart Cursor for changes to take effect!")

if __name__ == "__main__":
    update_mcp_config()



