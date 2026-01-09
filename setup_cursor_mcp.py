#!/usr/bin/env python3
"""
Setup script for Cursor MCP configuration.
This script helps set up the WandB MCP server configuration for Cursor.
"""
import os
import sys
import json
import platform
from pathlib import Path


def get_cursor_mcp_config_path() -> Path:
    """
    Get the path to Cursor's MCP configuration file based on the operating system.
    
    Returns:
        Path to the MCP configuration file
    """
    system = platform.system()
    
    if system == "Windows":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable not found")
        config_path = Path(appdata) / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
    elif system == "Darwin":  # macOS
        home = Path.home()
        config_path = home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
    else:  # Linux
        home = Path.home()
        config_path = home / ".config" / "Cursor" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json"
    
    return config_path


def get_project_root() -> Path:
    """Get the project root directory (where this script is located)."""
    return Path(__file__).parent.absolute()


def get_wandb_api_key() -> str:
    """
    Get WandB API key from environment variable or prompt user.
    
    Returns:
        WandB API key
    """
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY environment variable not set.")
        print("You can either:")
        print("  1. Set it: export WANDB_API_KEY=your_key_here")
        print("  2. Run: wandb login")
        print("  3. Enter it now (will be saved to config):")
        api_key = input("WandB API Key (or press Enter to skip): ").strip()
        if not api_key:
            print("Warning: No API key provided. You'll need to set WANDB_API_KEY or use wandb login.")
    return api_key


def create_mcp_config(project_root: Path, api_key: str = None) -> dict:
    """
    Create MCP configuration dictionary.
    
    Args:
        project_root: Path to project root directory
        api_key: Optional WandB API key (if None, will use environment variable)
        
    Returns:
        MCP configuration dictionary
    """
    # Use the currently running Python (e.g., your conda env) so Cursor calls the same environment
    config = {
        "mcpServers": {
            "wandb": {
                "command": sys.executable,
                "args": ["-m", "mcp_wandb_server.server"],
                "cwd": str(project_root)
            }
        }
    }
    
    # Add API key to env if provided
    if api_key:
        config["mcpServers"]["wandb"]["env"] = {
            "WANDB_API_KEY": api_key
        }
    
    return config


def merge_with_existing_config(new_config: dict, existing_path: Path) -> dict:
    """
    Merge new config with existing MCP configuration.
    
    Args:
        new_config: New configuration to add
        existing_path: Path to existing config file
        
    Returns:
        Merged configuration
    """
    if existing_path.exists():
        try:
            with open(existing_path, "r") as f:
                existing = json.load(f)
            
            # Merge mcpServers
            if "mcpServers" not in existing:
                existing["mcpServers"] = {}
            
            existing["mcpServers"].update(new_config["mcpServers"])
            return existing
        except Exception as e:
            print(f"Warning: Could not read existing config: {e}")
            print("Creating new configuration file.")
    
    return new_config


def main():
    """Main setup function."""
    print("=" * 60)
    print("Cursor MCP Server Setup for WandB")
    print("=" * 60)
    print()
    
    # Get paths
    project_root = get_project_root()
    config_path = get_cursor_mcp_config_path()
    
    print(f"Project root: {project_root}")
    print(f"Config path: {config_path}")
    print()
    
    # Check if project root exists and has mcp_wandb_server
    if not (project_root / "mcp_wandb_server").exists():
        print("Error: mcp_wandb_server directory not found in project root!")
        print(f"Expected: {project_root / 'mcp_wandb_server'}")
        sys.exit(1)
    
    # Get API key
    api_key = get_wandb_api_key()
    
    # Create configuration
    new_config = create_mcp_config(project_root, api_key if api_key else None)
    
    # Merge with existing config if it exists
    final_config = merge_with_existing_config(new_config, config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    try:
        with open(config_path, "w") as f:
            json.dump(final_config, f, indent=2)
        print(f"✓ Configuration written to: {config_path}")
    except Exception as e:
        print(f"Error writing configuration: {e}")
        sys.exit(1)
    
    print()
    print("Setup complete! Please restart Cursor for the changes to take effect.")
    print()
    print("To verify:")
    print("  1. Restart Cursor")
    print("  2. Ask Cursor: 'What are the recent wandb runs?'")
    print()


if __name__ == "__main__":
    main()

