"""Configuration for WandB MCP Server."""
import os
from typing import Optional


class WandBConfig:
    """Configuration for WandB API connection."""
    
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize WandB configuration.

        Args:
            project: WandB project name (default: env WANDB_PROJECT or "Chess-GRPO-Bot")
            entity: WandB entity/team name (defaults to user's entity)
            api_key: WandB API key (defaults to WANDB_API_KEY env var or wandb config)
        """
        self.project = project or os.getenv("WANDB_PROJECT", "Chess-GRPO-Bot")
        self.entity = entity or os.getenv("WANDB_ENTITY")
        self.api_key = api_key or os.getenv("WANDB_API_KEY")
        
    def get_api_kwargs(self) -> dict:
        """Get kwargs for wandb.Api() initialization."""
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


# Global config instance
_config: Optional[WandBConfig] = None


def get_config() -> WandBConfig:
    """Get the global WandB configuration."""
    global _config
    if _config is None:
        _config = WandBConfig()
    return _config


def set_config(config: WandBConfig) -> None:
    """Set the global WandB configuration."""
    global _config
    _config = config

