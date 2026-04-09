"""Configuration for Lightning.ai MCP Server."""
import os
from typing import Optional


class LightningConfig:
    def __init__(self):
        self.studio_name: str = os.environ.get("LIGHTNING_STUDIO_NAME", "grpo-chess")
        # Your Lightning.ai username (visible in your profile URL: lightning.ai/<username>).
        # Required — the SDK needs to know who owns the studio.
        # Auth is handled transparently via ~/.lightning/credentials.json (set by `lightning login`).
        self.teamspace: Optional[str] = os.environ.get("LIGHTNING_TEAMSPACE") or None
        self.user: Optional[str] = os.environ.get("LIGHTNING_USER") or None


_config: Optional[LightningConfig] = None


def get_config() -> LightningConfig:
    global _config
    if _config is None:
        _config = LightningConfig()
    return _config
