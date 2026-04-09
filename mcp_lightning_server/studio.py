"""Lightning.ai Studio wrapper."""
from lightning_sdk import Studio

from .config import get_config


def get_studio() -> Studio:
    config = get_config()
    kwargs = {"name": config.studio_name}
    if config.teamspace:
        kwargs["teamspace"] = config.teamspace
    if config.user:
        kwargs["user"] = config.user
    return Studio(**kwargs)


def get_jobs_plugin(studio: Studio):
    studio.install_plugin("jobs")
    return studio.installed_plugins["jobs"]
