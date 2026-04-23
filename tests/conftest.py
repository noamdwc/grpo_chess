"""Pytest configuration for the test suite."""
import os
import sys
from pathlib import Path

import pytest

# JAX must see this before any import to enable real float64 tests on CPU.
os.environ.setdefault("JAX_ENABLE_X64", "True")

# Add the project root to sys.path so that 'src' module can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Filter known third-party warnings that clutter test output."""
    # python-chess deprecations on Python 3.14+
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning:chess.engine")
    # haiku positional arg deprecation on Python 3.14+
    config.addinivalue_line("filterwarnings", "ignore:'maxsplit' is passed as positional argument:DeprecationWarning:haiku")
    # Lightning informational warnings irrelevant in tests
    config.addinivalue_line("filterwarnings", "ignore:GPU available but not used:UserWarning:pytorch_lightning")
    config.addinivalue_line("filterwarnings", "ignore:.*does not have many workers.*:UserWarning:pytorch_lightning")
    config.addinivalue_line("filterwarnings", "ignore:.*persistent_workers.*:UserWarning:pytorch_lightning")
    config.addinivalue_line("filterwarnings", "ignore:.*self.log.*trainer.*not registered.*:UserWarning:pytorch_lightning")
    config.addinivalue_line("filterwarnings", "ignore:.*validation_step.*val_dataloader.*:UserWarning:pytorch_lightning")
    config.addinivalue_line("filterwarnings", "ignore:.*Checkpoint directory.*not empty.*:UserWarning:pytorch_lightning")


def pytest_addoption(parser):
    parser.addoption(
        "--bc-checkpoint",
        action="store",
        default=None,
        help="Path to a BC checkpoint for warmstart smoke test.",
    )
    parser.addoption(
        "--dm-av-checkpoint",
        action="store",
        default=None,
        help="Path to a DM-AV checkpoint for warmstart smoke test.",
    )


@pytest.fixture
def bc_checkpoint_path(request) -> str | None:
    return request.config.getoption("--bc-checkpoint") or os.environ.get("GRPO_CHESS_BC_CHECKPOINT")


@pytest.fixture
def dm_av_checkpoint_path(request) -> str | None:
    return request.config.getoption("--dm-av-checkpoint") or os.environ.get("GRPO_CHESS_DM_AV_CHECKPOINT")
