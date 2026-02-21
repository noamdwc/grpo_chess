import sys

from src.checkpoint_compat import register_legacy_checkpoint_aliases


def test_register_legacy_checkpoint_aliases_sets_grpo_self_play_alias():
    sys.modules.pop("src.grpo_self_play", None)
    register_legacy_checkpoint_aliases()
    assert "src.grpo_self_play" in sys.modules
