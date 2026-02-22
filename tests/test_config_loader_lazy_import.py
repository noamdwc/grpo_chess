import importlib
import sys


def test_config_loader_import_does_not_eagerly_import_grpo_model():
    sys.modules.pop("src.configs.config_loader", None)
    sys.modules.pop("src.grpo_logic.model", None)

    importlib.import_module("src.configs.config_loader")

    assert "src.grpo_logic.model" not in sys.modules


def test_load_grpo_config_imports_grpo_model_on_demand():
    module = importlib.import_module("src.configs.config_loader")
    sys.modules.pop("src.grpo_logic.model", None)

    cfg = module.load_grpo_config("default.yaml")

    assert "src.grpo_logic.model" in sys.modules
    assert cfg is not None
