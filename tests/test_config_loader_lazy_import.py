import importlib
import sys


def test_config_loader_import_does_not_eagerly_import_grpo_model(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.configs.config_loader", raising=False)
    monkeypatch.delitem(sys.modules, "src.grpo_logic.model", raising=False)

    importlib.import_module("src.configs.config_loader")

    assert "src.grpo_logic.model" not in sys.modules


def test_load_grpo_config_imports_grpo_model_on_demand(monkeypatch):
    module = importlib.import_module("src.configs.config_loader")
    monkeypatch.delitem(sys.modules, "src.grpo_logic.model", raising=False)

    cfg = module.load_grpo_config("default.yaml")

    assert "src.grpo_logic.model" in sys.modules
    assert cfg is not None
