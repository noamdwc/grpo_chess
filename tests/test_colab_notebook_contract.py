import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_NOTEBOOK = REPO_ROOT / "chess_model_run_git.ipynb"
COLAB_CONFIG = REPO_ROOT / "src" / "configs" / "grpo_colab_main.yaml"


def _notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_main_colab_notebook_exists():
    assert MAIN_NOTEBOOK.exists(), "Main Colab notebook should be committed at repo root"


def test_main_notebook_targets_reasoning_grpo_entrypoint():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert "from src.train_self_play import train as grpo_train" in source
    assert "grpo_train(" in source
    assert "dataloader_kwargs" not in source
    assert "src.pretrain.pretrain" not in source
    assert "src.distill.distill" not in source


def test_main_notebook_exposes_mode_and_drive_parameters():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert 'MODE = "smoke"' in source or 'MODE = "full"' in source
    assert "DRIVE_ROOT" in source
    assert "BASE_CHECKPOINT_PATH" in source
    assert "RESUME_CHECKPOINT_OR_DIR" in source


def test_main_notebook_bootstraps_missing_base_checkpoint():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert "https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip" in source
    assert "src.dm_port.convert_jax" in source
    assert "if not base_checkpoint.exists()" in source


def test_committed_colab_config_loads_with_current_config_loader():
    from src.configs.config_loader import load_experiment_config

    assert COLAB_CONFIG.exists(), "Notebook template config should be committed under src/configs"
    cfg = load_experiment_config("grpo_colab_main.yaml")
    assert cfg.model.base_checkpoint
    assert cfg.training.checkpoint_dir
    assert cfg.reasoning.max_think_tokens >= cfg.reasoning.min_think_tokens
