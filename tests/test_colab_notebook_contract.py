import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_NOTEBOOK = REPO_ROOT / "chess_model_run_git.ipynb"
DEBUG_NOTEBOOK = REPO_ROOT / "chess_stockfish_eval_debug_colab.ipynb"
COLAB_CONFIG = REPO_ROOT / "src" / "configs" / "grpo_colab_main.yaml"


def _notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_main_colab_notebook_exists():
    assert MAIN_NOTEBOOK.exists(), "Main Colab notebook should be committed at repo root"


def test_debug_colab_notebook_exists():
    assert DEBUG_NOTEBOOK.exists(), "Debug Colab notebook should be committed at repo root"


def test_main_notebook_targets_reasoning_grpo_entrypoint():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert "from src.train_self_play import train as grpo_train" in source
    assert "grpo_train(" in source
    assert "eval_stockfish_init" in source
    assert "eval_stockfish/" in source
    assert "dataloader_kwargs" not in source
    assert "src.pretrain.pretrain" not in source
    assert "src.distill.distill" not in source
    assert 'os.chdir("/content")' in source
    assert "!git submodule update --init --recursive" in source
    assert '%pip install -q --force-reinstall --no-cache-dir pillow' in source
    assert 'os.kill(os.getpid(), signal.SIGKILL)' in source
    assert 'Restarting runtime to load fresh binary modules' in source
    assert "!apt-get -qq install -y stockfish" in source


def test_main_notebook_exposes_mode_and_drive_parameters():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert 'MODE = "smoke"' in source or 'MODE = "full"' in source
    assert 'CHECKPOINT_FAMILY = "dm_9m"' in source or 'CHECKPOINT_FAMILY = "bc_pt"' in source
    assert "DRIVE_ROOT" in source
    assert "BASE_CHECKPOINT_PATH" in source
    assert "RESUME_CHECKPOINT_OR_DIR" in source


def test_main_notebook_keeps_wandb_available_in_smoke_mode():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert 'USE_WANDB = True' in source
    assert '"use_wandb": False' not in source
    assert 'mode != "smoke"' not in source
    assert "from google.colab import userdata" in source
    assert 'os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")' in source
    assert "!wandb login" in source


def test_main_notebook_bootstraps_missing_base_checkpoint():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert "https://storage.googleapis.com/searchless_chess/checkpoints/9M.zip" in source
    assert "https://storage.googleapis.com/searchless_chess/checkpoints/136M.zip" in source
    assert "src.dm_port.convert_jax" in source
    assert "if not base_checkpoint.exists()" in source
    assert "scripts/setup_distill_deps.sh --checkpoint 9M --skip-checkpoint" in source
    assert "capture_output=True" in source
    assert "Converter stdout:" in source
    assert "Converter stderr:" in source
    assert "searchless_chess/checkpoints/136M" in source
    assert "symlink_to" in source


def test_main_notebook_rewrites_dm_av_embedding_sources_for_bc_warmstart():
    source = _notebook_sources(MAIN_NOTEBOOK)
    assert 'dm_av_checkpoint = drive_root / "base" / "9M.pt"' in source
    assert 'config_data["model"]["dm_av_embedding_source"] = str(dm_av_checkpoint)' in source
    assert 'config_data["rival"]["frozen_dm_9m"]["dm_av_embedding_source"] = str(dm_av_checkpoint)' in source


def test_committed_colab_config_loads_with_current_config_loader():
    from src.configs.config_loader import load_experiment_config

    assert COLAB_CONFIG.exists(), "Notebook template config should be committed under src/configs"
    cfg = load_experiment_config("grpo_colab_main.yaml")
    assert cfg.model.base_checkpoint
    assert cfg.training.checkpoint_dir
    assert cfg.reasoning.max_think_tokens >= cfg.reasoning.min_think_tokens
    assert cfg.leaf_evaluator.mode == "dm_value_head"
    assert cfg.leaf_evaluator.dm_value_head.model == "136M"


def test_debug_notebook_exposes_drive_checkpoint_eval_flow():
    source = _notebook_sources(DEBUG_NOTEBOOK)
    assert "CHECKPOINT_ROOT" in source
    assert "CHECKPOINT_FILTER" in source
    assert "CHECKPOINT_INDEX" in source
    assert "ReasoningEvaluator" in source
    assert "selected_checkpoint" in source
    assert "single_evaluation()" in source
    assert "resolve_stockfish_path" in source
    assert "All 3 callback-style eval attempts failed" in source
