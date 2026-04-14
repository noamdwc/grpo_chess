import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LADDER_NOTEBOOK = REPO_ROOT / "chess_grpo_experiment_ladder_colab.ipynb"


def _notebook_sources(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def test_ladder_notebook_exists():
    assert LADDER_NOTEBOOK.exists(), "Automated ladder Colab notebook should be committed at repo root"


def test_ladder_notebook_uses_existing_grpo_runner_flow_and_new_ladder_helpers():
    source = _notebook_sources(LADDER_NOTEBOOK)
    assert "from src.train_self_play import train as grpo_train" in source
    assert "from src.colab_experiment_ladder import" in source
    assert "STAGE_1_CONFIGS" in source
    assert "select_stage2_bracket" in source
    assert "choose_confirmation_source" in source
    assert "wandb.Api()" in source
    assert "run_experiment(" in source
    assert "stage1_results" in source
    assert "stage2_result" in source
    assert "confirmation_source" in source
