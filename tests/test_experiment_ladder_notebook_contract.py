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
    assert "HistoryWindow" in source
    assert "RunAssessment" in source
    assert "STAGE_1_CONFIGS" in source
    assert "select_stage2_bracket" in source
    assert "choose_confirmation_source" in source
    assert "scan_history" in source
    assert "run.id" in source or '"run_id"' in source
    assert "LAUNCHED_RUNS" in source
    assert "CapturingWandbLogger" in source
    assert "scan_history returned no rows" in source
    assert "summary_fallback" in source
    assert "run_experiment(" in source
    assert "stage1_results" in source
    assert "stage2_result" in source
    assert "confirmation_source" in source
    assert "decision_records" in source
    assert "failed_runs" in source
    assert "should_skip_stage2" in source
    assert "compute_confirmation_epochs" in source
    assert "resume_from_checkpoint" in source
    assert "stage_boundaries" not in source
