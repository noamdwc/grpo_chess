from src.colab_experiment_ladder import (
    RunAssessment,
    choose_confirmation_source,
    select_best_stage1_run,
    select_stage2_bracket,
)


def test_select_stage2_bracket_uses_low_midpoint_when_a1_wins():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.60, stable=True, eval_score=0.01),
        "A2": RunAssessment(name="A2", movement_score=0.45, stable=True, eval_score=0.00),
        "A3": RunAssessment(name="A3", movement_score=0.30, stable=False, eval_score=0.00),
    }

    assert select_stage2_bracket(results, best_stage1="A1") == "grpo_colab_bracket_b1_mid_low.yaml"


def test_select_stage2_bracket_uses_high_midpoint_when_a2_wins_and_a1_is_inert():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.05, stable=True, eval_score=0.00, inert=True),
        "A2": RunAssessment(name="A2", movement_score=0.50, stable=True, eval_score=0.01),
        "A3": RunAssessment(name="A3", movement_score=0.40, stable=True, eval_score=0.00),
    }

    assert select_stage2_bracket(results, best_stage1="A2") == "grpo_colab_bracket_b1_mid_high.yaml"


def test_select_stage2_bracket_uses_low_midpoint_when_a2_wins_and_a3_is_unstable():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.35, stable=True, eval_score=0.00),
        "A2": RunAssessment(name="A2", movement_score=0.55, stable=True, eval_score=0.01),
        "A3": RunAssessment(name="A3", movement_score=0.70, stable=False, eval_score=-0.01, unstable=True),
    }

    assert select_stage2_bracket(results, best_stage1="A2") == "grpo_colab_bracket_b1_mid_low.yaml"


def test_select_stage2_bracket_uses_high_midpoint_when_a3_wins():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.20, stable=True, eval_score=0.00),
        "A2": RunAssessment(name="A2", movement_score=0.45, stable=True, eval_score=0.01),
        "A3": RunAssessment(name="A3", movement_score=0.65, stable=True, eval_score=0.01),
    }

    assert select_stage2_bracket(results, best_stage1="A3") == "grpo_colab_bracket_b1_mid_high.yaml"


def test_select_best_stage1_run_prioritizes_movement_then_stability_then_eval():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.10, stable=True, eval_score=0.03),
        "A2": RunAssessment(name="A2", movement_score=0.50, stable=True, eval_score=0.00),
        "A3": RunAssessment(name="A3", movement_score=0.70, stable=False, eval_score=0.04, unstable=True),
    }

    assert select_best_stage1_run(results) == "A2"


def test_choose_confirmation_source_falls_back_to_best_stage1_when_stage2_is_inconclusive():
    best_stage1 = RunAssessment(name="A2", movement_score=0.50, stable=True, eval_score=0.01)
    stage2 = RunAssessment(
        name="B1",
        movement_score=0.51,
        stable=True,
        eval_score=0.01,
        inconclusive=True,
        config_name="grpo_colab_bracket_b1_mid_high.yaml",
    )

    assert choose_confirmation_source(best_stage1, stage2) == "A2"


def test_choose_confirmation_source_uses_stage2_when_clearly_better():
    best_stage1 = RunAssessment(name="A2", movement_score=0.50, stable=True, eval_score=0.01)
    stage2 = RunAssessment(
        name="B1",
        movement_score=0.60,
        stable=True,
        eval_score=0.02,
        config_name="grpo_colab_bracket_b1_mid_high.yaml",
    )

    assert choose_confirmation_source(best_stage1, stage2) == "B1"
