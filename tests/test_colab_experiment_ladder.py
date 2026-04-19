import math

import pytest

from src.colab_experiment_ladder import (
    MIN_USABLE_WINDOW,
    HistoryWindow,
    RunAssessment,
    assess_recent_history,
    build_assessment_from_wandb_metrics,
    choose_confirmation_source,
    compute_confirmation_epochs,
    select_best_stage1_run,
    select_stage2_bracket,
    should_skip_stage2,
)


def _healthy_row(i: int = 0) -> dict:
    return {
        "train/ratio_step": 1.01 + 0.001 * i,
        "train/clip_fraction_step": 0.05 + 0.002 * i,
        "train/ppo_loss_step": -0.0008 - 0.00002 * i,
        "train/kl_divergence_step": 0.0009 + 0.00003 * i,
        "train_total_loss_step": 0.018 + 0.0004 * i,
    }


def _healthy_summary() -> dict:
    return {
        "train/ratio_step": 1.012,
        "train/clip_fraction_step": 0.06,
        "train/ppo_loss_step": -0.0008,
        "train/kl_divergence_step": 0.0009,
        "train_total_loss_step": 0.019,
        "eval_stockfish/score": 0.02,
    }


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


def test_assess_recent_history_marks_dead_run_from_inert_window():
    window = HistoryWindow(
        ratio_mean=1.0004,
        ratio_std=0.0003,
        clip_fraction_mean=0.002,
        clip_fraction_peak=0.006,
        ppo_abs_mean=0.00008,
        kl_abs_mean=0.00007,
        loss_mean=0.018,
        loss_std=0.0004,
        eval_score=0.0,
        point_count=12,
    )

    assessment = assess_recent_history(
        run_name="A1",
        config_name="probe_a1.yaml",
        run_id="run-a1",
        window=window,
        epochs_completed=22,
    )

    assert assessment.inert is True
    assert assessment.unstable is False
    assert assessment.stop_reason == "dead_run"


def test_assess_recent_history_marks_instability_from_explosive_window():
    window = HistoryWindow(
        ratio_mean=1.08,
        ratio_std=0.09,
        clip_fraction_mean=0.42,
        clip_fraction_peak=0.71,
        ppo_abs_mean=0.004,
        kl_abs_mean=0.006,
        loss_mean=0.035,
        loss_std=0.02,
        eval_score=0.0,
        point_count=12,
    )

    assessment = assess_recent_history(
        run_name="A3",
        config_name="probe_a3.yaml",
        run_id="run-a3",
        window=window,
        epochs_completed=24,
    )

    assert assessment.unstable is True
    assert assessment.stable is False
    assert assessment.stop_reason == "instability"


def test_assess_recent_history_prefers_moderate_non_inert_window():
    window = HistoryWindow(
        ratio_mean=1.012,
        ratio_std=0.01,
        clip_fraction_mean=0.08,
        clip_fraction_peak=0.18,
        ppo_abs_mean=0.0008,
        kl_abs_mean=0.0009,
        loss_mean=0.019,
        loss_std=0.001,
        eval_score=0.01,
        point_count=12,
    )

    assessment = assess_recent_history(
        run_name="A2",
        config_name="probe_a2.yaml",
        run_id="run-a2",
        window=window,
        epochs_completed=28,
    )

    assert assessment.inert is False
    assert assessment.unstable is False
    assert assessment.stable is True
    assert assessment.stop_reason == ""


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


def test_should_skip_stage2_when_single_stage1_winner_is_clearly_dominant():
    results = {
        "A1": RunAssessment(name="A1", movement_score=0.62, stable=True, eval_score=0.02),
        "A2": RunAssessment(name="A2", movement_score=0.03, stable=True, eval_score=0.0, inert=True),
        "A3": RunAssessment(name="A3", movement_score=0.70, stable=False, eval_score=-0.01, unstable=True),
    }

    assert should_skip_stage2(results, best_stage1="A1") is True


def test_compute_confirmation_epochs_allows_short_run_when_late_in_session():
    epochs = compute_confirmation_epochs(
        remaining_hours=1.4,
        estimated_min_per_epoch=2.8,
        reserve_minutes=30,
        target_epochs=300,
    )

    assert 0 <= epochs < 30


def test_build_assessment_from_wandb_metrics_marks_inconclusive_when_history_is_empty():
    assessment = build_assessment_from_wandb_metrics(
        run_name="A3",
        config_name="probe_a3.yaml",
        run_id="run-a3",
        epochs_completed=20,
        rows=[],
        summary=_healthy_summary(),
    )

    assert assessment.failed is False
    assert assessment.inconclusive is True
    assert assessment.inert is False
    assert assessment.unstable is False
    assert assessment.stable is True
    assert assessment.stop_reason == ""
    assert assessment.run_id == "run-a3"
    assert "insufficient_history" in assessment.decision_reason


def test_build_assessment_from_wandb_metrics_marks_inconclusive_when_rows_are_sparse():
    # Dangerous middle case: non-empty history with valid-row count below MIN_USABLE_WINDOW.
    # Build (MIN_USABLE_WINDOW - 2) fully valid rows plus noise rows each missing one metric.
    rows: list[dict] = [_healthy_row(i) for i in range(MIN_USABLE_WINDOW - 2)]
    for key in (
        "train/ratio_step",
        "train/clip_fraction_step",
        "train/ppo_loss_step",
        "train/kl_divergence_step",
        "train_total_loss_step",
    ):
        noisy = _healthy_row(99)
        noisy.pop(key)
        rows.append(noisy)

    assessment = build_assessment_from_wandb_metrics(
        run_name="A2",
        config_name="probe_a2.yaml",
        run_id="run-a2",
        epochs_completed=30,
        rows=rows,
        summary=_healthy_summary(),
    )

    assert assessment.failed is False
    assert assessment.inconclusive is True
    assert assessment.inert is False
    assert assessment.unstable is False
    assert "insufficient_history" in assessment.decision_reason

    other = RunAssessment(
        name="A1", movement_score=0.30, stable=True, eval_score=0.0
    )
    results = {"A2": assessment, "A1": other}
    # Inconclusive winner candidate must not be promotable for Stage 2 skip,
    # and must be filtered out of Stage 1 winner selection.
    assert should_skip_stage2(results, best_stage1="A1") is False
    assert select_best_stage1_run(results) == "A1"


def test_build_assessment_from_wandb_metrics_uses_window_when_history_is_healthy():
    rows = [_healthy_row(i) for i in range(MIN_USABLE_WINDOW + 4)]

    assessment = build_assessment_from_wandb_metrics(
        run_name="A2",
        config_name="probe_a2.yaml",
        run_id="run-a2",
        epochs_completed=30,
        rows=rows,
        summary=_healthy_summary(),
    )

    assert assessment.failed is False
    assert assessment.inconclusive is False
    assert assessment.stable is True
    assert assessment.unstable is False
    # Real window must produce non-zero std, not the summary-snapshot zero.
    assert assessment.metrics["ratio_std"] > 0.0
    assert assessment.metrics["loss_std"] > 0.0
    assert assessment.metrics["point_count"] >= MIN_USABLE_WINDOW
    assert "insufficient_history" not in assessment.decision_reason


def test_build_assessment_excludes_rows_with_nan_or_missing_values():
    rows: list[dict] = [_healthy_row(i) for i in range(MIN_USABLE_WINDOW)]
    # Add two rows that must be excluded from the window.
    bad_nan = _healthy_row(42)
    bad_nan["train/ratio_step"] = float("nan")
    rows.append(bad_nan)
    bad_missing = _healthy_row(43)
    bad_missing.pop("train/kl_divergence_step")
    rows.append(bad_missing)

    assessment = build_assessment_from_wandb_metrics(
        run_name="A2",
        config_name="probe_a2.yaml",
        run_id="run-a2",
        epochs_completed=30,
        rows=rows,
        summary=_healthy_summary(),
    )

    assert assessment.inconclusive is False
    # point_count must reflect only fully-valid rows, ignoring the two bad rows.
    assert assessment.metrics["point_count"] == MIN_USABLE_WINDOW
    assert math.isfinite(assessment.metrics["ratio_mean"])


def test_select_best_stage1_run_raises_when_all_results_are_inconclusive():
    results = {
        name: RunAssessment(
            name=name,
            movement_score=0.2,
            stable=True,
            eval_score=0.0,
            inconclusive=True,
        )
        for name in ("A1", "A2", "A3")
    }

    with pytest.raises(ValueError):
        select_best_stage1_run(results)
