from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
import math
import statistics
from collections.abc import Mapping


MID_LOW_CONFIG = "grpo_colab_bracket_b1_mid_low.yaml"
MID_HIGH_CONFIG = "grpo_colab_bracket_b1_mid_high.yaml"

MIN_USABLE_WINDOW = 8
_REQUIRED_STEP_KEYS = (
    "train/ratio_step",
    "train/clip_fraction_step",
    "train/ppo_loss_step",
    "train/kl_divergence_step",
    "train_total_loss_step",
)


@dataclass(frozen=True)
class HistoryWindow:
    ratio_mean: float
    ratio_std: float
    clip_fraction_mean: float
    clip_fraction_peak: float
    ppo_abs_mean: float
    kl_abs_mean: float
    loss_mean: float
    loss_std: float
    eval_score: float
    point_count: int


@dataclass(frozen=True)
class RunAssessment:
    name: str
    movement_score: float
    stable: bool
    eval_score: float
    inert: bool = False
    unstable: bool = False
    inconclusive: bool = False
    failed: bool = False
    config_name: str = ""
    run_id: str = ""
    epochs_completed: int = 0
    stop_reason: str = ""
    decision_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def promotable(self) -> bool:
        return not self.failed and not self.inert and self.stable and not self.unstable


def _movement_score(window: HistoryWindow) -> float:
    ratio_delta = abs(window.ratio_mean - 1.0)
    clipped = min(window.clip_fraction_mean, 0.30)
    ppo_signal = min(window.ppo_abs_mean, 0.0025)
    kl_signal = min(window.kl_abs_mean, 0.003)
    return (
        ratio_delta * 40.0
        + clipped * 4.0
        + ppo_signal * 400.0
        + kl_signal * 250.0
    )


def assess_recent_history(
    *,
    run_name: str,
    config_name: str,
    run_id: str,
    window: HistoryWindow,
    epochs_completed: int,
) -> RunAssessment:
    ratio_delta = abs(window.ratio_mean - 1.0)
    inert = (
        ratio_delta < 0.002
        and window.clip_fraction_mean < 0.01
        and window.clip_fraction_peak < 0.03
        and window.ppo_abs_mean < 3e-4
        and window.kl_abs_mean < 3e-4
    )
    unstable = (
        window.clip_fraction_mean > 0.30
        or window.clip_fraction_peak > 0.60
        or ratio_delta > 0.08
        or window.ratio_std > 0.05
        or window.ppo_abs_mean > 0.003
        or window.kl_abs_mean > 0.004
        or window.loss_std > max(0.02, window.loss_mean * 0.75)
    )
    stable = not unstable
    stop_reason = "dead_run" if inert else ("instability" if unstable else "")
    movement_score = _movement_score(window)
    decision_reason = (
        "inert window at checkpoint boundary"
        if inert
        else "persistent instability in recent window"
        if unstable
        else "moderate non-inert movement with acceptable stability"
    )
    return RunAssessment(
        name=run_name,
        config_name=config_name,
        run_id=run_id,
        movement_score=movement_score,
        stable=stable,
        eval_score=window.eval_score,
        inert=inert,
        unstable=unstable,
        epochs_completed=epochs_completed,
        stop_reason=stop_reason,
        decision_reason=decision_reason,
        metrics={
            "ratio_mean": window.ratio_mean,
            "ratio_std": window.ratio_std,
            "clip_fraction_mean": window.clip_fraction_mean,
            "clip_fraction_peak": window.clip_fraction_peak,
            "ppo_abs_mean": window.ppo_abs_mean,
            "kl_abs_mean": window.kl_abs_mean,
            "loss_mean": window.loss_mean,
            "loss_std": window.loss_std,
            "eval_score": window.eval_score,
            "point_count": float(window.point_count),
        },
    )


def _finite_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _row_is_valid(row: Mapping[str, object]) -> bool:
    for key in _REQUIRED_STEP_KEYS:
        if key not in row:
            return False
        value = row[key]
        if value is None:
            return False
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def _summarize_history_window(rows: list[dict], eval_score: float) -> HistoryWindow:
    valid_rows = [row for row in rows if _row_is_valid(row)]
    if len(valid_rows) < MIN_USABLE_WINDOW:
        raise RuntimeError(
            f"insufficient usable rows: {len(valid_rows)} < {MIN_USABLE_WINDOW}"
        )

    ratios = [float(row["train/ratio_step"]) for row in valid_rows]
    clips = [float(row["train/clip_fraction_step"]) for row in valid_rows]
    ppo = [abs(float(row["train/ppo_loss_step"])) for row in valid_rows]
    kl = [abs(float(row["train/kl_divergence_step"])) for row in valid_rows]
    losses = [float(row["train_total_loss_step"]) for row in valid_rows]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return statistics.pstdev(values)

    return HistoryWindow(
        ratio_mean=_mean(ratios),
        ratio_std=_std(ratios),
        clip_fraction_mean=_mean(clips),
        clip_fraction_peak=max(clips),
        ppo_abs_mean=_mean(ppo),
        kl_abs_mean=_mean(kl),
        loss_mean=_mean(losses),
        loss_std=_std(losses),
        eval_score=eval_score,
        point_count=len(valid_rows),
    )


def _summary_float(summary: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return value if math.isfinite(value) else default


def _summary_history_window(summary: Mapping[str, object]) -> HistoryWindow:
    ratio = _summary_float(summary, "train/ratio_step", 1.0)
    clip_fraction = _summary_float(summary, "train/clip_fraction_step", 0.0)
    ppo_abs = abs(_summary_float(summary, "train/ppo_loss_step", 0.0))
    kl_abs = abs(_summary_float(summary, "train/kl_divergence_step", 0.0))
    loss = _summary_float(summary, "train_total_loss_step", _summary_float(summary, "train/loss_step", 0.0))
    eval_score = _summary_float(summary, "eval_stockfish/score", 0.0)
    return HistoryWindow(
        ratio_mean=ratio,
        ratio_std=0.0,
        clip_fraction_mean=clip_fraction,
        clip_fraction_peak=clip_fraction,
        ppo_abs_mean=ppo_abs,
        kl_abs_mean=kl_abs,
        loss_mean=loss,
        loss_std=0.0,
        eval_score=eval_score,
        point_count=1,
    )


def build_assessment_from_wandb_metrics(
    *,
    run_name: str,
    config_name: str,
    run_id: str,
    epochs_completed: int,
    rows: list[dict],
    summary: Mapping[str, object],
    history_error: Exception | None = None,
) -> RunAssessment:
    fallback_detail = "no rows returned"
    if rows:
        recent_rows = rows[-12:]
        eval_history = _finite_values(rows, "eval_stockfish/score")
        summary_eval = _summary_float(summary, "eval_stockfish/score", 0.0)
        try:
            window = _summarize_history_window(
                recent_rows,
                eval_history[-1] if eval_history else summary_eval,
            )
            return assess_recent_history(
                run_name=run_name,
                config_name=config_name,
                run_id=run_id,
                window=window,
                epochs_completed=epochs_completed,
            )
        except RuntimeError as exc:
            fallback_detail = str(exc)

    window = _summary_history_window(summary)
    assessment = assess_recent_history(
        run_name=run_name,
        config_name=config_name,
        run_id=run_id,
        window=window,
        epochs_completed=epochs_completed,
    )
    fallback_reason = f"insufficient_history: {fallback_detail}"
    if history_error is not None:
        fallback_reason += f" ({history_error})"
    return dataclasses.replace(
        assessment,
        decision_reason=fallback_reason,
        inconclusive=True,
        inert=False,
        unstable=False,
        stable=True,
        stop_reason="",
    )


def build_failed_assessment(
    *,
    run_name: str,
    config_name: str,
    run_id: str = "",
    epochs_completed: int = 0,
    reason: str = "",
) -> RunAssessment:
    return RunAssessment(
        name=run_name,
        config_name=config_name,
        run_id=run_id,
        movement_score=-1.0,
        stable=False,
        eval_score=-1.0,
        failed=True,
        stop_reason="failed",
        epochs_completed=epochs_completed,
        decision_reason=reason or "run failed before producing a promotable assessment",
    )


def _ranking_key(result: RunAssessment) -> tuple[float, ...]:
    # Rank by promotability first, then prefer less aggressive stable movement over risky ambiguity.
    return (
        0.0 if result.failed else 1.0,
        0.0 if result.inert else 1.0,
        1.0 if result.stable and not result.unstable else 0.0,
        result.movement_score,
        result.eval_score,
        -result.epochs_completed,
    )


def select_best_stage1_run(results: dict[str, RunAssessment]) -> str:
    valid = {
        name: result
        for name, result in results.items()
        if not result.failed and not result.inconclusive
    }
    if not valid:
        raise ValueError("no usable stage 1 results; all inconclusive or failed")
    return max(valid.items(), key=lambda item: _ranking_key(item[1]))[0]


def should_skip_stage2(results: dict[str, RunAssessment], *, best_stage1: str) -> bool:
    if best_stage1 not in results:
        raise KeyError(f"Missing best_stage1 result: {best_stage1}")
    winner = results[best_stage1]
    if winner.inconclusive:
        return False
    contenders = [result for name, result in results.items() if name != best_stage1 and not result.failed]
    if any(result.inconclusive for result in contenders):
        return False
    if not winner.promotable:
        return False
    if not contenders:
        return True

    contenders.sort(key=_ranking_key, reverse=True)
    runner_up = contenders[0]
    if not runner_up.promotable and winner.movement_score >= 0.50:
        return True
    return winner.movement_score - runner_up.movement_score >= 0.25 and runner_up.inert


def select_stage2_bracket(results: dict[str, RunAssessment], *, best_stage1: str) -> str:
    if best_stage1 not in results:
        raise KeyError(f"Missing best_stage1 result: {best_stage1}")

    if best_stage1 == "A1":
        return MID_LOW_CONFIG
    if best_stage1 == "A3":
        return MID_HIGH_CONFIG
    if best_stage1 != "A2":
        raise ValueError(f"Unsupported stage 1 winner: {best_stage1}")

    a1 = results.get("A1")
    a3 = results.get("A3")
    if a1 is not None and a1.inert:
        return MID_HIGH_CONFIG
    if a3 is not None and a3.unstable:
        return MID_LOW_CONFIG
    return MID_LOW_CONFIG


def choose_confirmation_source(
    best_stage1: RunAssessment,
    stage2_result: RunAssessment | None,
    *,
    best_stage1_key: str | None = None,
    stage2_key: str = "B1",
) -> str:
    """Return the stage key ("A1"/"A2"/"A3" or "B1") to confirm from.

    best_stage1_key overrides best_stage1.name; callers that store the run
    label (not the probe key) in .name must pass best_stage1_key explicitly.
    """
    winner_key = best_stage1_key if best_stage1_key is not None else best_stage1.name
    if stage2_result is None:
        return winner_key
    if (
        stage2_result.failed
        or stage2_result.inconclusive
        or not stage2_result.stable
        or stage2_result.unstable
    ):
        return winner_key

    # Movement score weights ratio / clip / PPO / KL into an approximately 0-3 band.
    # Require a visible gain, or a smaller movement gain plus a real eval nudge.
    movement_gain = stage2_result.movement_score - best_stage1.movement_score
    eval_gain = stage2_result.eval_score - best_stage1.eval_score
    clearly_better = movement_gain >= 0.15 or (movement_gain >= 0.08 and eval_gain >= 0.01)
    return stage2_key if clearly_better else winner_key


def resolve_stage3_target(
    *,
    stage1_results: dict[str, RunAssessment],
    best_stage1_key: str,
    stage2_result: RunAssessment | None,
    stage2_config: str | None,
    stage1_configs: Mapping[str, str],
    stage2_key: str = "B1",
) -> tuple[str, str, int]:
    source_key = choose_confirmation_source(
        stage1_results[best_stage1_key],
        stage2_result,
        best_stage1_key=best_stage1_key,
        stage2_key=stage2_key,
    )
    if source_key == stage2_key and stage2_result is not None:
        if stage2_config is None:
            raise ValueError(
                f"stage2_key={stage2_key!r} selected but stage2_config is None"
            )
        return source_key, stage2_config, stage2_result.epochs_completed
    if source_key not in stage1_configs:
        # Guards the recurring 'grpo-threshold-ladder-b1-eNNN' KeyError: a run
        # label leaked into the source key instead of a probe key.
        raise KeyError(
            f"resolve_stage3_target produced source_key={source_key!r}, "
            f"which is neither stage2_key={stage2_key!r} nor a key of "
            f"stage1_configs={list(stage1_configs)!r}"
        )
    return (
        source_key,
        stage1_configs[source_key],
        stage1_results[source_key].epochs_completed,
    )


def compute_confirmation_epochs(
    *,
    remaining_hours: float,
    estimated_min_per_epoch: float,
    reserve_minutes: int,
    target_epochs: int,
) -> int:
    usable_minutes = max(0.0, remaining_hours * 60.0 - reserve_minutes)
    if usable_minutes <= 0:
        return 0
    epochs = int(usable_minutes / max(estimated_min_per_epoch, 0.1))
    return max(0, min(target_epochs, epochs))
