from __future__ import annotations

from dataclasses import dataclass


MID_LOW_CONFIG = "grpo_colab_bracket_b1_mid_low.yaml"
MID_HIGH_CONFIG = "grpo_colab_bracket_b1_mid_high.yaml"


@dataclass(frozen=True)
class RunAssessment:
    name: str
    movement_score: float
    stable: bool
    eval_score: float
    inert: bool = False
    unstable: bool = False
    inconclusive: bool = False
    config_name: str = ""

    @property
    def promotable(self) -> bool:
        return not self.inert and self.stable and not self.unstable


def _ranking_key(result: RunAssessment) -> tuple[float, ...]:
    return (
        0.0 if result.inert else 1.0,
        1.0 if result.stable and not result.unstable else 0.0,
        result.movement_score,
        result.eval_score,
    )


def select_best_stage1_run(results: dict[str, RunAssessment]) -> str:
    if not results:
        raise ValueError("results must not be empty")
    return max(results.items(), key=lambda item: _ranking_key(item[1]))[0]


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


def choose_confirmation_source(best_stage1: RunAssessment, stage2_result: RunAssessment | None) -> str:
    if stage2_result is None:
        return best_stage1.name
    if stage2_result.inconclusive or not stage2_result.stable or stage2_result.unstable:
        return best_stage1.name

    movement_gain = stage2_result.movement_score - best_stage1.movement_score
    eval_gain = stage2_result.eval_score - best_stage1.eval_score
    clearly_better = movement_gain >= 0.05 or (movement_gain >= 0.02 and eval_gain >= 0.01)
    return stage2_result.name if clearly_better else best_stage1.name
