#!/usr/bin/env python3
"""Generate interview presentation visuals from repository artifacts."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "interview_presentation" / "assets"


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected source file: {path}")
    return path.read_text(encoding="utf-8")


def parse_teacher_baselines() -> list[dict[str, float]]:
    doc = _read(ROOT / "research_docs" / "2026-02-24_deepmind-136m-teacher-baseline.md")
    pattern = re.compile(
        r"\|\s*DeepMind\s+(\d+M)\s*\|\s*\**([0-9.]+)\**\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*\|\s*\**([+\-]?[0-9.]+)\**"
    )
    rows: list[dict[str, float]] = []
    for m in pattern.finditer(doc):
        rows.append(
            {
                "model": m.group(1),
                "score": float(m.group(2)),
                "wins": int(m.group(3)),
                "draws": int(m.group(4)),
                "losses": int(m.group(5)),
                "elo": float(m.group(6)),
            }
        )
    if not rows:
        raise ValueError("Failed to parse teacher baseline table")

    order = {"9M": 0, "136M": 1, "270M": 2}
    rows.sort(key=lambda r: order.get(r["model"], 99))
    return rows


def parse_distill_quality_trend() -> list[dict[str, float]]:
    doc = _read(ROOT / "research_docs" / "2026-02-22_distill-quality-collapse-sparse-labels.md")
    pattern = re.compile(
        r"\|\s*`([^`]+)`\s*\|\s*([0-9\-]+)\s*\|\s*`([0-9.]+)`\s*\|\s*`([0-9.]+)`\s*\|\s*`([0-9.]+)`\s*\|"
    )
    rows: list[dict[str, float]] = []
    for m in pattern.finditer(doc):
        rows.append(
            {
                "run": m.group(1),
                "date": m.group(2),
                "top1": float(m.group(3)),
                "top5": float(m.group(4)),
                "entropy": float(m.group(5)),
            }
        )
    if not rows:
        raise ValueError("Failed to parse distill quality trend table")
    return rows


def parse_k_hist() -> dict[int, int]:
    doc = _read(ROOT / "research_docs" / "2026-02-22_distill-quality-collapse-sparse-labels.md")
    m = re.search(r"k_hist\s*=\s*(\{[^\n]+\})", doc)
    if not m:
        raise ValueError("Failed to find k_hist in distill collapse doc")
    raw = m.group(1)
    normalized = re.sub(r"\((\d+)\)", r"\1", raw)
    parsed = ast.literal_eval(normalized)
    return {int(k): int(v) for k, v in parsed.items()}


def parse_loss_budget_components() -> list[dict[str, float]]:
    doc = _read(ROOT / "research_docs" / "2026-02-06_loss-budget-and-monitor-analysis.md")
    pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([+\-]?[0-9.]+)\s*\|\s*([+\-]?[0-9.]+)\s*\|\s*([+\-]?[0-9.]+)\s*\|\s*([+\-]?[0-9.]+)\s*\|"
    )
    rows: list[dict[str, float]] = []
    for m in pattern.finditer(doc):
        rows.append(
            {
                "step": int(m.group(1)),
                "ppo": float(m.group(2)),
                "kl": float(m.group(3)),
                "entropy": float(m.group(4)),
                "total": float(m.group(5)),
            }
        )
    if not rows:
        raise ValueError("Failed to parse loss budget table")
    return rows


def parse_data_coverage_values() -> tuple[dict[str, float], float]:
    default_cfg = yaml.safe_load(_read(ROOT / "src" / "configs" / "default.yaml"))
    pretrain_cfg = yaml.safe_load(_read(ROOT / "src" / "configs" / "pretrain.yaml"))
    phases = default_cfg["dataset"]["phase_distribution"]
    eval_fraction = float(pretrain_cfg["dataset"]["eval_fraction"])
    return {str(k): float(v) for k, v in phases.items()}, eval_fraction


def draw_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    nodes = [
        (0.4, 2.4, 2.0, 1.0, "Data\n(Pretrain +\nSelf-Play Seeds)"),
        (3.0, 2.4, 2.0, 1.0, "Pretrain\n(Supervised)"),
        (5.6, 2.4, 2.0, 1.0, "Distill\n(Teacher -> Student)"),
        (8.2, 2.4, 2.0, 1.0, "GRPO\nSelf-Play RL"),
        (10.4, 2.4, 1.4, 1.0, "Eval\nvs Stockfish"),
        (8.2, 0.8, 2.0, 1.0, "Metrics +\nReports")
    ]

    for x, y, w, h, label in nodes:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6,
            edgecolor="#2B2D42",
            facecolor="#EDF2F4",
        )
        ax.add_patch(box)
        ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=10)

    arrow_style = dict(arrowstyle="->", lw=1.8, color="#1D3557")
    ax.annotate("", xy=(3.0, 2.9), xytext=(2.4, 2.9), arrowprops=arrow_style)
    ax.annotate("", xy=(5.6, 2.9), xytext=(5.0, 2.9), arrowprops=arrow_style)
    ax.annotate("", xy=(8.2, 2.9), xytext=(7.6, 2.9), arrowprops=arrow_style)
    ax.annotate("", xy=(10.4, 2.9), xytext=(10.2, 2.9), arrowprops=arrow_style)
    ax.annotate("", xy=(9.2, 1.8), xytext=(9.2, 2.4), arrowprops=arrow_style)

    ax.text(6.0, 3.75, "Searchless chess policy learning pipeline", ha="center", fontsize=13, fontweight="bold")
    ax.text(6.0, 0.25, "No MCTS/tree-search in training loop; evaluator uses Stockfish only for reward/benchmarking", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "pipeline_overview.svg")
    fig.savefig(ASSETS / "pipeline_overview.png", dpi=200)
    plt.close(fig)


def plot_teacher_baselines(rows: list[dict[str, float]]) -> None:
    labels = [r["model"] for r in rows]
    scores = [r["score"] for r in rows]
    elos = [r["elo"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, scores, color=["#8ECAE6", "#219EBC", "#023047"])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Score vs Stockfish skill=2")
    ax.set_title("Teacher model baseline strength (32 games each)")

    for bar, score, elo in zip(bars, scores, elos):
        ax.text(bar.get_x() + bar.get_width() / 2.0, score + 0.015, f"{score:.3f}\nElo {elo:+.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "teacher_baseline_scores.png", dpi=220)
    plt.close(fig)


def plot_distill_trend(rows: list[dict[str, float]]) -> None:
    run_labels = [r["run"] for r in rows]
    top1 = [r["top1"] for r in rows]
    top5 = [r["top5"] for r in rows]

    x = list(range(len(rows)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar([i - width / 2 for i in x], top1, width=width, label="val/top1_match", color="#E76F51")
    ax.bar([i + width / 2 for i in x], top5, width=width, label="val/top5_match", color="#2A9D8F")
    ax.set_xticks(x)
    ax.set_xticklabels(run_labels)
    ax.set_ylim(0, max(top5) * 1.25)
    ax.set_title("Distillation quality collapse across runs")
    ax.set_ylabel("Validation match")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS / "distill_quality_trend.png", dpi=220)
    plt.close(fig)


def plot_label_sparsity(k_hist: dict[int, int]) -> None:
    ks = sorted(k_hist)
    counts = [k_hist[k] for k in ks]
    total = sum(counts)
    fractions = [c / total for c in counts]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar([str(k) for k in ks], fractions, color="#F4A261")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Teacher label support k (moves per sample)")
    ax.set_ylabel("Fraction of samples")
    ax.set_title("Converted DeepMind labels are mostly k=1")

    for b, frac, count in zip(bars, fractions, counts):
        ax.text(b.get_x() + b.get_width() / 2.0, frac + 0.015, f"{frac:.1%}\n({count:,})", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS / "label_sparsity_breakdown.png", dpi=220)
    plt.close(fig)


def plot_loss_budget(rows: list[dict[str, float]]) -> None:
    steps = [str(r["step"]) for r in rows]
    ppo = [abs(r["ppo"]) for r in rows]
    kl = [abs(r["kl"]) for r in rows]
    entropy = [abs(r["entropy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(steps, entropy, label="|Entropy term|", color="#457B9D")
    ax.bar(steps, ppo, bottom=entropy, label="|PPO term|", color="#E63946")
    stacked = [e + p for e, p in zip(entropy, ppo)]
    ax.bar(steps, kl, bottom=stacked, label="|KL term|", color="#A8DADC")

    ax.set_title("GRPO loss budget composition (failure run)")
    ax.set_xlabel("Training step snapshot")
    ax.set_ylabel("Absolute contribution magnitude")
    ax.legend()

    fig.tight_layout()
    fig.savefig(ASSETS / "grpo_loss_budget_components.png", dpi=220)
    plt.close(fig)


def plot_data_coverage(phases: dict[str, float], eval_fraction: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    phase_keys = list(phases.keys())
    phase_vals = [phases[k] for k in phase_keys]
    axes[0].bar(phase_keys, phase_vals, color=["#264653", "#2A9D8F", "#E9C46A"])
    axes[0].set_ylim(0, 0.5)
    axes[0].set_title("GRPO phase sampling mix")
    axes[0].set_ylabel("Configured proportion")

    train_fraction = 1.0 - eval_fraction
    axes[1].bar(["train", "eval"], [train_fraction, eval_fraction], color=["#1D3557", "#E63946"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Pretrain split fraction")

    fig.tight_layout()
    fig.savefig(ASSETS / "data_coverage_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    baselines = parse_teacher_baselines()
    trend = parse_distill_quality_trend()
    k_hist = parse_k_hist()
    budget = parse_loss_budget_components()
    phases, eval_fraction = parse_data_coverage_values()

    draw_pipeline_diagram()
    plot_teacher_baselines(baselines)
    plot_distill_trend(trend)
    plot_label_sparsity(k_hist)
    plot_loss_budget(budget)
    plot_data_coverage(phases, eval_fraction)

    print("Generated assets:")
    for path in sorted(ASSETS.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
