"""Evaluate the DeepMind teacher model vs Stockfish.

Usage:
    python -m src.distill.eval_teacher
    python -m src.distill.eval_teacher --config distill.yaml
"""
import argparse

from src.configs.config_loader import load_yaml_file, dict_to_dataclass
from src.distill.teacher import build_teacher_engine, DeepMindPlayer
from src.distill.generate_dataset import GenerateConfig
from src.eval_utils import EvalConfig, evaluate_policy_vs_stockfish
from src.chess.stockfish import StockfishPlayer, StockfishConfig, resolve_stockfish_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepMind teacher vs Stockfish")
    parser.add_argument("--config", default="distill.yaml")
    args = parser.parse_args()

    data = load_yaml_file(args.config)
    gen_cfg = dict_to_dataclass(GenerateConfig, data.get("generate", {}))
    eval_cfg = dict_to_dataclass(EvalConfig, data.get("eval", {}))
    sf_cfg = dict_to_dataclass(StockfishConfig, data.get("stockfish", {}))

    # Resolve Stockfish binary path (handles local vs server installs)
    resolved_path = resolve_stockfish_path(sf_cfg.path)
    from dataclasses import replace
    sf_cfg = replace(sf_cfg, path=resolved_path)

    engine, bucket_values = build_teacher_engine(
        model_name=gen_cfg.teacher_model,
        checkpoint_dir=gen_cfg.checkpoint_dir,
        checkpoint_step=gen_cfg.checkpoint_step,
        batch_size=1,
        use_half=True,
    )

    results, _, _ = evaluate_policy_vs_stockfish(
        DeepMindPlayer(engine, bucket_values),
        StockfishPlayer(sf_cfg),
        eval_cfg,
    )
    print(f"\nDeepMind {gen_cfg.teacher_model} vs Stockfish skill={sf_cfg.skill_level}")
    print(f"  Score: {results['score']:.3f}  ({results['wins']}W / {results['draws']}D / {results['losses']}L)")
    print(f"  Elo diff: {results['elo_diff_vs_stockfish_approx']:+.0f}")


if __name__ == "__main__":
    main()
