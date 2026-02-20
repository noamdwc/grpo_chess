#!/usr/bin/env python3
"""Load a checkpoint and evaluate it against Stockfish."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.chess.stockfish import StockfishManager
from src.configs.config_loader import load_experiment_config
from src.evaluator import Evaluator
from src.models import ChessTransformer


def _extract_model_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    """Extract policy model weights from supported checkpoint formats."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        if not isinstance(sd, dict):
            raise ValueError("checkpoint['state_dict'] is not a dict")

        # Lightning checkpoints can prefix keys with module names.
        extracted: dict[str, torch.Tensor] = {}
        for key, value in sd.items():
            if key.startswith("policy_model."):
                extracted[key[len("policy_model."):]] = value
            elif key.startswith("model."):
                extracted[key[len("model."):]] = value

        if extracted:
            return extracted
        return sd

    if isinstance(ckpt, dict):
        # Assume raw state dict.
        return ckpt

    raise ValueError(f"Unsupported checkpoint object type: {type(ckpt)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="default.yaml", help="Config YAML (under src/configs)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--games", type=int, default=None, help="Override eval.games")
    parser.add_argument("--max-plies", type=int, default=None, help="Override eval.max_plies")
    parser.add_argument("--seed", type=int, default=None, help="Override eval.seed")
    parser.add_argument("--randomize-opening", dest="randomize_opening", action="store_true", help="Enable random opening moves")
    parser.add_argument("--no-randomize-opening", dest="randomize_opening", action="store_false", help="Disable random opening moves")
    parser.set_defaults(randomize_opening=None)
    parser.add_argument("--opening-plies", type=int, default=None, help="Override eval.opening_plies")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Override stockfish.path")
    parser.add_argument("--stockfish-skill-level", type=int, default=None, help="Override stockfish.skill_level")
    parser.add_argument("--stockfish-movetime-ms", type=int, default=None, help="Override stockfish.movetime_ms")
    parser.add_argument("--disable-search", action="store_true", help="Ignore searcher config even if present")
    parser.add_argument("--save-pgn", type=str, default=None, help="Optional path to save PGNs")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save results JSON")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_experiment_config(args.config)
    eval_cfg = config.eval
    stockfish_cfg = config.stockfish
    searcher_cfg = None if args.disable_search else config.searcher

    if args.games is not None:
        eval_cfg = replace(eval_cfg, games=args.games)
    if args.max_plies is not None:
        eval_cfg = replace(eval_cfg, max_plies=args.max_plies)
    if args.seed is not None:
        eval_cfg = replace(eval_cfg, seed=args.seed)
    if args.randomize_opening is not None:
        eval_cfg = replace(eval_cfg, randomize_opening=args.randomize_opening)
    if args.opening_plies is not None:
        eval_cfg = replace(eval_cfg, opening_plies=args.opening_plies)

    if args.stockfish_path is not None:
        stockfish_cfg = replace(stockfish_cfg, path=args.stockfish_path)
    if args.stockfish_skill_level is not None:
        stockfish_cfg = replace(stockfish_cfg, skill_level=args.stockfish_skill_level)
    if args.stockfish_movetime_ms is not None:
        stockfish_cfg = replace(stockfish_cfg, movetime_ms=args.stockfish_movetime_ms)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Eval config: games={eval_cfg.games}, seed={eval_cfg.seed}, max_plies={eval_cfg.max_plies}")
    print(
        "Stockfish config: "
        f"path={stockfish_cfg.path}, skill={stockfish_cfg.skill_level}, movetime_ms={stockfish_cfg.movetime_ms}"
    )

    model = ChessTransformer(config.transformer).to(device)
    model.eval()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = _extract_model_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    evaluator = Evaluator(
        eval_cfg=eval_cfg,
        policy_cfg=config.policy,
        searcher_cfg=searcher_cfg,
        stockfish_cfg=stockfish_cfg,
    )

    try:
        results, _, pgns = evaluator.single_evaluation(model)
    finally:
        # Ensure no dangling engines on interrupted runs.
        StockfishManager.close_all()

    print("\nEvaluation results:")
    print(json.dumps(results, indent=2, default=str))

    if args.save_pgn:
        pgn_path = Path(args.save_pgn)
        pgn_path.parent.mkdir(parents=True, exist_ok=True)
        pgn_path.write_text("\n\n".join(pgns))
        print(f"Saved PGNs to: {pgn_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"Saved results JSON to: {json_path}")


if __name__ == "__main__":
    main()
