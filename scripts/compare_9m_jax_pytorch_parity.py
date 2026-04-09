#!/usr/bin/env python3
"""Compare DeepMind JAX 9M checkpoint against converted PyTorch 9M model.

This script runs two parity checks:
1. Move-level parity on a deterministic FEN suite.
2. Aggregate Stockfish evaluation parity.

Exit code is 0 only when both parity checks pass configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

# Ensure repo root is on sys.path for src.* imports when executed as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.chess.stockfish import StockfishConfig, StockfishPlayer, resolve_stockfish_path
from src.distill.convert_jax_checkpoint import convert as convert_jax_checkpoint
from src.distill.teacher import DeepMindPlayer, build_teacher_engine
from src.eval_utils import EvalConfig, evaluate_policy_vs_stockfish
from src.models_9m import Searchless9MTransformer
from src.searchless_chess_imports import MOVE_TO_ACTION, tokenize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare JAX 9M checkpoint parity with PyTorch 9M conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", default="searchless_chess/checkpoints")
    parser.add_argument("--checkpoint-step", type=int, default=6_400_000)
    parser.add_argument("--model-name", default="9M", choices=["9M"])

    parser.add_argument(
        "--pytorch-checkpoint",
        default=None,
        help="Optional path to converted PyTorch checkpoint (.pt).",
    )
    parser.add_argument(
        "--auto-convert-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-convert from JAX if --pytorch-checkpoint is missing.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--fen-count", type=int, default=200)
    parser.add_argument("--fen-seed", type=int, default=0)
    parser.add_argument("--fen-max-random-plies", type=int, default=30)

    parser.add_argument("--stockfish-games", type=int, default=8)
    parser.add_argument("--stockfish-skill-level", type=int, default=2)
    parser.add_argument("--stockfish-movetime-ms", type=int, default=20)
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--max-plies", type=int, default=200)

    parser.add_argument("--move-match-threshold", type=float, default=1.0)
    parser.add_argument("--score-delta-threshold", type=float, default=0.0)

    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def _bucket_values(num_buckets: int = 128) -> np.ndarray:
    full = np.linspace(0.0, 1.0, num_buckets + 1)
    return (full[:-1] + full[1:]) / 2


def _ensure_pytorch_checkpoint(args: argparse.Namespace) -> Path:
    if args.pytorch_checkpoint:
        checkpoint_path = Path(args.pytorch_checkpoint)
        if checkpoint_path.exists():
            return checkpoint_path
        if not args.auto_convert_if_missing:
            raise FileNotFoundError(
                f"PyTorch checkpoint not found and auto-convert disabled: {checkpoint_path}"
            )
    else:
        checkpoint_path = None
        if not args.auto_convert_if_missing:
            raise ValueError("--pytorch-checkpoint is required when --no-auto-convert-if-missing is used")

    out_path = Path(tempfile.mkdtemp(prefix="grpo_9m_parity_")) / "jax_9m_converted.pt"
    convert_jax_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        step=args.checkpoint_step,
        output=str(out_path),
        model_name=args.model_name,
    )
    return out_path


def _normalize_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_pytorch_model(checkpoint_path: Path, device: torch.device) -> Searchless9MTransformer:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = Searchless9MTransformer().to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Strict checkpoint load failed for Searchless9MTransformer. "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


class Pytorch9MPlayer:
    """PyTorch mirror of DeepMind action-value move selection used by DeepMindPlayer."""

    def __init__(self, model: Searchless9MTransformer, bucket_values: np.ndarray, device: torch.device):
        self._model = model
        self._bucket_values = bucket_values.astype(np.float64, copy=False)
        self._device = device

    @staticmethod
    def _build_targets(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
        sorted_legal = sorted(
            [m for m in board.legal_moves if m.uci() in MOVE_TO_ACTION],
            key=lambda m: MOVE_TO_ACTION[m.uci()],
        )
        if not sorted_legal:
            return [], np.empty((0, 79), dtype=np.int64)

        legal_actions = np.array([MOVE_TO_ACTION[m.uci()] for m in sorted_legal], dtype=np.int64)
        legal_actions = np.expand_dims(legal_actions, axis=-1)  # [N, 1]
        dummy_return_bucket = np.zeros((len(sorted_legal), 1), dtype=np.int64)  # [N, 1]

        tokenized_fen = tokenize(board.fen()).astype(np.int64)  # [77]
        fen_stack = np.stack([tokenized_fen] * len(sorted_legal), axis=0)  # [N, 77]

        targets = np.concatenate([fen_stack, legal_actions, dummy_return_bucket], axis=1)  # [N, 79]
        return sorted_legal, targets

    @staticmethod
    def _shift_right(targets: np.ndarray) -> np.ndarray:
        # Match searchless_chess/src/transformer.py shift_right exactly.
        bos = np.zeros((targets.shape[0], 1), dtype=np.int64)
        padded = np.concatenate([bos, targets], axis=1)
        return padded[:, :-1]

    @torch.no_grad()
    def act(self, board: chess.Board) -> chess.Move:
        sorted_legal, targets = self._build_targets(board)
        if not sorted_legal:
            # Rare fallback when a legal move is missing from MOVE_TO_ACTION.
            return random.choice(list(board.legal_moves))

        shifted_inputs = self._shift_right(targets)
        x = torch.from_numpy(shifted_inputs).to(self._device, dtype=torch.long)
        log_probs = self._model(x)[:, -1, :].detach().cpu().numpy()  # [N, 128]
        win_probs = np.exp(log_probs) @ self._bucket_values
        best_idx = int(np.argmax(win_probs))
        return sorted_legal[best_idx]


def _generate_fens(count: int, seed: int, max_random_plies: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    attempts = 0
    max_attempts = max(10 * count, 500)

    while len(fens) < count and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        plies = rng.randint(0, max_random_plies)
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))

        if board.is_game_over(claim_draw=True):
            continue

        mapped_legal_count = sum(1 for m in board.legal_moves if m.uci() in MOVE_TO_ACTION)
        if mapped_legal_count < 2:
            continue

        fens.append(board.fen())

    if len(fens) < count:
        raise RuntimeError(
            f"Failed to generate enough valid FENs: requested={count}, generated={len(fens)}"
        )
    return fens


def _compare_moves_on_fens(
    fens: list[str],
    jax_player: DeepMindPlayer,
    torch_player: Pytorch9MPlayer,
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    matches = 0

    for fen in fens:
        board = chess.Board(fen)
        jax_move = jax_player.act(board).uci()
        torch_move = torch_player.act(chess.Board(fen)).uci()
        if jax_move == torch_move:
            matches += 1
        else:
            mismatches.append({"fen": fen, "jax_move": jax_move, "pytorch_move": torch_move})

    total = len(fens)
    match_rate = matches / total if total else 0.0
    return {
        "total_positions": total,
        "matches": matches,
        "mismatches": total - matches,
        "match_rate": match_rate,
        "mismatch_examples": mismatches[:20],
    }


def _run_stockfish_eval(
    player: Any,
    eval_cfg: EvalConfig,
    sf_cfg: StockfishConfig,
) -> dict[str, Any]:
    results, _, _ = evaluate_policy_vs_stockfish(player, StockfishPlayer(sf_cfg), eval_cfg)
    return {
        "games": results["games"],
        "wins": results["wins"],
        "draws": results["draws"],
        "losses": results["losses"],
        "score": results["score"],
        "elo_diff_vs_stockfish_approx": results["elo_diff_vs_stockfish_approx"],
        "termination_reasons": results["termination_reasons"],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> None:
    args = _parse_args()
    device = _normalize_device(args.device)

    checkpoint_path = _ensure_pytorch_checkpoint(args)

    jax_engine, bucket_values = build_teacher_engine(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        batch_size=1,
        use_half=False,
    )
    jax_player = DeepMindPlayer(jax_engine, bucket_values)

    pytorch_model = _load_pytorch_model(checkpoint_path, device=device)
    torch_player = Pytorch9MPlayer(pytorch_model, bucket_values=bucket_values, device=device)

    fens = _generate_fens(
        count=args.fen_count,
        seed=args.fen_seed,
        max_random_plies=args.fen_max_random_plies,
    )

    move_parity = _compare_moves_on_fens(fens=fens, jax_player=jax_player, torch_player=torch_player)

    sf_path = resolve_stockfish_path(args.stockfish_path)
    eval_cfg = EvalConfig(
        games=args.stockfish_games,
        seed=args.fen_seed,
        max_plies=args.max_plies,
        randomize_opening=False,
        opening_plies=0,
    )
    sf_cfg = StockfishConfig(
        path=sf_path,
        skill_level=args.stockfish_skill_level,
        movetime_ms=args.stockfish_movetime_ms,
        threads=1,
        hash_mb=128,
    )

    jax_eval = _run_stockfish_eval(jax_player, eval_cfg=eval_cfg, sf_cfg=sf_cfg)
    torch_eval = _run_stockfish_eval(torch_player, eval_cfg=eval_cfg, sf_cfg=sf_cfg)

    score_delta = abs(float(jax_eval["score"]) - float(torch_eval["score"]))

    gates = {
        "move_parity_pass": move_parity["match_rate"] >= args.move_match_threshold,
        "stockfish_score_parity_pass": score_delta <= args.score_delta_threshold,
    }
    gates["overall_pass"] = gates["move_parity_pass"] and gates["stockfish_score_parity_pass"]

    report = {
        "inputs": {
            "checkpoint_dir": args.checkpoint_dir,
            "checkpoint_step": args.checkpoint_step,
            "model_name": args.model_name,
            "pytorch_checkpoint": str(checkpoint_path),
            "device": str(device),
            "fen_count": args.fen_count,
            "fen_seed": args.fen_seed,
            "fen_max_random_plies": args.fen_max_random_plies,
            "move_match_threshold": args.move_match_threshold,
            "stockfish_games": args.stockfish_games,
            "stockfish_skill_level": args.stockfish_skill_level,
            "stockfish_movetime_ms": args.stockfish_movetime_ms,
            "score_delta_threshold": args.score_delta_threshold,
            "stockfish_path": sf_path,
            "eval_cfg": asdict(eval_cfg),
            "stockfish_cfg": asdict(sf_cfg),
        },
        "move_parity": move_parity,
        "stockfish_eval": {
            "jax": jax_eval,
            "pytorch": torch_eval,
            "abs_score_delta": score_delta,
            "wins_delta": int(jax_eval["wins"]) - int(torch_eval["wins"]),
            "draws_delta": int(jax_eval["draws"]) - int(torch_eval["draws"]),
            "losses_delta": int(jax_eval["losses"]) - int(torch_eval["losses"]),
        },
        "gates": gates,
    }

    print("\n=== Move-level parity ===")
    print(
        f"positions={move_parity['total_positions']} "
        f"match_rate={move_parity['match_rate']:.4f} "
        f"matches={move_parity['matches']} mismatches={move_parity['mismatches']}"
    )
    if move_parity["mismatch_examples"]:
        print("mismatch_examples (up to 5):")
        for item in move_parity["mismatch_examples"][:5]:
            print(
                f"  fen={item['fen']} | "
                f"jax={item['jax_move']} | pytorch={item['pytorch_move']}"
            )

    print("\n=== Stockfish parity ===")
    print(
        f"jax_score={jax_eval['score']:.4f} pytorch_score={torch_eval['score']:.4f} "
        f"abs_delta={score_delta:.4f}"
    )
    print(
        f"jax W/D/L={jax_eval['wins']}/{jax_eval['draws']}/{jax_eval['losses']} | "
        f"pytorch W/D/L={torch_eval['wins']}/{torch_eval['draws']}/{torch_eval['losses']}"
    )

    print("\n=== Gates ===")
    print(json.dumps(gates, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"Saved parity report: {out_path}")

    raise SystemExit(0 if gates["overall_pass"] else 1)


if __name__ == "__main__":
    main()
