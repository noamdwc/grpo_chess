#!/usr/bin/env python3
"""Local-only parity diagnostic: JAX 9M vs PyTorch DM port.

Runs two things:

  1. Forward-pass gate — delegates to `tests/test_dm_port_parity.py`
     (float64 < 1e-10, float32 < 5e-4). If either fails, stop.
  2. 64-game Stockfish match per variant, using identical openings,
     identical colors, identical Stockfish (nodes-bounded, threads=1),
     greedy model moves. Report per-variant W/D/L + cross-variant
     move-for-move divergence.

Not collected by `pytest tests/`. Invoke explicitly:

    ~/miniconda3/envs/grpo_chess/bin/python scripts/compare_jax_pytorch_port.py \
        --report-path research_docs/runs/jax_pt_parity.md
"""
from __future__ import annotations

import os

os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import datetime as dt
import importlib.util
import io
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from src.chess.stockfish import resolve_stockfish_path
from src.eval_utils import estimate_elo_diff


REPO_ROOT = _REPO_ROOT
DM_PT_CKPT = REPO_ROOT / "checkpoints" / "dm_port" / "9M.pt"
JAX_CKPT_DIR = REPO_ROOT / "searchless_chess" / "checkpoints"
JAX_9M_CKPT = JAX_CKPT_DIR / "9M"


# --------------------------------------------------------------------------- #
# Prereq checks                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Prereqs:
    stockfish_path: str


def _check_prereqs(auto_convert: bool) -> Prereqs:
    missing = []
    if not JAX_9M_CKPT.exists():
        missing.append(
            f"JAX 9M checkpoint at {JAX_9M_CKPT} — run "
            f"`git submodule update --init` and `bash searchless_chess/checkpoints/download.sh`"
        )
    if importlib.util.find_spec("jax") is None:
        missing.append("JAX — install into the grpo_chess env")
    if not DM_PT_CKPT.exists():
        if auto_convert:
            print(f"[prereq] {DM_PT_CKPT} missing; running src.dm_port.convert_jax to produce it...")
            from src.dm_port.convert_jax import convert as convert_jax
            convert_jax(argparse.Namespace(
                model="9M",
                checkpoint_dir=str(JAX_CKPT_DIR),
                checkpoint_step=6_400_000,
                out=str(DM_PT_CKPT),
                inspect=False,
            ))
        else:
            missing.append(
                f"PT port checkpoint at {DM_PT_CKPT} — run "
                f"`python -m src.dm_port.convert_jax --model 9M --out {DM_PT_CKPT}`"
            )
    try:
        sf_path = resolve_stockfish_path(None)
    except FileNotFoundError as e:
        missing.append(str(e))
        sf_path = ""
    if missing:
        print("Missing prerequisites:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)
    return Prereqs(stockfish_path=sf_path)


# --------------------------------------------------------------------------- #
# Forward-pass gate (delegates to tests/test_dm_port_parity.py)               #
# --------------------------------------------------------------------------- #

def _run_forward_gate() -> dict[str, Any]:
    """Invoke the existing parity test functions directly. Returns result dict."""
    import test_dm_port_parity as parity
    result: dict[str, Any] = {"float64": {}, "float32": {}, "passed": True, "failures": []}

    # float64 hard gate
    try:
        torch_out = parity._torch_logits(torch.float64)
        jax_out = parity._jax_logits(parity.FIXED_FENS, dtype="float64")
        diff = float(np.abs(torch_out - jax_out).max())
        result["float64"] = {"max_abs_diff": diff, "threshold": 1e-10, "pass": diff < 1e-10}
        if diff >= 1e-10:
            result["passed"] = False
            result["failures"].append(f"float64 max_abs_diff={diff:.3e} ≥ 1e-10")
    except Exception as e:  # noqa: BLE001
        result["passed"] = False
        result["failures"].append(f"float64 gate error: {e!r}")
        result["float64"] = {"error": repr(e)}

    # float32 sanity
    try:
        torch_out = parity._torch_logits(torch.float32).astype(np.float32)
        jax_out = parity._jax_logits(parity.FIXED_FENS, dtype="float32").astype(np.float32)
        diff = float(np.abs(torch_out - jax_out).max())
        result["float32"] = {"max_abs_diff": diff, "threshold": 5e-4, "pass": diff < 5e-4}
        if diff >= 5e-4:
            result["passed"] = False
            result["failures"].append(f"float32 max_abs_diff={diff:.3e} ≥ 5e-4")
    except Exception as e:  # noqa: BLE001
        result["passed"] = False
        result["failures"].append(f"float32 gate error: {e!r}")
        result["float32"] = {"error": repr(e)}
    return result


# --------------------------------------------------------------------------- #
# Players                                                                     #
# --------------------------------------------------------------------------- #

def _build_jax_player() -> Callable[[chess.Board], chess.Move]:
    """Return a greedy JAX play(board) wrapping ActionValueEngine."""
    from src.distill.teacher import build_teacher_engine
    engine, _ = build_teacher_engine(
        model_name="9M",
        checkpoint_dir=str(JAX_CKPT_DIR),
        checkpoint_step=6_400_000,
        batch_size=1,
        use_half=False,
    )
    # ActionValueEngine selects argmax when temperature is None — greedy.
    assert engine.temperature is None, (
        f"JAX engine temperature must be None (greedy); got {engine.temperature}"
    )
    return engine.play


def _build_pt_player() -> Callable[[chess.Board], chess.Move]:
    """Greedy PT-port play(board) mirroring ActionValueEngine.analyse exactly."""
    from src.dm_port.transformer import DMTransformer, DMTransformerConfig
    from src.distill.teacher import _load_sc_module, _load_sc_engine_module
    sc_utils = _load_sc_module("utils")
    sc_tokenizer = _load_sc_module("tokenizer")
    sc_engine = _load_sc_engine_module("engine")

    ckpt = torch.load(DM_PT_CKPT, weights_only=False, map_location="cpu")
    cfg = DMTransformerConfig(**ckpt["config"])
    model = DMTransformer(cfg).eval()
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"PT port state_dict mismatch: missing={missing} unexpected={unexpected}")

    _, bucket_values = sc_utils.get_uniform_buckets_edges_values(128)

    def play(board: chess.Board) -> chess.Move:
        sorted_legal = sc_engine.get_ordered_legal_moves(board)
        legal_actions = np.array(
            [sc_utils.MOVE_TO_ACTION[m.uci()] for m in sorted_legal], dtype=np.int32
        )[:, None]
        dummy_buckets = np.zeros((len(legal_actions), 1), dtype=np.int32)
        tok_fen = np.asarray(sc_tokenizer.tokenize(board.fen()), dtype=np.int32)
        seqs = np.concatenate(
            [np.stack([tok_fen] * len(legal_actions)), legal_actions, dummy_buckets],
            axis=1,
        )
        with torch.no_grad():
            log_probs = model(torch.from_numpy(seqs).long())[:, -1, :].numpy()
        probs = np.exp(log_probs)
        win_probs = probs @ bucket_values
        # Repetition tweak: moves leading to claimable 3x/5x → 50% win prob.
        for i, mv in enumerate(sorted_legal):
            board.push(mv)
            if board.is_fivefold_repetition() or board.can_claim_threefold_repetition():
                win_probs[i] = 0.5
            board.pop()
        return sorted_legal[int(np.argmax(win_probs))]

    return play


# --------------------------------------------------------------------------- #
# Match                                                                       #
# --------------------------------------------------------------------------- #

# Reuse the 32 curated FENs from the forward-pass parity test as openings.
def _opening_set() -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    import test_dm_port_parity as parity
    return list(parity.FIXED_FENS)


@dataclass
class GameRecord:
    opening_idx: int
    opening_fen: str
    model_is_white: bool
    model_moves: list[str] = field(default_factory=list)  # UCI, in play order
    full_moves: list[str] = field(default_factory=list)   # UCI, all plies
    result: str = "*"       # "1-0", "0-1", "1/2-1/2", "*"
    termination: str = ""
    pgn: str = ""


def _open_stockfish(path: str, skill: int, hash_mb: int) -> chess.engine.SimpleEngine:
    sf = chess.engine.SimpleEngine.popen_uci(path)
    sf.configure({"Threads": 1, "Hash": hash_mb, "Skill Level": skill})
    return sf


def _play_game(
    model_play: Callable[[chess.Board], chess.Move],
    stockfish: chess.engine.SimpleEngine,
    opening_fen: str,
    model_is_white: bool,
    nodes: int,
    max_plies: int,
) -> GameRecord:
    board = chess.Board(opening_fen)
    rec = GameRecord(opening_idx=-1, opening_fen=opening_fen, model_is_white=model_is_white)
    limit = chess.engine.Limit(nodes=nodes)

    while not board.is_game_over(claim_draw=True) and board.ply() - chess.Board(opening_fen).ply() < max_plies:
        model_to_move = (board.turn == chess.WHITE) == model_is_white
        if model_to_move:
            mv = model_play(board)
            rec.model_moves.append(mv.uci())
        else:
            sf_res = stockfish.play(board, limit)
            mv = sf_res.move
            if mv is None:
                break
        rec.full_moves.append(mv.uci())
        board.push(mv)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        rec.result = "*"
        rec.termination = "max_plies" if not board.is_game_over(claim_draw=True) else "unknown"
    else:
        rec.result = outcome.result()
        rec.termination = outcome.termination.name

    game = chess.pgn.Game()
    game.headers["FEN"] = opening_fen
    game.headers["SetUp"] = "1"
    game.headers["Result"] = rec.result
    game.headers["White"] = "Model" if model_is_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_is_white else "Model"
    node = game
    b = chess.Board(opening_fen)
    for uci in rec.full_moves:
        mv = chess.Move.from_uci(uci)
        node = node.add_variation(mv)
        b.push(mv)
    buf = io.StringIO()
    game.accept(chess.pgn.FileExporter(buf))
    rec.pgn = buf.getvalue()
    return rec


def _match(
    model_play: Callable[[chess.Board], chess.Move],
    sf_path: str,
    openings: list[str],
    num_games: int,
    nodes: int,
    skill: int,
    hash_mb: int,
    max_plies: int,
) -> list[GameRecord]:
    # num_games = 2 * len(openings)  at the default 64
    games: list[GameRecord] = []
    sf = _open_stockfish(sf_path, skill=skill, hash_mb=hash_mb)
    try:
        for g in range(num_games):
            opening_idx = (g // 2) % len(openings)
            model_is_white = (g % 2 == 0)
            rec = _play_game(
                model_play, sf,
                opening_fen=openings[opening_idx],
                model_is_white=model_is_white,
                nodes=nodes,
                max_plies=max_plies,
            )
            rec.opening_idx = opening_idx
            games.append(rec)
            print(f"  game {g+1}/{num_games} opening={opening_idx} model_white={model_is_white} "
                  f"result={rec.result} plies={len(rec.full_moves)} term={rec.termination}")
    finally:
        sf.quit()
    return games


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def _aggregate_wdl(games: list[GameRecord]) -> dict[str, Any]:
    w = d = l = 0
    term_counts: Counter[str] = Counter()
    for g in games:
        term_counts[g.termination] += 1
        if g.result == "1-0":
            if g.model_is_white: w += 1
            else: l += 1
        elif g.result == "0-1":
            if g.model_is_white: l += 1
            else: w += 1
        elif g.result == "1/2-1/2":
            d += 1
        # "*" counted as neither
    total = w + d + l
    score = (w + 0.5 * d) / total if total else 0.0
    return {
        "games": total,
        "wins": w, "draws": d, "losses": l,
        "score": score,
        "elo_diff_approx": estimate_elo_diff(score) if total else 0.0,
        "terminations": dict(term_counts),
        "unfinished": sum(1 for g in games if g.result == "*"),
    }


def _pairwise_divergence(jax_games: list[GameRecord], pt_games: list[GameRecord]) -> dict[str, Any]:
    assert len(jax_games) == len(pt_games)
    diverged = 0
    first_div_plies: list[int] = []
    result_xtab: Counter[tuple[str, str]] = Counter()
    examples: list[dict[str, Any]] = []
    for i, (j, p) in enumerate(zip(jax_games, pt_games)):
        result_xtab[(j.result, p.result)] += 1
        n = min(len(j.model_moves), len(p.model_moves))
        first = None
        for k in range(n):
            if j.model_moves[k] != p.model_moves[k]:
                first = k
                break
        if first is None and len(j.model_moves) != len(p.model_moves):
            first = n  # one side played longer
        if first is not None:
            diverged += 1
            first_div_plies.append(first)
            if len(examples) < 5:
                examples.append({
                    "game": i,
                    "opening_idx": j.opening_idx,
                    "model_is_white": j.model_is_white,
                    "first_divergence_model_ply": first,
                    "jax_move": j.model_moves[first] if first < len(j.model_moves) else None,
                    "pt_move": p.model_moves[first] if first < len(p.model_moves) else None,
                })
    hist: Counter[int] = Counter(first_div_plies)
    return {
        "diverged_games": diverged,
        "total_games": len(jax_games),
        "first_divergence_ply_histogram": dict(sorted(hist.items())),
        "first_divergence_ply_stats": (
            {
                "min": min(first_div_plies),
                "p50": statistics.median(first_div_plies),
                "max": max(first_div_plies),
            } if first_div_plies else {}
        ),
        "result_crosstab": {f"{k[0]}|{k[1]}": v for k, v in sorted(result_xtab.items())},
        "examples": examples,
    }


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #

def _render_report(
    args: argparse.Namespace,
    sf_version: str,
    forward: dict[str, Any],
    jax_wdl: dict[str, Any] | None,
    pt_wdl: dict[str, Any] | None,
    pairwise: dict[str, Any] | None,
    overall_pass: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# JAX vs PyTorch DM-port parity report")
    lines.append(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Configuration")
    lines.append(f"- Stockfish: `{sf_version}`")
    lines.append(f"- Nodes/move: {args.stockfish_nodes}")
    lines.append(f"- Skill Level: {args.stockfish_skill}")
    lines.append(f"- Hash (MB): {args.stockfish_hash_mb}  |  Threads: 1")
    lines.append(f"- Games per variant: {args.num_games}")
    lines.append(f"- PT checkpoint: `{DM_PT_CKPT}`")
    lines.append(f"- JAX checkpoint: `{JAX_9M_CKPT}`\n")

    lines.append("## Forward-pass gate")
    if forward.get("skipped"):
        lines.append("- _SKIPPED via --skip-forward-gate_")
    if "float64" in forward and forward["float64"]:
        fd = forward["float64"]
        if "max_abs_diff" in fd:
            lines.append(f"- float64 max_abs_diff = {fd['max_abs_diff']:.3e}  (threshold 1e-10) → {'PASS' if fd['pass'] else 'FAIL'}")
        else:
            lines.append(f"- float64 gate error: {fd.get('error')}")
    if "float32" in forward and forward["float32"]:
        fd = forward["float32"]
        if "max_abs_diff" in fd:
            lines.append(f"- float32 max_abs_diff = {fd['max_abs_diff']:.3e}  (threshold 5e-4) → {'PASS' if fd['pass'] else 'FAIL'}")
        else:
            lines.append(f"- float32 gate error: {fd.get('error')}")
    if forward.get("failures"):
        lines.append(f"\n**Gate failures:** {forward['failures']}")

    if jax_wdl is not None and pt_wdl is not None:
        lines.append("\n## Per-variant match results")
        for name, wdl in [("JAX", jax_wdl), ("PT port", pt_wdl)]:
            lines.append(
                f"- **{name}**: W/D/L = {wdl['wins']}/{wdl['draws']}/{wdl['losses']}  "
                f"score = {wdl['score']:.3f}  Elo Δ = {wdl['elo_diff_approx']:+.0f}  "
                f"unfinished = {wdl['unfinished']}"
            )
            lines.append(f"  - terminations: {wdl['terminations']}")

    if pairwise is not None:
        lines.append("\n## Cross-variant move-for-move")
        lines.append(f"- Diverged games: **{pairwise['diverged_games']} / {pairwise['total_games']}**")
        if pairwise["first_divergence_ply_stats"]:
            s = pairwise["first_divergence_ply_stats"]
            lines.append(f"- First-divergence ply (model-side): min={s['min']} p50={s['p50']} max={s['max']}")
        lines.append(f"- First-divergence ply histogram: `{pairwise['first_divergence_ply_histogram']}`")
        lines.append(f"- Result cross-tab (jax|pt): `{pairwise['result_crosstab']}`")
        if pairwise["examples"]:
            lines.append("- Examples:")
            for ex in pairwise["examples"]:
                lines.append(f"  - game {ex['game']} op{ex['opening_idx']} mw={ex['model_is_white']} "
                             f"ply={ex['first_divergence_model_ply']} jax={ex['jax_move']} pt={ex['pt_move']}")

    lines.append("\n## Verdict")
    lines.append(f"**{'PASS' if overall_pass else 'FAIL'}**")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-games", type=int, default=64)
    p.add_argument("--stockfish-nodes", type=int, default=100_000)
    p.add_argument("--stockfish-skill", type=int, default=20)
    p.add_argument("--stockfish-hash-mb", type=int, default=16)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report-path", default="research_docs/runs/jax_pt_parity.md")
    p.add_argument("--skip-forward-gate", action="store_true",
                   help="Skip the forward-pass precondition (useful for iterating on the match code).")
    p.add_argument("--no-auto-convert", action="store_true",
                   help="Do not auto-run convert_jax if the PT checkpoint is missing.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    prereqs = _check_prereqs(auto_convert=not args.no_auto_convert)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Forward-pass gate
    if args.skip_forward_gate:
        print("[forward-gate] SKIPPED by flag")
        forward = {"passed": True, "skipped": True, "failures": [], "float64": {}, "float32": {}}
    else:
        print("[forward-gate] Running float64 and float32 parity checks (this loads JAX)...")
        forward = _run_forward_gate()
        for dtype in ("float64", "float32"):
            fd = forward[dtype]
            if "max_abs_diff" in fd:
                print(f"[forward-gate] {dtype}: max_abs_diff = {fd['max_abs_diff']:.3e} → {'PASS' if fd['pass'] else 'FAIL'}")
            else:
                print(f"[forward-gate] {dtype}: error = {fd.get('error')}")
        if not forward["passed"]:
            report = _render_report(
                args=args, sf_version="(not launched)", forward=forward,
                jax_wdl=None, pt_wdl=None, pairwise=None, overall_pass=False,
            )
            _write_report(args.report_path, report)
            print("\n" + report)
            return 1

    # Stockfish launch + version capture
    sf = _open_stockfish(prereqs.stockfish_path, skill=args.stockfish_skill, hash_mb=args.stockfish_hash_mb)
    sf_version = getattr(sf, "id", {}).get("name", "unknown")
    sf.quit()
    print(f"[stockfish] {sf_version} at {prereqs.stockfish_path}")

    # Build players
    print("[players] Building JAX engine...")
    jax_play = _build_jax_player()
    print("[players] Building PT-port engine...")
    pt_play = _build_pt_player()

    # Openings
    openings = _opening_set()
    needed_openings = (args.num_games + 1) // 2
    if needed_openings > len(openings):
        raise ValueError(f"Need {needed_openings} openings; only have {len(openings)}. "
                         "Reduce --num-games or extend the opening set.")

    # Matches
    print(f"\n[match] JAX vs Stockfish ({args.num_games} games)...")
    jax_games = _match(
        jax_play, prereqs.stockfish_path, openings[:needed_openings],
        num_games=args.num_games, nodes=args.stockfish_nodes,
        skill=args.stockfish_skill, hash_mb=args.stockfish_hash_mb,
        max_plies=args.max_plies,
    )
    print(f"\n[match] PT port vs Stockfish ({args.num_games} games)...")
    pt_games = _match(
        pt_play, prereqs.stockfish_path, openings[:needed_openings],
        num_games=args.num_games, nodes=args.stockfish_nodes,
        skill=args.stockfish_skill, hash_mb=args.stockfish_hash_mb,
        max_plies=args.max_plies,
    )

    jax_wdl = _aggregate_wdl(jax_games)
    pt_wdl = _aggregate_wdl(pt_games)
    pairwise = _pairwise_divergence(jax_games, pt_games)

    overall_pass = forward.get("passed", True) and pairwise["diverged_games"] == 0
    report = _render_report(args, sf_version, forward, jax_wdl, pt_wdl, pairwise, overall_pass)
    _write_report(args.report_path, report)
    print("\n" + report)

    # Also save JSON alongside
    json_path = Path(args.report_path).with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "forward": forward,
        "jax_wdl": jax_wdl,
        "pt_wdl": pt_wdl,
        "pairwise": pairwise,
        "stockfish_version": sf_version,
        "args": vars(args),
    }, indent=2, default=str))
    print(f"[report] Markdown: {args.report_path}")
    print(f"[report] JSON:     {json_path}")

    return 0 if overall_pass else 1


def _write_report(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


if __name__ == "__main__":
    sys.exit(main())
