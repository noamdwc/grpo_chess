"""Warmstart provenance records and contract for ReasoningModel.

See docs/superpowers/specs/2026-04-22-warmstart-provenance-test-design.md for
the full contract. This module only defines data structures and pure
helpers. The loader in src/reasoning/model.py is what fills the report in.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _searchless_src_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "searchless_chess" / "src"
        if (candidate / "utils.py").exists() and (candidate / "tokenizer.py").exists():
            return candidate
    raise FileNotFoundError("Unable to locate searchless_chess/src for warmstart fingerprints")


def _load_searchless_module(name: str):
    src_dir = _searchless_src_dir()
    spec = importlib.util.spec_from_file_location(f"searchless_{name}", src_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load searchless_chess module {name!r}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical_fen_characters() -> list[str]:
    tokenizer_path = _searchless_src_dir() / "tokenizer.py"
    module = ast.parse(tokenizer_path.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_CHARACTERS":
                    return list(ast.literal_eval(node.value))
    raise ValueError(f"Unable to parse _CHARACTERS from {tokenizer_path}")


def canonical_fen_tokenizer_fingerprint() -> str:
    """Hash of the canonical FEN character->ID mapping."""
    characters = _canonical_fen_characters()
    payload = "\n".join(f"{i}\t{c}" for i, c in enumerate(characters)).encode("utf-8")
    return _sha256(payload)


def canonical_action_vocab_fingerprint() -> str:
    """Hash of the canonical UCI action vocabulary."""
    move_to_action = dict(_load_searchless_module("utils").MOVE_TO_ACTION)

    ordered = sorted(move_to_action.items(), key=lambda kv: kv[1])
    payload = "\n".join(f"{idx}\t{move}" for move, idx in ordered).encode("utf-8")
    return _sha256(payload)
