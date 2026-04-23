"""Warmstart provenance records and contract for ReasoningModel.

See docs/superpowers/specs/2026-04-22-warmstart-provenance-test-design.md for
the full contract. This module only defines data structures and pure
helpers. The loader in src/reasoning/model.py is what fills the report in.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


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


@dataclass(frozen=True)
class WarmstartContract:
    name: str
    checkpoint_family: str
    allowed_new_modules: tuple[str, ...]
    allowed_new_rows: tuple[dict[str, Any], ...]
    require_fingerprints: bool = True
    allowed_warning_tags: tuple[str, ...] = ()


DM_AV_CONTRACT = WarmstartContract(
    name="dm_av_v1",
    checkpoint_family="dm_action_value",
    allowed_new_modules=("value_head", "dm.output_linear"),
    allowed_new_rows=(
        {"param": "dm.token_embedding.weight", "start": 1968, "end": 1971},
    ),
    require_fingerprints=True,
    allowed_warning_tags=(),
)

BC_CONTRACT = WarmstartContract(
    name="bc_with_dm_av_move_input_init_v1",
    checkpoint_family="dm_behavioral_cloning",
    allowed_new_modules=("value_head",),
    allowed_new_rows=(
        {"param": "dm.token_embedding.weight", "start": 1968, "end": 1971},
        {"param": "dm.output_linear.weight", "start": 1968, "end": 1971},
        {"param": "dm.output_linear.bias", "start": 1968, "end": 1971},
    ),
    require_fingerprints=True,
    allowed_warning_tags=(),
)

DM_AV_CONTRACT_LEGACY = dataclasses.replace(
    DM_AV_CONTRACT,
    require_fingerprints=False,
    allowed_warning_tags=("missing_fingerprint",),
)

BC_CONTRACT_LEGACY = dataclasses.replace(
    BC_CONTRACT,
    require_fingerprints=False,
    allowed_warning_tags=("missing_fingerprint",),
)


@dataclass
class ProvenanceReport:
    contract_version: str
    checkpoint_family: str
    sources: dict[str, dict[str, Any]]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    vocabularies: dict[str, Any] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)
