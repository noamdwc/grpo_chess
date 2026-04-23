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

    def verify(self, target_state: dict[str, Any], loaded_sources: dict[str, dict[str, Any]]) -> "ProvenanceReport":
        """Compare claimed provenance spans against actual tensors."""
        import torch as _torch

        for param_name, pspec in self.params.items():
            tgt = target_state[param_name]
            if pspec["status"] == "new_init":
                pspec["verified_not_equal_to_any_source"] = _verify_new_init(tgt, loaded_sources)
                continue
            if pspec["status"] == "full_copy":
                src_key = pspec["source_key"]
                src = loaded_sources[pspec["source"]][src_key]
                pspec["verified_equal"] = bool(_torch.equal(tgt, src))
                continue
            for span in pspec.get("spans", []):
                ts, te = span["target_start"], span["target_end"]
                tgt_slice = tgt[ts:te]
                if span["source"] == "new_init":
                    span["verified_not_equal_to_any_source"] = _verify_new_init(tgt_slice, loaded_sources)
                    continue
                if span["source"] == "copy_of_last_pretrained_row":
                    source_name = _find_source_with_key(loaded_sources, span["source_key"])
                    src = loaded_sources[source_name][span["source_key"]]
                    ss, se = span["source_start"], span["source_end"]
                    expected = src[ss:se].expand_as(tgt_slice)
                    span["verified_equal"] = bool(_torch.equal(tgt_slice, expected))
                    continue
                src = loaded_sources[span["source"]][span["source_key"]]
                ss, se = span["source_start"], span["source_end"]
                span["verified_equal"] = bool(_torch.equal(tgt_slice, src[ss:se]))
        return self

    def assert_contract(self, contract: WarmstartContract) -> None:
        """Assert that this provenance report satisfies the warmstart contract."""
        errors: list[str] = []
        if self.checkpoint_family != contract.checkpoint_family:
            errors.append(
                f"checkpoint_family mismatch: report={self.checkpoint_family!r} "
                f"contract={contract.checkpoint_family!r} ({contract.name})"
            )

        allowed_rows: dict[str, list[tuple[int, int]]] = {}
        for entry in contract.allowed_new_rows:
            allowed_rows.setdefault(entry["param"], []).append((entry["start"], entry["end"]))

        for pname, pspec in self.params.items():
            if pspec["status"] == "full_copy":
                if not pspec.get("verified_equal", False):
                    errors.append(f"verification failed (full_copy): {pname}")
                continue

            if pspec["status"] == "new_init":
                module = pname.rsplit(".", 1)[0]
                allowed_module = any(
                    pname == allowed
                    or module == allowed
                    or module.startswith(allowed + ".")
                    or pname.startswith(allowed + ".")
                    for allowed in contract.allowed_new_modules
                )
                if not allowed_module:
                    top = pname.split(".", 1)[0]
                    if (
                        top not in contract.allowed_new_modules
                        and module not in contract.allowed_new_modules
                        and not any(pname.startswith(allowed + ".") for allowed in contract.allowed_new_modules)
                    ):
                        errors.append(
                            f"unexpected new_init parameter: {pname} "
                            f"(not in allowed_new_modules={list(contract.allowed_new_modules)})"
                        )
                continue

            for span in pspec.get("spans", []):
                if span["source"] == "new_init":
                    ts, te = span["target_start"], span["target_end"]
                    param_allowlist = allowed_rows.get(pname, [])
                    if not any(ts >= start and te <= end for (start, end) in param_allowlist):
                        errors.append(
                            f"unexpected new-init rows in {pname}: [{ts}, {te})  "
                            f"(allowed: {param_allowlist}); semantic_role="
                            f"{span.get('semantic_role', '?')}"
                        )
                elif not span.get("verified_equal", False):
                    ts, te = span["target_start"], span["target_end"]
                    errors.append(
                        f"verification failed: {pname} rows [{ts}, {te}) "
                        f"claimed from {span['source']}/{span.get('source_key')}"
                    )

        for warning in self.warnings:
            if warning["tag"] not in contract.allowed_warning_tags:
                errors.append(
                    f"unacceptable warning tag={warning['tag']!r} "
                    f"message={warning['message']!r}"
                )

        if errors:
            header = (
                f"Warmstart contract violation "
                f"(contract={contract.name}, family={contract.checkpoint_family}):\n"
            )
            raise AssertionError(header + "\n".join(f"  - {error}" for error in errors))


def _find_source_with_key(loaded_sources: dict[str, dict[str, Any]], key: str) -> str:
    for name, state_dict in loaded_sources.items():
        if key in state_dict:
            return name
    raise KeyError(f"No loaded source provides key {key!r}")


def _verify_new_init(tensor: Any, loaded_sources: dict[str, dict[str, Any]]) -> bool:
    """Weak sanity check for new-init spans."""
    import torch as _torch

    for state_dict in loaded_sources.values():
        for src in state_dict.values():
            if src.shape == tensor.shape and _torch.equal(tensor, src):
                return False
    return True
