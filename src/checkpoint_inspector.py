"""Utilities for inspecting checkpoint family and interface shapes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess
import sys
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    checkpoint_format: str
    family: str
    input_vocab_size: int
    output_size: int
    positional_length: int
    checkpoint_step: int | None = None


def _infer_family(input_vocab_size: int, output_size: int) -> str:
    if input_vocab_size == 1968 and output_size == 128:
        return "dm_action_value"
    if input_vocab_size == 31 and output_size == 1968:
        return "dm_behavioral_cloning"
    if input_vocab_size == 31 and output_size == 128:
        return "dm_state_value"
    if input_vocab_size == 1971 and output_size == 1971:
        return "reasoning_policy"
    return "unknown"


def _inspect_torch_pt(path: Path) -> CheckpointInfo:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    input_vocab_size = int(cfg["vocab_size"])
    output_size = int(cfg["output_size"])
    positional_length = int(cfg["max_sequence_length"])
    return CheckpointInfo(
        path=str(path),
        checkpoint_format="torch_pt",
        family=_infer_family(input_vocab_size, output_size),
        input_vocab_size=input_vocab_size,
        output_size=output_size,
        positional_length=positional_length,
    )


def _orbax_tree_at_path(path: Path) -> tuple[dict[str, Any], int | None]:
    import orbax.checkpoint as ocp

    if path.is_dir():
        steps = ocp.utils.checkpoint_steps(str(path))
        if steps and (path / str(steps[-1]) / "params").exists():
            step = int(steps[-1])
            meta = ocp.StandardCheckpointer().metadata(str(path / str(step) / "params"))
            if meta.item_metadata is None:
                raise RuntimeError("orbax_metadata_unavailable")
            return meta.item_metadata.tree, step

        meta = ocp.StandardCheckpointer().metadata(str(path))
        if meta.item_metadata is None:
            raise RuntimeError("orbax_metadata_unavailable")
        tree = meta.item_metadata.tree
        if "params" in tree:
            return tree["params"], 0
        return tree, None
    raise FileNotFoundError(path)


def _inspect_orbax_via_subprocess(path: Path) -> CheckpointInfo:
    abs_path = str(path.resolve())
    code = f"""
import json
import os
import orbax.checkpoint as ocp
path = {abs_path!r}
steps = ocp.utils.checkpoint_steps(path)
params_path = os.path.join(path, str(steps[-1]), 'params') if steps else None
if steps and os.path.exists(params_path):
    meta = ocp.StandardCheckpointer().metadata(params_path)
    tree = meta.item_metadata.tree
    embed0 = tree['embed']['embeddings'].shape[0]
    embed1 = tree['embed_1']['embeddings'].shape[0]
    dense = max((k for k in tree if k.startswith('linear_')), key=lambda item: int(item.split('_')[1]))
    out = tree[dense]['b'].shape[0]
    step = int(steps[-1])
else:
    restored = ocp.StandardCheckpointer().restore(path)
    tree = restored['params']
    embed0 = tree['Embed_0']['embedding']['value'].shape[0]
    embed1 = tree['Embed_1']['embedding']['value'].shape[0]
    dense = max((k for k in tree if k.startswith('Dense_')), key=lambda item: int(item.split('_')[1]))
    out = tree[dense]['bias'].shape[0]
    step = 0
print(json.dumps({{'input_vocab_size': int(embed0), 'positional_length': int(embed1), 'output_size': int(out), 'checkpoint_step': step}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ.copy(), "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    )
    data = json.loads(proc.stdout.strip())
    return CheckpointInfo(
        path=abs_path,
        checkpoint_format="orbax",
        family=_infer_family(data["input_vocab_size"], data["output_size"]),
        input_vocab_size=int(data["input_vocab_size"]),
        output_size=int(data["output_size"]),
        positional_length=int(data["positional_length"]),
        checkpoint_step=int(data["checkpoint_step"]),
    )


def _inspect_orbax(path: Path) -> CheckpointInfo:
    try:
        tree, step = _orbax_tree_at_path(path)
    except RuntimeError as exc:
        if str(exc) == "orbax_metadata_unavailable":
            return _inspect_orbax_via_subprocess(path)
        raise
    if "embed" in tree:
        input_vocab_size = int(tree["embed"]["embeddings"].shape[0])
        positional_length = int(tree["embed_1"]["embeddings"].shape[0])
        dense = max((key for key in tree if key.startswith("linear_")), key=lambda item: int(item.split("_")[1]))
        output_size = int(tree[dense]["b"].shape[0])
    else:
        input_vocab_size = int(tree["Embed_0"]["embedding"]["value"].shape[0])
        positional_length = int(tree["Embed_1"]["embedding"]["value"].shape[0])
        dense = max((key for key in tree if key.startswith("Dense_")), key=lambda item: int(item.split("_")[1]))
        output_size = int(tree[dense]["bias"].shape[0])
    return CheckpointInfo(
        path=str(path),
        checkpoint_format="orbax",
        family=_infer_family(input_vocab_size, output_size),
        input_vocab_size=input_vocab_size,
        output_size=output_size,
        positional_length=positional_length,
        checkpoint_step=step,
    )


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    path = Path(path)
    if path.is_file():
        return _inspect_torch_pt(path)
    return _inspect_orbax(path)


def _main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    info = inspect_checkpoint(args.path)
    print(json.dumps(info.__dict__, indent=2))


if __name__ == "__main__":
    _main()
