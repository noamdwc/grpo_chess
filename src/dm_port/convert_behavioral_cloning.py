"""Convert the released 9M behavioral-cloning Orbax checkpoint to a PyTorch checkpoint.

Usage:
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
    ~/miniconda3/envs/grpo_chess/bin/python -m src.dm_port.convert_behavioral_cloning \\
        --checkpoint-dir searchless_chess/checkpoints/9M_behavioral_cloning \\
        --out checkpoints/dm_port/9M_behavioral_cloning.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
import torch

from src.dm_port.transformer import DMTransformer, DMTransformerConfig


def _tensor(arr) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32).copy())


def convert(checkpoint_dir: str, out: str) -> None:
    restored = ocp.StandardCheckpointer().restore(checkpoint_dir)
    params = restored["params"]

    state_dict: dict[str, torch.Tensor] = {}
    state_dict["token_embedding.weight"] = _tensor(params["Embed_0"]["embedding"]["value"])
    state_dict["pos_embedding.weight"] = _tensor(params["Embed_1"]["embedding"]["value"])

    for i in range(8):
        state_dict[f"layers.{i}.attn_ln.weight"] = _tensor(params[f"LayerNorm_{2 * i}"]["scale"])
        state_dict[f"layers.{i}.attn_ln.bias"] = _tensor(params[f"LayerNorm_{2 * i}"]["bias"])
        state_dict[f"layers.{i}.mlp_ln.weight"] = _tensor(params[f"LayerNorm_{2 * i + 1}"]["scale"])
        state_dict[f"layers.{i}.mlp_ln.bias"] = _tensor(params[f"LayerNorm_{2 * i + 1}"]["bias"])

        attn = params[f"MultiHeadDotProductAttention_{i}"]
        q = np.asarray(attn["query"]["kernel"], dtype=np.float32).reshape(256, 256)
        k = np.asarray(attn["key"]["kernel"], dtype=np.float32).reshape(256, 256)
        v = np.asarray(attn["value"]["kernel"], dtype=np.float32).reshape(256, 256)
        o = np.asarray(attn["out"]["kernel"], dtype=np.float32).reshape(256, 256)
        state_dict[f"layers.{i}.attn.q_proj.weight"] = torch.from_numpy(q.T.copy())
        state_dict[f"layers.{i}.attn.k_proj.weight"] = torch.from_numpy(k.T.copy())
        state_dict[f"layers.{i}.attn.v_proj.weight"] = torch.from_numpy(v.T.copy())
        state_dict[f"layers.{i}.attn.out_proj.weight"] = torch.from_numpy(o.T.copy())

        state_dict[f"layers.{i}.mlp.w1.weight"] = torch.from_numpy(
            np.asarray(params[f"Dense_{3 * i}"]["kernel"]["value"], dtype=np.float32).T.copy()
        )
        state_dict[f"layers.{i}.mlp.w2.weight"] = torch.from_numpy(
            np.asarray(params[f"Dense_{3 * i + 1}"]["kernel"]["value"], dtype=np.float32).T.copy()
        )
        state_dict[f"layers.{i}.mlp.w3.weight"] = torch.from_numpy(
            np.asarray(params[f"Dense_{3 * i + 2}"]["kernel"]["value"], dtype=np.float32).T.copy()
        )

    state_dict["post_ln.weight"] = _tensor(params["LayerNorm_16"]["scale"])
    state_dict["post_ln.bias"] = _tensor(params["LayerNorm_16"]["bias"])
    state_dict["output_linear.weight"] = torch.from_numpy(
        np.asarray(params["Dense_24"]["kernel"]["value"], dtype=np.float32).T.copy()
    )
    state_dict["output_linear.bias"] = _tensor(params["Dense_24"]["bias"])

    config = DMTransformerConfig(
        vocab_size=31,
        output_size=1968,
        embedding_dim=256,
        num_layers=8,
        num_heads=8,
        max_sequence_length=78,
    )
    model = DMTransformer(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch. Missing={missing} Unexpected={unexpected}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "config": config.__dict__}, out_path)
    meta = {
        "family": "dm_behavioral_cloning",
        "input_vocab_size": config.vocab_size,
        "output_size": config.output_size,
        "positional_length": config.max_sequence_length,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="searchless_chess/checkpoints/9M_behavioral_cloning")
    parser.add_argument("--out", default="checkpoints/dm_port/9M_behavioral_cloning.pt")
    args = parser.parse_args()
    convert(args.checkpoint_dir, args.out)


if __name__ == "__main__":
    _main()
