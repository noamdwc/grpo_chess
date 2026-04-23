"""One-shot JAX → PyTorch weight converter for the DeepMind searchless chess model.

Usage:
    ~/miniconda3/envs/grpo_chess/bin/python -m src.dm_port.convert_jax \
        --model 9M --out checkpoints/dm_port/9M.pt

Run with --inspect to print the JAX parameter tree without converting (use this
first to discover the exact key layout for a new model size).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jax import random as jrandom

from src.dm_port.transformer import DMTransformer, DMTransformerConfig
from src.distill.teacher import MODEL_CONFIGS, _load_sc_module
from src.reasoning.warmstart_provenance import (
    canonical_action_vocab_fingerprint,
    canonical_fen_tokenizer_fingerprint,
)


def _load_jax_params(model_name: str, checkpoint_dir: str, checkpoint_step: int) -> dict:
    """Load DM checkpoint params into a flat dict of {path: np.ndarray}.

    Reuses the searchless_chess plumbing already in src/distill/teacher.py.
    We don't need the engine — only the `params` tree.
    """
    sc_transformer = _load_sc_module("transformer")
    sc_training_utils = _load_sc_module("training_utils")
    sc_utils = _load_sc_module("utils")
    sc_tokenizer = _load_sc_module("tokenizer")

    cfg = MODEL_CONFIGS[model_name]
    predictor = sc_transformer.build_transformer_predictor(
        config=sc_transformer.TransformerConfig(
            vocab_size=sc_utils.NUM_ACTIONS,
            output_size=128,
            pos_encodings=sc_transformer.PositionalEncodings.LEARNED,
            max_sequence_length=sc_tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            embedding_dim=cfg["embedding_dim"],
            apply_post_ln=True,
            apply_qk_layernorm=False,
            use_causal_mask=False,
        )
    )
    ckpt_dir = os.path.abspath(os.path.join(checkpoint_dir, model_name))
    params = sc_training_utils.load_parameters(
        checkpoint_dir=ckpt_dir,
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=checkpoint_step,
    )
    flat = {}

    def walk(prefix: str, tree: Any) -> None:
        if hasattr(tree, "items"):
            for k, v in tree.items():
                walk(f"{prefix}/{k}" if prefix else k, v)
        else:
            flat[prefix] = np.asarray(tree)

    walk("", params)
    return flat


def _tensor(arr: np.ndarray, transpose: bool = False) -> torch.Tensor:
    """Convert a numpy array (from JAX) into a float32 torch tensor.

    JAX hk.Linear stores weights as (in, out); nn.Linear uses (out, in), so
    pass transpose=True for Linear weights.
    """
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return t.T.contiguous() if transpose else t


def inspect(args: argparse.Namespace) -> None:
    flat = _load_jax_params(args.model, args.checkpoint_dir, args.checkpoint_step)
    for k, v in sorted(flat.items()):
        print(f"{k:80s}  {tuple(v.shape)}  {v.dtype}")
    print(f"\nTotal params: {sum(v.size for v in flat.values())}")


def convert(args: argparse.Namespace) -> None:
    flat = _load_jax_params(args.model, args.checkpoint_dir, args.checkpoint_step)
    cfg = MODEL_CONFIGS[args.model]
    num_layers = cfg["num_layers"]
    embedding_dim = cfg["embedding_dim"]

    # Track JAX keys we consume so we can detect silent drops (e.g. an
    # unexpected extra module or a miscounted layer index).
    consumed: set[str] = set()

    def take(key: str) -> Any:
        consumed.add(key)
        return flat[key]

    state_dict: dict[str, torch.Tensor] = {}

    # --- Embeddings ---
    # Haiku key layout discovered via --inspect (see convert_jax_layout.md):
    # No module prefix — keys are at root level.
    # Token embedding: embed/embeddings  shape (vocab_size, embedding_dim)
    # Position embedding: embed_1/embeddings  shape (max_seq_len, embedding_dim)
    token_key = "embed/embeddings"
    pos_key = "embed_1/embeddings"
    state_dict["token_embedding.weight"] = _tensor(take(token_key))
    state_dict["pos_embedding.weight"] = _tensor(take(pos_key))

    # --- Per-layer blocks ---
    # Haiku auto-numbers modules in call order across the whole model:
    #
    # LayerNorms: layer_norm_0, layer_norm_1, ..., layer_norm_{2*num_layers}
    #   layer_norm_{2*i}   = block i attn pre-LN
    #   layer_norm_{2*i+1} = block i mlp pre-LN
    #   layer_norm_{2*num_layers} = post_ln (final)
    #
    # Attention Q/K/V/Out: under named module multi_head_dot_product_attention_{i}
    #   multi_head_dot_product_attention_{i}/linear/w   = Q  shape (embedding_dim, embedding_dim)
    #   multi_head_dot_product_attention_{i}/linear_1/w = K
    #   multi_head_dot_product_attention_{i}/linear_2/w = V
    #   multi_head_dot_product_attention_{i}/linear_3/w = Out
    #
    # MLP (SwiGLU): linear_0..linear_{3*num_layers - 1}, stride 3 per block
    #   linear_{3*i}   = w1  shape (embedding_dim, ffn_dim)
    #   linear_{3*i+1} = w2  shape (embedding_dim, ffn_dim)
    #   linear_{3*i+2} = w3  shape (ffn_dim, embedding_dim)
    #
    # Output head: linear_{3*num_layers}/w and linear_{3*num_layers}/b
    for i in range(num_layers):
        ln_attn_idx = 2 * i
        ln_mlp_idx = 2 * i + 1
        mlp_w1_idx = 3 * i
        mlp_w2_idx = 3 * i + 1
        mlp_w3_idx = 3 * i + 2

        # Attention pre-LN (scale=weight, offset=bias in Haiku)
        attn_ln_name = "layer_norm" if i == 0 else f"layer_norm_{ln_attn_idx}"
        state_dict[f"layers.{i}.attn_ln.weight"] = _tensor(take(f"{attn_ln_name}/scale"))
        state_dict[f"layers.{i}.attn_ln.bias"] = _tensor(take(f"{attn_ln_name}/offset"))

        # Attention projections — live under named module, not global counter
        # multi_head_dot_product_attention_{i} for i>0, no suffix for i=0
        attn_name = (
            "multi_head_dot_product_attention"
            if i == 0
            else f"multi_head_dot_product_attention_{i}"
        )
        # linear/w=Q, linear_1/w=K, linear_2/w=V, linear_3/w=Out
        state_dict[f"layers.{i}.attn.q_proj.weight"] = _tensor(
            take(f"{attn_name}/linear/w"), transpose=True
        )
        state_dict[f"layers.{i}.attn.k_proj.weight"] = _tensor(
            take(f"{attn_name}/linear_1/w"), transpose=True
        )
        state_dict[f"layers.{i}.attn.v_proj.weight"] = _tensor(
            take(f"{attn_name}/linear_2/w"), transpose=True
        )
        state_dict[f"layers.{i}.attn.out_proj.weight"] = _tensor(
            take(f"{attn_name}/linear_3/w"), transpose=True
        )

        # MLP pre-LN
        mlp_ln_name = f"layer_norm_{ln_mlp_idx}"
        state_dict[f"layers.{i}.mlp_ln.weight"] = _tensor(take(f"{mlp_ln_name}/scale"))
        state_dict[f"layers.{i}.mlp_ln.bias"] = _tensor(take(f"{mlp_ln_name}/offset"))

        # MLP linears (all bias=False; transpose)
        w1_name = "linear" if mlp_w1_idx == 0 else f"linear_{mlp_w1_idx}"
        w2_name = f"linear_{mlp_w2_idx}"
        w3_name = f"linear_{mlp_w3_idx}"
        state_dict[f"layers.{i}.mlp.w1.weight"] = _tensor(take(f"{w1_name}/w"), transpose=True)
        state_dict[f"layers.{i}.mlp.w2.weight"] = _tensor(take(f"{w2_name}/w"), transpose=True)
        state_dict[f"layers.{i}.mlp.w3.weight"] = _tensor(take(f"{w3_name}/w"), transpose=True)

    # --- Post-LN ---
    post_ln_idx = 2 * num_layers
    state_dict["post_ln.weight"] = _tensor(take(f"layer_norm_{post_ln_idx}/scale"))
    state_dict["post_ln.bias"] = _tensor(take(f"layer_norm_{post_ln_idx}/offset"))

    # --- Output head ---
    out_lin_idx = 3 * num_layers
    out_lin_name = "linear" if out_lin_idx == 0 else f"linear_{out_lin_idx}"
    state_dict["output_linear.weight"] = _tensor(take(f"{out_lin_name}/w"), transpose=True)
    state_dict["output_linear.bias"] = _tensor(take(f"{out_lin_name}/b"))

    # Catch silent drops on the JAX side (e.g. an unexpected bias, an extra
    # module, or a miscounted layer index). strict=False below only guards
    # the PyTorch-side keys.
    leftover = set(flat.keys()) - consumed
    if leftover:
        raise RuntimeError(
            f"JAX params not consumed by converter: {sorted(leftover)}\n"
            "Re-run with --inspect and fix the mapping in convert_jax.py."
        )

    # --- Sanity load ---
    parity_cfg = DMTransformerConfig(
        vocab_size=flat[token_key].shape[0],
        output_size=state_dict["output_linear.weight"].shape[0],
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=cfg["num_heads"],
        max_sequence_length=flat[pos_key].shape[0],
    )
    model = DMTransformer(parity_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch.\nMissing: {missing}\nUnexpected: {unexpected}\n"
            "Re-read the --inspect output and fix the mapping in convert_jax.py."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "family": "dm_action_value",
        "model": args.model,
        "checkpoint_step": args.checkpoint_step,
        "input_vocab_size": parity_cfg.vocab_size,
        "output_size": parity_cfg.output_size,
        "positional_length": parity_cfg.max_sequence_length,
        "num_params": sum(t.numel() for t in state_dict.values()),
        "param_keys": sorted(state_dict.keys()),
        "fen_tokenizer_fingerprint": canonical_fen_tokenizer_fingerprint(),
        "action_vocab_fingerprint": canonical_action_vocab_fingerprint(),
    }
    torch.save(
        {
            "state_dict": state_dict,
            "config": parity_cfg.__dict__,
            "meta": meta,
        },
        out_path,
    )
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} ({sum(t.numel() for t in state_dict.values())} params)")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="9M", choices=["9M", "136M"])
    parser.add_argument("--checkpoint-dir", default="searchless_chess/checkpoints")
    parser.add_argument("--checkpoint-step", type=int, default=6_400_000)
    parser.add_argument("--out", default="checkpoints/dm_port/9M.pt")
    parser.add_argument("--inspect", action="store_true", help="Print JAX param tree and exit")
    args = parser.parse_args()
    if args.inspect:
        inspect(args)
        return
    convert(args)


if __name__ == "__main__":
    _main()
