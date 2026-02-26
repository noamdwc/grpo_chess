"""
Convert the DeepMind searchless_chess 9M Orbax checkpoint to a PyTorch state dict.

Usage:
    python -m src.distill.convert_jax_checkpoint \
        --checkpoint_dir searchless_chess/checkpoints \
        --step 6400000 \
        --output checkpoints/jax_9m_converted.pt

The resulting .pt file contains {"model_state_dict": <state_dict>} and can be
loaded into src.models_9m.Searchless9MTransformer with strict=True.

JAX Haiku param key structure (9M, 8 layers)
---------------------------------------------
Haiku auto-numbers top-scope module instantiations in call order within
transformer_decoder (which is the hk.transform root).

Call order:
  embed_sequences:
    embed/embed_lookup:0          [1968, 256]  — token embedding
    embed_1/embed_lookup:0        [79,   256]  — learned positional encoding

  For layer i = 0..7:
    layer_norm_{2i}/scale|offset  [256]        — pre-attention LayerNorm
    # Haiku stores nested-module params with "/" in the top-level dict key:
    multi_head_dot_product_attention_{i}/linear   → {"w": [256, 256]}  Q
    multi_head_dot_product_attention_{i}/linear_1 → {"w": [256, 256]}  K
    multi_head_dot_product_attention_{i}/linear_2 → {"w": [256, 256]}  V
    multi_head_dot_product_attention_{i}/linear_3 → {"w": [256, 256]}  O
    layer_norm_{2i+1}/scale|offset [256]       — pre-MLP LayerNorm
    linear_{3i}/w                 [256, 1024]  — gate_proj  (MLP)
    linear_{3i+1}/w               [256, 1024]  — up_proj    (MLP)
    linear_{3i+2}/w               [1024, 256]  — down_proj  (MLP)

  After all layers:
    layer_norm_16/scale|offset    [256]        — final post-LayerNorm
    linear_24/w                   [256, 128]   — output head weight
    linear_24/b                   [128]        — output head bias

  (layer_norm_0 is accessed as "layer_norm", linear_0 as "linear", etc.)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path so src.* imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Key-name helpers (Haiku auto-numbering convention)
# ---------------------------------------------------------------------------

def _ln(i: int) -> str:
    """Return the Haiku name for the i-th LayerNorm call."""
    return "layer_norm" if i == 0 else f"layer_norm_{i}"


def _attn(i: int) -> str:
    """Return the Haiku name for the i-th MultiHeadDotProductAttention."""
    return (
        "multi_head_dot_product_attention"
        if i == 0
        else f"multi_head_dot_product_attention_{i}"
    )


def _lin(i: int) -> str:
    """Return the Haiku name for the i-th top-scope Linear."""
    return "linear" if i == 0 else f"linear_{i}"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    checkpoint_dir: str,
    step: int,
    output: str,
    model_name: str = "9M",
    wandb_project: str = "chess-grpo-pretrain",
    wandb_tags: list = None,
    wandb_run_name: str = None,
) -> None:
    import jax
    import jax.numpy as jnp

    # Patch apache_beam stub so searchless_chess modules load without Beam installed
    try:
        import apache_beam  # noqa: F401
    except ImportError:
        import types

        def _stub(name):
            m = types.ModuleType(name)
            m.__path__ = []
            return m

        beam = _stub("apache_beam")
        coders_mod = _stub("apache_beam.coders")

        class _Dummy:
            def __init__(self, *a, **kw): pass

        for c in ("StrUtf8Coder", "BigIntegerCoder", "FloatCoder", "TupleCoder"):
            setattr(coders_mod, c, _Dummy)

        beam.coders = coders_mod
        sys.modules.setdefault("apache_beam", beam)
        sys.modules.setdefault("apache_beam.coders", coders_mod)

    if not hasattr(jax.sharding, "PositionalSharding"):
        jax.sharding.PositionalSharding = type("PositionalSharding", (), {})

    # Add searchless_chess/src to path for direct imports
    sc_src = _REPO_ROOT / "searchless_chess" / "src"
    sys.path.insert(0, str(sc_src.parent))

    from searchless_chess.src import transformer as sc_transformer
    from searchless_chess.src import training_utils as sc_training_utils
    from searchless_chess.src import utils as sc_utils
    from searchless_chess.src import tokenizer as sc_tokenizer

    # Replicate the 9M config used in teacher.py
    config = sc_transformer.TransformerConfig(
        vocab_size=sc_utils.NUM_ACTIONS,      # 1968
        output_size=128,
        embedding_dim=256,
        num_layers=8,
        num_heads=8,
        pos_encodings=sc_transformer.PositionalEncodings.LEARNED,
        max_sequence_length=sc_tokenizer.SEQUENCE_LENGTH + 2,  # 79
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = sc_transformer.build_transformer_predictor(config)
    dummy = np.ones((1, 1), dtype=np.uint32)
    init_params = predictor.initial_params(jax.random.PRNGKey(0), dummy)

    ckpt_path = os.path.abspath(os.path.join(checkpoint_dir, model_name))
    print(f"Loading checkpoint from {ckpt_path} at step {step} ...")
    params = sc_training_utils.load_parameters(
        checkpoint_dir=ckpt_path,
        params=init_params,
        step=step,
    )

    # Print all JAX keys for verification
    print("\n=== JAX 9M param keys ===")
    flat = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        key = "/".join(str(p.key) for p in path)
        arr = np.array(leaf)
        flat[key] = arr
        print(f"  {key}: {arr.shape}")

    def t(x) -> torch.Tensor:
        return torch.from_numpy(np.array(x, dtype=np.float32))

    def get(scope: str, subkey: str) -> np.ndarray:
        """Look up params[scope][subkey] with a clear error on missing key."""
        if scope not in params:
            raise KeyError(
                f"Expected JAX scope '{scope}' not found.\n"
                f"Available top-level keys: {sorted(params.keys())}"
            )
        if subkey not in params[scope]:
            raise KeyError(
                f"Expected key '{subkey}' not found in scope '{scope}'.\n"
                f"Available keys: {sorted(params[scope].keys())}"
            )
        return params[scope][subkey]

    sd: dict[str, torch.Tensor] = {}

    # --- Embeddings ---
    sd["embedding.weight"]    = t(get("embed",   "embeddings"))   # [1968, 256]
    sd["pos_encoding.weight"] = t(get("embed_1", "embeddings"))   # [79,   256]

    for i in range(8):
        # --- Layer norms ---
        sd[f"layers.{i}.norm1.weight"] = t(get(_ln(2 * i),     "scale"))
        sd[f"layers.{i}.norm1.bias"]   = t(get(_ln(2 * i),     "offset"))
        sd[f"layers.{i}.norm2.weight"] = t(get(_ln(2 * i + 1), "scale"))
        sd[f"layers.{i}.norm2.bias"]   = t(get(_ln(2 * i + 1), "offset"))

        # --- Attention (Q, K, V, O) ---
        # Haiku stores nested-module params with "/" in the top-level key
        Q = t(get(f"{_attn(i)}/linear",   "w"))  # [256, 256]  JAX: [in, out]
        K = t(get(f"{_attn(i)}/linear_1", "w"))  # [256, 256]
        V = t(get(f"{_attn(i)}/linear_2", "w"))  # [256, 256]
        O = t(get(f"{_attn(i)}/linear_3", "w"))  # [256, 256]

        # PyTorch in_proj_weight = [3*embed_dim, embed_dim] = concat(Q.T, K.T, V.T)
        sd[f"layers.{i}.attn.in_proj_weight"]  = torch.cat([Q.T, K.T, V.T], dim=0)  # [768, 256]
        # out_proj.weight = [embed_dim, embed_dim], PyTorch convention [out, in]
        sd[f"layers.{i}.attn.out_proj.weight"] = O.T  # [256, 256]

        # --- SwiGLU FFN (3 top-scope Linears per layer) ---
        b = 3 * i
        # JAX weights are [in_features, out_features]; PyTorch wants [out, in]
        sd[f"layers.{i}.ffn.gate_proj.weight"] = t(get(_lin(b),     "w")).T  # [1024, 256]
        sd[f"layers.{i}.ffn.up_proj.weight"]   = t(get(_lin(b + 1), "w")).T  # [1024, 256]
        sd[f"layers.{i}.ffn.down_proj.weight"] = t(get(_lin(b + 2), "w")).T  # [256, 1024]

    # --- Final norm and output head ---
    sd["final_norm.weight"]  = t(get(_ln(16), "scale"))
    sd["final_norm.bias"]    = t(get(_ln(16), "offset"))
    sd["policy_head.weight"] = t(get(_lin(24), "w")).T  # [256, 128].T → [128, 256]
    sd["policy_head.bias"]   = t(get(_lin(24), "b"))    # [128]

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save({"model_state_dict": sd}, output)
    print(f"\nSaved {len(sd)} tensors → {output}")

    # --- Verification: load into Searchless9MTransformer ---
    from src.models_9m import Searchless9MTransformer
    model = Searchless9MTransformer()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"WARNING — missing keys: {missing}")
        print(f"WARNING — unexpected keys: {unexpected}")
    else:
        print("strict=True load succeeded — all keys matched.")

    # Quick forward pass to confirm no shape errors
    dummy_input = torch.zeros((1, 79), dtype=torch.long)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Forward pass OK — output shape: {out.shape}")  # expect [1, 79, 128]

    n_params = sum(p.numel() for p in model.parameters())
    summary = {
        "model_name": model_name,
        "step": step,
        "output_path": output,
        "n_params": n_params,
        "n_state_dict_keys": len(sd),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "forward_pass_output_shape": list(out.shape),
        "load_strict": True,
    }

    # Log to WandB if API key is available
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    if wandb_key:
        try:
            import wandb
            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"convert-jax-{model_name.lower()}-{step}",
                tags=(wandb_tags or []) + ["jax-conversion", model_name],
                config={
                    "model_name": model_name,
                    "checkpoint_step": step,
                    "checkpoint_dir": checkpoint_dir,
                    "output_path": output,
                    "n_params": n_params,
                },
            )
            wandb.summary.update(summary)
            wandb.finish()
            print(f"WandB run logged → {run.url}")
        except Exception as e:
            print(f"WandB logging skipped: {e}")
    else:
        print("No WANDB_API_KEY found — skipping WandB logging.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 9M JAX Orbax checkpoint → PyTorch state dict"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="searchless_chess/checkpoints",
        help="Root checkpoint dir; model subdir is appended automatically.",
    )
    parser.add_argument("--step", type=int, default=6400000)
    parser.add_argument("--output", default="checkpoints/jax_9m_converted.pt")
    parser.add_argument("--model_name", default="9M")
    parser.add_argument("--wandb_project", default="chess-grpo-pretrain")
    parser.add_argument("--wandb_tags", default="", help="Comma-separated WandB tags")
    parser.add_argument("--wandb_run_name", default="")
    args = parser.parse_args()

    convert(
        checkpoint_dir=args.checkpoint_dir,
        step=args.step,
        output=args.output,
        model_name=args.model_name,
        wandb_project=args.wandb_project,
        wandb_tags=[t for t in args.wandb_tags.split(",") if t],
        wandb_run_name=args.wandb_run_name or None,
    )


if __name__ == "__main__":
    main()
