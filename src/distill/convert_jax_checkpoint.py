"""
Convert DeepMind searchless_chess Orbax checkpoints to PyTorch state dicts.

Usage:
    python -m src.distill.convert_jax_checkpoint \
        --checkpoint_dir searchless_chess/checkpoints \
        --model_name 136M \
        --step 6400000 \
        --output checkpoints/jax_136m_converted.pt

The resulting .pt file contains {"model_state_dict": <state_dict>} and can be
loaded into src.models_9m.Searchless9MTransformer with strict=True by passing
the matching Searchless9MConfig for the selected model size.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure repo root is on sys.path so src.* imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

MODEL_CONFIGS = {
    "9M": {"num_layers": 8, "embedding_dim": 256, "num_heads": 8},
    "136M": {"num_layers": 8, "embedding_dim": 1024, "num_heads": 8},
    "270M": {"num_layers": 16, "embedding_dim": 1024, "num_heads": 8},
    "local": {"num_layers": 4, "embedding_dim": 64, "num_heads": 4},
}


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


def _resolve_model_config(model_name: str) -> dict[str, int]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name {model_name!r}. Expected one of: {sorted(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[model_name]
    embed_dim = int(cfg["embedding_dim"])
    return {
        "num_layers": int(cfg["num_layers"]),
        "embed_dim": embed_dim,
        "num_heads": int(cfg["num_heads"]),
        "ffn_dim": 4 * embed_dim,
    }


def _verify_converted_state_dict(
    sd: dict[str, torch.Tensor],
    model_cfg: dict[str, int],
) -> tuple[dict[str, object], dict[str, object]]:
    """Best-effort verification against the legacy 9M PyTorch model, if present."""
    try:
        models_9m = importlib.import_module("src.models_9m")
    except ModuleNotFoundError:
        return (
            {
                "verification_skipped": True,
                "verification_reason": "src.models_9m not available",
            },
            {},
        )

    model = models_9m.Searchless9MTransformer(
        models_9m.Searchless9MConfig(
            embed_dim=model_cfg["embed_dim"],
            num_layers=model_cfg["num_layers"],
            num_heads=model_cfg["num_heads"],
            ffn_dim=model_cfg["ffn_dim"],
        )
    )
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print(f"WARNING — missing keys: {missing}")
        print(f"WARNING — unexpected keys: {unexpected}")
    else:
        print("strict=True load succeeded — all keys matched.")

    dummy_input = torch.zeros((1, 79), dtype=torch.long)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"Forward pass OK — output shape: {out.shape}")

    summary = {
        "verification_skipped": False,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "forward_pass_output_shape": list(out.shape),
        "load_strict": True,
    }
    extras = {
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return summary, extras


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

    model_cfg = _resolve_model_config(model_name)

    # Replicate the teacher TransformerConfig for the selected model size.
    config = sc_transformer.TransformerConfig(
        vocab_size=sc_utils.NUM_ACTIONS,      # 1968
        output_size=128,
        embedding_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
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
    print(f"\n=== JAX {model_name} param keys ===")
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

    num_layers = model_cfg["num_layers"]
    for i in range(num_layers):
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
    final_ln_idx = 2 * num_layers
    head_lin_idx = 3 * num_layers
    sd["final_norm.weight"]  = t(get(_ln(final_ln_idx), "scale"))
    sd["final_norm.bias"]    = t(get(_ln(final_ln_idx), "offset"))
    sd["policy_head.weight"] = t(get(_lin(head_lin_idx), "w")).T
    sd["policy_head.bias"]   = t(get(_lin(head_lin_idx), "b"))

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save({"model_state_dict": sd}, output)
    print(f"\nSaved {len(sd)} tensors → {output}")

    verification_summary, verification_extras = _verify_converted_state_dict(sd, model_cfg)
    if verification_summary.get("verification_skipped"):
        print(f"Verification skipped — {verification_summary['verification_reason']}")

    summary = {
        "model_name": model_name,
        "embed_dim": model_cfg["embed_dim"],
        "num_layers": model_cfg["num_layers"],
        "num_heads": model_cfg["num_heads"],
        "ffn_dim": model_cfg["ffn_dim"],
        "step": step,
        "output_path": output,
        "n_state_dict_keys": len(sd),
    }
    summary.update(verification_summary)
    summary.update(verification_extras)

    # Log to WandB if API key is available
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    if wandb_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = wandb_key
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
                    "n_params": verification_extras.get("n_params"),
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
        description="Convert JAX Orbax checkpoint (9M/136M/270M/local) → PyTorch state dict"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="searchless_chess/checkpoints",
        help="Root checkpoint dir; model subdir is appended automatically.",
    )
    parser.add_argument("--step", type=int, default=6400000)
    parser.add_argument("--output", default="checkpoints/jax_9m_converted.pt")
    parser.add_argument("--model_name", default="9M", choices=sorted(MODEL_CONFIGS.keys()))
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
