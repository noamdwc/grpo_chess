"""Phase-0 gate: PyTorch DM port must match the JAX reference on a fixed batch.

Contract (both must pass):
  - **Hard gate:** ``max_abs_diff < 1e-10`` in **float64**. This is the real
    numerical contract — it proves the PyTorch port is mathematically
    equivalent to the JAX reference, up to machine epsilon.
  - **Sanity check:** ``max_abs_diff < 5e-4`` in **float32**. This is loose
    because XLA and ATen CPU matmul kernels accumulate reductions in
    different orders, causing ~1e-4 drift on ``log_softmax`` outputs
    through an 8-layer / 256-dim / 1024-wide-MLP transformer. The drift
    is CPU-kernel artifact, NOT a bug in the port. See the float64 test
    above for the real contract.

If the float64 test fails: fix ``src/dm_port/transformer.py`` or the weight
converter (``src/dm_port/convert_jax.py``). Debug order:
  (a) print per-layer activation deltas (everything is bit-exact up to the
      first MLP in float64 — so narrow down by layer),
  (b) check LayerNorm epsilon,
  (c) check Q/K/V weight transpose in convert_jax.py,
  (d) check PE indexing.

If ONLY the float32 test fails while float64 still passes at 1e-10, the port
is correct and the float32 threshold is the thing to reconsider — but only
after confirming float64 parity is still tight.

Skipped if the converted checkpoint is missing (run src.dm_port.convert_jax).
"""
# JAX defaults to float32 on CPU. We need float64 for the hard-gate test,
# and that requires enabling x64 BEFORE any jax import — so set the env var
# here at module load, long before `_jax_logits` imports JAX.
import os
os.environ.setdefault("JAX_ENABLE_X64", "True")

from pathlib import Path

import numpy as np
import pytest
import torch

from src.searchless_chess_imports import tokenize

CKPT = Path("checkpoints/dm_port/9M.pt")

pytestmark = pytest.mark.skipif(
    not CKPT.exists(),
    reason="DM port checkpoint missing; run src.dm_port.convert_jax first.",
)

# 32 diverse FENs spanning openings, middlegames, endgames, tactical
# positions, castling rights, en-passant squares, and both sides to move.
FIXED_FENS = [
    # Openings (white to move)
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 5",
    # Openings (black to move)
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    # Middlegames with castling done
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 6",
    "r2q1rk1/pp1bbppp/2np1n2/2p1p3/4P3/1BPP1N1P/PP1N1PP1/R1BQ1RK1 w - - 2 9",
    "r1bq1rk1/pp2bppp/2np1n2/4p3/4P3/1BN1BN2/PPP2PPP/R2Q1RK1 b - - 5 8",
    # Middlegames with tactical complexity
    "r3k2r/pbppqppp/1pn2n2/4p3/1bB1P3/2NP1N2/PPPBQPPP/R3K2R w KQkq - 0 8",
    "r1bqk2r/ppp2ppp/2n1pn2/3p4/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 1 6",
    "2rq1rk1/pb1nbppp/1p2pn2/3p4/2PP4/P1N1PN2/1PQ1BPPP/R1B2RK1 b - - 3 11",
    # Sharp tactical positions
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
    "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3",
    # Endgames — king and pawn
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "8/5k2/8/4K3/8/8/3P4/8 b - - 0 1",
    "8/2k5/8/3P4/3K4/8/8/8 w - - 0 1",
    # Endgames — rook
    "8/8/8/3k4/8/8/4K3/R7 w - - 0 1",
    "8/5k2/8/8/8/5K2/r7/8 b - - 0 1",
    "4k3/8/8/8/8/8/R7/4K3 w - - 0 1",
    # Endgames — minor pieces
    "8/8/4k3/8/2N5/4K3/8/8 w - - 0 1",
    "8/8/3k4/8/3B4/3K4/8/8 b - - 0 1",
    "8/2k5/8/3B4/4N3/3K4/8/8 w - - 0 1",
    # Endgame — queen vs rook
    "8/8/8/4k3/8/8/4K3/3Q4 w - - 0 1",
    "8/8/5k2/8/8/5K2/8/4q3 b - - 0 1",
    # En-passant edge cases
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/1pp1pppp/8/p2pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3",
    # Castling rights preserved, sharp middlegame
    "r3k2r/pp1n1ppp/2pbpn2/q7/2BP4/2N1PN2/PP2QPPP/R1B2RK1 b kq - 5 10",
    # Black to move, complex middlegame
    "r2qkb1r/pp2nppp/2n1p3/3pP3/3P4/2PB1N2/P4PPP/RNBQK2R b KQkq - 0 8",
    # Promotion-adjacent endgame
    "8/1P6/8/8/8/8/6p1/4K2k w - - 0 1",
]

assert len(FIXED_FENS) == 32, f"Expected 32 FENs, got {len(FIXED_FENS)}"


def _tokenize_batch(fens) -> np.ndarray:
    return np.stack([np.asarray(tokenize(f), dtype=np.int64) for f in fens])


def _jax_logits(fens, *, dtype: str, model_name: str = "9M") -> np.ndarray:
    """Run the JAX reference model on the given FENs at the given dtype.

    Args:
        fens: list of FEN strings.
        dtype: "float32" or "float64". JAX_ENABLE_X64=True is set at module
            load, so float64 actually computes at 64 bits.

    Returns:
        ndarray [B, T, 128] at the requested dtype.
    """
    assert dtype in ("float32", "float64"), dtype
    from src.distill.teacher import MODEL_CONFIGS, _load_sc_module
    import jax
    import jax.numpy as jnp
    from jax import random as jrandom

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
    params = sc_training_utils.load_parameters(
        checkpoint_dir=os.path.abspath(f"searchless_chess/checkpoints/{model_name}"),
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=6_400_000,
    )

    target_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    params = jax.tree_util.tree_map(
        lambda x: x.astype(target_dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )

    tokens = _tokenize_batch(fens).astype(np.int32)
    log_probs = predictor.predict(params=params, targets=tokens, rng=None)
    return np.asarray(log_probs, dtype=np.dtype(dtype))


def _torch_logits(dtype: torch.dtype) -> np.ndarray:
    """Run the PyTorch DM port on FIXED_FENS at the given dtype."""
    from src.dm_port.transformer import DMTransformer, DMTransformerConfig

    ckpt = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = DMTransformerConfig(**ckpt["config"])
    model = DMTransformer(cfg).eval().to(dtype)
    # state_dict is saved as float32; to() above already handles the cast.
    model.load_state_dict({k: v.to(dtype) for k, v in ckpt["state_dict"].items()})

    tokens = _tokenize_batch(FIXED_FENS)
    with torch.no_grad():
        out = model(torch.from_numpy(tokens))
    return out.cpu().numpy()


def test_port_matches_jax_float64():
    """Hard gate: mathematical equivalence at float64 (~machine epsilon)."""
    torch_out = _torch_logits(torch.float64)
    jax_out = _jax_logits(FIXED_FENS, dtype="float64")

    assert torch_out.shape == jax_out.shape == (32, 77, 128), (
        f"Shape mismatch: torch={torch_out.shape}, jax={jax_out.shape}"
    )

    diff = float(np.abs(torch_out - jax_out).max())
    print(f"\nfloat64 max_abs_diff = {diff:.3e}")
    assert diff < 1e-10, (
        f"float64 max_abs_diff {diff:.3e} exceeds 1e-10 hard-gate threshold. "
        "The PyTorch DM port is NOT mathematically equivalent to the JAX "
        "reference. This is a real bug — fix src/dm_port/transformer.py or "
        "src/dm_port/convert_jax.py. Likely culprits: LN epsilon, PE indexing, "
        "weight transpose in convert_jax.py, residual scaling, attention einsum."
    )


def test_port_matches_jax_float32():
    """Sanity check: ~1e-4 CPU-kernel drift on log_softmax is expected.

    This threshold is deliberately loose (5e-4) because XLA and ATen matmul
    kernels accumulate in different orders. The real contract is the float64
    test above — if that passes at 1e-10, the port is correct regardless of
    what happens here.
    """
    torch_out = _torch_logits(torch.float32).astype(np.float32)
    jax_out = _jax_logits(FIXED_FENS, dtype="float32").astype(np.float32)

    assert torch_out.shape == jax_out.shape == (32, 77, 128), (
        f"Shape mismatch: torch={torch_out.shape}, jax={jax_out.shape}"
    )

    diff = float(np.abs(torch_out - jax_out).max())
    print(f"\nfloat32 max_abs_diff = {diff:.3e}")
    assert diff < 5e-4, (
        f"float32 max_abs_diff {diff:.3e} exceeds 5e-4 sanity threshold. "
        "This is well beyond CPU-kernel drift. If the float64 test above "
        "still passes at 1e-10, something unusual is going on in float32 "
        "accumulation. Investigate — do NOT just relax this threshold."
    )
