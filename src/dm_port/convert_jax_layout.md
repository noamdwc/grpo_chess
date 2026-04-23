# JAX Haiku Parameter Layout — DM 9M Searchless Chess

Observed via `--inspect` on checkpoint step 6,400,000. 8,954,240 total params.

## Key structure

There is **no module prefix** — all keys are at the root of the param tree.

## Embeddings

- `embed/embeddings` — token embedding, shape `(1968, 256)` (vocab_size × embedding_dim)
- `embed_1/embeddings` — positional embedding, shape `(79, 256)` (max_seq_len × embedding_dim)

## Layer Norms

Haiku numbers LayerNorms globally in call order. With 8 blocks and 2 LNs per block plus 1 post-LN:

- `layer_norm/scale`, `layer_norm/offset` — block 0 attention pre-LN (index 0, no number suffix)
- `layer_norm_1/...` — block 0 MLP pre-LN
- `layer_norm_{2*i}/...` — block i attention pre-LN (i=1..7 have numeric suffix)
- `layer_norm_{2*i+1}/...` — block i MLP pre-LN
- `layer_norm_16/...` — post-LN (the final LayerNorm after all blocks)

## Attention Q/K/V/Out

Each block has its own **named** Haiku sub-module (not the global counter):

- Block 0: `multi_head_dot_product_attention/linear/w` (Q), `.../linear_1/w` (K), `.../linear_2/w` (V), `.../linear_3/w` (Out)
- Block i (i≥1): `multi_head_dot_product_attention_{i}/linear/w`, etc.

All shapes are `(256, 256)` — (embedding_dim, embedding_dim). Transpose required for nn.Linear.

## MLP (SwiGLU w1/w2/w3)

MLP linears use the **global** counter (`linear`, `linear_1`, ..., `linear_23`), 3 per block:

- Block i: `linear_{3*i}` = w1 (256→1024), `linear_{3*i+1}` = w2 (256→1024), `linear_{3*i+2}` = w3 (1024→256)
- Block 0 uses `linear/w`, `linear_1/w`, `linear_2/w` (the first `linear` omits its `_0` suffix per Haiku convention). Blocks 1-7 use `linear_3/w` through `linear_23/w`.

## Output head

`linear_24/w` (256→128) and `linear_24/b` (128,) — the only linear with a bias.

## Gotchas

- The first LayerNorm and first MLP Linear have **no numeric suffix** (Haiku omits `_0`).
- Attention linears live under named sub-modules, not the global `linear_N` counter.
- All hk.Linear weights are stored as `(in_features, out_features)` → transpose on load for nn.Linear `(out_features, in_features)`.
