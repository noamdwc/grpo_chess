import torch
import torch.nn as nn

from src.models import ChessTransformer, ChessTransformerConfig


class _PassThroughEncoder(nn.Module):
    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:  # noqa: D401
        return x


def _build_readout_probe(readout: str) -> ChessTransformer:
    cfg = ChessTransformerConfig(
        vocab_size=16,
        embed_dim=4,
        num_layers=1,
        num_heads=2,
        action_dim=4,
        readout=readout,
        head_mult=1,
        activation="relu",
        ffn_mult=1,
    )
    model = ChessTransformer(cfg)
    model.transformer = _PassThroughEncoder()
    model.policy_head = nn.Identity()
    with torch.no_grad():
        model.pos_encoding.zero_()
        values = torch.arange(model.embedding.num_embeddings, dtype=torch.float32).unsqueeze(1)
        model.embedding.weight.copy_(values.repeat(1, model.embedding.embedding_dim))
    return model


def test_last_readout_uses_last_non_pad_token() -> None:
    model = _build_readout_probe("last")
    x = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    out = model(x)

    expected = torch.tensor([[2.0, 2.0, 2.0, 2.0], [5.0, 5.0, 5.0, 5.0]])
    assert torch.allclose(out, expected)


def test_cls_readout_prepends_cls_token() -> None:
    model = _build_readout_probe("cls")
    x = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long)

    out = model(x)

    # cls_token_id defaults to vocab_size when readout='cls' (here: 16).
    expected = torch.full((2, 4), 16.0)
    assert torch.allclose(out, expected)

