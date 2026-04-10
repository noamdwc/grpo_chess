"""Construction + forward smoke tests for ReasoningModel."""
import copy
from pathlib import Path

import pytest
import torch

from src.reasoning.tokens import TOTAL_VOCAB_SIZE, FEN_LEN

CKPT = Path("checkpoints/dm_port/9M.pt")
requires_ckpt = pytest.mark.skipif(
    not CKPT.exists(),
    reason="DM port checkpoint missing; run Phase 0 first.",
)


@requires_ckpt
def test_construction_extends_vocab_and_pe_correctly():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    cfg = ReasoningModelConfig(
        dm_checkpoint=str(CKPT),
        max_seq_len=120,
        value_head_hidden=64,
    )
    model = ReasoningModel(cfg).eval()
    # Embedding: base 1968 + 3 specials = 1971
    assert model.dm.token_embedding.weight.shape[0] == TOTAL_VOCAB_SIZE
    # PE: extended to max_seq_len
    assert model.dm.pos_embedding.weight.shape[0] == cfg.max_seq_len
    # Output head: full vocab
    assert model.policy_head.out_features == TOTAL_VOCAB_SIZE


@requires_ckpt
def test_forward_shapes_with_padding():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=100)
    model = ReasoningModel(cfg).eval()
    B, T = 2, 90
    seq = torch.zeros((B, T), dtype=torch.long)
    seq_lens = torch.tensor([85, 90])
    with torch.no_grad():
        out = model(seq, seq_lens)
    assert out.logits.shape == (B, T, TOTAL_VOCAB_SIZE)
    assert out.hidden.shape[:2] == (B, T)


@requires_ckpt
def test_construction_copies_last_dm_positional_row_into_extension():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    ckpt = torch.load(CKPT, weights_only=False)
    dm_pe = ckpt["state_dict"]["pos_embedding.weight"]
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    assert torch.equal(model.dm.pos_embedding.weight[: dm_pe.shape[0]], dm_pe)
    extended = model.dm.pos_embedding.weight[dm_pe.shape[0] :]
    assert torch.equal(extended, dm_pe[-1].expand_as(extended))


@requires_ckpt
def test_construction_preserves_pretrained_token_rows():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    ckpt = torch.load(CKPT, weights_only=False)
    dm_tok = ckpt["state_dict"]["token_embedding.weight"]
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    assert torch.equal(model.dm.token_embedding.weight[: dm_tok.shape[0]], dm_tok)


def test_construction_rejects_too_short_max_seq_len():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    with pytest.raises(ValueError, match=f"max_seq_len must be at least {FEN_LEN}"):
        ReasoningModel(ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=FEN_LEN - 1))


@requires_ckpt
def test_construction_rejects_missing_checkpoint_body_keys(monkeypatch):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    ckpt = torch.load(CKPT, weights_only=False)
    broken = copy.deepcopy(ckpt)
    broken["state_dict"].pop("layers.0.attn_ln.weight")

    real_load = torch.load

    def fake_load(path, *args, **kwargs):
        if path == str(CKPT):
            return broken
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="Missing checkpoint keys"):
        ReasoningModel(ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120))
