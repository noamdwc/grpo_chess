import sys

import pytest

from src.checkpoint_compat import (
    load_checkpoint_with_compat,
    load_state_dict_with_checkpoint_compat,
    register_legacy_checkpoint_aliases,
)
from src.models import ChessTransformer, ChessTransformerConfig


def _build_model(readout: str) -> ChessTransformer:
    return ChessTransformer(
        ChessTransformerConfig(
            vocab_size=16,
            embed_dim=8,
            num_layers=1,
            num_heads=1,
            action_dim=32,
            readout=readout,
            dropout=0.0,
        )
    )


def test_register_legacy_checkpoint_aliases_sets_grpo_self_play_alias():
    sys.modules.pop("src.grpo_self_play", None)
    register_legacy_checkpoint_aliases()
    assert "src.grpo_self_play" in sys.modules


def test_load_state_dict_with_checkpoint_compat_raises_cls_readout_message() -> None:
    legacy_model = _build_model("mean")
    cls_model = _build_model("cls")

    with pytest.raises(RuntimeError) as exc_info:
        load_state_dict_with_checkpoint_compat(
            cls_model,
            legacy_model.state_dict(),
            checkpoint_path="legacy-pretrain.pt",
            strict=False,
        )

    message = str(exc_info.value)
    assert "legacy-pretrain.pt" in message
    assert "transformer.readout='cls'" in message
    assert "embedding.weight shape mismatch" in message


def test_load_state_dict_with_checkpoint_compat_preserves_non_cls_errors() -> None:
    model = _build_model("last")
    incompatible_state = model.state_dict()
    incompatible_state["embedding.weight"] = incompatible_state["embedding.weight"][:-1]

    with pytest.raises(RuntimeError, match="size mismatch for embedding.weight"):
        load_state_dict_with_checkpoint_compat(
            model,
            incompatible_state,
            checkpoint_path="broken.pt",
            strict=False,
        )


def test_load_checkpoint_with_compat_retries_pretrainconfig_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_load(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise AttributeError("Can't get attribute 'PretrainConfig' on <module 'src.distill.distill'>")
        return {"model_state_dict": {"x": 1}}

    alias_count = {"n": 0}

    def fake_register_aliases() -> None:
        alias_count["n"] += 1

    monkeypatch.setattr("src.checkpoint_compat.torch.load", fake_load)
    monkeypatch.setattr("src.checkpoint_compat._register_legacy_dataclass_aliases", fake_register_aliases)

    ckpt = load_checkpoint_with_compat("legacy.pt", map_location="cpu", weights_only=False)
    assert ckpt == {"model_state_dict": {"x": 1}}
    assert call_count["n"] == 2
    assert alias_count["n"] == 1


def test_load_checkpoint_with_compat_preserves_unrelated_attribute_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(*args, **kwargs):
        raise AttributeError("Can't get attribute 'SomethingElse' on <module 'x'>")

    monkeypatch.setattr("src.checkpoint_compat.torch.load", fake_load)

    with pytest.raises(AttributeError, match="SomethingElse"):
        load_checkpoint_with_compat("legacy.pt", map_location="cpu", weights_only=False)
