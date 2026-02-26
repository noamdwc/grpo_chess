import importlib
import sys


def test_import_src_does_not_eagerly_import_pytorch_lightning(monkeypatch):
    monkeypatch.delitem(sys.modules, "src", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch_lightning", raising=False)

    src = importlib.import_module("src")

    assert "pytorch_lightning" not in sys.modules
    # Accessing a lightweight lazy export should still work.
    assert src.ChessTransformerConfig is not None
