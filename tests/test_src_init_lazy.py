import importlib
import sys


def test_import_src_does_not_eagerly_import_pytorch_lightning():
    sys.modules.pop("src", None)
    sys.modules.pop("pytorch_lightning", None)

    src = importlib.import_module("src")

    assert "pytorch_lightning" not in sys.modules
    # Accessing a lightweight lazy export should still work.
    assert src.ChessTransformerConfig is not None

