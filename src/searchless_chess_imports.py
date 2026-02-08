"""Imports from the searchless_chess submodule (google-deepmind/searchless_chess)."""
import importlib.util
from pathlib import Path

_SC_SRC = Path(__file__).resolve().parent.parent / "searchless_chess" / "src"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SC_SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils = _load("utils")
_tokenizer = _load("tokenizer")

ACTION_TO_MOVE = _utils.ACTION_TO_MOVE
MOVE_TO_ACTION = _utils.MOVE_TO_ACTION
tokenize = _tokenizer.tokenize
SEQUENCE_LENGTH = _tokenizer.SEQUENCE_LENGTH
