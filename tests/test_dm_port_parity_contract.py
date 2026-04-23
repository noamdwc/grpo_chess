import importlib
import importlib.util
from pathlib import Path

PARITY_TEST_PATH = Path(__file__).with_name("test_dm_port_parity.py")
PARITY_CKPT_PATH = PARITY_TEST_PATH.parent.parent / "checkpoints" / "dm_port" / "9M.pt"
JAX_CKPT_PATH = PARITY_TEST_PATH.parent.parent / "searchless_chess" / "checkpoints" / "9M"
EXPECTED_SKIP_REASON = (
    "DM port checkpoint, JAX checkpoint, or JAX environment missing; run "
    "src.dm_port.convert_jax, sync searchless_chess checkpoints, and install JAX."
)


def _load_parity_module(
    monkeypatch,
    *,
    parity_checkpoint_exists=True,
    jax_checkpoint_exists=True,
    jax_import_mode="missing",
):
    real_import_module = importlib.import_module
    real_path_exists = Path.exists

    def fake_import_module(name, package=None):
        if name == "jax":
            if jax_import_mode == "missing":
                raise ImportError("simulated missing jax")
            if jax_import_mode == "forbid":
                raise AssertionError("jax import should not be attempted")
            if jax_import_mode == "real":
                return real_import_module(name, package)
            raise ValueError(f"Unsupported jax_import_mode: {jax_import_mode}")
        return real_import_module(name, package)

    def fake_exists(self):
        if self.resolve() == PARITY_CKPT_PATH.resolve():
            return parity_checkpoint_exists
        if self.resolve() == JAX_CKPT_PATH.resolve():
            return jax_checkpoint_exists
        return real_path_exists(self)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(Path, "exists", fake_exists)

    spec = importlib.util.spec_from_file_location(
        "dm_port_parity_contract_mod",
        PARITY_TEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dm_port_parity_skips_when_jax_environment_missing(monkeypatch):
    module = _load_parity_module(
        monkeypatch,
        parity_checkpoint_exists=True,
        jax_checkpoint_exists=True,
        jax_import_mode="missing",
    )

    assert module._parity_test_skip_reason() == EXPECTED_SKIP_REASON


def test_dm_port_parity_does_not_hide_non_import_jax_failures(monkeypatch):
    real_import_module = importlib.import_module
    real_path_exists = Path.exists

    def fake_import_module(name, package=None):
        if name == "jax":
            raise RuntimeError("broken jax runtime")
        return real_import_module(name, package)

    def fake_exists(self):
        if self.resolve() == PARITY_CKPT_PATH.resolve():
            return True
        if self.resolve() == JAX_CKPT_PATH.resolve():
            return True
        return real_path_exists(self)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(Path, "exists", fake_exists)

    spec = importlib.util.spec_from_file_location(
        "dm_port_parity_contract_broken_jax_mod",
        PARITY_TEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    import pytest

    with pytest.raises(RuntimeError, match="broken jax runtime"):
        spec.loader.exec_module(module)


def test_dm_port_parity_skips_when_checkpoint_is_missing(monkeypatch):
    module = _load_parity_module(
        monkeypatch,
        parity_checkpoint_exists=False,
        jax_checkpoint_exists=True,
        jax_import_mode="forbid",
    )

    assert module._parity_test_skip_reason() == EXPECTED_SKIP_REASON


def test_dm_port_parity_skips_when_jax_checkpoint_is_missing(monkeypatch):
    module = _load_parity_module(
        monkeypatch,
        parity_checkpoint_exists=True,
        jax_checkpoint_exists=False,
        jax_import_mode="forbid",
    )

    assert module._parity_test_skip_reason() == EXPECTED_SKIP_REASON
