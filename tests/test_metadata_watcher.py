"""Tests for scripts/watch_and_touch_metadata.py."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path


def _load_module():
    script_path = Path(__file__).parent.parent / "scripts" / "watch_and_touch_metadata.py"
    spec = importlib.util.spec_from_file_location("watch_and_touch_metadata", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _logger() -> logging.Logger:
    logger = logging.getLogger("test_metadata_watcher")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    return logger


def test_process_candidate_updates_mtime_and_atime(tmp_path):
    module = _load_module()
    file_path = tmp_path / "incoming.txt"
    file_path.write_text("x", encoding="utf-8")

    before = file_path.stat()
    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger())
    assert normalizer.process_candidate(file_path)

    after = file_path.stat()
    assert after.st_mtime_ns >= before.st_mtime_ns
    assert after.st_atime_ns >= before.st_atime_ns


def test_process_candidate_ignores_excluded_paths(tmp_path):
    module = _load_module()
    excluded_dir = tmp_path / ".git"
    excluded_dir.mkdir()
    file_path = excluded_dir / "ignored.txt"
    file_path.write_text("x", encoding="utf-8")

    calls = []

    def fake_touch(path):
        calls.append(path)
        return True

    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger(), touch_func=fake_touch)
    assert not normalizer.process_candidate(file_path)
    assert not calls


def test_process_candidate_dry_run_does_not_modify(tmp_path):
    module = _load_module()
    file_path = tmp_path / "incoming.txt"
    file_path.write_text("x", encoding="utf-8")

    before = file_path.stat().st_mtime_ns
    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger(), dry_run=True)
    assert normalizer.process_candidate(file_path)
    after = file_path.stat().st_mtime_ns

    assert after == before


def test_process_candidate_ignores_symlink(tmp_path):
    module = _load_module()
    target = tmp_path / "target.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    os.symlink(target, link)

    calls = []

    def fake_touch(path):
        calls.append(path)
        return True

    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger(), touch_func=fake_touch)
    assert not normalizer.process_candidate(link)
    assert not calls


def test_process_candidate_missing_file_does_not_crash(tmp_path):
    module = _load_module()
    missing = tmp_path / "missing.txt"

    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger())
    assert not normalizer.process_candidate(missing)


def test_process_candidate_deduplicates_bursty_events(tmp_path):
    module = _load_module()
    file_path = tmp_path / "incoming.txt"
    file_path.write_text("x", encoding="utf-8")

    calls = []

    def fake_touch(path):
        calls.append(path)
        return True

    normalizer = module.MetadataNormalizer(
        tmp_path,
        logger=_logger(),
        touch_func=fake_touch,
        cooldown_ms=5000,
    )
    assert normalizer.process_candidate(file_path)
    assert not normalizer.process_candidate(file_path)
    assert len(calls) == 1


def test_move_into_directory_can_be_processed(tmp_path):
    module = _load_module()
    src = tmp_path / "outside.txt"
    src.write_text("x", encoding="utf-8")

    dst_dir = tmp_path / "incoming"
    dst_dir.mkdir()
    dst = dst_dir / "moved.txt"
    src.rename(dst)

    normalizer = module.MetadataNormalizer(tmp_path, logger=_logger())
    assert normalizer.process_candidate(dst)
