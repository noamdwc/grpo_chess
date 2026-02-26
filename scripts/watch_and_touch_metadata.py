#!/usr/bin/env python3
"""Watch a directory and normalize file timestamps for incoming files."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

DEFAULT_EXCLUDES = [
    ".git",
    ".pytest_cache",
    "__pycache__",
    "checkpoints",
    "lightning_logs",
]

IGNORED_SUFFIXES = ("~", ".swp", ".tmp")
IGNORED_NAMES = {".DS_Store"}


def build_logger(log_level: str) -> logging.Logger:
    logger = logging.getLogger("metadata_watcher")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger


def _is_temp_file(path: Path) -> bool:
    name = path.name
    if name in IGNORED_NAMES:
        return True
    return name.endswith(IGNORED_SUFFIXES)


def touch_file_metadata(path: Path, *, dry_run: bool, logger: logging.Logger) -> bool:
    if dry_run:
        logger.info("[dry-run] would touch %s", path)
        return True

    now = time.time()
    os.utime(path, times=(now, now), follow_symlinks=False)
    logger.info("touched %s", path)
    return True


class MetadataNormalizer:
    """Core metadata normalization logic used by CLI and watcher callbacks."""

    def __init__(
        self,
        root: Path,
        *,
        excludes: list[str] | None = None,
        dry_run: bool = False,
        cooldown_ms: int = 500,
        logger: logging.Logger | None = None,
        touch_func: Callable[[Path], bool] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.excludes = list(excludes or DEFAULT_EXCLUDES)
        self.dry_run = dry_run
        self.cooldown_sec = max(0, cooldown_ms) / 1000.0
        self.logger = logger or build_logger("INFO")
        self._touch_func = touch_func or (
            lambda p: touch_file_metadata(p, dry_run=self.dry_run, logger=self.logger)
        )
        self._recent: dict[tuple[str, int], float] = {}

    def _is_excluded(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return True

        parts = rel.parts
        return any(part in self.excludes for part in parts)

    def _prune_recent(self, now: float) -> None:
        cutoff = now - self.cooldown_sec
        stale = [key for key, ts in self._recent.items() if ts < cutoff]
        for key in stale:
            self._recent.pop(key, None)

    def process_candidate(self, path: str | Path) -> bool:
        candidate = Path(path)

        if _is_temp_file(candidate):
            return False

        if self._is_excluded(candidate):
            return False

        try:
            if candidate.is_symlink() or not candidate.is_file():
                return False

            stat_result = candidate.stat()
            key = (str(candidate.resolve()), stat_result.st_mtime_ns)

            now_mono = time.monotonic()
            self._prune_recent(now_mono)
            if key in self._recent:
                return False
            self._recent[key] = now_mono

            return self._touch_func(candidate)
        except FileNotFoundError:
            self.logger.debug("file disappeared before processing: %s", candidate)
            return False
        except PermissionError:
            self.logger.warning("permission denied touching %s", candidate)
            return False
        except OSError as exc:
            self.logger.warning("failed touching %s: %s", candidate, exc)
            return False


def run_watch_loop(normalizer: MetadataNormalizer) -> int:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("Missing dependency: watchdog. Install with `pip install watchdog`.", file=sys.stderr)
        return 1

    class Handler(FileSystemEventHandler):
        def on_created(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            normalizer.process_candidate(event.src_path)

        def on_moved(self, event) -> None:  # type: ignore[override]
            if event.is_directory:
                return
            normalizer.process_candidate(event.dest_path)

    observer = Observer()
    observer.schedule(Handler(), str(normalizer.root), recursive=True)
    observer.start()
    normalizer.logger.info("watching %s", normalizer.root)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        normalizer.logger.info("stopping watcher")
    finally:
        observer.stop()
        observer.join()

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch folder and normalize metadata (mtime+atime) for incoming files."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory to watch")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional path segment to exclude (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log actions without changing files")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger level",
    )
    parser.add_argument(
        "--once",
        type=Path,
        help="Process a single file and exit (no watcher required)",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=500,
        help="Dedup window in milliseconds",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid --root directory: {root}", file=sys.stderr)
        return 1

    excludes = DEFAULT_EXCLUDES + list(args.exclude)
    logger = build_logger(args.log_level)
    normalizer = MetadataNormalizer(
        root,
        excludes=excludes,
        dry_run=args.dry_run,
        cooldown_ms=args.cooldown_ms,
        logger=logger,
    )

    if args.once is not None:
        target = args.once
        if not target.is_absolute():
            target = root / target
        normalizer.process_candidate(target)
        return 0

    return run_watch_loop(normalizer)


if __name__ == "__main__":
    raise SystemExit(main())
