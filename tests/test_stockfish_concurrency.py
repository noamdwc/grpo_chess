"""Test that Stockfish is both thread-safe and process-safe.

This test verifies that multiple threads and processes can safely
use Stockfish engines concurrently without race conditions or crashes.

Process safety tests run Stockfish in separate subprocesses (not
ProcessPoolExecutor) because python-chess SimpleEngine uses asyncio/threads
internally and can hang when used inside ProcessPoolExecutor workers.
"""
import os
import sys
import shutil
import subprocess
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pytest
import chess

from src.grpo_self_play.chess.stockfish import (
    StockfishManager,
    StockfishConfig,
    stockfish_analyse,
    stockfish_play,
    DEFAULT_STOCKFISH_TIMEOUT,
)


def get_stockfish_path() -> str:
    """Find Stockfish binary path."""
    # Check common locations
    paths_to_try = [
        shutil.which("stockfish"),  # In PATH
        "/opt/homebrew/bin/stockfish",  # macOS Homebrew ARM
        "/usr/local/bin/stockfish",  # macOS Homebrew Intel
        "/usr/games/stockfish",  # Linux
        "/usr/bin/stockfish",  # Linux alternative
    ]
    for path in paths_to_try:
        if path and os.path.isfile(path):
            return path
    return None


# Test positions (various game states)
TEST_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1.e4
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Italian opening
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Giuoco Piano
    "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",  # Sicilian
]


def _get_test_config() -> StockfishConfig:
    """Get test config with correct stockfish path (needed for subprocess workers)."""
    path = get_stockfish_path()
    if path is None:
        raise RuntimeError("Stockfish not found")
    return StockfishConfig(path=path, skill_level=1, movetime_ms=10)


@pytest.fixture(scope="module", autouse=True)
def require_stockfish():
    """Skip all tests if stockfish is not available."""
    path = get_stockfish_path()
    if path is None:
        pytest.skip("Stockfish not found on system")


def _worker_analyse(args):
    """Worker function for analyse tests (works in both threads and processes)."""
    engine_name, fen, idx = args
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=5)
    cfg = _get_test_config()

    result = stockfish_analyse(engine_name, board, limit, timeout=DEFAULT_STOCKFISH_TIMEOUT, cfg=cfg)
    return idx, fen, result is not None


def _worker_play(args):
    """Worker function for play tests (works in both threads and processes)."""
    engine_name, fen, idx = args
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=5)
    cfg = _get_test_config()

    move = stockfish_play(engine_name, board, limit, timeout=DEFAULT_STOCKFISH_TIMEOUT, cfg=cfg)
    is_valid = move is not None and move in board.legal_moves
    return idx, fen, is_valid


def _process_worker_analyse(args):
    """Process worker - engines are recreated per process."""
    engine_name, fen, idx = args
    # In a new process, engine will be created fresh due to ensure_pid()
    return _worker_analyse(args)


def _process_worker_play(args):
    """Process worker - engines are recreated per process."""
    engine_name, fen, idx = args
    return _worker_play(args)


def _check_engine_not_inherited(name):
    """In a child process, the engine should not be inherited (module-level for pickling)."""
    StockfishManager.ensure_pid()
    return not StockfishManager.is_name_registered(name)


def _run_stockfish_task_in_subprocess(mode: str, engine_name: str, pos_index: int, idx: int, timeout: float = 60) -> tuple[int, bool]:
    """Run one analyse or play task in a separate process. Returns (idx, success)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get stockfish path in parent process and pass it to child
    stockfish_path = get_stockfish_path() or "/usr/games/stockfish"
    # Pass repo root as first arg so the child uses the same path (works in Colab / any cwd).
    cmd = [
        sys.executable,
        "-c",
        """
import sys
import traceback
root = sys.argv[1]
sys.path.insert(0, root)
pos_index = int(sys.argv[2])
idx = int(sys.argv[3])
mode = sys.argv[4]
engine_name = sys.argv[5]
stockfish_path = sys.argv[6]
try:
    import chess
    from src.grpo_self_play.chess.stockfish import (
        StockfishManager, StockfishConfig, stockfish_analyse, stockfish_play,
        DEFAULT_STOCKFISH_TIMEOUT,
    )
    TEST_POSITIONS = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",
    ]
    fen = TEST_POSITIONS[pos_index % len(TEST_POSITIONS)]
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=5)
    cfg = StockfishConfig(path=stockfish_path, skill_level=1, movetime_ms=10)
    if mode == "analyse":
        result = stockfish_analyse(engine_name, board, limit, timeout=DEFAULT_STOCKFISH_TIMEOUT, cfg=cfg)
        success = result is not None
    else:
        move = stockfish_play(engine_name, board, limit, timeout=DEFAULT_STOCKFISH_TIMEOUT, cfg=cfg)
        success = move is not None and move in board.legal_moves
    print(idx, 1 if success else 0, sep="\\t")
except Exception as e:
    print(f"SUBPROCESS_ERROR: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
finally:
    try:
        from src.grpo_self_play.chess.stockfish import StockfishManager
        StockfishManager.close_all()
    except Exception:
        pass
""",
        root,
        str(pos_index),
        str(idx),
        mode,
        engine_name,
        stockfish_path,
    ]
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.pathsep.join([root, env.get("PYTHONPATH", "")]).rstrip(os.path.pathsep)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
            env=env,
        )
        if proc.returncode != 0:
            # Log stderr for debugging
            if proc.stderr:
                print(f"Subprocess {idx} stderr: {proc.stderr[:500]}", file=sys.stderr)
            return idx, False
        line = (proc.stdout or "").strip().split("\n")[-1]
        parts = line.split("\t")
        if len(parts) >= 2:
            return int(parts[0]), parts[1] == "1"
        return idx, False
    except subprocess.TimeoutExpired:
        print(f"Subprocess {idx} timed out after {timeout}s", file=sys.stderr)
        return idx, False
    except ValueError as e:
        print(f"Subprocess {idx} value error: {e}", file=sys.stderr)
        return idx, False


class TestStockfishThreadSafety:
    """Test thread safety of Stockfish operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up engines after each test."""
        yield
        StockfishManager.close_all()

    def test_concurrent_analyse_same_engine(self):
        """Multiple threads calling analyse on the same engine name."""
        engine_name = "thread_test_analyse"
        num_tasks = 20

        # Create tasks - multiple positions, same engine
        tasks = [
            (engine_name, TEST_POSITIONS[i % len(TEST_POSITIONS)], i)
            for i in range(num_tasks)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker_analyse, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        # All should succeed
        assert len(results) == num_tasks
        successes = sum(1 for _, _, success in results if success)
        assert successes == num_tasks, f"Only {successes}/{num_tasks} analyses succeeded"

    def test_concurrent_play_same_engine(self):
        """Multiple threads calling play on the same engine name."""
        engine_name = "thread_test_play"
        num_tasks = 20

        tasks = [
            (engine_name, TEST_POSITIONS[i % len(TEST_POSITIONS)], i)
            for i in range(num_tasks)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker_play, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == num_tasks
        successes = sum(1 for _, _, success in results if success)
        assert successes == num_tasks, f"Only {successes}/{num_tasks} plays succeeded"

    def test_concurrent_mixed_operations(self):
        """Multiple threads doing both analyse and play on the same engine."""
        engine_name = "thread_test_mixed"
        num_tasks = 10

        analyse_tasks = [
            (engine_name, TEST_POSITIONS[i % len(TEST_POSITIONS)], i)
            for i in range(num_tasks)
        ]
        play_tasks = [
            (engine_name, TEST_POSITIONS[i % len(TEST_POSITIONS)], i + num_tasks)
            for i in range(num_tasks)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            analyse_futures = [executor.submit(_worker_analyse, task) for task in analyse_tasks]
            play_futures = [executor.submit(_worker_play, task) for task in play_tasks]

            for future in as_completed(analyse_futures + play_futures):
                results.append(future.result())

        assert len(results) == num_tasks * 2
        successes = sum(1 for _, _, success in results if success)
        assert successes == num_tasks * 2, f"Only {successes}/{num_tasks * 2} operations succeeded"

    def test_multiple_engines_concurrent(self):
        """Multiple threads using different engine names concurrently."""
        num_engines = 4
        num_tasks_per_engine = 5

        tasks = []
        for eng_idx in range(num_engines):
            engine_name = f"thread_test_multi_{eng_idx}"
            for task_idx in range(num_tasks_per_engine):
                fen = TEST_POSITIONS[(eng_idx + task_idx) % len(TEST_POSITIONS)]
                tasks.append((engine_name, fen, eng_idx * num_tasks_per_engine + task_idx))

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_worker_analyse, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        total = num_engines * num_tasks_per_engine
        assert len(results) == total
        successes = sum(1 for _, _, success in results if success)
        assert successes == total, f"Only {successes}/{total} operations succeeded"


class TestStockfishProcessSafety:
    """Test process safety of Stockfish operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up engines after each test."""
        yield
        StockfishManager.close_all()

    def test_concurrent_analyse_multiple_processes(self):
        """Multiple processes calling analyse concurrently (via subprocess, not ProcessPoolExecutor)."""
        engine_name = "process_test_analyse"
        num_tasks = 4

        # Run each task in a separate OS process to avoid ProcessPoolExecutor + python-chess hang.
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_run_stockfish_task_in_subprocess, "analyse", engine_name, i, i, 60)
                for i in range(num_tasks)
            ]
            for future in as_completed(futures, timeout=90):
                results.append(future.result())

        assert len(results) == num_tasks
        successes = sum(1 for _, success in results if success)
        assert successes == num_tasks, f"Only {successes}/{num_tasks} analyses succeeded"

    def test_concurrent_play_multiple_processes(self):
        """Multiple processes calling play concurrently (via subprocess, not ProcessPoolExecutor)."""
        engine_name = "process_test_play"
        num_tasks = 4

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_run_stockfish_task_in_subprocess, "play", engine_name, i, i, 60)
                for i in range(num_tasks)
            ]
            for future in as_completed(futures, timeout=90):
                results.append(future.result())

        assert len(results) == num_tasks
        successes = sum(1 for _, success in results if success)
        assert successes == num_tasks, f"Only {successes}/{num_tasks} plays succeeded"

    def test_process_isolation(self):
        """Verify that engines are properly isolated between processes."""
        # Create an engine in the main process
        main_engine_name = "main_process_engine"
        cfg = _get_test_config()
        StockfishManager.get_engine(main_engine_name, cfg)

        assert StockfishManager.is_name_registered(main_engine_name)

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(_check_engine_not_inherited, main_engine_name)
            result = future.result(timeout=30)

        assert result, "Child process should not inherit parent's engine"
        # Main process should still have the engine
        assert StockfishManager.is_name_registered(main_engine_name)


class TestStockfishStress:
    """Stress tests for Stockfish concurrency."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up engines after each test."""
        yield
        StockfishManager.close_all()

    def test_high_concurrency_threads(self):
        """Stress test with many concurrent threads."""
        engine_name = "stress_thread"
        num_tasks = 50

        tasks = [
            (engine_name, TEST_POSITIONS[i % len(TEST_POSITIONS)], i)
            for i in range(num_tasks)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(_worker_play, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        assert len(results) == num_tasks
        successes = sum(1 for _, _, success in results if success)
        # Allow some tolerance for timeouts under heavy load
        assert successes >= num_tasks * 0.9, f"Only {successes}/{num_tasks} operations succeeded"

    def test_mixed_threads_and_processes(self):
        """Test with both threads and processes (subprocess) running concurrently."""
        thread_engine = "mixed_thread"
        process_engine = "mixed_process"
        num_thread_tasks = 10
        num_process_tasks = 4

        thread_tasks = [
            (thread_engine, TEST_POSITIONS[i % len(TEST_POSITIONS)], i)
            for i in range(num_thread_tasks)
        ]

        thread_results = []
        process_results = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            thread_futures = [executor.submit(_worker_play, task) for task in thread_tasks]
            process_futures = [
                executor.submit(_run_stockfish_task_in_subprocess, "play", process_engine, i, i, 60)
                for i in range(num_process_tasks)
            ]
            for future in as_completed(thread_futures + process_futures, timeout=90):
                r = future.result()
                if len(r) == 3:  # thread result (idx, fen, success)
                    thread_results.append(r)
                else:  # subprocess result (idx, success)
                    process_results.append(r)

        thread_successes = sum(1 for _, _, success in thread_results if success)
        process_successes = sum(1 for _, success in process_results if success)

        assert thread_successes == num_thread_tasks, f"Thread: {thread_successes}/{num_thread_tasks}"
        assert process_successes == num_process_tasks, f"Process: {process_successes}/{num_process_tasks}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
