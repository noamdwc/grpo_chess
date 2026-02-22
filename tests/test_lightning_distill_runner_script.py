from pathlib import Path


SCRIPT_PATH = Path("scripts/lightning/run_distill_labelsafe.sh")


def _script_text() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_runner_uses_non_debug_conversion_defaults():
    text = _script_text()
    assert 'DISTILL_MODE="${DISTILL_MODE:-fast}"' in text
    assert 'DEEPMIND_NUM_SHARDS="${DEEPMIND_NUM_SHARDS:-8}"' in text
    assert 'DEEPMIND_MIN_WIN_PROB="${DEEPMIND_MIN_WIN_PROB:-0.55}"' in text
    assert 'DISTILL_MAX_SHARDS="${DISTILL_MAX_SHARDS:-8}"' in text


def test_runner_includes_dataset_metadata_and_rebuild_controls():
    text = _script_text()
    assert 'DISTILL_FORCE_REBUILD_DATA="${DISTILL_FORCE_REBUILD_DATA:-0}"' in text
    assert 'DISTILL_DATASET_TAG="${DISTILL_DATASET_TAG:-${DISTILL_DATASET_TAG_DEFAULT}}"' in text
    assert 'DISTILL_REMOTE_DATA_DIR_DEFAULT="distill_data_quality"' in text
    assert 'PERSIST_DATA_REMOTE_DIR="${PERSIST_DATA_REMOTE_DIR:-${DISTILL_REMOTE_DATA_DIR_DEFAULT}}"' in text
    assert 'DATASET_META_PATH="${DATA_DIR}/_build_meta.json"' in text
    assert 'DATASET_QUALITY_STATS_PATH="${DATA_DIR}/dataset_quality_stats.json"' in text
    assert 'read_dataset_meta_tag()' in text
    assert 'write_dataset_meta()' in text
    assert 'run_dataset_preflight()' in text
    assert 'dataset tag mismatch' in text


def test_runner_writes_quality_gate_to_generated_config():
    text = _script_text()
    assert "quality_gate_min_val_top1: ${DISTILL_QUALITY_GATE_MIN_VAL_TOP1}" in text
    assert "quality_gate_epoch: ${DISTILL_QUALITY_GATE_EPOCH}" in text


def test_runner_supports_quality_mode_dataset_generation():
    text = _script_text()
    assert 'if [[ "${DISTILL_MODE}" == "fast" ]]; then' in text
    assert "python -m src.distill.convert_deepmind_data" in text
    assert 'bash scripts/setup_distill_deps.sh --checkpoint "${TEACHER_MODEL}"' in text
    assert "-m src.distill.generate_dataset" in text
