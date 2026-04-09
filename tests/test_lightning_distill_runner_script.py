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
    assert 'DISABLE_LIGHTNING_SITECUSTOMIZE="${DISABLE_LIGHTNING_SITECUSTOMIZE:-1}"' in text
    assert 'sitecustomize.py' in text
    assert 'QUALITY_REPAIR_PANDAS_ABI="${QUALITY_REPAIR_PANDAS_ABI:-1}"' in text
    assert 'LIGHTNING_SHARED_CACHE_DIR="${LIGHTNING_SHARED_CACHE_DIR:-/teamspace/uploads/grpo_chess_artifacts}"' in text
    assert 'if mkdir -p "${LIGHTNING_SHARED_CACHE_DIR}" 2>/dev/null; then' in text
    assert 'AUTO_USE_EFS_CONNECTION_CACHE="${AUTO_USE_EFS_CONNECTION_CACHE:-1}"' in text
    assert 'if [[ "${AUTO_USE_EFS_CONNECTION_CACHE}" == "1" ]] && [[ -d "/teamspace/efs_connections" ]]; then' in text


def test_runner_writes_quality_gate_to_generated_config():
    text = _script_text()
    assert "quality_gate_min_val_top1: ${DISTILL_QUALITY_GATE_MIN_VAL_TOP1}" in text
    assert "quality_gate_epoch: ${DISTILL_QUALITY_GATE_EPOCH}" in text
    assert "distill_lambda_soft: ${DISTILL_LAMBDA_SOFT}" in text
    assert "distill_lambda_hard: ${DISTILL_LAMBDA_HARD}" in text
    assert "distill_target_top1_mix_alpha: ${DISTILL_TARGET_TOP1_MIX_ALPHA}" in text
    assert 'teacher_target_score_norm: "${TEACHER_TARGET_SCORE_NORM}"' in text


def test_runner_supports_quality_mode_dataset_generation():
    text = _script_text()
    assert 'if [[ "${DISTILL_MODE}" == "fast" ]]; then' in text
    assert "python -m src.distill.convert_deepmind_data" in text
    assert 'bash scripts/setup_distill_deps.sh --checkpoint "${TEACHER_MODEL}"' in text
    assert 'ensure_quality_generation_stack "${QUALITY_REPAIR_PANDAS_ABI}"' in text
    assert "-m src.distill.generate_dataset" in text
    assert 'restore_distill_runtime_numpy()' in text
    assert 'restore_distill_runtime_numpy "${DISTILL_TORCH_RUNTIME_NUMPY}"' in text
