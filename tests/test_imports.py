"""Smoke tests: ensure the public repo imports work without ~/git/ml_sessions."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_ml_sessions_compat_config():
    import ml_sessions_compat.config as cfg
    assert hasattr(cfg, "DATA_DIR")


def test_ml_sessions_compat_features():
    from ml_sessions_compat.features.technical import merge_timeframes, get_feature_columns
    from ml_sessions_compat.features.temporal import TEMPORAL_FEATURE_NAMES
    assert callable(merge_timeframes)
    assert callable(get_feature_columns)
    assert isinstance(TEMPORAL_FEATURE_NAMES, list)


def test_pipeline_merge_timeframes_internal():
    from pipeline.merge_timeframes_internal import merge_timeframes, get_feature_columns
    assert callable(merge_timeframes)
    assert callable(get_feature_columns)
