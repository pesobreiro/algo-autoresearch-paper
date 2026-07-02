"""Local replacement for ml_sessions/features/technical.py.

Re-exports the audited merge logic already present in the repo.
"""
from pipeline.merge_timeframes_internal import merge_timeframes, get_feature_columns

__all__ = ["merge_timeframes", "get_feature_columns"]
