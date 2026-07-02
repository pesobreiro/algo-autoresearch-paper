"""
Orchestrator for the algo_autoresearch pipeline.

Coordinates: generate_labels → train → backtest
With label cache keyed by hash of the input parameters.
"""
import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from pipeline.generate_labels import generate_labels
from pipeline.train import train_model
from pipeline.backtest import run_backtest


@dataclass
class PipelineResult:
    """Complete result of one pipeline iteration."""
    success: bool
    metrics: dict = field(default_factory=dict)
    stats_labels: dict = field(default_factory=dict)
    stats_train: dict = field(default_factory=dict)
    labels_reused: bool = False
    total_duration_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    # Backward-compatible property aliases for external callers
    @property
    def sucesso(self) -> bool:
        return self.success

    @property
    def metricas(self) -> dict:
        return self.metrics

    @property
    def stats_treino(self) -> dict:
        return self.stats_train

    @property
    def labels_reutilizados(self) -> bool:
        return self.labels_reused

    @property
    def duracao_total_segundos(self) -> float:
        return self.total_duration_seconds

    @property
    def erro(self) -> Optional[str]:
        return self.error


def load_params(params_path: Path) -> dict:
    """Load research_params.py as a dict."""
    spec = importlib.util.spec_from_file_location("research_params", params_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return {
        'FEATURES':               getattr(mod, 'FEATURES', []),
        'TIMEFRAMES':             getattr(mod, 'TIMEFRAMES', ['15m', '4h']),
        'STOCH_RSI_PERIOD':       getattr(mod, 'STOCH_RSI_PERIOD', 14),
        'ADX_PERIOD':             getattr(mod, 'ADX_PERIOD', 14),
        'EMA_FAST':               getattr(mod, 'EMA_FAST', 12),
        'EMA_SLOW':               getattr(mod, 'EMA_SLOW', 26),
        'BB_PERIOD':              getattr(mod, 'BB_PERIOD', 20),
        'ENTRY_STOCH_THRESHOLD':  getattr(mod, 'ENTRY_STOCH_THRESHOLD', 20),
        'ENTRY_ADX_THRESHOLD':    getattr(mod, 'ENTRY_ADX_THRESHOLD', 25),
        'N_ESTIMATORS':           getattr(mod, 'N_ESTIMATORS', 200),
        'MAX_DEPTH':              getattr(mod, 'MAX_DEPTH', 6),
        'LEARNING_RATE':          getattr(mod, 'LEARNING_RATE', 0.05),
        'MIN_CHILD_WEIGHT':       getattr(mod, 'MIN_CHILD_WEIGHT', 5),
        'GAMMA':                  getattr(mod, 'GAMMA', 1.0),
        'SUBSAMPLE':              getattr(mod, 'SUBSAMPLE', 0.7),
        'COLSAMPLE_BYTREE':       getattr(mod, 'COLSAMPLE_BYTREE', 0.7),
        'REG_ALPHA':              getattr(mod, 'REG_ALPHA', 0.5),
        'REG_LAMBDA':             getattr(mod, 'REG_LAMBDA', 1.0),
        'SL_RANGE':               getattr(mod, 'SL_RANGE', (0.5, 12.0)),
        'TP_RANGE':               getattr(mod, 'TP_RANGE', (1.0, 40.0)),
        'THRESHOLD_RANGE':        getattr(mod, 'THRESHOLD_RANGE', (0.30, 0.80)),
        'N_TRIALS':               getattr(mod, 'N_TRIALS', 120),
        # XGBoost Optuna ranges (optional — activate hyperparameter optimization)
        'DEPTH_RANGE':            getattr(mod, 'DEPTH_RANGE', None),
        'LR_RANGE':               getattr(mod, 'LR_RANGE', None),
        'ESTIMATORS_RANGE':       getattr(mod, 'ESTIMATORS_RANGE', None),
        'ALPHA_RANGE':            getattr(mod, 'ALPHA_RANGE', None),
        'LAMBDA_RANGE':           getattr(mod, 'LAMBDA_RANGE', None),
        'N_TRIALS_XGB':           getattr(mod, 'N_TRIALS_XGB', 0),
        'OBJECTIVE_MODE':         getattr(mod, 'OBJECTIVE_MODE', 'score'),
    }


def hash_entry_params(params: dict, train_start: int = 2019, train_end: int = 2021) -> str:
    """Hash of the parameters that affect label generation."""
    key = {
        'ENTRY_STOCH_THRESHOLD': params['ENTRY_STOCH_THRESHOLD'],
        'ENTRY_ADX_THRESHOLD':   params['ENTRY_ADX_THRESHOLD'],
        'SL_RANGE':              list(params.get('SL_RANGE', (0.5, 12.0))),
        'TP_RANGE':              list(params.get('TP_RANGE', (1.0, 40.0))),
        'THRESHOLD_RANGE':       list(params.get('THRESHOLD_RANGE', (0.30, 0.80))),
        'train_start':           train_start,
        'train_end':             train_end,
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:8]


def hash_params_complete(params: dict) -> str:
    """Hash of all parameters (to identify the iteration)."""
    serializable = {k: (sorted(v) if isinstance(v, list) else v)
                    for k, v in params.items()}
    return hashlib.md5(json.dumps(serializable, sort_keys=True).encode()).hexdigest()[:8]


def run_pipeline(config: dict, params_path: Path,
                 cache_dir: Path) -> PipelineResult:
    """
    Run the complete pipeline: labels → train → backtest.

    Label cache: labels are reused if ENTRY_* params did not change.
    Train and backtest are always re-executed.

    Args:
        config: system configuration (config.yaml)
        params_path: path to research_params.py
        cache_dir: directory for label and model cache

    Returns:
        PipelineResult with metrics and metadata
    """
    t_start = time.time()

    try:
        params = load_params(params_path)
    except Exception as e:
        return PipelineResult(success=False, error=f"Error loading params: {e}")

    train_start   = config['pipeline'].get('train_start', 2017)
    train_end     = config['pipeline'].get('train_end', 2024)
    params_hash   = hash_entry_params(params, train_start=train_start, train_end=train_end)
    full_hash = hash_params_complete(params)

    labels_dir  = cache_dir / 'labels'
    model_dir   = cache_dir / 'models' / full_hash
    ticker      = config['pipeline']['ticker']
    labels_path = labels_dir / f'labels_{ticker}_{params_hash}.parquet'
    hash_file   = labels_dir / f'hash_atual_{ticker}.txt'

    labels_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Labels (with cache) ---
    labels_reused = False
    stats_labels = {}

    previous_hash = hash_file.read_text().strip() if hash_file.exists() else ""

    if labels_path.exists() and previous_hash == params_hash:
        print(f"\n  [CACHE] Labels reused (hash={params_hash})")
        labels_reused = True
        # Recover feature_cols from parquet
        import pandas as pd
        df_tmp = pd.read_parquet(labels_path)
        exclude = {'ticker', 'entry_idx', 'timestamp', 'label', 'pnl_pct', 'sl_pct', 'tp_pct'}
        feature_cols = [c for c in df_tmp.columns if c not in exclude]
        stats_labels = {'feature_cols': feature_cols, 'n_samples': len(df_tmp)}
    else:
        print(f"\n  [LABELS] Generating new labels (hash={params_hash})...")
        try:
            # Build feature_cols from FEATURES × TIMEFRAMES
            from pipeline.features_catalog import get_feature_columns as cat_get_cols
            feature_cols_target = cat_get_cols(params['FEATURES'], params['TIMEFRAMES'])
            stats_labels = generate_labels(
                config, params, labels_path,
                feature_cols_override=None  # use whitelist from get_feature_columns
            )
            feature_cols = stats_labels.get('feature_cols', [])
            hash_file.write_text(params_hash)
        except Exception as e:
            return PipelineResult(
                success=False,
                error=f"Error in label generation: {e}",
                total_duration_seconds=time.time() - t_start,
            )

    # --- Step 2: Train ---
    try:
        stats_train = train_model(
            config, params, labels_path, model_dir,
            feature_cols=stats_labels.get('feature_cols')
        )
    except Exception as e:
        return PipelineResult(
            success=False,
            stats_labels=stats_labels,
            labels_reused=labels_reused,
            error=f"Error in training: {e}",
            total_duration_seconds=time.time() - t_start,
        )

    # --- Step 3: Backtest ---
    try:
        objective_mode = params.get('OBJECTIVE_MODE', 'score')
        metrics = run_backtest(config, params, model_dir, objective_mode=objective_mode)
    except Exception as e:
        return PipelineResult(
            success=False,
            stats_labels=stats_labels,
            stats_train=stats_train,
            labels_reused=labels_reused,
            error=f"Error in backtest: {e}",
            total_duration_seconds=time.time() - t_start,
        )

    duration = time.time() - t_start

    # Propagate training stats to metrics (used by the agent)
    metrics['top_features']    = stats_train.get('top_features', [])
    metrics['bottom_features'] = stats_train.get('bottom_features', [])
    metrics['cv_auc_mean']     = stats_train.get('cv_auc_mean', 0.0)
    metrics['cv_auc_std']      = stats_train.get('cv_auc_std', 0.0)
    metrics['xgb_optuna_best'] = stats_train.get('xgb_optuna_best', {})

    return PipelineResult(
        success=True,
        metrics=metrics,
        stats_labels=stats_labels,
        stats_train=stats_train,
        labels_reused=labels_reused,
        total_duration_seconds=duration,
    )
