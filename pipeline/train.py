"""
XGBoost model training for the algo_autoresearch pipeline.

Adapted from btc_only_repro/04_train.py to read hyperparameters
from research_params.py.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {'ticker', 'entry_idx', 'timestamp', 'open_time', 'label',
               'pnl_pct', 'sl_pct', 'tp_pct', 'symbol', 'year'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in
            ['float64', 'float32', 'int64', 'int32', 'int16', 'int8']]


def walk_forward_validation(X, y, params, sample_weight=None, n_splits=5) -> list:
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        m  = XGBClassifier(**params)
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        m.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        auc = roc_auc_score(y[te_idx], m.predict_proba(X[te_idx])[:, 1])
        results.append(auc)
        print(f"    Fold {fold+1}: ROC-AUC = {auc:.4f}")
    return results


def _xgb_optuna_study(X, y, base_params: dict, xgb_ranges: dict,
                      sample_weight, n_trials: int) -> dict:
    """Optuna study for XGBoost hyperparameters. Returns the best parameter dict."""

    def objective(trial):
        p = base_params.copy()
        if 'DEPTH_RANGE' in xgb_ranges:
            p['max_depth'] = trial.suggest_int('max_depth', *xgb_ranges['DEPTH_RANGE'])
        if 'LR_RANGE' in xgb_ranges:
            p['learning_rate'] = trial.suggest_float('learning_rate', *xgb_ranges['LR_RANGE'], log=True)
        if 'ESTIMATORS_RANGE' in xgb_ranges:
            p['n_estimators'] = trial.suggest_int('n_estimators', *xgb_ranges['ESTIMATORS_RANGE'])
        if 'ALPHA_RANGE' in xgb_ranges:
            p['reg_alpha'] = trial.suggest_float('reg_alpha', *xgb_ranges['ALPHA_RANGE'], log=True)
        if 'LAMBDA_RANGE' in xgb_ranges:
            p['reg_lambda'] = trial.suggest_float('reg_lambda', *xgb_ranges['LAMBDA_RANGE'], log=True)

        tscv = TimeSeriesSplit(n_splits=3)
        aucs = []
        for tr_idx, te_idx in tscv.split(X):
            m  = XGBClassifier(**p)
            sw = sample_weight[tr_idx] if sample_weight is not None else None
            m.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
            aucs.append(roc_auc_score(y[te_idx], m.predict_proba(X[te_idx])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=max(5, n_trials // 4)),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params

    # Merge with base_params
    result = base_params.copy()
    result.update(best)
    return result, study.best_value


def train_model(config: dict, params: dict, data_path: Path,
                model_dir: Path, feature_cols: list = None) -> dict:
    """
    Train an XGBoost model with hyperparameters from research_params.py.

    Args:
        config: system configuration
        params: parameters from research_params.py
        data_path: path to the labels parquet
        model_dir: directory where the model will be saved
        feature_cols: list of columns (if None, auto-detected)

    Returns:
        dict with training metrics
    """
    xgb_params = {
        'n_estimators':     params['N_ESTIMATORS'],
        'max_depth':        params['MAX_DEPTH'],
        'learning_rate':    params['LEARNING_RATE'],
        'min_child_weight': params['MIN_CHILD_WEIGHT'],
        'gamma':            params['GAMMA'],
        'subsample':        params['SUBSAMPLE'],
        'colsample_bytree': params['COLSAMPLE_BYTREE'],
        'reg_alpha':        params.get('REG_ALPHA', 0.5),
        'reg_lambda':       params.get('REG_LAMBDA', 1.0),
        'random_state':     42,
        'n_jobs':           -1,
        'eval_metric':      'logloss',
    }

    print(f"\n{'='*70}")
    print(f"MODEL TRAINING")
    print(f"  XGBoost: n_est={xgb_params['n_estimators']}, depth={xgb_params['max_depth']}, "
          f"lr={xgb_params['learning_rate']}")
    print(f"{'='*70}")

    print(f"  Loading: {data_path}")
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp').reset_index(drop=True)

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Use only columns that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]

    print(f"  Samples: {len(df):,} | Features: {len(feature_cols)}")
    print(f"  Positives: {df['label'].mean()*100:.1f}%")

    X = df[feature_cols].ffill().fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df['label'].values

    # Sample weights: penalize errors more on trades with larger PnL magnitude
    # Trades with high pnl_pct (large gains or losses) receive more weight
    if 'pnl_pct' in df.columns:
        raw_w = df['pnl_pct'].abs().fillna(0).values
        # Normalize to mean=1 (does not change absolute loss scale)
        mean_w = raw_w.mean()
        sample_weight = raw_w / (mean_w + 1e-9) if mean_w > 0 else np.ones(len(df))
        print(f"  Sample weights: pnl_pct magnitude (min={raw_w.min():.2f}% max={raw_w.max():.2f}% mean={mean_w:.2f}%)")
    else:
        sample_weight = np.ones(len(df))
        print("  Sample weights: uniform (pnl_pct not available)")

    # Optuna XGBoost: active if the LLM defines ranges instead of fixed values
    xgb_ranges = {k: params[k] for k in
                  ('DEPTH_RANGE', 'LR_RANGE', 'ESTIMATORS_RANGE', 'ALPHA_RANGE', 'LAMBDA_RANGE')
                  if params.get(k) is not None}
    n_trials_xgb = params.get('N_TRIALS_XGB', 0)
    xgb_optuna_best = {}

    if xgb_ranges and n_trials_xgb > 0:
        print(f"\n  Optuna XGBoost ({n_trials_xgb} trials, ranges: {list(xgb_ranges.keys())})...")
        xgb_params, best_auc = _xgb_optuna_study(X, y, xgb_params, xgb_ranges, sample_weight, n_trials_xgb)
        xgb_optuna_best = {k: xgb_params.get(k) for k in ('max_depth', 'learning_rate', 'n_estimators', 'reg_alpha', 'reg_lambda')}
        print(f"  Optuna XGB best: depth={xgb_params.get('max_depth')} lr={xgb_params.get('learning_rate'):.4f} "
              f"n_est={xgb_params.get('n_estimators')} alpha={xgb_params.get('reg_alpha'):.3f} "
              f"lambda={xgb_params.get('reg_lambda'):.3f} → AUC={best_auc:.4f}")

    print("\n  Walk-forward CV (5 folds)...")
    start = datetime.now()
    cv_aucs = walk_forward_validation(X, y, xgb_params, sample_weight=sample_weight)
    print(f"  Average AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    print(f"\n  Final training on {len(df):,} samples...")
    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X, y, sample_weight=sample_weight)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path   = model_dir / 'xgboost_best.joblib'
    feature_path = model_dir / 'feature_names.txt'

    joblib.dump(final_model, model_path)
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    elapsed = (datetime.now() - start).total_seconds()

    # Sorted feature importance
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    top_features    = [(f, round(float(v), 4)) for f, v in sorted_imp[:8]]
    bottom_features = [(f, round(float(v), 4)) for f, v in sorted_imp[-5:] if v < 0.02]

    print(f"\n  Top features: {', '.join(f'{f}={v:.3f}' for f,v in top_features[:5])}")
    if bottom_features:
        print(f"  Weak features (<0.02): {', '.join(f for f,_ in bottom_features)}")

    train_duration_seconds = elapsed

    stats = {
        'n_samples':       len(df),
        'n_features':      len(feature_cols),
        'xgb_optuna_best': xgb_optuna_best,
        'cv_auc_mean':     float(np.mean(cv_aucs)),
        'cv_auc_std':   float(np.std(cv_aucs)),
        'model_path':   str(model_path),
        'feature_cols': feature_cols,
        'top_features':    top_features,
        'bottom_features': bottom_features,
        'duracao_treino_segundos': train_duration_seconds,
    }

    print(f"  Model saved: {model_path}")
    return stats


# Backward-compatible alias for external callers
treinar_modelo = train_model
