"""
Backtest with Bayesian Optimization (Optuna) for the algo_autoresearch pipeline.

Instead of a static SL/TP/threshold grid, uses TPE Sampler to explore
the space intelligently. Penalizes configs with inconsistent degradation
between years to avoid overfitting the training period.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
import joblib
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from numba import njit

import ml_sessions_compat.config as ml_config
from ml_sessions_compat.features.technical import merge_timeframes

# --- Module-level parquet cache (OOM mitigation S8) ---
# Each entry: (ticker, exchange, tf) → full DataFrame
# Avoids re-reading from disk and RSS accumulation at each iteration
_PARQUET_CACHE: dict = {}


def find_parquet(data_dir: str, ticker: str, tf: str, exchange: str = 'binance') -> str | None:
    candidates = []
    if exchange and exchange != 'binance':
        candidates.append(f'{ticker}_{tf}_usdt_{exchange}.parquet')
    candidates.append(f'{ticker}_{tf}_usdt_binance.parquet')
    candidates.append(f'{ticker}_{tf}_usdt.parquet')
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_model(model_dir: Path):
    model_path = model_dir / 'xgboost_best.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    model = joblib.load(model_path)
    feature_path = model_dir / 'feature_names.txt'
    with open(feature_path) as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return model, feature_names


def _load_parquet_cached(data_dir: str, ticker: str, tf: str, exchange: str) -> pd.DataFrame | None:
    """Read parquet from disk only once per process; reuse from cache in subsequent iterations."""
    key = (ticker, exchange, tf)
    if key not in _PARQUET_CACHE:
        fpath = find_parquet(data_dir, ticker, tf, exchange)
        if fpath is None:
            _PARQUET_CACHE[key] = None
        else:
            df = pd.read_parquet(fpath)
            df['timestamp'] = pd.to_datetime(df['open_time']).astype('datetime64[ns]')
            _PARQUET_CACHE[key] = df
    return _PARQUET_CACHE[key]


def load_and_prepare(year: int, model, feature_names: list, ticker: str,
                     exchange: str = 'binance') -> dict | None:
    data_dir = ml_config.DATA_DIR

    df_15m_full = _load_parquet_cached(data_dir, ticker, '15m', exchange)
    if df_15m_full is None:
        return None

    df_15m = df_15m_full[df_15m_full['timestamp'].dt.year == year].sort_values('timestamp').reset_index(drop=True)

    if len(df_15m) < 100:
        return None

    higher_tf = {}
    for tf_key, tf_code in [('04h', '04h'), ('01d', '01d')]:
        df_full = _load_parquet_cached(data_dir, ticker, tf_code, exchange)
        if df_full is not None:
            df = df_full[df_full['timestamp'].dt.year >= year - 1].sort_values('timestamp').reset_index(drop=True)
            higher_tf[tf_key] = df

    df = merge_timeframes(df_15m, higher_tf)
    if df is None or len(df) < 100:
        return None

    df = df.sort_values('timestamp').reset_index(drop=True)

    X = np.zeros((len(df), len(feature_names)), dtype=np.float64)
    for j, f in enumerate(feature_names):
        if f in df.columns:
            X[:, j] = df[f].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(X)[:, 1].astype(np.float64)

    n = len(df)
    return {
        'high':       df['high'].to_numpy(dtype=np.float64),
        'low':        df['low'].to_numpy(dtype=np.float64),
        'close':      df['close'].to_numpy(dtype=np.float64),
        'adx':        df['adx_15m'].to_numpy(dtype=np.float64) if 'adx_15m' in df.columns else np.zeros(n),
        'atr_regime': df['atr_regime_15m'].to_numpy(dtype=np.float64) if 'atr_regime_15m' in df.columns else np.zeros(n),
        'probs':      probs,
        'n':          n,
    }


@njit
def simulate_numba(high, low, close, probs, adx,
                   sl_pct, tp_pct, threshold, adx_min,
                   initial=10_000.0, slippage=0.001, max_pos=1, fee_pct=0.001):
    capital   = initial
    n_trades  = 0
    n_wins    = 0

    pos_entry  = np.zeros(max_pos, dtype=np.float64)
    pos_size   = np.zeros(max_pos, dtype=np.float64)
    pos_sl     = np.zeros(max_pos, dtype=np.float64)
    pos_tp     = np.zeros(max_pos, dtype=np.float64)
    pos_count  = 0

    CANDLES_PER_DAY    = 96
    day_start_eq       = initial
    daily_sum          = 0.0
    daily_sq_sum       = 0.0
    downside_sq_sum    = 0.0   # sum of squared negative daily returns
    n_days             = 0

    peak   = initial
    max_dd = 0.0

    for i in range(len(high)):
        new_count = 0
        for j in range(pos_count):
            e = pos_entry[j]; sz = pos_size[j]
            if low[i] <= pos_sl[j]:
                capital  += sz * (pos_sl[j] / e) * (1.0 - fee_pct)
                n_trades += 1
            elif high[i] >= pos_tp[j]:
                capital  += sz * (pos_tp[j] / e) * (1.0 - fee_pct)
                n_trades += 1
                n_wins   += 1
            else:
                pos_entry[new_count] = e
                pos_size[new_count]  = sz
                pos_sl[new_count]    = pos_sl[j]
                pos_tp[new_count]    = pos_tp[j]
                new_count += 1
        pos_count = new_count

        if pos_count < max_pos and probs[i] >= threshold and adx[i] > adx_min:
            # Dynamic sizing: scale between 50% and 100% of the base slot
            # according to model confidence above the threshold
            confidence  = (probs[i] - threshold) / (1.0 - threshold + 1e-9)
            factor      = 0.5 + 0.5 * confidence   # [0.50, 1.00]
            slot_size   = (capital / max_pos) * factor
            if slot_size > 50.0:
                ep = close[i] * (1.0 + slippage)
                capital -= slot_size
                pos_entry[pos_count] = ep
                pos_size[pos_count]  = slot_size
                pos_sl[pos_count]    = ep * (1.0 - sl_pct / 100.0)
                pos_tp[pos_count]    = ep * (1.0 + tp_pct / 100.0)
                pos_count += 1

        open_val = 0.0
        for j in range(pos_count):
            open_val += pos_size[j]
        equity_now = capital + open_val

        if equity_now > peak: peak = equity_now
        dd = (equity_now - peak) / peak
        if dd < max_dd: max_dd = dd

        if (i + 1) % CANDLES_PER_DAY == 0:
            if day_start_eq > 0:
                dr = (equity_now - day_start_eq) / day_start_eq
                daily_sum    += dr
                daily_sq_sum += dr * dr
                if dr < 0.0:
                    downside_sq_sum += dr * dr
                n_days       += 1
            day_start_eq = equity_now

    for j in range(pos_count):
        capital  += pos_size[j] * (close[-1] / pos_entry[j]) * (1.0 - fee_pct)
        n_trades += 1
        if close[-1] > pos_entry[j]: n_wins += 1

    total_ret = (capital / initial - 1.0) * 100.0

    sharpe_raw  = 0.0
    sortino_raw = 0.0
    if n_days > 1:
        mean_dr = daily_sum / n_days
        var_dr  = daily_sq_sum / n_days - mean_dr * mean_dr
        if var_dr > 0.0:
            sharpe_raw = (mean_dr / (var_dr ** 0.5)) * (365.0 ** 0.5)
        downside_std = (downside_sq_sum / n_days) ** 0.5
        if downside_std > 0.0:
            sortino_raw = (mean_dr / downside_std) * (365.0 ** 0.5)

    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    return total_ret, sharpe_raw, max_dd * 100.0, n_trades, win_rate, sortino_raw


@njit
def simulate_numba_equity(high, low, close, probs, adx,
                          sl_pct, tp_pct, threshold, adx_min,
                          initial=10_000.0, slippage=0.001, max_pos=1, fee_pct=0.001):
    """Same as simulate_numba but also returns equity curve and max drawdown indices."""
    capital  = initial
    n_trades = 0
    n_wins   = 0

    pos_entry = np.zeros(max_pos, dtype=np.float64)
    pos_size  = np.zeros(max_pos, dtype=np.float64)
    pos_sl    = np.zeros(max_pos, dtype=np.float64)
    pos_tp    = np.zeros(max_pos, dtype=np.float64)
    pos_count = 0

    n = len(high)
    equity = np.zeros(n, dtype=np.float64)

    peak      = initial
    max_dd    = 0.0
    peak_idx  = 0
    dd_start  = 0
    dd_end    = 0

    CANDLES_PER_DAY  = 96
    day_start_eq     = initial
    daily_sum        = 0.0
    daily_sq_sum     = 0.0
    downside_sq_sum  = 0.0
    n_days           = 0

    for i in range(n):
        new_count = 0
        for j in range(pos_count):
            e = pos_entry[j]; sz = pos_size[j]
            if low[i] <= pos_sl[j]:
                capital  += sz * (pos_sl[j] / e) * (1.0 - fee_pct)
                n_trades += 1
            elif high[i] >= pos_tp[j]:
                capital  += sz * (pos_tp[j] / e) * (1.0 - fee_pct)
                n_trades += 1
                n_wins   += 1
            else:
                pos_entry[new_count] = e
                pos_size[new_count]  = sz
                pos_sl[new_count]    = pos_sl[j]
                pos_tp[new_count]    = pos_tp[j]
                new_count += 1
        pos_count = new_count

        if pos_count < max_pos and probs[i] >= threshold and adx[i] > adx_min:
            confidence = (probs[i] - threshold) / (1.0 - threshold + 1e-9)
            factor     = 0.5 + 0.5 * confidence
            slot_size = (capital / max_pos) * factor
            if slot_size > 50.0:
                ep = close[i] * (1.0 + slippage)
                capital -= slot_size
                pos_entry[pos_count] = ep
                pos_size[pos_count]  = slot_size
                pos_sl[pos_count]    = ep * (1.0 - sl_pct / 100.0)
                pos_tp[pos_count]    = ep * (1.0 + tp_pct / 100.0)
                pos_count += 1

        open_val = 0.0
        for j in range(pos_count):
            open_val += pos_size[j]
        equity_now  = capital + open_val
        equity[i]   = equity_now

        if equity_now > peak:
            peak     = equity_now
            peak_idx = i
        dd = (equity_now - peak) / peak
        if dd < max_dd:
            max_dd   = dd
            dd_start = peak_idx
            dd_end   = i

        if (i + 1) % CANDLES_PER_DAY == 0:
            if day_start_eq > 0:
                dr = (equity_now - day_start_eq) / day_start_eq
                daily_sum    += dr
                daily_sq_sum += dr * dr
                if dr < 0.0:
                    downside_sq_sum += dr * dr
                n_days       += 1
            day_start_eq = equity_now

    for j in range(pos_count):
        capital  += pos_size[j] * (close[-1] / pos_entry[j]) * (1.0 - fee_pct)
        n_trades += 1
        if close[-1] > pos_entry[j]: n_wins += 1

    total_ret   = (capital / initial - 1.0) * 100.0
    sharpe_raw  = 0.0
    sortino_raw = 0.0
    if n_days > 1:
        mean_dr = daily_sum / n_days
        var_dr  = daily_sq_sum / n_days - mean_dr * mean_dr
        if var_dr > 0.0:
            sharpe_raw = (mean_dr / (var_dr ** 0.5)) * (365.0 ** 0.5)
        downside_std = (downside_sq_sum / n_days) ** 0.5
        if downside_std > 0.0:
            sortino_raw = (mean_dr / downside_std) * (365.0 ** 0.5)
    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    return total_ret, sharpe_raw, max_dd * 100.0, n_trades, win_rate, sortino_raw, equity, dd_start, dd_end


def calculate_score(sharpe: float, return_pct: float, drawdown_pct: float,
                    weights: dict = None) -> float:
    """
    Composite score: tanh(S/2)*0.50 + tanh(R/100)*0.30 - (abs(DD)/100)*0.20
    """
    if weights is None:
        weights = {'sharpe': 0.50, 'return': 0.30, 'drawdown': 0.20}

    s = float(np.tanh(sharpe / 2)) * weights['sharpe']
    r = float(np.tanh(return_pct / 100)) * weights['return']
    d = (abs(drawdown_pct) / 100) * weights['drawdown']
    return s + r - d


def run_backtest(config: dict, params: dict, model_dir: Path,
                 objective_mode: str = 'score') -> dict:
    """
    Backtest with Bayesian Optimization (Optuna).

    Each trial tests an (SL, TP, threshold) suggested by the TPE Sampler.
    objective_mode='score': maximize average composite score over OOS years (S1-S4).
    objective_mode='profit': maximize average OOS return % directly (S5+).
    """
    ticker           = config['pipeline']['ticker']
    exchange         = config['pipeline'].get('exchange', 'binance')
    validation_years = config['pipeline'].get('validation_years', [2023, 2024])
    holdout_years    = config['pipeline'].get('holdout_years', [2025])
    max_pos          = config['pipeline'].get('max_positions', 1)
    w          = config.get('scoring', {})
    weights    = {
        'sharpe':   w.get('sharpe_weight', 0.50),
        'return':   w.get('return_weight', 0.30),
        'drawdown': w.get('drawdown_weight', 0.20),
    }
    min_trades  = w.get('min_trades', 10)
    max_dd_gate       = w.get('max_dd_gate', -30.0)
    min_sharpe_gate   = w.get('min_sharpe_gate', 0.5)
    min_trades_profit = w.get('min_trades_profit', min_trades)
    atr_kill          = w.get('atr_regime_kill', float('inf'))  # kill-switch: zero probs if atr_regime > limit
    n_trials    = params.get('N_TRIALS', 120)
    sl_range    = params.get('SL_RANGE', (0.5, 12.0))
    tp_range    = params.get('TP_RANGE', (1.0, 40.0))
    thr_range   = params.get('THRESHOLD_RANGE', (0.30, 0.80))

    print(f"\n{'='*70}")
    print(f"BACKTEST (Optuna TPE) — {ticker.upper()}  [mode: {objective_mode.upper()}]")
    print(f"  Optuna validation: {validation_years} | True holdout (passive): {holdout_years} | Trials: {n_trials}")
    print(f"  SL: {sl_range} | TP: {tp_range} | Thr: {thr_range} | Max pos: {max_pos}")
    print(f"{'='*70}")

    model, feature_names = load_model(model_dir)
    print(f"  Features: {len(feature_names)}")

    # Compile Numba
    print("  Compiling Numba...", end='', flush=True)
    _d = np.ones(10, dtype=np.float64)
    simulate_numba(_d, _d * 0.99, _d, _d * 0.5, _d * 30.0, 1.0, 2.0, 0.5, 0.0, max_pos=max_pos)
    print(" done")

    # Load validation data (for Optuna)
    print("  Loading validation data...", end='', flush=True)
    data_by_year_val = {}
    for year in validation_years:
        data = load_and_prepare(year, model, feature_names, ticker, exchange)
        if data is not None:
            data_by_year_val[year] = data
            print(f" {year}({data['n']} bars)", end='', flush=True)
        else:
            print(f" {year}(SKIP)", end='', flush=True)
    print()

    # Load holdout data (passive — Optuna never touches it)
    print("  Loading holdout data (passive)...", end='', flush=True)
    data_by_year_holdout = {}
    for year in holdout_years:
        data = load_and_prepare(year, model, feature_names, ticker, exchange)
        if data is not None:
            data_by_year_holdout[year] = data
            print(f" {year}({data['n']} bars)", end='', flush=True)
        else:
            print(f" {year}(SKIP)", end='', flush=True)
    print()

    if not data_by_year_val:
        return {
            'sharpe_raw': 0.0, 'retorno_anual_pct': 0.0,
            'max_drawdown_pct': 100.0, 'win_rate_pct': 0.0,
            'n_trades': 0, 'score_composto': -1.0,
            'sl_pct': 0.0, 'tp_pct': 0.0, 'threshold': 0.0,
            'equity_500_final': 500.0, 'equity_500_por_ano': {},
            'retorno_total_oos_pct': 0.0,
            'sharpe_validation': 0.0, 'sharpe_holdout': 0.0,
        }

    def objective(trial: optuna.Trial) -> float:
        sl  = trial.suggest_float('sl_pct',    sl_range[0],  sl_range[1])
        tp  = trial.suggest_float('tp_pct',    tp_range[0],  tp_range[1])
        thr = trial.suggest_float('threshold', thr_range[0], thr_range[1])

        if objective_mode == 'profit':
            # --- profit mode: maximize average return % (validation years only) ---
            return_accumulated = 0.0
            for year, data in data_by_year_val.items():
                probs_safe = np.where(data['atr_regime'] > atr_kill, 0.0, data['probs'])
                ret, sharpe, max_dd, n_trades, _, _sortino = simulate_numba(
                    data['high'], data['low'], data['close'],
                    probs_safe, data['adx'],
                    sl, tp, thr, 0.0, max_pos=max_pos, fee_pct=0.002,
                )
                if n_trades < min_trades_profit:
                    return -999.0
                if max_dd < max_dd_gate:
                    return -999.0
                if sharpe < min_sharpe_gate:
                    return -999.0
                return_accumulated += ret
            if not data_by_year_val:
                return -999.0
            return return_accumulated / len(data_by_year_val)

        else:
            # --- score mode: maximize worst sub-window (anti-overfitting robustness) ---
            # Maximize min(sharpe) instead of mean(sharpe): penalizes configs that
            # perform well in one validation year but collapse in the other.
            sharpes_val = []
            for year, data in data_by_year_val.items():
                probs_safe = np.where(data['atr_regime'] > atr_kill, 0.0, data['probs'])
                ret, sharpe, max_dd, n_trades, _, _sortino = simulate_numba(
                    data['high'], data['low'], data['close'],
                    probs_safe, data['adx'],
                    sl, tp, thr, 0.0, max_pos=max_pos, fee_pct=0.002,
                )
                if n_trades < min_trades:
                    return -1.0
                sharpes_val.append(sharpe)

            if not sharpes_val:
                return -1.0

            return float(min(sharpes_val))  # worst sub-window

    # Create and optimize Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_score = study.best_value

    sl_b   = best['sl_pct']
    tp_b   = best['tp_pct']
    thr_b  = best['threshold']

    best_label = f"return={best_score:+.1f}%" if objective_mode == 'profit' else f"score={best_score:.4f}"
    print(f"\n  Best config: SL={sl_b:.2f}%, TP={tp_b:.2f}%, T={thr_b:.2f}  ({best_label})")

    # Compute final metrics — validation (years Optuna optimized)
    sharpe_total_val = return_total_val = max_dd_total_val = win_rate_total_val = trades_total_val = sortino_total_val = 0.0
    equity_val = 500.0
    equity_by_year_val = {}

    for year, data in sorted(data_by_year_val.items()):
        probs_safe = np.where(data['atr_regime'] > atr_kill, 0.0, data['probs'])
        ret, sharpe, max_dd, n_trades, win_rate, sortino = simulate_numba(
            data['high'], data['low'], data['close'],
            probs_safe, data['adx'],
            sl_b, tp_b, thr_b, 0.0, max_pos=max_pos, fee_pct=0.002,
        )
        equity_val = equity_val * (1 + ret / 100)
        equity_by_year_val[year] = round(equity_val, 2)
        sharpe_total_val   += sharpe
        sortino_total_val  += sortino
        return_total_val   += ret
        max_dd_total_val    = min(max_dd_total_val, max_dd)
        win_rate_total_val  = win_rate
        trades_total_val   += n_trades

    n_years_val     = max(len(data_by_year_val), 1)
    sharpe_validation = sharpe_total_val / n_years_val
    sortino_val    = sortino_total_val / n_years_val
    return_avg_val = return_total_val / n_years_val

    print(f"  [VAL] Sharpe(val)={sharpe_validation:.2f} | Sortino={sortino_val:.2f} | "
          f"Average return={return_avg_val:+.1f}% | DD={abs(max_dd_total_val):.1f}%")
    print(f"  Equity val (€500): ", end="")
    for yr, eq in equity_by_year_val.items():
        print(f"{yr}→ €{eq:.0f}  ", end="")
    print()

    # Compute final metrics — holdout (passive, never touched by Optuna)
    sharpe_total_hld = return_total_hld = max_dd_total_hld = win_rate_total_hld = trades_total_hld = sortino_total_hld = 0.0
    equity_hld = 500.0
    equity_by_year_hld = {}

    for year, data in sorted(data_by_year_holdout.items()):
        probs_safe = np.where(data['atr_regime'] > atr_kill, 0.0, data['probs'])
        ret, sharpe, max_dd, n_trades, win_rate, sortino = simulate_numba(
            data['high'], data['low'], data['close'],
            probs_safe, data['adx'],
            sl_b, tp_b, thr_b, 0.0, max_pos=max_pos, fee_pct=0.002,
        )
        equity_hld = equity_hld * (1 + ret / 100)
        equity_by_year_hld[year] = round(equity_hld, 2)
        sharpe_total_hld   += sharpe
        sortino_total_hld  += sortino
        return_total_hld   += ret
        max_dd_total_hld    = min(max_dd_total_hld, max_dd)
        win_rate_total_hld  = win_rate
        trades_total_hld   += n_trades

    n_years_hld = max(len(data_by_year_holdout), 1)
    sharpe_holdout    = sharpe_total_hld / n_years_hld if data_by_year_holdout else 0.0
    return_avg_hld = return_total_hld / n_years_hld if data_by_year_holdout else 0.0
    return_oos_hld   = (equity_hld / 500.0 - 1) * 100
    profit_hld         = equity_hld - 500.0

    print(f"  [HOLDOUT/passive] Sharpe={sharpe_holdout:.2f} | Return={return_avg_hld:+.1f}% | "
          f"DD={abs(max_dd_total_hld):.1f}% | Equity: ", end="")
    for yr, eq in equity_by_year_hld.items():
        print(f"{yr}→ €{eq:.0f}  ", end="")
    print(f"| Total: {return_oos_hld:+.1f}% (€{profit_hld:+.0f})")

    # Aggregated metrics for return dict (uses validation as primary)
    sharpe_avg  = sharpe_validation
    sortino_avg = sortino_val
    return_avg = return_avg_val
    max_dd_total  = max_dd_total_val
    win_rate_total = win_rate_total_val
    trades_total  = trades_total_val
    equity        = equity_val
    equity_by_year = {**equity_by_year_val, **equity_by_year_hld}
    return_oos   = return_oos_hld  # reported OOS return = holdout (true OOS)
    profit         = profit_hld

    # Drawdown forensics — use simulate_numba_equity on the first validation year
    drawdown_forensics = {}
    try:
        forensic_year = sorted(data_by_year_val.keys())[0]
        data_f = data_by_year_val[forensic_year]
        probs_f = np.where(data_f['atr_regime'] > atr_kill, 0.0, data_f['probs'])
        _, _, _, _, _, _sortino, equity_curve, dd_start, dd_end = simulate_numba_equity(
            data_f['high'], data_f['low'], data_f['close'],
            probs_f, data_f['adx'],
            sl_b, tp_b, thr_b, 0.0,
            initial=10_000.0, max_pos=max_pos,
        )
        CANDLES_PER_DAY = 96
        if dd_end > dd_start:
            dd_adx_mean  = float(data_f['adx'][dd_start:dd_end + 1].mean())
            dd_adx_min   = float(data_f['adx'][dd_start:dd_end + 1].min())
            dd_dur_days  = (dd_end - dd_start) / CANDLES_PER_DAY
            dd_depth_pct = float((equity_curve[dd_end] / equity_curve[dd_start] - 1) * 100)

            # Classify market regime during the DD
            if dd_adx_mean < 20:
                regime = "Sideways / No Trend (ADX < 20)"
            elif dd_adx_mean < 30:
                regime = "Weak Trend (ADX 20-30)"
            else:
                regime = "Strong Trend (ADX > 30)"

            drawdown_forensics = {
                'ano':          forensic_year,
                'dur_dias':     round(dd_dur_days, 1),
                'profundidade': round(dd_depth_pct, 2),
                'adx_medio':    round(dd_adx_mean, 1),
                'adx_minimo':   round(dd_adx_min, 1),
                'regime':       regime,
            }
            print(f"  DD forensics ({forensic_year}): {dd_dur_days:.0f} days | ADX avg={dd_adx_mean:.1f} | {regime}")
    except Exception:
        pass

    # Optuna parameter importance (how much each dimension contributed to the score)
    optuna_param_importance = {}
    try:
        from optuna.importance import get_param_importances
        raw_imp = get_param_importances(study)
        optuna_param_importance = {k: round(float(v), 4) for k, v in raw_imp.items()}
        # Map internal names to research_params names
        name_map = {'sl_pct': 'SL_RANGE', 'tp_pct': 'TP_RANGE', 'threshold': 'THRESHOLD_RANGE'}
        optuna_param_importance = {name_map.get(k, k): v for k, v in optuna_param_importance.items()}
        sorted_imp = sorted(optuna_param_importance.items(), key=lambda x: -x[1])
        print(f"  Optuna param importance: " + " | ".join(f"{k}={v:.3f}" for k, v in sorted_imp))
    except Exception:
        pass

    return {
        'sharpe_raw':               sharpe_avg,       # = sharpe_validation (primary)
        'sharpe_validation':        sharpe_validation,  # average sharpe over validation years
        'sharpe_holdout':           sharpe_holdout,     # passive — never use for acceptance
        'sortino_raw':              sortino_avg,
        'retorno_anual_pct':        return_avg,
        'max_drawdown_pct':         max_dd_total,
        'win_rate_pct':             win_rate_total,
        'n_trades':                 int(trades_total),
        'score_composto':           best_score,
        'sl_pct':                   sl_b,
        'tp_pct':                   tp_b,
        'threshold':                thr_b,
        'equity_500_final':         round(equity_val, 2),
        'equity_500_por_ano':       equity_by_year,
        'retorno_total_oos_pct':    round(return_oos_hld, 2),  # holdout OOS
        'n_trials_optuna':          n_trials,
        'optuna_param_importance':  optuna_param_importance,
        'drawdown_forensics':       drawdown_forensics,
    }
