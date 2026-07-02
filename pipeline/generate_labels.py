"""
Label generation for the algo_autoresearch pipeline.

Adapted from btc_only_repro/02_generate_labels.py to read parameters
from research_params.py instead of hardcoded constants.

Entry signal: stoch_rsi_k < ENTRY_STOCH_THRESHOLD AND adx > ENTRY_ADX_THRESHOLD AND ema_diff > 0
SL/TP: bounds passed to Optuna in backtest; labels generated with average combo of the ranges.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import pandas as pd
import numpy as np
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from numba import njit

import ml_sessions_compat.config as ml_config
from ml_sessions_compat.features.technical import merge_timeframes, get_feature_columns


@njit
def simulate_trades_numba(high, low, close, stoch_k, adx, ema_diff,
                          sl_pct, tp_pct,
                          stoch_threshold=20.0, adx_min=25.0, max_pos=5):
    """
    Trade simulation with a parametrizable entry signal.

    Entry: stoch_k < stoch_threshold AND adx > adx_min AND ema_diff > 0
    Exit: SL or TP (triple-barrier)
    """
    n = len(close)

    pos_entry_idxs   = np.zeros(max_pos, dtype=np.int64)
    pos_entry_prices = np.zeros(max_pos, dtype=np.float64)
    pos_sls          = np.zeros(max_pos, dtype=np.float64)
    pos_tps          = np.zeros(max_pos, dtype=np.float64)
    pos_count = 0

    res_entry_idxs = np.zeros(n, dtype=np.int64)
    res_labels     = np.zeros(n, dtype=np.int64)
    res_pnls       = np.zeros(n, dtype=np.float64)
    res_count = 0

    for i in range(50, n - 1):
        # Check exits
        active_count = 0
        for j in range(pos_count):
            p_idx   = pos_entry_idxs[j]
            p_price = pos_entry_prices[j]
            p_sl    = pos_sls[j]
            p_tp    = pos_tps[j]

            if i <= p_idx:
                pos_entry_idxs[active_count]   = p_idx
                pos_entry_prices[active_count] = p_price
                pos_sls[active_count]          = p_sl
                pos_tps[active_count]          = p_tp
                active_count += 1
                continue

            sl_price = p_price * (1.0 - p_sl / 100.0)
            tp_price = p_price * (1.0 + p_tp / 100.0)

            if low[i] <= sl_price:
                res_entry_idxs[res_count] = p_idx
                res_labels[res_count]     = 0
                res_pnls[res_count]       = -p_sl
                res_count += 1
            elif high[i] >= tp_price:
                res_entry_idxs[res_count] = p_idx
                res_labels[res_count]     = 1
                res_pnls[res_count]       = p_tp
                res_count += 1
            else:
                pos_entry_idxs[active_count]   = p_idx
                pos_entry_prices[active_count] = p_price
                pos_sls[active_count]          = p_sl
                pos_tps[active_count]          = p_tp
                active_count += 1

        pos_count = active_count

        # Check entries
        if pos_count < max_pos:
            if stoch_k[i] < stoch_threshold and adx[i] > adx_min and ema_diff[i] > 0:
                pos_entry_idxs[pos_count]   = i
                pos_entry_prices[pos_count] = close[i]
                pos_sls[pos_count]          = sl_pct
                pos_tps[pos_count]          = tp_pct
                pos_count += 1

    return res_entry_idxs[:res_count], res_labels[:res_count], res_pnls[:res_count]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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


def load_ticker_data(ticker: str, exchange: str, train_start: int, train_end: int) -> tuple:
    data_dir = ml_config.DATA_DIR

    f15m = find_parquet(data_dir, ticker, '15m', exchange)
    if f15m is None:
        raise FileNotFoundError(f"15m file not found for {ticker} in {data_dir}")

    df_15m = pd.read_parquet(f15m)
    df_15m['timestamp'] = pd.to_datetime(df_15m['open_time']).astype('datetime64[ns]')
    df_15m = df_15m[
        (df_15m['timestamp'].dt.year >= train_start) &
        (df_15m['timestamp'].dt.year <= train_end)
    ].sort_values('timestamp').reset_index(drop=True)

    higher_tf = {}
    for tf_key, tf_code in [('04h', '04h'), ('01d', '01d')]:
        fpath = find_parquet(data_dir, ticker, tf_code, exchange)
        if fpath:
            df = pd.read_parquet(fpath)
            df['timestamp'] = pd.to_datetime(df['open_time']).astype('datetime64[ns]')
            higher_tf[tf_key] = df.sort_values('timestamp').reset_index(drop=True)

    return df_15m, higher_tf


def _add_macro_features(df: pd.DataFrame, higher_tf: dict,
                        ticker: str, exchange: str, data_dir: str) -> pd.DataFrame:
    """
    Add macro (market regime) features to the merged dataframe.

    Computed features (all relative, no absolute prices):
      dist_sma200_pct_1d — distance to SMA200 normalized (% of price)
      btc_trend_1d       — BTC above/below EMA50 (cross-asset, 0/1)
      atr_regime_*       — ATR / rolling_mean(ATR, 50) per timeframe
    """
    df = df.copy()

    # --- 1. dist_sma200_pct_1d ---
    df_1d = higher_tf.get('01d')
    if df_1d is not None and 'close' in df_1d.columns:
        tmp = df_1d[['timestamp', 'close']].copy().sort_values('timestamp')
        tmp['sma200'] = tmp['close'].rolling(200, min_periods=50).mean()
        tmp['dist_sma200_pct_1d'] = (tmp['close'] - tmp['sma200']) / tmp['close'].abs() * 100.0
        tmp['_date'] = tmp['timestamp'].dt.date
        tmp = tmp[['_date', 'dist_sma200_pct_1d']].dropna()
        df['_date'] = df['timestamp'].dt.date
        df = df.merge(tmp, on='_date', how='left')
        df.drop(columns='_date', inplace=True)
        df['dist_sma200_pct_1d'] = df['dist_sma200_pct_1d'].ffill().fillna(0.0)

    # --- 2. btc_trend_1d (cross-asset) ---
    btc_path = find_parquet(data_dir, 'btc', '01d', exchange)
    if btc_path is None:
        btc_path = find_parquet(data_dir, 'btc', '1d', exchange)
    if btc_path and os.path.exists(btc_path):
        df_btc = pd.read_parquet(btc_path)
        df_btc['timestamp'] = pd.to_datetime(df_btc['open_time']).astype('datetime64[ns]')
        df_btc = df_btc.sort_values('timestamp')
        df_btc['_ema50'] = df_btc['close'].ewm(span=50, adjust=False).mean()
        df_btc['btc_trend_1d'] = (df_btc['close'] > df_btc['_ema50']).astype(float)
        df_btc['_date'] = df_btc['timestamp'].dt.date
        df_btc = df_btc[['_date', 'btc_trend_1d']].dropna()
        df['_date'] = df['timestamp'].dt.date
        df = df.merge(df_btc, on='_date', how='left')
        df.drop(columns='_date', inplace=True)
        df['btc_trend_1d'] = df['btc_trend_1d'].ffill().fillna(0.5)

    # --- 3. atr_regime_* (ATR / rolling mean ratio) ---
    for tf in ['15m', '4h', '1d']:
        col = f'atr_pct_{tf}'
        if col in df.columns:
            roll = df[col].rolling(50, min_periods=20).mean()
            df[f'atr_regime_{tf}'] = (df[col] / roll.clip(lower=1e-9)).fillna(1.0)

    return df


def process_ticker(ticker, df_15m, higher_tf, params, feature_cols_override=None,
                   exchange='binance', data_dir=None):
    """Merge timeframes, compute features, simulate trades for all SL/TP combos."""
    log(f"  Merging timeframes for {ticker.upper()}...")
    df = merge_timeframes(df_15m, higher_tf)
    if df is None or len(df) < 200:
        raise ValueError(f"Insufficient data after merge: {len(df) if df is not None else 0} rows")

    df = df.sort_values('timestamp').reset_index(drop=True)
    log(f"  Merged: {len(df):,} rows  ({df['timestamp'].min().date()} – {df['timestamp'].max().date()})")

    log(f"  Calculating macro features (dist_sma200_pct, btc_trend, atr_regime)...")
    df = _add_macro_features(df, higher_tf, ticker, exchange, data_dir)

    high     = df['high'].to_numpy(np.float64)
    low      = df['low'].to_numpy(np.float64)
    close    = df['close'].to_numpy(np.float64)
    stoch_k  = df.get('stoch_rsi_k_15m', pd.Series(np.full(len(df), 50.0))).fillna(50.0).to_numpy(np.float64)
    adx_arr  = df.get('adx_15m', pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy(np.float64)
    ema_diff = df.get('ema_diff_15m', pd.Series(np.zeros(len(df)))).fillna(0.0).to_numpy(np.float64)

    # Training features — use whitelist from params or from get_feature_columns
    if feature_cols_override is not None:
        feature_cols = [c for c in feature_cols_override if c in df.columns]
    else:
        feature_cols = get_feature_columns(df)

    feature_data = {col: df[col].to_numpy(np.float64) for col in feature_cols}

    # With Optuna, the exact SL/TP is optimized in the backtest.
    # Here we generate labels with a single representative combo (midpoint of the ranges).
    sl_range = params.get('SL_RANGE', (1.0, 6.0))
    tp_range = params.get('TP_RANGE', (2.0, 20.0))
    sl_rep   = (sl_range[0] + sl_range[1]) / 2.0
    tp_rep   = (tp_range[0] + tp_range[1]) / 2.0
    all_combos = [(sl_rep, tp_rep)]
    stoch_thr  = float(params['ENTRY_STOCH_THRESHOLD'])
    adx_min    = float(params['ENTRY_ADX_THRESHOLD'])

    trades = []
    for ci, (sl_pct, tp_pct) in enumerate(all_combos):
        entry_idxs, labels, pnls = simulate_trades_numba(
            high, low, close, stoch_k, adx_arr, ema_diff,
            sl_pct, tp_pct, stoch_thr, adx_min, max_pos=5
        )
        log(f"    Representative combo: SL={sl_pct:.2f}%, TP={tp_pct:.2f}% → {len(entry_idxs)} trades")

        for k, idx in enumerate(entry_idxs):
            row = {
                'ticker':    ticker,
                'entry_idx': int(idx),
                'timestamp': df['timestamp'].iloc[idx],
                'label':     int(labels[k]),
                'pnl_pct':   float(pnls[k]),
                'sl_pct':    sl_pct,
                'tp_pct':    tp_pct,
            }
            for col in feature_cols:
                row[col] = feature_data[col][idx]
            trades.append(row)

    return trades, feature_cols


def aggregate_labels(trades_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    agg_dict = {col: 'mean' for col in feature_cols}
    agg_dict['label']     = lambda x: int(x.mean() >= 0.5)
    agg_dict['pnl_pct']   = 'mean'
    agg_dict['sl_pct']    = 'mean'
    agg_dict['tp_pct']    = 'mean'
    agg_dict['timestamp'] = 'first'
    return trades_df.groupby(['ticker', 'entry_idx'], sort=False).agg(agg_dict).reset_index()


def generate_labels(config: dict, params: dict, output_path: Path,
                    feature_cols_override=None) -> dict:
    """
    Main function — generates labels and saves them to output_path.

    Args:
        config: system configuration (ticker, exchange, etc.)
        params: parameters from research_params.py
        output_path: where to save the labels parquet
        feature_cols_override: list of columns to use (or None for auto)

    Returns:
        dict with statistics
    """
    ticker     = config['pipeline']['ticker']
    exchange   = config['pipeline'].get('exchange', 'binance')
    train_start = config['pipeline'].get('train_start', 2017)
    train_end   = config['pipeline'].get('train_end', 2024)

    print(f"\n{'='*70}")
    print(f"LABEL GENERATION — {ticker.upper()}")
    print(f"  Signal: stoch_rsi_k < {params['ENTRY_STOCH_THRESHOLD']} "
          f"AND adx > {params['ENTRY_ADX_THRESHOLD']} AND ema_diff > 0")
    sl_r = params.get('SL_RANGE', (1.0, 6.0))
    tp_r = params.get('TP_RANGE', (2.0, 20.0))
    print(f"  Optuna bounds: SL={sl_r}, TP={tp_r} | Labels using representative combo (mean)")
    print(f"{'='*70}")

    # Compile numba
    print("  Compiling Numba...", end='', flush=True)
    _d = np.ones(200, dtype=np.float64)
    simulate_trades_numba(_d, _d * 0.99, _d, _d * 10.0, _d * 30.0, _d, 1.0, 2.0)
    print(" done")

    log(f"Loading data {ticker.upper()} ({train_start}–{train_end})...")
    df_15m, higher_tf = load_ticker_data(ticker, exchange, train_start, train_end)
    log(f"15m rows: {len(df_15m):,} | Higher TFs: {list(higher_tf.keys())}")

    start = datetime.now()
    trades, feature_cols = process_ticker(ticker, df_15m, higher_tf, params, feature_cols_override,
                                          exchange=exchange, data_dir=ml_config.DATA_DIR)

    if not trades:
        raise ValueError("No trades generated — check parameters and data")

    trades_df = pd.DataFrame(trades)
    elapsed = (datetime.now() - start).total_seconds()
    log(f"Simulation completed: {len(trades):,} trade-events in {elapsed:.1f}s")

    log("Aggregating labels (majority)...")
    final_df = aggregate_labels(trades_df, feature_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)

    labels_duration_seconds = elapsed

    stats = {
        'n_samples': len(final_df),
        'n_positive': int(final_df['label'].sum()),
        'positive_rate': float(final_df['label'].mean()),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'duracao_labels_segundos': labels_duration_seconds,
    }

    print(f"\n  Samples: {stats['n_samples']:,}")
    print(f"  Positives: {stats['n_positive']:,} ({stats['positive_rate']*100:.1f}%)")
    print(f"  Features: {stats['n_features']}")
    print(f"  Saved to: {output_path}")

    return stats
