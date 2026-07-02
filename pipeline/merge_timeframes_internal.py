"""
Internal copy of the external feature-merge module for reproducibility and audit.

Original source: ~/git/ml_sessions/features/technical.py
Imported at runtime by pipeline/backtest.py and pipeline/generate_labels.py as
`features.technical.merge_timeframes`. This file preserves the same causal logic:
- CANDLE_OFFSETS shifts higher-timeframe timestamps from open to close (RULE 2).
- merge_asof(..., direction='backward') propagates each higher-timeframe value
  only to 15-minute bars at or after the corresponding close (RULE 3).

Copied on 2026-06-26 for the MDPI Analytics submission "Autonomous within Seasons:
An LLM-Driven Prescriptive Strategy Research Pipeline".
"""
import pandas as pd
import numpy as np

# Consistent column mapping for ALL timeframes
COLUMN_MAPPING = {
    'momentum_stoch_rsi_k': 'stoch_rsi_k',
    'momentum_stoch_rsi_d': 'stoch_rsi_d',
    'momentum_rsi': 'rsi',
    'trend_macd': 'macd',
    'trend_macd_signal': 'macd_signal',
    'trend_macd_diff': 'macd_hist',
    'volatility_atr': 'atr',
    'volatility_bbw': 'bb_width',
    'volatility_bbp': 'bb_position',
    'trend_adx': 'adx',
    'trend_ema_fast': 'ema_fast',
    'trend_ema_slow': 'ema_slow',
}

# Candle durations — shift timestamp to CLOSE time before merge_asof (RULE 2)
CANDLE_OFFSETS = {
    '04h': pd.Timedelta(hours=4),
    '01d': pd.Timedelta(days=1),
}


def rename_features(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Apply consistent column mapping for any timeframe.

    Adds normalized derived features (asset-agnostic).
    Absolute-value features (macd, atr, bb_width, ema) are kept with their
    suffix names so the training blacklist in get_feature_columns() can exclude them.
    """
    df = df.copy()

    for old_col, new_base in COLUMN_MAPPING.items():
        new_col = f'{new_base}_{suffix}'
        if old_col in df.columns:
            df[new_col] = df[old_col]

    # Scale Stoch RSI from 0-1 to 0-100
    if f'stoch_rsi_k_{suffix}' in df.columns:
        df[f'stoch_rsi_k_{suffix}'] = df[f'stoch_rsi_k_{suffix}'] * 100
    if f'stoch_rsi_d_{suffix}' in df.columns:
        df[f'stoch_rsi_d_{suffix}'] = df[f'stoch_rsi_d_{suffix}'] * 100

    # Derived normalized features
    if f'atr_{suffix}' in df.columns:
        df[f'atr_pct_{suffix}'] = df[f'atr_{suffix}'] / df['close'] * 100
    if f'ema_fast_{suffix}' in df.columns and f'ema_slow_{suffix}' in df.columns:
        df[f'ema_diff_{suffix}'] = (
            (df[f'ema_fast_{suffix}'] - df[f'ema_slow_{suffix}']) / df['close'] * 100
        )
        df[f'trend_{suffix}'] = (df[f'ema_fast_{suffix}'] > df[f'ema_slow_{suffix}']).astype(int)
    df[f'returns_1_{suffix}'] = df['close'].pct_change(1) * 100

    # Normalize MACD (asset-agnostic)
    if f'macd_{suffix}' in df.columns:
        df[f'macd_pct_{suffix}'] = df[f'macd_{suffix}'] / df['close'] * 100
    if f'macd_signal_{suffix}' in df.columns:
        df[f'macd_signal_pct_{suffix}'] = df[f'macd_signal_{suffix}'] / df['close'] * 100
    if f'macd_hist_{suffix}' in df.columns:
        df[f'macd_hist_pct_{suffix}'] = df[f'macd_hist_{suffix}'] / df['close'] * 100

    # Normalize BB Width (asset-agnostic)
    if f'bb_width_{suffix}' in df.columns:
        df[f'bb_width_pct_{suffix}'] = df[f'bb_width_{suffix}'] / df['close'] * 100

    # 15m-only features
    if suffix == '15m':
        df['volume_norm_15m'] = df['volume'] / df['volume'].rolling(20).mean()
        df['returns_5_15m'] = df['close'].pct_change(5) * 100

    return df


def merge_timeframes(df_15m: pd.DataFrame, btc_data: dict) -> pd.DataFrame:
    """
    Merge 15m ticker data with BTC higher-TF context.

    Applies candle offsets (RULE 2) and merge_asof direction='backward' (RULE 3).
    btc_data keys: '04h', '01d'.
    """
    if df_15m is None or len(df_15m) < 100:
        return None

    df = df_15m.copy()
    df = rename_features(df, '15m')
    df['timestamp'] = df['timestamp'].astype('datetime64[ns]')

    for tf, suffix in [('04h', '4h'), ('01d', '1d')]:
        if tf not in btc_data or len(btc_data[tf]) < 50:
            continue

        df_btc = btc_data[tf].copy()
        df_btc = rename_features(df_btc, suffix)

        # Shift timestamp from open_time to close_time (RULE 2)
        df_btc['timestamp'] = (df_btc['timestamp'] + CANDLE_OFFSETS[tf]).astype('datetime64[ns]')

        feature_cols = [c for c in df_btc.columns if suffix in c]
        df_btc = df_btc[['timestamp'] + feature_cols]

        df = pd.merge_asof(
            df.sort_values('timestamp'),
            df_btc.sort_values('timestamp'),
            on='timestamp',
            direction='backward',   # RULE 3
        )

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return engineered feature columns for training — WHITELIST approach.

    Only includes features explicitly constructed in rename_features() plus
    temporal features from features/temporal.py. Ignores all other parquet
    columns (ichimoku, raw EMAs, etc.) that would otherwise leak in via blacklist.

    Technical features per timeframe (15m, 4h, 1d):
      oscillators:  stoch_rsi_k, stoch_rsi_d, rsi, bb_position, adx
      normalized:   ema_diff, trend, returns_1, atr_pct, macd_pct,
                    macd_signal_pct, macd_hist_pct, bb_width_pct
      15m-only:     volume_norm, returns_5

    Temporal features (14): added by features/temporal.py
    """
    from ml_sessions_compat.features.temporal import TEMPORAL_FEATURE_NAMES

    # Oscillators: normalized by design (0-100 or ratio)
    oscillators = ['stoch_rsi_k', 'stoch_rsi_d', 'rsi', 'bb_position', 'adx']

    # Normalized derived features
    normalized = [
        'ema_diff', 'trend', 'returns_1',
        'atr_pct', 'macd_pct', 'macd_signal_pct', 'macd_hist_pct', 'bb_width_pct',
    ]

    suffixes = ['15m', '4h', '1d']
    feature_cols = []

    for suffix in suffixes:
        for base in oscillators + normalized:
            col = f'{base}_{suffix}'
            if col in df.columns:
                feature_cols.append(col)

    # 15m-only extras
    for col in ['volume_norm_15m', 'returns_5_15m']:
        if col in df.columns:
            feature_cols.append(col)

    # Temporal/session features (all 14 from temporal.py)
    for col in TEMPORAL_FEATURE_NAMES:
        if col in df.columns:
            feature_cols.append(col)

    return feature_cols
