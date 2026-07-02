"""
Catalog of allowed relative features.

All features in this catalog are guaranteed to be relative (no absolute prices).
The LLM agent may only use features from this catalog in research_params.py.

Structure: CATALOG[base_name] = description
Features are expanded by timeframe in run_pipeline.py.
"""

# Available features per timeframe (all relative)
CATALOG = {
    # Oscillators (0-100 by design)
    "stoch_rsi_k":     "Stochastic RSI %K (0-100)",
    "stoch_rsi_d":     "Stochastic RSI %D (0-100)",
    "rsi":             "Relative Strength Index (0-100)",
    "bb_position":     "Bollinger Band position (0=lower, 1=upper)",
    "adx":             "Average Directional Index (0-100)",

    # Normalized by price (ratio/percentage)
    "ema_diff":        "EMA fast-slow difference normalized by price (%)",
    "trend":           "EMA direction (0=down, 1=up, binary)",
    "returns_1":       "1-period percent return (%)",
    "atr_pct":         "ATR normalized by price (%)",
    "macd_pct":        "MACD normalized by price (%)",
    "macd_signal_pct": "MACD signal normalized by price (%)",
    "macd_hist_pct":   "MACD histogram normalized by price (%)",
    "bb_width_pct":    "Bollinger Bands width normalized (%)",

    # 15m only
    "volume_norm":     "Volume normalized by 20-period moving average (15m only)",
    "returns_5":       "5-period percent return (15m only)",

    # Macro / market regime (1d only)
    "dist_sma200_pct": "Distance to SMA200 normalized by price (%, 1d only)",
    "btc_trend":       "BTC above/below EMA50 (0=down, 1=up, 1d only, cross-asset)",
    "atr_regime":      "Current ATR / 50-period moving average of ATR (volatility ratio, all TFs)",
}

# Features exclusive to the 15m timeframe (_15m suffix)
FEATURES_15M_ONLY = {"volume_norm", "returns_5"}

# Features exclusive to the 1d timeframe (_1d suffix)
FEATURES_1D_ONLY = {"dist_sma200_pct", "btc_trend"}

# Forbidden features (absolute prices — NEVER include)
FORBIDDEN = {
    "close", "open", "high", "low", "volume",
    "ema_fast", "ema_slow", "atr", "bb_width",
    "macd", "macd_signal", "macd_hist",
}

# Supported timeframes
SUPPORTED_TIMEFRAMES = ["15m", "4h", "1d"]


def _normalize_name(feat: str) -> str:
    """
    Normalize a feature name by removing timeframe suffixes if present.
    Ex: 'returns_5_15m' -> 'returns_5', 'stoch_rsi_k_4h' -> 'stoch_rsi_k'
    """
    for tf in SUPPORTED_TIMEFRAMES:
        if feat.endswith(f"_{tf}"):
            return feat[: -(len(tf) + 1)]
    return feat


def get_feature_columns(features: list, timeframes: list) -> list:
    """
    Expand features and timeframes into concrete column names.

    Accepts both base names ('returns_5') and full names ('returns_5_15m').

    Args:
        features: list of base names (ex: ["stoch_rsi_k", "returns_5_15m"])
        timeframes: list of timeframes (ex: ["15m", "4h"])

    Returns:
        List of column names (ex: ["stoch_rsi_k_15m", "rsi_15m", ...])
    """
    cols = []
    for tf in timeframes:
        for feat_raw in features:
            feat = _normalize_name(feat_raw)  # strip suffix if present
            if feat in FORBIDDEN:
                raise ValueError(f"Forbidden feature: '{feat}' (absolute price)")
            if feat not in CATALOG:
                raise ValueError(f"Feature not in catalog: '{feat}' (original: '{feat_raw}')")
            if feat in FEATURES_15M_ONLY and tf != "15m":
                continue  # volume_norm and returns_5 only exist in 15m
            if feat in FEATURES_1D_ONLY and tf != "1d":
                continue  # dist_sma200_pct and btc_trend only exist in 1d
            col = f"{feat}_{tf}"
            if col not in cols:  # avoid duplicates if LLM passes both
                cols.append(col)
    return cols


def validate_features(features: list) -> tuple[bool, str]:
    """Validate a feature list against the catalog (accepts names with or without timeframe suffix)."""
    for feat_raw in features:
        feat = _normalize_name(feat_raw)
        if feat in FORBIDDEN:
            return False, f"Forbidden feature: '{feat}'"
        if feat not in CATALOG:
            return False, f"Feature not in catalog: '{feat}' (original: '{feat_raw}')"
    return True, "OK"


def validate_timeframes(timeframes: list) -> tuple[bool, str]:
    """Validate a list of timeframes."""
    for tf in timeframes:
        if tf not in SUPPORTED_TIMEFRAMES:
            return False, f"Invalid timeframe: '{tf}'. Supported: {SUPPORTED_TIMEFRAMES}"
    if len(timeframes) == 0:
        return False, "Timeframe list empty"
    return True, "OK"
