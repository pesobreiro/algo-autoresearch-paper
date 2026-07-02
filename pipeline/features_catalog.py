"""
Catálogo de features relativas permitidas.

Todas as features neste catálogo são garantidamente relativas (sem preços absolutos).
O agente LLM só pode usar features deste catálogo em research_params.py.

Estrutura: CATALOG[base_name] = descrição
Features são expandidas por timeframe em run_pipeline.py.
"""

# Features disponíveis por timeframe (todas relativas)
CATALOG = {
    # Osciladores (0-100 por design)
    "stoch_rsi_k":     "Stochastic RSI %K (0-100)",
    "stoch_rsi_d":     "Stochastic RSI %D (0-100)",
    "rsi":             "Relative Strength Index (0-100)",
    "bb_position":     "Bollinger Band position (0=lower, 1=upper)",
    "adx":             "Average Directional Index (0-100)",

    # Normalizados por preço (ratio/percentagem)
    "ema_diff":        "EMA fast-slow diferença normalizada pelo preço (%)",
    "trend":           "Direção da EMA (0=baixa, 1=alta, binário)",
    "returns_1":       "Retorno percentual 1 período (%)",
    "atr_pct":         "ATR normalizado pelo preço (%)",
    "macd_pct":        "MACD normalizado pelo preço (%)",
    "macd_signal_pct": "MACD signal normalizado pelo preço (%)",
    "macd_hist_pct":   "MACD histogram normalizado pelo preço (%)",
    "bb_width_pct":    "Largura das Bandas de Bollinger normalizada (%)",

    # 15m apenas
    "volume_norm":     "Volume normalizado pela média móvel de 20 períodos (apenas 15m)",
    "returns_5":       "Retorno percentual 5 períodos (apenas 15m)",

    # Macro / regime de mercado (apenas 1d)
    "dist_sma200_pct": "Distância à SMA200 normalizada pelo preço (%, apenas 1d)",
    "btc_trend":       "BTC acima/abaixo da EMA50 (0=baixo, 1=alto, apenas 1d, cross-asset)",
    "atr_regime":      "ATR atual / média móvel 50p do ATR (ratio de volatilidade, todos os TFs)",
}

# Features exclusivas do timeframe 15m (sufixo _15m)
FEATURES_15M_ONLY = {"volume_norm", "returns_5"}

# Features exclusivas do timeframe 1d (sufixo _1d)
FEATURES_1D_ONLY = {"dist_sma200_pct", "btc_trend"}

# Features proibidas (preços absolutos — NUNCA incluir)
FORBIDDEN = {
    "close", "open", "high", "low", "volume",
    "ema_fast", "ema_slow", "atr", "bb_width",
    "macd", "macd_signal", "macd_hist",
}

# Timeframes suportados
SUPPORTED_TIMEFRAMES = ["15m", "4h", "1d"]


def _normalizar_nome(feat: str) -> str:
    """
    Normaliza nome de feature removendo sufixos de timeframe se presentes.
    Ex: 'returns_5_15m' → 'returns_5', 'stoch_rsi_k_4h' → 'stoch_rsi_k'
    """
    for tf in SUPPORTED_TIMEFRAMES:
        if feat.endswith(f"_{tf}"):
            return feat[: -(len(tf) + 1)]
    return feat


def get_feature_columns(features: list, timeframes: list) -> list:
    """
    Expande features e timeframes para nomes de colunas concretos.

    Aceita tanto nomes base ('returns_5') como nomes completos ('returns_5_15m').

    Args:
        features: lista de base names (ex: ["stoch_rsi_k", "returns_5_15m"])
        timeframes: lista de timeframes (ex: ["15m", "4h"])

    Returns:
        Lista de nomes de colunas (ex: ["stoch_rsi_k_15m", "rsi_15m", ...])
    """
    cols = []
    for tf in timeframes:
        for feat_raw in features:
            feat = _normalizar_nome(feat_raw)  # strip sufixo se presente
            if feat in FORBIDDEN:
                raise ValueError(f"Feature proibida: '{feat}' (preço absoluto)")
            if feat not in CATALOG:
                raise ValueError(f"Feature não catalogada: '{feat}' (original: '{feat_raw}')")
            if feat in FEATURES_15M_ONLY and tf != "15m":
                continue  # volume_norm e returns_5 só existem em 15m
            if feat in FEATURES_1D_ONLY and tf != "1d":
                continue  # dist_sma200_pct e btc_trend só existem em 1d
            col = f"{feat}_{tf}"
            if col not in cols:  # evitar duplicados se LLM passar ambos
                cols.append(col)
    return cols


def validate_features(features: list) -> tuple[bool, str]:
    """Valida lista de features contra o catálogo (aceita nomes com ou sem sufixo de timeframe)."""
    for feat_raw in features:
        feat = _normalizar_nome(feat_raw)
        if feat in FORBIDDEN:
            return False, f"Feature proibida: '{feat}'"
        if feat not in CATALOG:
            return False, f"Feature não no catálogo: '{feat}' (original: '{feat_raw}')"
    return True, "OK"


def validate_timeframes(timeframes: list) -> tuple[bool, str]:
    """Valida lista de timeframes."""
    for tf in timeframes:
        if tf not in SUPPORTED_TIMEFRAMES:
            return False, f"Timeframe inválido: '{tf}'. Suportados: {SUPPORTED_TIMEFRAMES}"
    if len(timeframes) == 0:
        return False, "Lista de timeframes vazia"
    return True, "OK"
