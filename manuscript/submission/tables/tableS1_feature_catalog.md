## Table S1. Feature catalog

| Base name | Description | Timeframes available | Normalisation / bounds |
|---|---|---|---|
| `stoch_rsi_k` | Stochastic RSI %K | 15m, 4h, 1d | 0–100 |
| `stoch_rsi_d` | Stochastic RSI %D | 15m, 4h, 1d | 0–100 |
| `rsi` | Relative Strength Index | 15m, 4h, 1d | 0–100 |
| `bb_position` | Bollinger Band position | 15m, 4h, 1d | 0 = lower band, 1 = upper band |
| `adx` | Average Directional Index | 15m, 4h, 1d | 0–100 |
| `ema_diff` | EMA fast − EMA slow, normalised by price | 15m, 4h, 1d | Percentage of price |
| `trend` | EMA trend direction | 15m, 4h, 1d | 0 = down, 1 = up |
| `returns_1` | 1-period percentage return | 15m, 4h, 1d | Percentage |
| `atr_pct` | Average True Range normalised by price | 15m, 4h, 1d | Percentage of price |
| `macd_pct` | MACD line normalised by price | 15m, 4h, 1d | Percentage of price |
| `macd_signal_pct` | MACD signal line normalised by price | 15m, 4h, 1d | Percentage of price |
| `macd_hist_pct` | MACD histogram normalised by price | 15m, 4h, 1d | Percentage of price |
| `bb_width_pct` | Bollinger Band width normalised by price | 15m, 4h, 1d | Percentage of price |
| `volume_norm` | Volume normalised by 20-period moving average | 15m only | Ratio |
| `returns_5` | 5-period percentage return | 15m only | Percentage |
| `dist_sma200_pct` | Distance to SMA200 normalised by price | 1d only | Percentage of price |
| `btc_trend` | BTC above/below EMA50 | 1d only | 0 = below, 1 = above |
| `atr_regime` | Current ATR / 50-period moving average of ATR | 15m, 4h, 1d | Volatility ratio |

*Note: The catalog is enforced by the validator. Each base feature is expanded across its allowed timeframes, yielding concrete column names such as `rsi_15m`, `adx_4h`, or `btc_trend_1d`. Absolute prices and unnormalised quantities (e.g., `close`, `ema_fast`, `atr`) are forbidden.*
