# Deployment — production models

## Model 1 — S11 iter 1077 — BNB/USDT — live

**Iter 1077** — Season 11 — BNB/USDT 15m

```
SL=7.80%  TP=7.10%  Threshold=0.857  ADX_min=22
ATR kill-switch: atr_regime > 3.0 → no signal
Fee: 0.2% round-trip  |  Slippage: 0.1%
TFs: 15m + 4h + 1d
```

### Features

```
ema_diff_1d, macd_pct_1d, macd_signal_pct_1d, returns_5_15m, dist_sma200_pct,
btc_trend, atr_pct_1d, adx_1d, rsi_1d, stoch_rsi_k_1d, volume_norm, ema_diff_4h,
bb_width_pct_15m
```

### Performance (€500 initial capital per year, independent backtests)

| Year | Ret% | Sharpe | DD% | Trades | WR% | Equity |
|-----|------|--------|-----|--------|-----|--------|
| 2022 | +16.75% | 1.40 | -6.74% | 82 | 65.9% | €584 |
| 2023 | +7.80% | 1.05 | -5.86% | 27 | 70.4% | €539 |
| 2024 | +15.80% | 2.59 | -1.85% | 24 | 91.7% | €579 |
| **2025 OOS** | **+15.90%** | **2.87** | **-1.70%** | **24** | **91.7%** | **€580** |
| 2026 OOS | +1.94% | 4.17 | 0.00% | 4 | 75.0% | €510 |
| **Total** | **+58.2%** | — | **-6.74%** | **161** | — | — |

### Scripts

```bash
python deployment/backtest_deploy.py --iter 1077 --season 11 \
    --sl 7.80 --tp 7.10 --thr 0.857 --ticker bnb \
    --years 2022 2023 2024 2025 2026
```

### Model files

```
best_models/season_11/iter_1077/
├── model/          # XGBoost + feature names
└── meta.json       # params + season metrics
```

---

## Model 2 — S12 iter 5502 — BTC/USDT — candidate

**Iter 5502** — Season 12 — BTC/USDT 15m

```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
ATR kill-switch: atr_regime > 3.0 → no signal
Fee: 0.2% round-trip  |  Slippage: 0.1%
TFs: 15m + 4h + 1d
```

### Features
```
ema_diff_15m, macd_pct_4h, bb_width_pct_15m, dist_sma200_pct, btc_trend,
volume_norm, adx_15m, rsi_4h, macd_hist_pct_15m, stoch_rsi_k_1d,
atr_regime_4h, macd_signal_pct_1d, macd_pct_1d, bb_width_pct_1d,
atr_regime_15m, dist_sma200_pct_4h, macd_pct_15m, stoch_rsi_d_1d, ema_diff_4h
```

### Performance (€500 initial capital, carry-forward, compounded)

| Year | Ret% | Sharpe | DD% | Trades | WR% | Equity |
|-----|------|--------|-----|--------|-----|--------|
| 2021 | +3.7% | 0.69 | -4.6% | 13 | 69% | €519 |
| 2022 | +13.0% | 1.79 | -1.1% | 21 | 81% | €586 |
| 2023 | +14.7% | 2.16 | -1.5% | 20 | 90% | €672 |
| 2024 | +10.7% | 0.94 | -9.3% | 49 | 63% | €744 |
| **2025 OOS** | **+9.6%** | **1.27** | **-4.5%** | **32** | **66%** | **€816** |
| **Total** | **+63%** | — | **-9.3%** | **135** | **73%** | **816** |

### Evaluated alternatives (S12)

| Iter | sv | sh | DD% | Trades | Final capital | Notes |
|------|----|----|-----|--------|---------------|-------|
| **5502** | 1.39 | 1.15 | -9.3% | 135 | **€816** | **selected** |
| 5446 | 1.43 | 0.98 | -12.9% | 144 | €630 | high DD 2021–2022 |
| 1736 | 1.06 | 1.23 | -5.1% | 88 | €623 | weak val 2022–2023 |

### Scripts

```bash
python deployment/backtest_deploy.py --iter 5502 --season 12 \
    --sl 9.70 --tp 9.12 --thr 0.890 --ticker btc \
    --years 2021 2022 2023 2024 2025

# Without carry-forward (€500 independent per year)
python deployment/backtest_deploy.py --iter 5502 --season 12 \
    --sl 9.70 --tp 9.12 --thr 0.890 --ticker btc --no-compound
```

### Model files

```
best_models/season_12/iter_5502/
├── model/          # XGBoost + feature names
└── meta.json       # params + season metrics
```

---

## Cross-season comparison

| Model | Asset | Final Capital | Total Ret% | OOS Sharpe | Max DD | Trades/year |
|--------|--------|---------------|------------|------------|--------|------------|
| **S11 iter 1077** | BNB | — | **+58.2%** | **2.87** | **-6.74%** | ~32 |
| S12 iter 5502 | BTC | €816 | +63% | 1.27 | -9.3% | ~27 |

S11/BNB has the better OOS Sharpe and the smaller drawdown; S12/BTC produced slightly higher total return in the independent-backtest scenario. BNB's price microstructure is more favorable to mean-reversion than BTC's, which carries more institutional and macro noise.
