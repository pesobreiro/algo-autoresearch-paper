# Deployment — Modelos em Produção

## Modelo 1 — S11 iter 1066 — BNB/USDT — EM PRODUÇÃO

**Iter 1066** — Season 11 — BNB/USDT 15m

```
SL=9.00%  TP=9.69%  Threshold=0.868  ADX_min=22
ATR kill-switch: atr_regime > 3.0 → sem sinal
Fee: 0.2% round-trip  |  Slippage: 0.1%
TFs: 15m + 4h
```

### Performance (€500 inicial, carry-forward, composto)

| Ano | Ret% | Sharpe | DD% | Trades | WR% | Equity |
|-----|------|--------|-----|--------|-----|--------|
| 2022 | +20.4% | 1.31 | -9.6% | 66 | 65% | €602 |
| 2023 | +17.9% | 1.71 | -4.2% | 35 | 74% | €710 |
| 2024 | +20.6% | 2.09 | -2.1% | 31 | 84% | €856 |
| **2025 OOS** | **+26.8%** | **2.94** | **-1.9%** | **32** | **94%** | **€1085** |
| 2026 OOS | +1.9% | 2.99 | 0.0% | 2 | 100% | €1106 |
| **Total** | **+121%** | — | **-9.6%** | **166** | **76%** | **€1106** |

### Scripts

```bash
python deployment/backtest_deploy.py --iter 1066 --season 11 \
    --sl 9.00 --tp 9.69 --thr 0.868 --ticker bnb \
    --years 2022 2023 2024 2025 2026
```

### Modelo

```
best_models/season_11/iter_1066/
├── model/          # XGBoost + feature names
└── meta.json       # params + métricas da season
```

---

## Modelo 2 — S12 iter 5502 — BTC/USDC — CANDIDATO DEPLOY

**Iter 5502** — Season 12 — BTC/USDC 15m

```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
ATR kill-switch: atr_regime > 3.0 → sem sinal
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

### Performance (€500 inicial, carry-forward, composto)

| Ano | Ret% | Sharpe | DD% | Trades | WR% | Equity |
|-----|------|--------|-----|--------|-----|--------|
| 2021 | +3.7% | 0.69 | -4.6% | 13 | 69% | €519 |
| 2022 | +13.0% | 1.79 | -1.1% | 21 | 81% | €586 |
| 2023 | +14.7% | 2.16 | -1.5% | 20 | 90% | €672 |
| 2024 | +10.7% | 0.94 | -9.3% | 49 | 63% | €744 |
| **2025 OOS** | **+9.6%** | **1.27** | **-4.5%** | **32** | **66%** | **€816** |
| **Total** | **+63%** | — | **-9.3%** | **135** | **73%** | **€816** |

### Alternativas Avaliadas (S12)

| Iter | sv | sh | DD% | Trades | Capital final | Notas |
|------|----|----|-----|--------|---------------|-------|
| **5502** | 1.39 | 1.15 | -9.3% | 135 | **€816** | **seleccionado** |
| 5446 | 1.43 | 0.98 | -12.9% | 144 | €630 | DD alto 2021–2022 |
| 1736 | 1.06 | 1.23 | -5.1% | 88 | €623 | val fraca 2022–2023 |

### Scripts

```bash
python deployment/backtest_deploy.py --iter 5502 --season 12 \
    --sl 9.70 --tp 9.12 --thr 0.890 --ticker btc \
    --years 2021 2022 2023 2024 2025

# Sem carry-forward (€500 independentes por ano)
python deployment/backtest_deploy.py --iter 5502 --season 12 \
    --sl 9.70 --tp 9.12 --thr 0.890 --ticker btc --no-compound
```

### Modelo

```
best_models/season_12/iter_5502/
├── model/          # XGBoost + feature names
└── meta.json       # params + métricas da season
```

---

## Comparação Cross-Season

| Modelo | Activo | Capital Final | Ret% Total | Sharpe OOS | DD máx | Trades/ano |
|--------|--------|---------------|------------|------------|--------|------------|
| **S11 iter 1066** | BNB | **€1106** | **+121%** | **2.94** | -9.6% | ~33 |
| S12 iter 5502 | BTC | €816 | +63% | 1.27 | -9.3% | ~27 |

S11/BNB superior em todas as dimensões — BNB tem microestrutura mais favorável para mean-reversion que BTC (menos ruído institucional/macro).
