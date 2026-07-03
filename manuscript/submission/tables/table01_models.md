## Table 1. Case-study strategies

| Feature | S11 iter 1077 (production) | S12 iter 5502 (BTC replication) |
|---|---|---|
| Asset | BNB/USDT | BTC/USDT |
| Training period | 2022–2023 | 2017–2020 |
| Validation period | 2022–2024 (walk-forward) | 2021–2023 |
| Holdout period | 2025 | 2024–2025 |
| True OOS period | Jan–Mar 2026 | — |
| Validation AUC | 0.5721 | 0.5321 |
| Validation Sharpe | 1.69 | 1.39 |
| Holdout Sharpe | 2.87 | 1.15 |
| True OOS Sharpe | 4.17 | — |
| Total return (2022–2026) | 58.2 % (simple sum of annual returns) | 44.0 % (2021–2025, simple sum) |
| Holdout return (2024–2025) | — | 27.7 % (compounded) |
| Maximum drawdown | −6.74 % | −8.1 % |
| Average win rate | 65.9–91.7 % | 90.9 % |
| Number of trades | 161 (2022–2026) | 38 (2021–2025) |
| Stop-loss / Take-profit | 7.8 % / 7.1 % | 9.7 % / 9.1 % |
| Probability threshold | 0.857 | 0.890 |
| Regime filter | ADX ≥ 22, ATR kill 3σ | ADX implicit in features |
| Modeled costs | 0.20 % fee + 0.10 % slippage | 0.20 % fee + 0.10 % slippage |

*Note: S11 metrics are produced by `deployment/evaluate_models.py`: independent yearly backtests with initial capital 500, max_pos=5, ATR kill 3.0, and modeled costs 0.20 % fee + 0.10 % slippage. The holdout year (2025) and the true out-of-sample period (Jan–Mar 2026) were never used during hyper-parameter optimization.*
