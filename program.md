# Research Direction — algo_autoresearch S12

This file guides the LLM agent in its parameter-modification decisions.

---

## Season 12 — BTC/USDT — Completed

**107 accepted** in ~9900 iterations. Selected deploy model: **iter 5502**.

### Deploy Model
```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
TFs: 15m + 4h + 1d  |  REG_ALPHA=7.0  REG_LAMBDA=7.0
Final capital (2021→2025): €816 (+63.2%)  |  2025 OOS: Sharpe=1.27
```

---

## S12 Immutable Data Design

| Period | Role | Can Optuna use? |
|---------|-------|-------------------|
| 2017–2020 | Train (XGBoost) | — |
| 2021 | Validation | YES |
| 2022 | Validation (bear — FTX crash) | YES |
| 2023 | Validation (bull recovery) | YES |
| 2024 | **Holdout** | NO — passive |
| 2025 | **Holdout** | NO — passive |

---

## S12 Acceptance Criterion

```
cv_auc_mean ≥ 0.51
AND sharpe_validation ≥ 1.0
AND sharpe_holdout ≥ 0.3
```

---

## What You Can Modify

**Features** — any subset of the catalog:
- Oscillators: `stoch_rsi_k`, `stoch_rsi_d`, `rsi`, `bb_position`, `adx`
- Momentum: `ema_diff`, `trend`, `returns_1`, `returns_5`
- Volatility: `atr_pct`, `bb_width_pct`, `atr_regime`
- MACD: `macd_pct`, `macd_signal_pct`, `macd_hist_pct`
- Volume: `volume_norm`
- Macro (1d only): `dist_sma200_pct`, `btc_trend` ← **never remove these two**

**Timeframes** — any subset of `["15m", "4h", "1d"]`

**XGBoost** — explore:
- `N_ESTIMATORS`: 300–1000, `MAX_DEPTH`: 3–10, `LEARNING_RATE`: 0.01–0.30
- `MIN_CHILD_WEIGHT`: 5–80, `GAMMA`: 0–3, `SUBSAMPLE`: 0.6–1.0
- `COLSAMPLE_BYTREE`: 0.6–1.0, `REG_ALPHA`: 0–10, `REG_LAMBDA`: 0–10

**Optuna Bounds**:
- `SL_RANGE`: tuple (min, max) with 1.0 <= min < max <= 15.0
- `TP_RANGE`: tuple (min, max) with 2.0 <= min < max <= 20.0
- `THRESHOLD_RANGE`: tuple with 0.70 <= min < max <= 0.99

---

## Absolute Rules

1. **ONLY relative indicators** — never use absolute prices
2. Use only features from the catalog `pipeline/features_catalog.py`
3. `SL_RANGE`, `TP_RANGE`, `THRESHOLD_RANGE` are **tuples (min, max)** with min < max
4. **`OBJECTIVE_MODE = "score"`** — DO NOT remove, DO NOT change, DO NOT omit
5. `N_TRIALS` <= 200
6. `btc_trend` and `dist_sma200_pct` **never remove** — mandatory macro context

---

## S12 Notes

- **Dominant 1d features**: `bb_width_pct_1d`, `macd_signal_pct_1d`, `stoch_rsi_k_1d` — keep TF 1d
- **REG_ALPHA/LAMBDA=7.0** proved more stable than 3.5 for BTC
- **Threshold dominates** (Optuna importance ~0.85) — tuning THRESHOLD_RANGE is the main lever
- **2025 extreme regime**: BTC rallied to 125k and dropped to 60k — any model with sh>1.0 in 2025 is exceptional
- **Minimum THRESHOLD_RANGE**: do not go below 0.70 — generates too many false positives in BTC

---

## Top 5 S12 Accepted (by deploy score)

**#1 iter=5502** sv=1.39 | sh=1.15 | 2024=€577 | 2025=€638 | score=0.95
→ SL=9.70% TP=9.12% Thr=0.890 | TFs=['15m','4h','1d'] ← **SELECTED**

**#2 iter=5446** sv=1.43 | sh=0.98 | 2024=€622 | 2025=€665 | score=0.79
→ SL=10.39% TP=9.15% Thr=0.874 | TFs=['15m','4h','1d']

**#3 iter=1736** sv=1.06 | sh=1.23 | 2024=€612 | 2025=€646 | score=0.75
→ SL=7.03% TP=10.92% Thr=0.919 | TFs=['15m','4h']

**#4 iter=1692** sv=1.24 | sh=0.96 | 2024=€607 | 2025=€639 | score=0.98
→ SL=7.23% TP=9.99% Thr=0.897 | TFs=['15m','4h']

**#5 iter=8942** sv=1.25 | sh=0.67 | 2024=€684 | 2025=€630 | score=0.93
→ SL=7.43% TP=9.75% Thr=0.876 | TFs=['15m','4h','1d']

---
