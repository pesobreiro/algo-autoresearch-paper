# Direção da Pesquisa — algo_autoresearch S12

Este ficheiro guia o agente LLM nas suas decisões de modificação de parâmetros.

---

## Season 12 — BTC/USDC — Concluída

**107 aceites** em ~9900 iterações. Modelo de deploy seleccionado: **iter 5502**.

### Modelo Deploy
```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
TFs: 15m + 4h + 1d  |  REG_ALPHA=7.0  REG_LAMBDA=7.0
Capital final (2021→2025): €816 (+63.2%)  |  2025 OOS: Sharpe=1.27
```

---

## Design de Dados S12 — Imutável

| Período | Papel | Optuna pode usar? |
|---------|-------|-------------------|
| 2017–2020 | Train (XGBoost) | — |
| 2021 | Validação | ✅ SIM |
| 2022 | Validação (bear — FTX crash) | ✅ SIM |
| 2023 | Validação (bull recovery) | ✅ SIM |
| 2024 | **Holdout** | ❌ NÃO — passivo |
| 2025 | **Holdout** | ❌ NÃO — passivo |

---

## Critério de Aceitação S12

```
cv_auc_mean ≥ 0.51
AND sharpe_validation ≥ 1.0
AND sharpe_holdout ≥ 0.3
```

---

## O Que Podes Modificar

**Features** — qualquer subconjunto do catálogo:
- Osciladores: `stoch_rsi_k`, `stoch_rsi_d`, `rsi`, `bb_position`, `adx`
- Momentum: `ema_diff`, `trend`, `returns_1`, `returns_5`
- Volatilidade: `atr_pct`, `bb_width_pct`, `atr_regime`
- MACD: `macd_pct`, `macd_signal_pct`, `macd_hist_pct`
- Volume: `volume_norm`
- Macro (1d only): `dist_sma200_pct`, `btc_trend` ← **nunca remover estes dois**

**Timeframes** — qualquer subconjunto de `["15m", "4h", "1d"]`

**XGBoost** — explorar:
- `N_ESTIMATORS`: 300–1000, `MAX_DEPTH`: 3–10, `LEARNING_RATE`: 0.01–0.30
- `MIN_CHILD_WEIGHT`: 5–80, `GAMMA`: 0–3, `SUBSAMPLE`: 0.6–1.0
- `COLSAMPLE_BYTREE`: 0.6–1.0, `REG_ALPHA`: 0–10, `REG_LAMBDA`: 0–10

**Bounds Optuna**:
- `SL_RANGE`: tuplo (min, max) com 1.0 <= min < max <= 15.0
- `TP_RANGE`: tuplo (min, max) com 2.0 <= min < max <= 20.0
- `THRESHOLD_RANGE`: tuplo com 0.70 <= min < max <= 0.99

---

## Regras Absolutas

1. **APENAS indicadores relativos** — nunca usar preços absolutos
2. Usar apenas features do catálogo `pipeline/features_catalog.py`
3. `SL_RANGE`, `TP_RANGE`, `THRESHOLD_RANGE` são **tuplos (min, max)** com min < max
4. **`OBJECTIVE_MODE = "score"`** — NÃO remover, NÃO alterar, NÃO omitir
5. `N_TRIALS` <= 200
6. `btc_trend` e `dist_sma200_pct` **nunca remover** — contexto macro obrigatório

---

## Notas S12

- **Features 1d dominantes**: `bb_width_pct_1d`, `macd_signal_pct_1d`, `stoch_rsi_k_1d` — manter TF 1d
- **REG_ALPHA/LAMBDA=7.0** mostrou-se mais estável que 3.5 para BTC
- **Threshold dominante** (importância Optuna ~0.85) — afinar THRESHOLD_RANGE é a alavanca principal
- **2025 regime extremo**: BTC subiu a 125k e desceu a 60k — qualquer modelo com sh>1.0 em 2025 é excepcional
- **THRESHOLD_RANGE mínimo**: não descer abaixo de 0.70 — gera demasiados falsos positivos em BTC

---

## 📊 Top 5 Aceites S12 (por deploy score)

**#1 iter=5502** sv=1.39 | sh=1.15 | 2024=€577 | 2025=€638 | score=0.95
→ SL=9.70% TP=9.12% Thr=0.890 | TFs=['15m','4h','1d'] ← **SELECCIONADO**

**#2 iter=5446** sv=1.43 | sh=0.98 | 2024=€622 | 2025=€665 | score=0.79
→ SL=10.39% TP=9.15% Thr=0.874 | TFs=['15m','4h','1d']

**#3 iter=1736** sv=1.06 | sh=1.23 | 2024=€612 | 2025=€646 | score=0.75
→ SL=7.03% TP=10.92% Thr=0.919 | TFs=['15m','4h']

**#4 iter=1692** sv=1.24 | sh=0.96 | 2024=€607 | 2025=€639 | score=0.98
→ SL=7.23% TP=9.99% Thr=0.897 | TFs=['15m','4h']

**#5 iter=8942** sv=1.25 | sh=0.67 | 2024=€684 | 2025=€630 | score=0.93
→ SL=7.43% TP=9.75% Thr=0.876 | TFs=['15m','4h','1d']

---
