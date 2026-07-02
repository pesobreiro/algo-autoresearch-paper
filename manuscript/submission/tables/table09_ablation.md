## Table 9. Ablation study: random baselines versus LLM-selected configuration (S12 iter 5502)

| Baseline | Fixed component | Varied component | N configurations | Mean holdout Sharpe (2024–2025) | Best holdout Sharpe | Sharpe > 1.0 (%) |
|---|---|---|---:|---:|---:|---:|
| LLM-selected | Features, SL, TP, threshold from S12 iter 5502 | — | 1 | 1.15 | 1.15 | 100 |
| Random SL/TP/threshold | LLM-selected features | SL, TP, threshold uniform random | 200 | 0.90 ± 0.31 | 1.68 | 38 |
| Random features | SL=9.70, TP=9.12, Thr=0.890 | Feature subset uniform random | 8 | 1.32 ± 0.29 | 1.60 | 75 |

*Note: The LLM-selected configuration was produced by the full pipeline: the LLM proposed the feature set and the Optuna TPE sampler tuned SL/TP/threshold on the 2021–2023 validation window. The random SL/TP/threshold baseline fixes the LLM feature set and draws 200 parameter triples uniformly from the same ranges used by Optuna. The random-feature baseline fixes the final SL/TP/threshold values and draws eight random feature subsets from the catalog, training an XGBoost classifier for each and then running a 30-trial random search over SL/TP/threshold. All backtests use max_pos = 5 and modelled costs of 0.20 % fee + 0.10 % slippage. The holdout window (2024–2025) was not used during training or Optuna optimisation. Because the random-feature sample is small and the random search over risk parameters is coarser than the original Optuna study, these figures are indicative rather than definitive; a full ablation would require thousands of additional training runs.*
