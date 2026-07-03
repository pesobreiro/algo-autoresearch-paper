## Table S2. Inter-season design decisions

| Transition | Trigger / observed outcome | Decision | Rationale |
|---|---|---|---|
| S4 → S5 | Best S4 results hit SL/TP ceiling (8 %); threshold < 0.80 was a dead zone | Expand `SL_RANGE`/`TP_RANGE` to 10 %; enforce `THRESHOLD_RANGE` ≥ 0.80 | Give Optuna room to explore wider exits and avoid low-threshold overtrading |
| S5 → S9 | Earlier seasons (S2–S8) had OOS contamination: Optuna touched 2025/2026 | Introduce passive holdout: `validation_years=[2023,2024]`, `holdout_years=[2025]`; Optuna never sees holdout | Restore holdout integrity; modern train window 2020–2022 |
| S9 → S10 | S9 validation [2023,2024] were both bull markets; best S9 holdout was negative | Add 2022 bear market to validation; introduce ATR kill-switch at 3× normal volatility | Force robustness to bear regimes and volatility spikes |
| S10 → S11 | S10 ratchet baseline caused progressive val-overfitting (holdout degraded) | Replace ratchet with triple gate: `AUC≥0.55`, `Sharpe(val)≥1.0`, `Sharpe(holdout)≥0.5`; no ratchet | Stop holdout degradation while retaining selection pressure |
| S11 → S12 | S11 succeeded on BNB; test cross-asset generalization | Switch to BTC/USDC; train 2017–2020, val 2021–2023, holdout 2024–2025; relax holdout gate to 0.3 | Verify asset-agnostic design on a different volatility regime |

*Note: Decisions were made by the human reviewer after each season based on the distribution of accepted strategies, failure modes, and holdout behavior. Each transition typically required 15–30 minutes of review.*
