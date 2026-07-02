# CHANGELOG

## 2026-06-26 — Manuscript normalisation and MDPI Analytics submission package

### Changed
- Normalised `manuscript/paper_draft_analytics.md`, `paper_draft_analytics_figures.md`, and `paper_draft_analytics.tex` for MDPI Analytics style:
  - Added author information and removed citations from the abstract (now ≤ 200 words).
  - Updated aggregate experiment counts to 34 333 iterations and 329 accepted strategies (from `season_summary.json`).
  - Aligned S11 iter 1077 reporting with `deployment/evaluate_models.py`: 161 trades, max drawdown −6.74%, and true-OOS 2026 Sharpe 4.17.
  - Corrected the selection-bias paragraph to reflect that the deploy score was computed for the 30 top holdout candidates re-evaluated from Season 11.
  - Expanded reference entries that previously used "et al." and corrected the Alpha-GPT arXiv identifier to 2308.00016.
- Updated `deployment/selected_model.json` and `manuscript/selected_model.json` to be consistent with `evaluate_models.py`.
- Updated `manuscript/analytics/tables/verified_facts.md`, `table01_models.md`, and `audit_report.md` to reflect the corrected S11 metrics.

### Added
- `manuscript/submission/` package with `manuscript.md`, `manuscript.tex`, `figures/`, `tables/`, `cover_letter.md`, `highlights.md`, `declarations.md`, and `README_submission.md`.

## 2026-03-25 — Season 12 — BTC/USDC — Conclusion and Deploy

### S12 Results
- **107 accepted** in ~9900 iterations (1.1% acceptance rate)
- **Selected deploy model**: iter 5502 — best val+holdout+equity OOS balance
- **Splits**: train 2017–2020, val 2021–2023, holdout 2024–2025
- **Triple gate**: AUC≥0.51 AND Sharpe(val)≥1.0 AND Sharpe(holdout)≥0.3

### Deploy Model — iter 5502
```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
TFs: 15m + 4h + 1d  |  REG_ALPHA=7.0  REG_LAMBDA=7.0
Features: ema_diff_15m, macd_pct_4h, bb_width_pct_15m, dist_sma200_pct, btc_trend,
          volume_norm, adx_15m, rsi_4h, macd_hist_pct_15m, stoch_rsi_k_1d,
          atr_regime_4h, macd_signal_pct_1d, macd_pct_1d, bb_width_pct_1d,
          atr_regime_15m, dist_sma200_pct_4h, macd_pct_15m, stoch_rsi_d_1d, ema_diff_4h
```

| Year | Sharpe | Ret% | DD% | Trades | WR% | Equity |
|-----|--------|------|-----|--------|-----|--------|
| 2021 | 0.69 | +3.7% | -4.6% | 13 | 69% | €519 |
| 2022 | 1.79 | +13.0% | -1.1% | 21 | 81% | €586 |
| 2023 | 2.16 | +14.7% | -1.5% | 20 | 90% | €672 |
| 2024 | 0.94 | +10.7% | -9.3% | 49 | 63% | €744 |
| **2025 OOS** | **1.27** | **+9.6%** | **-4.5%** | **32** | **66%** | **€816** |
| **Total** | — | **+63.2%** | **-9.3%** | **135** | **73%** | **€816** |

### Changed
- **`config.yaml`**: `ticker: bnb→btc`, `season: 11→12`, `max_positions: 5→2`, `accept_auc_min: 0.55→0.51`, `accept_sharpe_holdout_min: 0.5→0.3`, `train_start: 2019→2017`, `train_end: 2021→2020`, `validation_years: [2022,2023,2024]→[2021,2022,2023]`, `holdout_years: [2025]→[2024,2025]`, `max_tokens: 2048→4096`
- **`llm/start_server.sh`**: `ctx-size 8192→16384`, added `--flash-attn on`, removed `--cache-type-k/v q4_0` (incompatible with Qwen2.5-7B-Instruct — caused corrupted outputs)
- **`autoresearch/agent.py`**: history sent to LLM 5→30 iters
- **`autoresearch/runner.py`**: `listar_historico(10→30)`, `top_n_scores(5→10)` in two places
- **`main.py`**: bug fix — `_gerar_research_params` did not include `OBJECTIVE_MODE` in the focus template
- **`deployment/backtest_deploy.py`**: bug fix — hardcoded ticker `'bnb'` → `--ticker` argument

### S12 Lessons — BTC vs BNB
- BTC 2025 was an extreme regime (rally to 125k, drop to 60k) — not seen in training 2017–2020
- KV cache quantised (`--cache-type-k q4_0`) incompatible with Qwen2.5-7B-Instruct → corrupted outputs (AL_AL_AL...)
- Widening splits to 2017–2020 (train) + 2021–2023 (val) improved holdout stability
- 1d features dominated S12 importance: `bb_width_pct_1d`, `macd_signal_pct_1d`, `stoch_rsi_k_1d`

## 2026-03-22 — Season 11 Setup (BNB/USDT) — Triple Gate + Holdout Acceptance

### Changed
- **`config.yaml`**: `season: 10→11`, added `accept_sharpe_holdout_min: 0.5`
- **`autoresearch/runner.py`**: acceptance now uses **triple gate**: `AUC≥0.55 AND Sharpe(val)≥1.0 AND Sharpe(holdout)≥0.5`; removed baseline ratchet — any iteration passing all 3 gates is accepted; rejection message shows holdout gate
- **`pipeline/research_params.py`**: reset for S11 from S10 iter=417 (best holdout: AUC=0.563, Sharpe(val)=1.50, Sharpe(holdout)=2.82, equity 2025→€600)

### Added — Lesson from S10: Val Overfitting with Ratchet
- S10 (1148 iters, 3 accepted): baseline increased with each acceptance (ratchet) → LLM forced to find increasingly optimised configs for 2022-2024
- Progression of the 3 accepted: iter=326 (holdout=+2.41) → iter=582 (holdout=+0.17) → iter=727 (holdout=-1.02) — clear degradation
- **S11 fix (Option A)**: holdout enters as gate (`≥0.5`), no ratchet; 2025 is no longer true holdout but eliminates progressive val overfitting
- Starting point: iter=417 S10 — simpler features (11 vs 16), best holdout of all S10

## 2026-03-21 — Season 10 Setup (BNB/USDT) — Bear Market Validation

### Changed
- **`config.yaml`**: `season: 9→10`, `train_start: 2020→2019`, `train_end: 2022→2021`, `validation_years: [2022,2023,2024]` (was [2023,2024]), `scoring.atr_regime_kill: 3.0` (new)
- **`pipeline/backtest.py`**: `load_and_prepare()` adds `atr_regime` to data dict; `correr_backtest()` reads `atr_kill` from config; kill-switch applied in all `simulate_numba` calls (objective, val loop, holdout loop, forensics): `probs_safe = np.where(atr_regime > atr_kill, 0.0, probs)`
- **`pipeline/run_pipeline.py`**: defaults `hash_entry_params(train_start=2019, train_end=2021)`
- **`pipeline/research_params.py`**: reset S10 from S9 iter=311 (AUC=0.561, Sharpe=2.83)
- **`program.md`**: rewritten for S10 — data split with 2022 bear, ATR kill-switch, `min(Sharpe_val)` effect with 3 years

### Added — Lesson from S9: Regime Blindness
- S9 (311 iters, 5 accepted): validation [2023,2024] were both bull markets → Optuna never saw a crash
- iter=311 (best S9): Sharpe(val)=2.83 but Sharpe(holdout)=-0.28 — overfit to bull run
- iter=178 (most robust S9): Sharpe(val)=2.41, Sharpe(holdout)=+1.00 — survived by not trading in 2025
- **S10 fix**: `validation_years=[2022,2023,2024]` — 2022 (FTX/Luna, BNB -50%) dominates `min(Sharpe_val)`, forcing Optuna to find configs that survive bear markets
- **ATR kill-switch**: events with `atr_regime>3.0` (volatility 3× normal) are mechanically blocked — not delegated to ML

## 2026-03-21 — Season 9 Setup (BNB/USDT) — Clean OOS Design

### Changed
- **`config.yaml`**: `ticker: btc → bnb`, `season: 8 → 9`, `train_start: 2020`, `train_end: 2022`, `validation_years: [2023, 2024]`, `holdout_years: [2025]`, `baseline_override: 0.0`, `accept_auc_min: 0.55`, `accept_sharpe_min: 1.0`, `min_sharpe_gate: 0.3`
- **`pipeline/backtest.py`**: replaced single `years` with `validation_years` + `holdout_years`; Optuna uses only `data_por_ano_val`; score mode maximises `min(sharpe_val)` (worst sub-window) instead of mean; final metrics computed for val and holdout separately; `fee_pct=0.002` (0.2% round-trip) in all `simulate_numba`; return dict with `sharpe_validation` and `sharpe_holdout`
- **`pipeline/run_pipeline.py`**: `hash_entry_params()` accepts `train_start`+`train_end` — label cache invalidated between seasons with different train windows
- **`autoresearch/tracker.py`**: `top_n_scores()` and `melhor_score()` sort by `sharpe_validation` (with fallback to `score_composto`)
- **`autoresearch/runner.py`**: S9 acceptance uses dual-gate (`cv_auc_mean ≥ accept_auc_min AND sharpe_validation ≥ accept_sharpe_min`); holdout is passive — never used for filtering; `score_baseline` tracks `sharpe_validation`; console shows AUC + Sharpe(val) + Sharpe(holdout/passive); `limpar_cache()` passes `train_start/train_end` to hash
- **`pipeline/research_params.py`**: reset for S9/BNB from S5 iter 349 (Sharpe=2.37). `FEATURES` includes `btc_trend`+`dist_sma200_pct` mandatory; `N_TRIALS=200`; `OBJECTIVE_MODE="score"`
- **`program.md`**: rewritten for S9 — BNB asset, acceptance by AUC+Sharpe(val) gates, immutable data split, ATR kill-switch, passive holdout

### Fixed — 5 methodological improvements
1. **Holdout integrity (critical)**: Optuna never touches 2025. Seasons S2–S8 had OOS contamination (Optuna optimised on 2025+2026 which were "OOS")
2. **Optuna anti-overfitting**: N_TRIALS=200 (2 years of val cannot support more); objective = `min(sharpe_val)` (worst sub-window) instead of mean
3. **Round-trip fees**: `fee_pct=0.002` (0.2%) in all `simulate_numba` — previously only 0.1% on exit (missing entry fee)
4. **Modern train window**: 2020–2022 (excludes 2017–2019 with microstructure/liquidity different from modern markets)
5. **BNB asset**: less institutional noise than BTC (BTC has global ETFs + macro funds that hinder ML mean-reversion; S8 BTC: 134 iters, 0 accepted)

## 2026-03-21 — Season 8 Setup (BTC/USDT)

### Changed
- **`pipeline/research_params.py`**: reset for S8/BTC — `N_TRIALS=3000` (reduced from 10000, OOM mitigation), `SL_RANGE=(2.0,12.0)`, `TP_RANGE=(5.0,25.0)`, `THRESHOLD_RANGE=(0.85,0.95)` (opened for BTC exploration), `TIMEFRAMES=["15m","4h","1d"]`, `OBJECTIVE_MODE="profit"`
- **`config.yaml`**: `ticker: eth → btc`, `season: 7 → 8`, `baseline_override: 30.154715`
- **`program.md`**: rewritten for S8/BTC with cross-asset context (reference to BNB/ETH optima)

### Fixed (OOM mitigations diagnosed in S7)
- **`autoresearch/runner.py`**: added `import gc` and `gc.collect()` after `tracker.guardar_experiencia()` each iteration — forces release of Optuna/sklearn reference cycles before next iteration
- **`pipeline/backtest.py`**: added module-level `_PARQUET_CACHE: dict` and `_load_parquet_cached()` — each parquet file is read from disk **only once** per process and reused in subsequent iterations, eliminating the main RSS growth cause

## 2026-03-21 — OOM Diagnostic Season 7 (no fix applied)

### Diagnostic: OOM crashes during ~1095 iterations (S7)

**Evidence from kernel logs (`/var/log/syslog`):**

| Date       | PID     | RSS anon      | VM total |
|------------|---------|---------------|----------|
| 2026-03-16 | 3832181 | **7.7 GB**    | 45 GB    |
| 2026-03-17 | 2768659 | **25.5 GB**   | 64.5 GB  |
| 2026-03-20 | 2813875 | **26.7 GB**   | 67.2 GB  |
| 2026-03-21 | —       | killed (OOM)  | —        |

The Python process was killed by the kernel OOM killer **at least 4 times** during S7.
RSS grew from 7.7 GB → 25.5 GB between restarts, confirming **gradual memory leak** (not a single allocation).
The `s7.txt` log confirms one restart: loop resumed at iter 966 after previous crash (max accepted iter: 580).

**Identified causes (in estimated impact order):**

1. **`backtest.py` — `load_and_prepare()` without data cache** _(primary cause)_
   - Each iteration reads the full parquet file (2017–2024, ~258k rows × multiple TFs)
     into memory, merges timeframes, and filters by OOS year.
   - With 2 OOS years (2025 + 2026) and 10,000 Optuna trials per iteration, ~258k-row arrays
     are generated and passed to Numba each call — with no explicit `gc.collect()` after each iteration.
   - Python/glibc does not return arenas to the OS after freeing large objects: RSS grows
     to the high-water mark of all allocations and never drops.

2. **`backtest.py` — Optuna `TPEsampler` with 10,000 trials** _(secondary cause)_
   - Each `create_study()` is local and theoretically freed at the end of `correr_backtest()`.
   - However, `get_param_importances(study)` (line ~528) uses sklearn internally
     (Random Forest), which creates objects with reference cycles — delaying GC.
   - With 1000+ iterations × 10,000 trials, these structures accumulate until GC processes them.

3. **`generate_labels.py` + `backtest.py` — Numba `@njit` without prior warmup** _(tertiary cause)_
   - Functions `simulate_numba`, `simulate_numba_equity`, `simulate_trades_numba` are `@njit`.
   - The LLVM compiler keeps all compilation artefacts in memory (IR + machine code).
   - If arrays with distinct shapes are passed across iterations, Numba compiles
     new specialisations that stay in memory for the process lifetime.

4. **`runner.py` — no `gc.collect()` between iterations** _(amplifier)_
   - The main loop never calls `gc.collect()`. With reference cycles created
     by Optuna/sklearn, objects are not freed deterministically.

**Mitigations for next experiment (S8+):**

- [ ] Module-level cache of parquet DataFrames in `backtest.py` (read 1x, reuse between iterations)
- [ ] Call `gc.collect()` at the end of each iteration in `runner.py`
- [ ] Reduce `N_TRIALS` from 10,000 to 3,000–5,000 (exploration vs. memory trade-off)
- [ ] Consider isolated subprocess for the pipeline (each iteration in child process that exits cleanly)
- [ ] Increase swap or use `ulimit -v` to limit VM and detect crashes earlier

## 2026-03-15 (Season 5 — profit objective)

### Added
- **`OBJECTIVE_MODE` in Optuna** (`pipeline/backtest.py`): `correr_backtest()` accepts new parameter `objective_mode='score'|'profit'`. In `profit` mode, Optuna directly maximises `retorno_total_oos_pct` (average OOS return %) instead of the composite score. Composite score is still computed and reported in both modes.
- **Minimum Sharpe gate in profit mode** (`pipeline/backtest.py`): strategies with Sharpe < 0.5 are rejected by Optuna (`-999.0`). Avoids selecting roller-coaster strategies that maximise return at the cost of an erratic equity curve. Configurable via `config.scoring.min_sharpe_gate`.
- **DD gate in profit mode** (`pipeline/backtest.py`): `max_dd < max_dd_gate` (default -30%) returns `-999.0`. Configurable via `config.scoring.max_dd_gate`.
- **`min_trades_profit`** (`pipeline/backtest.py`): minimum trades in profit mode read from `config.scoring.min_trades_profit` (default inherits `min_trades`). Avoids overfitting to 3 "lucky" high-return trades.

### Changed
- **`pipeline/run_pipeline.py`**: `carregar_params()` reads `OBJECTIVE_MODE` (default `'score'`); `executar_pipeline()` passes `objective_mode` to `correr_backtest()`.
- **`pipeline/research_params.py`**: added `OBJECTIVE_MODE = "profit"` for Season 5 start.
- **`autoresearch/runner.py`**: accept/reject now mode-aware — uses `retorno_total_oos_pct` when `OBJECTIVE_MODE='profit'`, `score_composto` otherwise. Accept/reject messages show `%` instead of score in profit mode. Baseline when resuming search also reads the correct metric according to the last accepted iteration's mode.
- **`program.md`**: rewritten for S5 — primary objective is `retorno_total_oos_pct`, Sharpe/DD are context metrics. LLM instructed to prioritise return without sacrificing minimum Sharpe (>= 0.5).

## 2026-03-15 (session 5 — cont. 2)

### Changed
- **Exit fee added to Numba simulator** (`pipeline/backtest.py`): `simulate_numba` and `simulate_numba_equity` receive new parameter `fee_pct=0.001` (0.1%). At all 3 exit points (SL hit, TP hit, end-of-period), capital is now computed as `sz * (exit_price / entry_price) * (1 - fee_pct)` instead of `sz + sz * (exit_price/entry - 1)`. This models Binance+BNB taker fee (~0.075%) + exit slippage (~0.025%). Without this fix, 148 trades × 0.1% = ~15% drag was being ignored, overstating returns.

## 2026-03-15 (session 5 — cont.)

### Changed
- **`THRESHOLD_RANGE` minimum raised to 0.80** (`pipeline/research_params.py`, `autoresearch/runner.py`, `program.md`): empirical analysis of 2400+ S4 iterations shows threshold < 0.80 is a dead zone (99-100% negative results, avg 1000+ trades). Only threshold ≥ 0.85 produces positive results (avg 332 trades). Auto-corrector now enforces `_THR_LO_MIN=0.80` and `_THR_HI_MIN=0.90`.
- **`program.md` updated** with explicit empirical evidence of the threshold dead zone, so the LLM does not propose values below 0.80 again.

## 2026-03-15 (session 5)

### Changed
- **`SL_RANGE` and `TP_RANGE` expanded to 10.0** (`pipeline/research_params.py`, `autoresearch/runner.py`): best S4 results (iters 24, 25, 43) had SL=7.86–7.97% hitting the previous 8.0 ceiling. Expanded to `SL_RANGE=(1.0, 10.0)` and `TP_RANGE=(0.5, 10.0)` to give Optuna room to explore SL>8%. `THRESHOLD_RANGE` widened to `(0.6, 0.95)`.
- **Auto-corrector `_SL_HI_MIN` and `_TP_HI_MIN` raised to 10.0** (`autoresearch/runner.py`): the auto-corrector's minimum SL/TP ceiling barrier was raised from 7.0/5.0 to 10.0/10.0 to keep consistency with the new target bounds.

## 2026-03-14 (session 4)

### Added
- **Season support** (`main.py`, `runner.py`, `config.yaml`): `--season N` flag on all CLI commands. Each season uses a separate experiment directory (`experiments/` for S1, `experiments_s2/` for S2, etc.). `config.yaml` can set `agent.season` and `agent.baseline_override` to impose a minimum score when starting a new season.
- **Season 2 started**: `research_params.py` reset to S1 starting point (iter 32, score=0.5607) with XGBoost Optuna active (`N_TRIALS_XGB=30`, with DEPTH/LR/ESTIMATORS/ALPHA/LAMBDA ranges) and macro features `btc_trend` + `dist_sma200_pct` introduced.
- **`program.md` rewritten for S2**: describes current architecture without transient alerts; clear priorities (macro features, XGBoost Optuna, bounds shifting).
- **`new-season` command** (`main.py`): automatic transition ritual between seasons. Reads best result from current season, updates `config.yaml` (`season` and `baseline_override`) and regenerates `research_params.py` from the best result's `params_snapshot`. Supports `--dry-run`. If no accepted results, increments season without changing params.
- **Temperature curriculum** (`runner.py`, `agent.py`): LLM temperature adjusted automatically based on score trend. When score improves, temperature drops by `t_decay=0.92` (exploit). When `stagnation_threshold=5` iterations pass without improvement, temperature rises by `t_grow=1.08` (explore). Limits `[t_min=0.3, t_max=1.2]` configurable in `config.yaml`. Current temperature shown in each iteration header.
- **Rejection feedback loop** (`runner.py`, `agent.py`): last 5 validation rejections are accumulated and included in the next LLM prompt as `⚠️ RECENT REJECTIONS — DO NOT REPEAT`. Prevents the LLM from repeating invalid features (`macd_diff`, `macd_signal`) in consecutive iterations. List is cleared when a proposal passes validation.
- **Macro regime features** (`features_catalog.py`, `generate_labels.py`, `agent.py`):
  - `dist_sma200_pct` (1d-only): distance to SMA200 normalised by price (%), calculated from 1d ticker close
  - `btc_trend` (1d-only): BTC above/below EMA50 (0/1), cross-asset, loaded from `btc_01d_usdt_binance.parquet`
  - `atr_regime` (all TFs): current ATR / rolling_mean(ATR, 50) — relative volatility ratio
  - Added `FEATURES_1D_ONLY` to catalog (silent skip for TFs != 1d)
  - `_adicionar_macro_features()` in `generate_labels.py` computes and merges features into the merged dataframe
  - Agent system prompt updated with the 3 new features and usage notes

## 2026-03-14 (session 3)

### Added
- **Forensic drawdown analysis** (`backtest.py`): after each backtest, detects max drawdown period (peak→trough), calculates duration in days, average/minimum ADX and classifies market regime (Sideways/Weak Trend/Strong Trend). Included in LLM prompt with automatic correction suggestion.
- **`simulate_numba_equity`** (`backtest.py`): variant of Numba simulator that returns full equity curve + drawdown period indices.
- **XGBoost Optuna** (`train.py`): optional Optuna study for XGBoost hyperparameters (DEPTH_RANGE, LR_RANGE, ESTIMATORS_RANGE, ALPHA_RANGE, LAMBDA_RANGE). Activated with `N_TRIALS_XGB >= 20`. When active, Optuna finds best XGBoost config via 3-fold CV AUC before final training.
- `research_params.py`: new optional XGBoost Optuna fields documented (commented by default, N_TRIALS_XGB=0).
- Agent system prompt: instructions to activate XGBoost Optuna when AUC is stagnant.

### Changed
- `walk_forward_validation`: passes `sample_weight` to XGBoost in all CV folds.
- `run_pipeline.py`: propagates `xgb_optuna_best` and `drawdown_forensics` to iteration metrics.

## 2026-03-14 (session 2)

### Added
- **Dynamic position sizing** (`backtest.py`): capital allocation scales between 50%-100% of base slot according to model confidence above threshold. Trades with prob=0.95 receive 2× more capital than trades with prob=threshold+ε. Reduces drawdown without reducing number of trades.
- **Sample weights by PnL** (`train.py`): XGBoost penalised proportionally to each trade's PnL magnitude. Trades with large losses receive more weight — aligns loss function with drawdown protection.
- **Feature importance in LLM prompt** (`train.py`, `agent.py`): top 8 features and weak features (<0.02) included in agent context after each training.
- **Optuna parameter importance in prompt** (`backtest.py`, `agent.py`): relative importance of SL_RANGE/TP_RANGE/THRESHOLD_RANGE included in context after each backtest.
- **Score breakdown by component** (`agent.py`): LLM sees Sharpe/Return/DD contribution vs maximum possible.
- **Bayesian Optimisation (Optuna TPE)** (`backtest.py`): replaces static SL/TP grid; 120 trials per backtest with inter-year degradation penalty (15%).
- **Minimum TP/SL ratio 1.5×** in Optuna objective: prevents configs with almost zero R:R.
- `show_best.py`: script to display best setups with rich table and full detail.
- `program.md` and `README.md` updated to reflect Optuna architecture.

### Changed
- `generate_labels.py`: uses representative combo (midpoint of ranges) instead of SL/TP grid.
- `agent.py`: system prompt rewritten — LLM as bounds strategist, not exact-value guesser.
- `research_params.py`: migrated from `SL_GRID`/`TP_GRID` to `SL_RANGE`/`TP_RANGE`/`THRESHOLD_RANGE`/`N_TRIALS`.

## 2026-03-14

### Added
- Initial project structure `algo_autoresearch`
- Pipeline adapted from `btc_only_repro` with `research_params.py` support
- LLM agent (Qwen2.5-7B via llama.cpp) to propose parameter modifications
- Experiment logging system with tags and human notes
- CLI: `run`, `review`, `tag`, `analysis`, `setup`
- Relative features catalog (`features_catalog.py`)
- Composite score: tanh(S/2)*0.50 + tanh(R/100)*0.30 - (DD/100)*0.20
- Label cache keyed by input parameter hash
- AST validation of relative indicators in agent
- llama.cpp setup script with CUDA sm_89+sm_120 (RTX 5060)
