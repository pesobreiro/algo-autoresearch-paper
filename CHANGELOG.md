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

## 2026-03-25 — Season 12 — BTC/USDC — Conclusão e Deploy

### Resultados S12
- **107 aceites** em ~9900 iterações (1.1% taxa de aceitação)
- **Modelo seleccionado para deploy**: iter 5502 — melhor equilíbrio val+holdout+equity OOS
- **Splits**: train 2017–2020, val 2021–2023, holdout 2024–2025
- **Triple gate**: AUC≥0.51 AND Sharpe(val)≥1.0 AND Sharpe(holdout)≥0.3

### Modelo Deploy — iter 5502
```
SL=9.70%  TP=9.12%  Threshold=0.890  ADX_min=25
TFs: 15m + 4h + 1d  |  REG_ALPHA=7.0  REG_LAMBDA=7.0
Features: ema_diff_15m, macd_pct_4h, bb_width_pct_15m, dist_sma200_pct, btc_trend,
          volume_norm, adx_15m, rsi_4h, macd_hist_pct_15m, stoch_rsi_k_1d,
          atr_regime_4h, macd_signal_pct_1d, macd_pct_1d, bb_width_pct_1d,
          atr_regime_15m, dist_sma200_pct_4h, macd_pct_15m, stoch_rsi_d_1d, ema_diff_4h
```

| Ano | Sharpe | Ret% | DD% | Trades | WR% | Equity |
|-----|--------|------|-----|--------|-----|--------|
| 2021 | 0.69 | +3.7% | -4.6% | 13 | 69% | €519 |
| 2022 | 1.79 | +13.0% | -1.1% | 21 | 81% | €586 |
| 2023 | 2.16 | +14.7% | -1.5% | 20 | 90% | €672 |
| 2024 | 0.94 | +10.7% | -9.3% | 49 | 63% | €744 |
| **2025 OOS** | **1.27** | **+9.6%** | **-4.5%** | **32** | **66%** | **€816** |
| **Total** | — | **+63.2%** | **-9.3%** | **135** | **73%** | **€816** |

### Changed
- **`config.yaml`**: `ticker: bnb→btc`, `season: 11→12`, `max_positions: 5→2`, `accept_auc_min: 0.55→0.51`, `accept_sharpe_holdout_min: 0.5→0.3`, `train_start: 2019→2017`, `train_end: 2021→2020`, `validation_years: [2022,2023,2024]→[2021,2022,2023]`, `holdout_years: [2025]→[2024,2025]`, `max_tokens: 2048→4096`
- **`llm/start_server.sh`**: `ctx-size 8192→16384`, adicionado `--flash-attn on`, removido `--cache-type-k/v q4_0` (incompatível com Qwen2.5-7B-Instruct — causava outputs corrompidos)
- **`autoresearch/agent.py`**: histórico enviado ao LLM 5→30 iters
- **`autoresearch/runner.py`**: `listar_historico(10→30)`, `top_n_scores(5→10)` em dois locais
- **`main.py`**: bug fix — `_gerar_research_params` não incluía `OBJECTIVE_MODE` no template focus
- **`deployment/backtest_deploy.py`**: bug fix — ticker hardcoded `'bnb'` → argumento `--ticker`

### Lições S12 — BTC vs BNB
- BTC 2025 foi regime extremo (subida até 125k, descida a 60k) — não visto no treino 2017–2020
- KV cache quantizado (`--cache-type-k q4_0`) incompatível com Qwen2.5-7B-Instruct → outputs corrompidos (AL_AL_AL...)
- Alargamento splits para 2017–2020 (train) + 2021–2023 (val) melhorou estabilidade holdout
- Features 1d dominaram importance no S12: `bb_width_pct_1d`, `macd_signal_pct_1d`, `stoch_rsi_k_1d`

## 2026-03-22 — Season 11 Setup (BNB/USDT) — Triple Gate + Holdout Acceptance

### Changed
- **`config.yaml`**: `season: 10→11`, adicionado `accept_sharpe_holdout_min: 0.5`
- **`autoresearch/runner.py`**: aceitação agora usa **triple gate**: `AUC≥0.55 AND Sharpe(val)≥1.0 AND Sharpe(holdout)≥0.5`; removido ratchet de baseline — qualquer iteração que passe as 3 gates é aceite; mensagem de rejeição mostra gate holdout
- **`pipeline/research_params.py`**: reset para S11 a partir de S10 iter=417 (melhor holdout: AUC=0.563, Sharpe(val)=1.50, Sharpe(holdout)=2.82, equity 2025→€600)

### Added — Lição de S10: Val Overfitting com Ratchet
- S10 (1148 iters, 3 aceites): baseline crescia com cada aceite (ratchet) → LLM forçado a encontrar configs cada vez mais otimizadas para 2022-2024
- Progressão dos 3 aceites: iter=326 (holdout=+2.41) → iter=582 (holdout=+0.17) → iter=727 (holdout=-1.02) — degradação clara
- **Fix S11 (Opção A)**: holdout entra como gate (`≥0.5`), sem ratchet; 2025 deixa de ser true holdout mas elimina val overfitting progressivo
- Ponto de partida: iter=417 S10 — features mais simples (11 vs 16), melhor holdout de toda S10

## 2026-03-21 — Season 10 Setup (BNB/USDT) — Bear Market Validation

### Changed
- **`config.yaml`**: `season: 9→10`, `train_start: 2020→2019`, `train_end: 2022→2021`, `validation_years: [2022,2023,2024]` (era [2023,2024]), `scoring.atr_regime_kill: 3.0` (novo)
- **`pipeline/backtest.py`**: `load_and_prepare()` adiciona `atr_regime` ao dict de dados; `correr_backtest()` lê `atr_kill` do config; kill-switch aplicado em todas as chamadas `simulate_numba` (objective, val loop, holdout loop, forensics): `probs_safe = np.where(atr_regime > atr_kill, 0.0, probs)`
- **`pipeline/run_pipeline.py`**: defaults `hash_entry_params(train_start=2019, train_end=2021)`
- **`pipeline/research_params.py`**: reset S10 a partir de S9 iter=311 (AUC=0.561, Sharpe=2.83)
- **`program.md`**: reescrito para S10 — data split com 2022 bear, kill-switch ATR, efeito `min(Sharpe_val)` com 3 anos

### Added — Lição de S9: Regime Blindness
- S9 (311 iters, 5 aceites): validação [2023,2024] eram ambos bull markets → Optuna nunca viu crash
- iter=311 (melhor S9): Sharpe(val)=2.83 mas Sharpe(holdout)=-0.28 — overfit ao bull run
- iter=178 (mais robusto S9): Sharpe(val)=2.41, Sharpe(holdout)=+1.00 — sobreviveu por não operar em 2025
- **Fix S10**: `validation_years=[2022,2023,2024]` — 2022 (FTX/Luna, BNB -50%) domina `min(Sharpe_val)`, forçando Optuna a encontrar configs que sobrevivem a bear markets
- **Kill-switch ATR**: eventos com `atr_regime>3.0` (volatilidade 3× normal) são bloqueados mecanicamente — não delegados ao ML

## 2026-03-21 — Season 9 Setup (BNB/USDT) — Clean OOS Design

### Changed
- **`config.yaml`**: `ticker: btc → bnb`, `season: 8 → 9`, `train_start: 2020`, `train_end: 2022`, `validation_years: [2023, 2024]`, `holdout_years: [2025]`, `baseline_override: 0.0`, `accept_auc_min: 0.55`, `accept_sharpe_min: 1.0`, `min_sharpe_gate: 0.3`
- **`pipeline/backtest.py`**: substituído `years` único por `validation_years` + `holdout_years`; Optuna usa apenas `data_por_ano_val`; modo score maximiza `min(sharpe_val)` (pior sub-janela) em vez de mean; métricas finais calculadas para val e holdout separadamente; `fee_pct=0.002` (0.2% round-trip) em todos os `simulate_numba`; retorno dict com `sharpe_validation` e `sharpe_holdout`
- **`pipeline/run_pipeline.py`**: `hash_entry_params()` aceita `train_start`+`train_end` — cache de labels invalidado entre seasons com train windows diferentes
- **`autoresearch/tracker.py`**: `top_n_scores()` e `melhor_score()` ordenam por `sharpe_validation` (com fallback para `score_composto`)
- **`autoresearch/runner.py`**: aceitação S9 usa dual-gate (`cv_auc_mean ≥ accept_auc_min AND sharpe_validation ≥ accept_sharpe_min`); holdout é passivo — nunca usado para filtrar; `score_baseline` rastreia `sharpe_validation`; console mostra AUC + Sharpe(val) + Sharpe(holdout/passivo); `limpar_cache()` passa `train_start/train_end` ao hash
- **`pipeline/research_params.py`**: reset para S9/BNB a partir de S5 iter 349 (Sharpe=2.37). `FEATURES` inclui `btc_trend`+`dist_sma200_pct` obrigatórios; `N_TRIALS=200`; `OBJECTIVE_MODE="score"`
- **`program.md`**: reescrito para S9 — activo BNB, aceitação por gates AUC+Sharpe(val), data split imutável, kill-switch ATR, holdout passivo

### Fixed — 5 melhorias metodológicas
1. **Holdout integrity (crítico)**: Optuna nunca toca 2025. Seasons S2–S8 tinham contaminação OOS (Optuna otimizava em 2025+2026 que eram "OOS")
2. **Optuna anti-overfitting**: N_TRIALS=200 (2 anos de val não suportam mais); objetivo = `min(sharpe_val)` (pior sub-janela) em vez de mean
3. **Round-trip fees**: `fee_pct=0.002` (0.2%) em todos os `simulate_numba` — anteriormente só 0.1% na saída (entrada em falta)
4. **Train window moderna**: 2020–2022 (excluído 2017–2019 com microestrutura/liquidez diferente de mercados modernos)
5. **Activo BNB**: menos ruído institucional que BTC (BTC tem ETFs globais + macro funds que dificultam ML mean-reversion; S8 BTC: 134 iters, 0 aceites)

## 2026-03-21 — Season 8 Setup (BTC/USDT)

### Changed
- **`pipeline/research_params.py`**: reset para S8/BTC — `N_TRIALS=3000` (reduzido de 10000, mitigação OOM), `SL_RANGE=(2.0,12.0)`, `TP_RANGE=(5.0,25.0)`, `THRESHOLD_RANGE=(0.85,0.95)` (abertos para BTC explorar), `TIMEFRAMES=["15m","4h","1d"]`, `OBJECTIVE_MODE="profit"`
- **`config.yaml`**: `ticker: eth → btc`, `season: 7 → 8`, `baseline_override: 30.154715`
- **`program.md`**: reescrito para S8/BTC com contexto cross-asset (referência BNB/ETH óptimos)

### Fixed (mitigações OOM diagnosticadas em S7)
- **`autoresearch/runner.py`**: adicionado `import gc` e `gc.collect()` após `tracker.guardar_experiencia()` em cada iteração — força libertação de ciclos de referência Optuna/sklearn antes da próxima iteração
- **`pipeline/backtest.py`**: adicionado `_PARQUET_CACHE: dict` module-level e `_load_parquet_cached()` — cada ficheiro parquet é lido do disco **uma única vez** por processo e reutilizado nas iterações seguintes, eliminando a principal causa de crescimento de RSS

## 2026-03-21 — Diagnóstico OOM Season 7 (sem fix aplicado)

### Diagnóstico: Crashes por OOM durante ~1095 iterações (S7)

**Evidência dos kernel logs (`/var/log/syslog`):**

| Data       | PID     | RSS anon      | VM total |
|------------|---------|---------------|----------|
| 2026-03-16 | 3832181 | **7.7 GB**    | 45 GB    |
| 2026-03-17 | 2768659 | **25.5 GB**   | 64.5 GB  |
| 2026-03-20 | 2813875 | **26.7 GB**   | 67.2 GB  |
| 2026-03-21 | —       | killed (OOM)  | —        |

O processo Python foi morto pelo kernel OOM killer **pelo menos 4 vezes** durante a S7.
O RSS cresceu de 7.7 GB → 25.5 GB entre restarts, confirmando **memory leak gradual** (não alocação pontual).
O log `s7.txt` confirma um restart: loop retomou em iter 966 após crash anterior (iter máx aceite: 580).

**Causas identificadas (por ordem de impacto estimado):**

1. **`backtest.py` — `load_and_prepare()` sem cache de dados** _(causa primária)_
   - Em cada iteração, a função lê o ficheiro parquet completo (2017–2024, ~258k rows × múltiplos TFs)
     em memória, faz merge dos timeframes, e filtra por ano OOS.
   - Com 2 anos OOS (2025 + 2026) e 10,000 trials Optuna por iteração, são gerados e passados
     ao Numba arrays de ~258k linhas por chamada — sem qualquer `gc.collect()` explícito após cada iteração.
   - Python/glibc não devolve arenas de memória ao OS após liberar objetos grandes: o RSS cresce
     até ao high-water mark de todas as alocações e nunca desce.

2. **`backtest.py` — Optuna `TPEsampler` com 10,000 trials** _(causa secundária)_
   - Cada `create_study()` é local e teoricamente libertado no fim de `correr_backtest()`.
   - Porém, o `get_param_importances(study)` (linha ~528) usa sklearn internamente
     (Random Forest), que cria objectos com ciclos de referência — atrasando o GC.
   - Com 1000+ iterações × 10,000 trials, estas estruturas acumulam até o GC as processar.

3. **`generate_labels.py` + `backtest.py` — Numba `@njit` sem warmup antecipado** _(causa terciária)_
   - As funções `simulate_numba`, `simulate_numba_equity`, `simulate_trades_numba` são `@njit`.
   - O compilador LLVM mantém em memória todos os artefactos de compilação (IR + código máquina).
   - Se ao longo das iterações forem passados arrays com shapes distintos, Numba compila
     novas especializações que ficam em memória pelo resto do processo.

4. **`runner.py` — sem `gc.collect()` entre iterações** _(amplificador)_
   - O loop principal nunca chama `gc.collect()`. Com os ciclos de referência criados
     pelo Optuna/sklearn, os objectos não são libertados de forma determinística.

**Mitigações para a próxima experiment (S8+):**

- [ ] Cache module-level dos DataFrames parquet em `backtest.py` (ler 1x, reutilizar entre iterações)
- [ ] Chamar `gc.collect()` no fim de cada iteração em `runner.py`
- [ ] Reduzir `N_TRIALS` de 10,000 para 3,000–5,000 (trade-off exploração vs. memória)
- [ ] Considerar subprocess isolado para o pipeline (cada iteração em processo filho que morre limpo)
- [ ] Aumentar swap ou usar `ulimit -v` para limitar VM e detetar crashes mais cedo

## 2026-03-15 (Season 5 — objetivo profit)

### Added
- **`OBJECTIVE_MODE` no Optuna** (`pipeline/backtest.py`): `correr_backtest()` aceita novo parâmetro `objective_mode='score'|'profit'`. No modo `profit`, o Optuna maximiza diretamente `retorno_total_oos_pct` (retorno % médio OOS) em vez do score composto. Score composto continua a ser calculado e reportado em ambos os modos.
- **Gate Sharpe mínimo no modo profit** (`pipeline/backtest.py`): estratégias com Sharpe < 0.5 são rejeitadas pelo Optuna (`-999.0`). Evita selecionar estratégias de montanha-russa que maximizam retorno às custas de uma curva de equity errática. Configurável via `config.scoring.min_sharpe_gate`.
- **Gate DD no modo profit** (`pipeline/backtest.py`): `max_dd < max_dd_gate` (default -30%) retorna `-999.0`. Configurável via `config.scoring.max_dd_gate`.
- **`min_trades_profit`** (`pipeline/backtest.py`): mínimo de trades no modo profit lido de `config.scoring.min_trades_profit` (default herda `min_trades`). Evita overfitting a 3 trades "de sorte" com retorno alto.

### Changed
- **`pipeline/run_pipeline.py`**: `carregar_params()` lê `OBJECTIVE_MODE` (default `'score'`); `executar_pipeline()` passa `objective_mode` a `correr_backtest()`.
- **`pipeline/research_params.py`**: adicionado `OBJECTIVE_MODE = "profit"` para arranque da Season 5.
- **`autoresearch/runner.py`**: accept/reject agora mode-aware — usa `retorno_total_oos_pct` quando `OBJECTIVE_MODE='profit'`, `score_composto` caso contrário. Mensagens de aceitação/rejeição mostram `%` em vez de score no modo profit. Baseline ao retomar pesquisa também lê a métrica correta conforme o modo da última iteração aceite.
- **`program.md`**: reescrito para S5 — objetivo primário é `retorno_total_oos_pct`, Sharpe/DD são métricas de contexto. LLM instruído a priorizar retorno sem sacrificar Sharpe mínimo (>= 0.5).

## 2026-03-15 (sessão 5 — cont. 2)

### Changed
- **Exit fee adicionada ao simulador Numba** (`pipeline/backtest.py`): `simulate_numba` e `simulate_numba_equity` recebem novo parâmetro `fee_pct=0.001` (0.1%). Nos 3 pontos de saída (SL hit, TP hit, end-of-period), o capital agora é calculado como `sz * (exit_price / entry_price) * (1 - fee_pct)` em vez de `sz + sz * (exit_price/entry - 1)`. Isto modela a taxa taker Binance+BNB (~0.075%) + exit slippage (~0.025%). Sem esta correção, 148 trades × 0.1% = ~15% drag estava a ser ignorado, sobrestimando retornos.

## 2026-03-15 (sessão 5 — cont.)

### Changed
- **`THRESHOLD_RANGE` mínimo elevado para 0.80** (`pipeline/research_params.py`, `autoresearch/runner.py`, `program.md`): análise empírica de 2400+ iterações S4 mostra que threshold < 0.80 é zona morta (99-100% dos resultados negativos, avg 1000+ trades). Só threshold ≥ 0.85 produz resultados positivos (avg 332 trades). Auto-corrector agora enforce `_THR_LO_MIN=0.80` e `_THR_HI_MIN=0.90`.
- **`program.md` atualizado** com evidência empírica explícita da zona morta do threshold, para o LLM não voltar a propor valores abaixo de 0.80.

## 2026-03-15 (sessão 5)

### Changed
- **`SL_RANGE` e `TP_RANGE` expandidos até 10.0** (`pipeline/research_params.py`, `autoresearch/runner.py`): os melhores resultados S4 (iters 24, 25, 43) tinham SL=7.86–7.97% a atingir o teto anterior de 8.0. Expandido para `SL_RANGE=(1.0, 10.0)` e `TP_RANGE=(0.5, 10.0)` para dar espaço ao Optuna explorar SL>8%. `THRESHOLD_RANGE` alargado para `(0.6, 0.95)`.
- **Auto-corrector `_SL_HI_MIN` e `_TP_HI_MIN` elevados a 10.0** (`autoresearch/runner.py`): a barreira mínima do teto SL/TP no auto-corrector foi aumentada de 7.0/5.0 para 10.0/10.0 para manter coerência com os novos bounds alvo.

## 2026-03-14 (sessão 4)

### Added
- **Suporte a temporadas** (`main.py`, `runner.py`, `config.yaml`): flag `--season N` em todos os comandos CLI. Cada temporada usa um directório de experiências separado (`experiments/` para S1, `experiments_s2/` para S2, etc.). `config.yaml` pode definir `agent.season` e `agent.baseline_override` para impor um score mínimo ao iniciar uma temporada nova.
- **Season 2 iniciada**: `research_params.py` reset ao ponto de partida S1 (iter 32, score=0.5607) com XGBoost Optuna ativo (`N_TRIALS_XGB=30`, com DEPTH/LR/ESTIMATORS/ALPHA/LAMBDA ranges) e features macro `btc_trend` + `dist_sma200_pct` introduzidas.
- **`program.md` reescrito para S2**: descreve a arquitectura actual sem alertas transitórios; prioridades claras (macro features, XGBoost Optuna, bounds shifting).
- **Comando `new-season`** (`main.py`): ritual de transição automático entre temporadas. Lê o melhor resultado da temporada actual, actualiza `config.yaml` (`season` e `baseline_override`) e regera `research_params.py` a partir do `params_snapshot` do melhor resultado. Suporta `--dry-run`. Se não houver resultados aceites, incrementa a temporada sem alterar params.
- **Temperature curriculum** (`runner.py`, `agent.py`): temperatura do LLM ajustada automaticamente com base na tendência do score. Quando o score melhora, temperatura desce por `t_decay=0.92` (exploit). Quando `stagnation_threshold=5` iterações passam sem melhoria, temperatura sobe por `t_grow=1.08` (explore). Limites `[t_min=0.3, t_max=1.2]` configuráveis em `config.yaml`. Temperatura actual mostrada no header de cada iteração.
- **Rejection feedback loop** (`runner.py`, `agent.py`): últimas 5 rejeições de validação são acumuladas e incluídas no próximo prompt LLM como `⚠️ REJEIÇÕES RECENTES — NÃO REPETIR`. Impede o LLM de repetir features inválidas (`macd_diff`, `macd_signal`) em iterações consecutivas. Lista é limpa quando uma proposta passa a validação.
- **Features macro de regime de mercado** (`features_catalog.py`, `generate_labels.py`, `agent.py`):
  - `dist_sma200_pct` (1d-only): distância à SMA200 normalizada pelo preço (%), calculada da 1d close da ticker
  - `btc_trend` (1d-only): BTC acima/abaixo da EMA50 (0/1), cross-asset, carregado de `btc_01d_usdt_binance.parquet`
  - `atr_regime` (todos os TFs): ATR atual / rolling_mean(ATR, 50) — ratio de volatilidade relativa
  - Adicionado `FEATURES_1D_ONLY` ao catálogo (skip silencioso para TFs != 1d)
  - `_adicionar_macro_features()` em `generate_labels.py` computa e merge as features no dataframe merged
  - System prompt do agente atualizado com as 3 novas features e notas de uso

## 2026-03-14 (sessão 3)

### Added
- **Forensic drawdown analysis** (`backtest.py`): após cada backtest, deteta o período do max drawdown (peak→trough), calcula duração em dias, ADX médio/mínimo e classifica o regime de mercado (Lateral/Tendência Fraca/Forte). Incluído no prompt do LLM com sugestão automática de correção.
- **`simulate_numba_equity`** (`backtest.py`): variante do simulador Numba que retorna equity curve completa + índices do drawdown period.
- **XGBoost Optuna** (`train.py`): estudo Optuna opcional para hiperparâmetros XGBoost (DEPTH_RANGE, LR_RANGE, ESTIMATORS_RANGE, ALPHA_RANGE, LAMBDA_RANGE). Ativado com `N_TRIALS_XGB >= 20`. Quando ativo, a Optuna encontra o melhor config XGBoost via 3-fold CV AUC antes do treino final.
- `research_params.py`: novos campos opcionais XGBoost Optuna documentados (comentados por defeito, N_TRIALS_XGB=0).
- System prompt do agente: instruções para ativar XGBoost Optuna quando AUC estagnado.

### Changed
- `walk_forward_validation`: passa `sample_weight` ao XGBoost em todos os folds CV.
- `run_pipeline.py`: propaga `xgb_optuna_best` e `drawdown_forensics` para as métricas da iteração.

## 2026-03-14 (sessão 2)

### Added
- **Dynamic position sizing** (`backtest.py`): alocação de capital escala entre 50%-100% da slot base conforme confiança do modelo acima do threshold. Trades com prob=0.95 recebem 2× mais capital que trades com prob=threshold+ε. Reduz drawdown sem reduzir número de trades.
- **Sample weights por PnL** (`train.py`): XGBoost penalizado proporcionalmente à magnitude do PnL de cada trade. Trades com perdas grandes recebem mais peso — alinha função de perda com proteção de drawdown.
- **Feature importance no prompt LLM** (`train.py`, `agent.py`): top 8 features e features fracas (<0.02) incluídas no contexto do agente após cada treino.
- **Optuna parameter importance no prompt** (`backtest.py`, `agent.py`): importância relativa de SL_RANGE/TP_RANGE/THRESHOLD_RANGE incluída no contexto após cada backtest.
- **Score breakdown por componente** (`agent.py`): LLM vê contribuição Sharpe/Return/DD vs máximo possível.
- **Otimização Bayesiana (Optuna TPE)** (`backtest.py`): substitui grid SL/TP estático; 120 trials por backtest com penalização de degradação inter-anual (15%).
- **Ratio mínimo TP/SL 1.5×** no Optuna objective: previne configs com R:R quase nulo.
- `show_best.py`: script de apresentação dos melhores setups com tabela rich e detalhe completo.
- `program.md` e `README.md` atualizados para refletir arquitetura Optuna.

### Changed
- `generate_labels.py`: usa combo representativo (ponto médio dos ranges) em vez de grid SL/TP.
- `agent.py`: system prompt reescrito — LLM como estrategista de bounds, não adivinhador de valores exactos.
- `research_params.py`: migrado de `SL_GRID`/`TP_GRID` para `SL_RANGE`/`TP_RANGE`/`THRESHOLD_RANGE`/`N_TRIALS`.

## 2026-03-14

### Added
- Estrutura inicial do projeto `algo_autoresearch`
- Pipeline adaptado de `btc_only_repro` com suporte a `research_params.py`
- Agente LLM (Qwen2.5-7B via llama.cpp) para propor modificações de parâmetros
- Sistema de logging de experiências com tags e notas humanas
- CLI: `run`, `review`, `tag`, `analysis`, `setup`
- Catálogo de features relativas (`features_catalog.py`)
- Score composto: tanh(S/2)*0.50 + tanh(R/100)*0.30 - (DD/100)*0.20
- Cache de labels keyed por hash dos parâmetros de entrada
- Validação AST de indicadores relativos no agente
- Script de setup llama.cpp com CUDA sm_89+sm_120 (RTX 5060)
