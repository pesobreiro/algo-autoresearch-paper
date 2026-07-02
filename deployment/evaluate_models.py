"""
deployment/evaluate_models.py

Avalia os melhores modelos aceites para decisão de deploy.

Para cada modelo seleccionado:
  - Re-corre simulação com SL/TP/Threshold FIXOS (sem Optuna)
  - Anos avaliados: 2022, 2023, 2024 (val), 2025 (holdout), 2026 (true OOS)
  - 2026 = dados que nenhum modelo viu durante treino/optimização

Uso:
    python deployment/evaluate_models.py
    python deployment/evaluate_models.py --season 11 --top 30
    python deployment/evaluate_models.py --season 11 --top 50 --min-holdout 1.5
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.backtest import load_model, load_and_prepare, simulate_numba


EVAL_YEARS  = [2022, 2023, 2024, 2025, 2026]
VAL_YEARS   = [2022, 2023, 2024]
HOLDOUT_YR  = 2025
TRUE_OOS_YR = 2026

INITIAL_CAPITAL = 500.0
FEE_PCT         = 0.002   # 0.2% round-trip
SLIPPAGE        = 0.001
ATR_KILL        = 3.0
MAX_POS         = 5


def _run_year(model, feature_names, year, sl, tp, thr, adx_min, ticker='bnb'):
    data = load_and_prepare(year, model, feature_names, ticker)
    if data is None:
        return None

    probs_safe = np.where(data['atr_regime'] > ATR_KILL, 0.0, data['probs'])

    ret, sharpe, dd, trades, wr, sortino = simulate_numba(
        data['high'], data['low'], data['close'],
        probs_safe, data['adx'],
        sl, tp, thr, adx_min,
        initial=INITIAL_CAPITAL,
        slippage=SLIPPAGE,
        max_pos=MAX_POS,
        fee_pct=FEE_PCT,
    )
    return {
        'retorno_pct':  round(ret, 2),
        'sharpe':       round(sharpe, 3),
        'sortino':      round(sortino, 3),
        'max_dd_pct':   round(dd, 2),
        'n_trades':     int(trades),
        'win_rate_pct': round(wr, 1),
        'equity_final': round(INITIAL_CAPITAL * (1 + ret / 100), 2),
    }


def evaluate_model(iter_dir: Path, meta: dict) -> dict | None:
    model_dir = iter_dir / 'model'
    try:
        model, feature_names = load_model(model_dir)
    except Exception as e:
        print(f"  [ERRO] {iter_dir.name}: {e}")
        return None

    m       = meta['metricas']
    params  = meta['params']
    sl      = m['sl_pct']
    tp      = m['tp_pct']
    thr     = m['threshold']
    adx_min = params.get('ENTRY_ADX_THRESHOLD', 20)

    years_results = {}
    for year in EVAL_YEARS:
        r = _run_year(model, feature_names, year, sl, tp, thr, adx_min)
        years_results[year] = r

    # Métricas agregadas
    val_sharpes   = [years_results[y]['sharpe']      for y in VAL_YEARS   if years_results.get(y)]
    holdout       = years_results.get(HOLDOUT_YR)
    true_oos      = years_results.get(TRUE_OOS_YR)

    sharpe_val_mean = round(np.mean(val_sharpes), 3) if val_sharpes else 0.0
    sharpe_holdout  = holdout['sharpe']   if holdout  else None
    sharpe_2026     = true_oos['sharpe']  if true_oos else None
    ret_2026        = true_oos['retorno_pct'] if true_oos else None

    # Score de deploy: média de holdout + 2026 (ambos nunca otimizados)
    oos_scores = [s for s in [sharpe_holdout, sharpe_2026] if s is not None]
    deploy_score = round(np.mean(oos_scores), 3) if oos_scores else 0.0

    return {
        'iter':          meta['iteracao'],
        'season':        meta.get('season', '?'),
        'auc':           round(m['cv_auc_mean'], 4),
        'sharpe_val':    round(m['sharpe_validation'], 3),  # original (do Optuna)
        'sharpe_val_rerun':   sharpe_val_mean,              # re-calculado aqui
        'sharpe_holdout':     sharpe_holdout,               # 2025
        'sharpe_2026':        sharpe_2026,                  # true OOS
        'ret_2026_pct':       ret_2026,
        'deploy_score':       deploy_score,                 # média (holdout + 2026)
        'sl_pct':        round(sl, 2),
        'tp_pct':        round(tp, 2),
        'threshold':     round(thr, 3),
        'n_trades_val':  sum(years_results[y]['n_trades'] for y in VAL_YEARS if years_results.get(y)),
        'max_dd_val':    round(min(years_results[y]['max_dd_pct'] for y in VAL_YEARS if years_results.get(y)), 2),
        'by_year':       {str(y): years_results[y] for y in EVAL_YEARS},
    }


def load_candidates(season_dir: Path, top: int, min_holdout: float) -> list[dict]:
    """Carrega e filtra modelos por Sharpe(holdout) do meta.json."""
    candidates = []
    for iter_dir in sorted(season_dir.iterdir()):
        meta_path = iter_dir / 'meta.json'
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        sh = meta['metricas'].get('sharpe_holdout', 0)
        if sh >= min_holdout:
            candidates.append((sh, iter_dir, meta))

    candidates.sort(key=lambda x: -x[0])
    return [(d, m) for _, d, m in candidates[:top]]


def print_table(results: list[dict]):
    print()
    print(f"{'iter':>6} {'AUC':>7} {'val':>6} {'hld':>6} {'2026':>6} {'deploy':>7} "
          f"{'ret26%':>7} {'DD_v%':>7} {'trades_v':>8} {'SL':>5} {'TP':>5} {'Thr':>6}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -(x['deploy_score'] or 0)):
        sh26 = f"{r['sharpe_2026']:.2f}" if r['sharpe_2026'] is not None else "N/A"
        ret26 = f"{r['ret_2026_pct']:+.1f}" if r['ret_2026_pct'] is not None else "N/A"
        print(f"{r['iter']:>6} {r['auc']:>7.4f} {r['sharpe_val']:>6.2f} "
              f"{(r['sharpe_holdout'] or 0):>6.2f} {sh26:>6} {r['deploy_score']:>7.3f} "
              f"{ret26:>7} {r['max_dd_val']:>7.1f} {r['n_trades_val']:>8} "
              f"{r['sl_pct']:>5.1f} {r['tp_pct']:>5.1f} {r['threshold']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description='Avalia modelos para deploy')
    parser.add_argument('--season',      type=int,   default=11)
    parser.add_argument('--top',         type=int,   default=30,  help='Top N por holdout')
    parser.add_argument('--min-holdout', type=float, default=1.0, help='Sharpe holdout mínimo')
    parser.add_argument('--out',         type=str,   default='deployment/results')
    args = parser.parse_args()

    season_dir = BASE_DIR / f'best_models/season_{args.season}'
    if not season_dir.exists():
        print(f"ERRO: {season_dir} não existe")
        sys.exit(1)

    out_dir = BASE_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Avaliação de Deploy — Season {args.season} ===")
    print(f"Top {args.top} por Sharpe(holdout) ≥ {args.min_holdout}")
    print(f"Anos: {EVAL_YEARS} | 2026 = True OOS")
    print()

    candidates = load_candidates(season_dir, args.top, args.min_holdout)
    print(f"Candidatos encontrados: {len(candidates)}")
    print()

    results = []
    for i, (iter_dir, meta) in enumerate(candidates):
        iter_num = meta['iteracao']
        sh_h = meta['metricas'].get('sharpe_holdout', 0)
        print(f"[{i+1:>2}/{len(candidates)}] iter={iter_num:>4}  "
              f"AUC={meta['metricas']['cv_auc_mean']:.4f}  "
              f"Sharpe(val)={meta['metricas']['sharpe_validation']:.2f}  "
              f"Sharpe(holdout)={sh_h:.2f}", end='  ', flush=True)
        r = evaluate_model(iter_dir, meta)
        if r:
            sh26 = r['sharpe_2026']
            print(f"→ 2026={sh26:.2f}  deploy_score={r['deploy_score']:.3f}")
            results.append(r)
        else:
            print("→ ERRO")

    print_table(results)

    # Salvar resultados
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = out_dir / f'evaluation_s{args.season}_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump({'season': args.season, 'timestamp': ts,
                   'results': sorted(results, key=lambda x: -(x['deploy_score'] or 0))}, f, indent=2)
    print(f"\nResultados guardados em: {out_path}")

    # Top 5
    top5 = sorted(results, key=lambda x: -(x['deploy_score'] or 0))[:5]
    print("\n=== TOP 5 CANDIDATOS DEPLOY ===")
    for r in top5:
        sh26 = r['sharpe_2026']
        eq_2026 = r['by_year'].get('2026', {})
        print(f"  iter={r['iter']:>4}  deploy_score={r['deploy_score']:.3f}  "
              f"Sharpe(holdout)={r['sharpe_holdout']:.2f}  Sharpe(2026)={sh26:.2f}  "
              f"ret2026={r['ret_2026_pct']:+.1f}%  "
              f"equity2026=€{eq_2026.get('equity_final', 0):.0f}")


if __name__ == '__main__':
    main()
