"""
deployment/full_report.py

Full report de todos os modelos aceites com backtest composto full capital (max_pos=1).
"""
import sys
import json
import glob
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.backtest import load_model, load_and_prepare, simulate_numba

ATR_KILL = 3.0
FEE      = 0.002
SLIP     = 0.001
MAX_POS  = 1
INITIAL  = 500.0
YEARS    = [2022, 2023, 2024, 2025, 2026]


def run_model(meta_path: Path) -> dict | None:
    with open(meta_path) as f:
        meta = json.load(f)

    model_dir = meta_path.parent / 'model'
    try:
        model, feature_names = load_model(model_dir)
    except Exception as e:
        return None

    sl      = meta['metricas']['sl_pct']
    tp      = meta['metricas']['tp_pct']
    thr     = meta['metricas']['threshold']
    adx_min = meta['params'].get('ENTRY_ADX_THRESHOLD', 20)
    it      = meta['iteracao']

    capital      = INITIAL
    year_results = {}

    for year in YEARS:
        data = load_and_prepare(year, model, feature_names, 'bnb')
        if data is None:
            year_results[year] = None
            continue
        probs_safe = np.where(data['atr_regime'] > ATR_KILL, 0.0, data['probs'])
        ret, sharpe, dd, trades, wr, _ = simulate_numba(
            data['high'], data['low'], data['close'],
            probs_safe, data['adx'],
            sl, tp, thr, adx_min,
            initial=capital, slippage=SLIP, max_pos=MAX_POS, fee_pct=FEE)
        capital = capital * (1 + ret / 100)
        year_results[year] = {'ret': round(ret, 1), 'sharpe': round(sharpe, 2),
                               'dd': round(dd, 1), 'trades': int(trades), 'wr': round(wr, 1)}

    return {
        'iter':          it,
        'auc':           round(meta['metricas']['cv_auc_mean'], 4),
        'val':           round(meta['metricas']['sharpe_validation'], 2),
        'holdout':       round(meta['metricas']['sharpe_holdout'], 2),
        'sl':            round(sl, 2),
        'tp':            round(tp, 2),
        'thr':           round(thr, 3),
        'capital_final': round(capital, 2),
        'ret_total':     round((capital / INITIAL - 1) * 100, 1),
        'years':         year_results,
    }


def main():
    # Carregar todos os aceites de S11
    accepted = []
    for f in sorted(glob.glob(str(BASE_DIR / 'experiments_s11/iter_*.json'))):
        d = json.load(open(f))
        if d.get('status') == 'aceite':
            it = d['iteracao']
            meta_path = BASE_DIR / f'best_models/season_11/iter_{it:04d}/meta.json'
            if meta_path.exists():
                accepted.append((it, meta_path))

    print(f'Aceites com modelo guardado: {len(accepted)}')
    print(f'Capital inicial: €{INITIAL:.0f}  Max pos: {MAX_POS} (full capital)  Composto: sim')
    print()

    results = []
    for i, (it, meta_path) in enumerate(accepted):
        r = run_model(meta_path)
        if r:
            results.append(r)
            print(f'[{i+1:>3}/{len(accepted)}] iter={it:>4}  '
                  f'€{r["capital_final"]:>7.0f}  {r["ret_total"]:>+6.1f}%  '
                  f'(val={r["val"]:.2f} hld={r["holdout"]:.2f})', flush=True)

    results.sort(key=lambda x: -x['capital_final'])

    print()
    print(f'{"="*130}')
    print(f'  RANKING FINAL — Full Capital Composto — {len(results)} modelos')
    print(f'{"="*130}')
    print(f'  {"#":>3} {"iter":>6} {"AUC":>7} {"val":>5} {"hld":>5} '
          f'{"SL%":>5} {"TP%":>5} {"Thr":>6}  '
          f'{"2022":>7} {"2023":>7} {"2024":>7} {"2025":>7} {"2026":>7}  '
          f'{"€FINAL":>8} {"TOT%":>7}')
    print(f'  {"-"*125}')

    def yr(r, year):
        y = r['years'].get(year)
        return f'{y["ret"]:+.1f}%' if y else '  N/A '

    for rank, r in enumerate(results, 1):
        print(f'  {rank:>3} {r["iter"]:>6} {r["auc"]:>7.4f} {r["val"]:>5.2f} {r["holdout"]:>5.2f} '
              f'{r["sl"]:>5.1f} {r["tp"]:>5.1f} {r["thr"]:>6.3f}  '
              f'{yr(r,2022):>7} {yr(r,2023):>7} {yr(r,2024):>7} {yr(r,2025):>7} {yr(r,2026):>7}  '
              f'€{r["capital_final"]:>7.0f} {r["ret_total"]:>+7.1f}%')

    out_path = BASE_DIR / 'deployment/results/full_report_s11_full_capital.json'
    with open(out_path, 'w') as f:
        json.dump({'n_models': len(results), 'initial': INITIAL, 'max_pos': MAX_POS,
                   'results': results}, f, indent=2)
    print()
    print(f'Guardado: {out_path}')
    print(f'Melhor: iter={results[0]["iter"]}  €{results[0]["capital_final"]:.0f}  {results[0]["ret_total"]:+.1f}%')
    print(f'Pior:   iter={results[-1]["iter"]}  €{results[-1]["capital_final"]:.0f}  {results[-1]["ret_total"]:+.1f}%')


if __name__ == '__main__':
    main()
