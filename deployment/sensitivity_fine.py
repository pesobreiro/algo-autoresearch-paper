"""
deployment/sensitivity_fine.py

Fine sensitivity analysis for iter 1077 — optimal zone:
  SL:        6.0 → 7.0  (step 0.5)
  TP:        6.0 → 7.0  (step 0.5)
  Threshold: ±1% around 0.857  (step 0.01)

Metrics per year (2022-2026) + deploy_score = average Sharpe(2025+2026).
"""
import sys
import json
import numpy as np
from pathlib import Path
from itertools import product

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.backtest import load_model, load_and_prepare, simulate_numba

ITER    = 1077
SEASON  = 11
ATR_KILL = 3.0
FEE      = 0.002
SLIP     = 0.001
MAX_POS  = 1
INITIAL  = 500.0
YEARS    = [2022, 2023, 2024, 2025, 2026]
OOS_YEARS = [2025, 2026]

SL_VALS  = [6.0, 6.5, 7.0]
TP_VALS  = [6.0, 6.5, 7.0]
THR_VALS = [round(v, 3) for v in np.arange(0.847, 0.878, 0.010)]  # 0.847 0.857 0.867

iter_dir = BASE_DIR / f'best_models/season_{SEASON}/iter_{ITER:04d}'
with open(iter_dir / 'meta.json') as f:
    meta = json.load(f)
model, feature_names = load_model(iter_dir / 'model')
adx_min = meta['params']['ENTRY_ADX_THRESHOLD']

# Pre-load data
print("Loading data...", flush=True)
data_cache = {}
for year in YEARS:
    d = load_and_prepare(year, model, feature_names, 'bnb')
    if d is not None:
        d['probs_safe'] = np.where(d['atr_regime'] > ATR_KILL, 0.0, d['probs'])
        data_cache[year] = d
print(f"Available years: {list(data_cache.keys())}\n")

grid = list(product(SL_VALS, TP_VALS, THR_VALS))
print(f"Grid: {len(SL_VALS)} SL × {len(TP_VALS)} TP × {len(THR_VALS)} Thr = {len(grid)} combinations")
print(f"SL:  {SL_VALS}")
print(f"TP:  {TP_VALS}")
print(f"Thr: {THR_VALS}")
print()

results = []
for sl, tp, thr in grid:
    yr = {}
    for year, data in data_cache.items():
        ret, sharpe, dd, trades, wr, sortino = simulate_numba(
            data['high'], data['low'], data['close'],
            data['probs_safe'], data['adx'],
            sl, tp, thr, adx_min,
            initial=INITIAL, slippage=SLIP, max_pos=MAX_POS, fee_pct=FEE)
        yr[year] = {'ret': round(ret,1), 'sharpe': round(sharpe,2),
                    'dd': round(dd,1), 'trades': int(trades), 'wr': round(wr,1)}

    oos_sharpes = [yr[y]['sharpe'] for y in OOS_YEARS if y in yr]
    deploy_score = round(np.mean(oos_sharpes), 3) if oos_sharpes else 0.0
    val_sharpes  = [yr[y]['sharpe'] for y in [2022,2023,2024] if y in yr]
    sharpe_val   = round(np.mean(val_sharpes), 2) if val_sharpes else 0.0
    min_sharpe   = round(min(yr[y]['sharpe'] for y in yr), 2)

    results.append({
        'sl': sl, 'tp': tp, 'thr': thr,
        'deploy_score': deploy_score,
        'sharpe_val': sharpe_val,
        'sharpe_2025': yr.get(2025, {}).get('sharpe', 0),
        'sharpe_2026': yr.get(2026, {}).get('sharpe', 0),
        'min_sharpe': min_sharpe,
        'years': yr,
    })

results.sort(key=lambda x: -x['deploy_score'])

# Table
print(f"{'SL':>5} {'TP':>5} {'Thr':>6}  {'deploy':>7} {'val':>6} {'2025':>6} {'2026':>6} {'minS':>6}  "
      f"{'2022ret':>7} {'2023ret':>7} {'2024ret':>7} {'2025ret':>7} {'2026ret':>7}")
print("-" * 100)
for r in results:
    y = r['years']
    def yr_ret(y_, yr_):
        return f'{yr_.get(y_, {}).get("ret", 0):+.1f}%' if y_ in yr_ else '  N/A'
    print(f"{r['sl']:>5.1f} {r['tp']:>5.1f} {r['thr']:>6.3f}  "
          f"{r['deploy_score']:>7.3f} {r['sharpe_val']:>6.2f} "
          f"{r['sharpe_2025']:>6.2f} {r['sharpe_2026']:>6.2f} {r['min_sharpe']:>6.2f}  "
          f"{yr_ret(2022,y):>7} {yr_ret(2023,y):>7} {yr_ret(2024,y):>7} "
          f"{yr_ret(2025,y):>7} {yr_ret(2026,y):>7}")

print()
best = results[0]
print(f"=== BEST COMBINATION ===")
print(f"SL={best['sl']}%  TP={best['tp']}%  Thr={best['thr']}  deploy_score={best['deploy_score']}")
print(f"Sharpe val={best['sharpe_val']}  2025={best['sharpe_2025']}  2026={best['sharpe_2026']}  min={best['min_sharpe']}")

# Reference: original params
ref = next((r for r in results if r['sl']==7.8 and r['tp']==7.1 and r['thr']==0.857), None)
if not ref:
    # show the closest one
    print(f"\n(Original params SL=7.8/TP=7.1/Thr=0.857 outside this analysis grid)")
