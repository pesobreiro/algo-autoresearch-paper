"""
deployment/exit_slippage_sensitivity.py

Stress-test de slippage nas saídas (stop-loss, take-profit, eoy close)
para os modelos selecionados S11 iter 1077 e S12 iter 5502.

Mantém a fee de 0.2% round-trip e o slippage de entrada de 0.1%,
adicionando slippage simétrico nas saídas: 0.0%, 0.1%, 0.2%, 0.3%.
"""
import sys
import json
import numpy as np
from pathlib import Path
from numba import njit

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.backtest import load_model, load_and_prepare

CANDLES_PER_DAY = 96
FEE_PCT = 0.002
ENTRY_SLIPPAGE = 0.001


@njit
def simulate_with_exit_slippage(high, low, close, probs, adx,
                                sl_pct, tp_pct, threshold, adx_min,
                                initial=500.0, max_pos=5,
                                exit_slippage=0.0):
    capital = initial
    n_trades = 0
    n_wins = 0

    pos_entry = np.zeros(max_pos, dtype=np.float64)
    pos_size = np.zeros(max_pos, dtype=np.float64)
    pos_sl = np.zeros(max_pos, dtype=np.float64)
    pos_tp = np.zeros(max_pos, dtype=np.float64)
    pos_count = 0

    day_start_eq = initial
    daily_sum = 0.0
    daily_sq_sum = 0.0
    downside_sq_sum = 0.0
    n_days = 0

    peak = initial
    max_dd = 0.0

    for i in range(len(high)):
        new_count = 0
        for j in range(pos_count):
            e = pos_entry[j]
            sz = pos_size[j]
            exited = False
            win = False
            exit_price = 0.0

            if low[i] <= pos_sl[j]:
                # Saída por SL com slippage desfavorável (preço pior para posição longa)
                exit_price = pos_sl[j] * (1.0 - exit_slippage)
                if exit_price <= 0:
                    exit_price = pos_sl[j]
                exited = True
            elif high[i] >= pos_tp[j]:
                # Saída por TP com slippage desfavorável (preço pior para posição longa)
                exit_price = pos_tp[j] * (1.0 - exit_slippage)
                exited = True
                win = True

            if exited:
                capital += sz * (exit_price / e) * (1.0 - FEE_PCT)
                n_trades += 1
                if win:
                    n_wins += 1
            else:
                pos_entry[new_count] = e
                pos_size[new_count] = sz
                pos_sl[new_count] = pos_sl[j]
                pos_tp[new_count] = pos_tp[j]
                new_count += 1
        pos_count = new_count

        if pos_count < max_pos and probs[i] >= threshold and adx[i] > adx_min:
            confianca = (probs[i] - threshold) / (1.0 - threshold + 1e-9)
            fator = 0.5 + 0.5 * confianca
            slot_size = (capital / max_pos) * fator
            if slot_size > 50.0:
                ep = close[i] * (1.0 + ENTRY_SLIPPAGE)
                capital -= slot_size
                pos_entry[pos_count] = ep
                pos_size[pos_count] = slot_size
                pos_sl[pos_count] = ep * (1.0 - sl_pct / 100.0)
                pos_tp[pos_count] = ep * (1.0 + tp_pct / 100.0)
                pos_count += 1

        open_val = 0.0
        for j in range(pos_count):
            open_val += pos_size[j]
        equity_now = capital + open_val

        if equity_now > peak:
            peak = equity_now
        dd = (equity_now - peak) / peak
        if dd < max_dd:
            max_dd = dd

        if (i + 1) % CANDLES_PER_DAY == 0:
            if day_start_eq > 0:
                dr = (equity_now - day_start_eq) / day_start_eq
                daily_sum += dr
                daily_sq_sum += dr * dr
                if dr < 0.0:
                    downside_sq_sum += dr * dr
                n_days += 1
            day_start_eq = equity_now

    # Fechar posições em aberto no fim com slippage desfavorável
    for j in range(pos_count):
        e = pos_entry[j]
        sz = pos_size[j]
        exit_price = close[-1] * (1.0 - exit_slippage)
        capital += sz * (exit_price / e) * (1.0 - FEE_PCT)
        n_trades += 1
        if close[-1] > e:
            n_wins += 1

    total_ret = (capital / initial - 1.0) * 100.0

    sharpe = 0.0
    sortino = 0.0
    if n_days > 1:
        mean_dr = daily_sum / n_days
        var_dr = daily_sq_sum / n_days - mean_dr * mean_dr
        if var_dr > 0.0:
            sharpe = (mean_dr / (var_dr ** 0.5)) * (365.0 ** 0.5)
        downside_std = (downside_sq_sum / n_days) ** 0.5
        if downside_std > 0.0:
            sortino = (mean_dr / downside_std) * (365.0 ** 0.5)

    win_rate = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0
    return total_ret, sharpe, max_dd * 100.0, n_trades, win_rate, sortino


def evaluate_case(season, iter_num, sl, tp, thr, ticker, years, label):
    iter_dir = BASE_DIR / f'best_models/season_{season}/iter_{iter_num:04d}'
    with open(iter_dir / 'meta.json') as f:
        meta = json.load(f)
    model, feature_names = load_model(iter_dir / 'model')
    adx_min = meta['params']['ENTRY_ADX_THRESHOLD']

    data_cache = {}
    for year in years:
        d = load_and_prepare(year, model, feature_names, ticker)
        if d is not None:
            d['probs_safe'] = np.where(d['atr_regime'] > 3.0, 0.0, d['probs'])
            data_cache[year] = d

    exit_slippages = [0.0, 0.001, 0.002, 0.003]
    results = []

    for ex_slip in exit_slippages:
        year_metrics = []
        for year, data in data_cache.items():
            ret, sharpe, dd, trades, wr, sortino = simulate_with_exit_slippage(
                data['high'], data['low'], data['close'],
                data['probs_safe'], data['adx'],
                sl, tp, thr, adx_min,
                initial=500.0, max_pos=5, exit_slippage=ex_slip)
            year_metrics.append({
                'year': year, 'ret': ret, 'sharpe': sharpe,
                'dd': dd, 'trades': trades, 'wr': wr, 'sortino': sortino})

        # Agregar por período: val (exclui holdout), holdout, 2026 se existir
        val_years = [y for y in years if y not in (2025, 2026)]
        holdout_years = [y for y in years if y == 2025]
        oos2026 = [y for y in years if y == 2026]

        def agg(ms):
            if not ms:
                return {}
            rets = [m['ret'] for m in ms]
            sharpes = [m['sharpe'] for m in ms]
            dds = [m['dd'] for m in ms]
            trades = sum(m['trades'] for m in ms)
            return {
                'ret': round(sum(rets), 2),
                'sharpe': round(np.mean(sharpes), 2),
                'dd': round(min(dds), 2),
                'trades': trades,
            }

        results.append({
            'exit_slippage': ex_slip,
            'val': agg([m for m in year_metrics if m['year'] in val_years]),
            'holdout': agg([m for m in year_metrics if m['year'] in holdout_years]),
            'oos2026': agg([m for m in year_metrics if m['year'] in oos2026]),
            'all': agg(year_metrics),
        })

    print(f"\n=== {label} ===")
    print(f"{'Exit slip':>9} {'Val Sharpe':>11} {'Hold Sharpe':>12} {'2026 Sharpe':>12} "
          f"{'All Ret%':>10} {'All DD%':>9} {'All Trades':>11}")
    print("-" * 90)
    for r in results:
        vsh = f"{r['val'].get('sharpe', 0):.2f}" if r['val'] else "—"
        hsh = f"{r['holdout'].get('sharpe', 0):.2f}" if r['holdout'] else "—"
        osh = f"{r['oos2026'].get('sharpe', 0):.2f}" if r['oos2026'] else "—"
        print(f"{r['exit_slippage']*100:>7.1f}%   {vsh:>10}  {hsh:>11}  {osh:>11}  "
              f"{r['all'].get('ret', 0):>+8.1f}%  {r['all'].get('dd', 0):>7.1f}%  {r['all'].get('trades', 0):>9}")

    return results


def main():
    print("Stress-test de slippage nas saídas")
    print("Fee: 0.2% round-trip | Slippage entrada: 0.1%")

    s11 = evaluate_case(
        season=11, iter_num=1077, sl=7.8, tp=7.1, thr=0.857,
        ticker='bnb', years=[2022, 2023, 2024, 2025, 2026],
        label='S11 iter 1077 (BNB/USDT)')

    s12 = evaluate_case(
        season=12, iter_num=5502, sl=9.7, tp=9.1, thr=0.890,
        ticker='btc', years=[2021, 2022, 2023, 2024, 2025],
        label='S12 iter 5502 (BTC/USDC)')

    # Guardar resultados
    out_dir = BASE_DIR / 'deployment/results'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'exit_slippage_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump({'s11': s11, 's12': s12}, f, indent=2)
    print(f"\nResultados guardados em: {out_path}")


if __name__ == '__main__':
    main()
