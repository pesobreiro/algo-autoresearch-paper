"""
deployment/backtest_deploy.py

Backtest completo com registo de trades individuais para avaliação de deploy.

Uso:
    python deployment/backtest_deploy.py
    python deployment/backtest_deploy.py --iter 1077 --sl 6.5 --tp 6.5 --thr 0.857
    python deployment/backtest_deploy.py --years 2025 2026
"""
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.backtest import load_model, load_and_prepare, find_parquet
import ml_sessions_compat.config as ml_config


ATR_KILL  = 3.0
SLIPPAGE  = 0.001
FEE_PCT   = 0.002
MAX_POS   = 5
INITIAL   = 500.0


def load_timestamps(year: int, ticker: str, exchange: str = 'binance') -> pd.Series | None:
    """Carrega timestamps do parquet 15m para mapear índices → datas."""
    fpath = find_parquet(ml_config.DATA_DIR, ticker, '15m', exchange)
    if fpath is None:
        return None
    df = pd.read_parquet(fpath)
    df['timestamp'] = pd.to_datetime(df['open_time'])
    df = df[df['timestamp'].dt.year == year].sort_values('timestamp').reset_index(drop=True)
    return df['timestamp']


def simulate_with_trades(data: dict, sl_pct: float, tp_pct: float,
                         threshold: float, adx_min: float,
                         timestamps: pd.Series,
                         initial: float = INITIAL,
                         max_pos: int = MAX_POS) -> tuple[dict, list[dict]]:
    """
    Simulação Python com registo completo de trades.
    Retorna (métricas, lista_de_trades).
    """
    high   = data['high']
    low    = data['low']
    close  = data['close']
    probs  = data['probs_safe']
    adx    = data['adx']
    n      = len(high)

    capital    = initial
    n_trades   = 0
    n_wins     = 0
    trades_log = []

    # Posições abertas: lista de dicts
    positions  = []

    peak    = initial
    max_dd  = 0.0
    equity_curve = np.zeros(n)

    CANDLES_PER_DAY = 96
    day_start_eq    = initial
    daily_rets      = []

    def get_ts(i):
        if timestamps is not None and i < len(timestamps):
            return timestamps.iloc[i]
        return None

    for i in range(n):
        # --- Fechar posições ---
        # Recolher trades fechados primeiro; equity_after calculado depois de processar todas as posições
        still_open    = []
        candle_closed = []
        for pos in positions:
            if low[i] <= pos['sl_price']:
                pnl_pct = (pos['sl_price'] / pos['entry_price'] - 1.0) * 100 - FEE_PCT * 100
                pnl_eur = pos['size'] * (pos['sl_price'] / pos['entry_price']) * (1 - FEE_PCT) - pos['size']
                capital += pos['size'] + pnl_eur
                n_trades += 1
                candle_closed.append({**pos,
                    'exit_idx':   i,
                    'exit_time':  get_ts(i),
                    'exit_price': round(pos['sl_price'], 4),
                    'exit_type':  'SL',
                    'pnl_pct':    round(pnl_pct, 3),
                    'pnl_eur':    round(pnl_eur, 4),
                })
            elif high[i] >= pos['tp_price']:
                pnl_pct = (pos['tp_price'] / pos['entry_price'] - 1.0) * 100 - FEE_PCT * 100
                pnl_eur = pos['size'] * (pos['tp_price'] / pos['entry_price']) * (1 - FEE_PCT) - pos['size']
                capital += pos['size'] + pnl_eur
                n_trades += 1
                n_wins   += 1
                candle_closed.append({**pos,
                    'exit_idx':   i,
                    'exit_time':  get_ts(i),
                    'exit_price': round(pos['tp_price'], 4),
                    'exit_type':  'TP',
                    'pnl_pct':    round(pnl_pct, 3),
                    'pnl_eur':    round(pnl_eur, 4),
                })
            else:
                still_open.append(pos)
        positions = still_open
        # equity_after correcto: capital + todas as posições ainda abertas após este candle
        equity_candle = round(capital + sum(p['size'] for p in positions), 2)
        for t in candle_closed:
            trades_log.append({**t, 'equity_after': equity_candle})

        # --- Abrir posição ---
        if len(positions) < max_pos and probs[i] >= threshold and adx[i] > adx_min:
            confianca = (probs[i] - threshold) / (1.0 - threshold + 1e-9)
            fator     = 0.5 + 0.5 * confianca
            slot_size = (capital / max_pos) * fator
            if slot_size > 50.0:
                ep = close[i] * (1.0 + SLIPPAGE)
                capital -= slot_size
                positions.append({
                    'entry_idx':   i,
                    'entry_time':  get_ts(i),
                    'entry_price': round(ep, 4),
                    'sl_price':    round(ep * (1 - sl_pct / 100), 4),
                    'tp_price':    round(ep * (1 + tp_pct / 100), 4),
                    'size':        round(slot_size, 4),
                    'prob':        round(float(probs[i]), 4),
                    'adx':         round(float(adx[i]), 2),
                })

        # --- Equity ---
        open_val   = sum(p['size'] for p in positions)
        equity_now = capital + open_val
        equity_curve[i] = equity_now

        if equity_now > peak:
            peak = equity_now
        dd = (equity_now - peak) / peak
        if dd < max_dd:
            max_dd = dd

        if (i + 1) % CANDLES_PER_DAY == 0:
            if day_start_eq > 0:
                daily_rets.append((equity_now - day_start_eq) / day_start_eq)
            day_start_eq = equity_now

    # --- Fechar posições em aberto no fim do ano ---
    ts_last   = timestamps.iloc[-1] if timestamps is not None else None
    eoy_closed = []
    for pos in positions:
        pnl_pct = (close[-1] / pos['entry_price'] - 1.0) * 100 - FEE_PCT * 100
        pnl_eur = pos['size'] * (close[-1] / pos['entry_price']) * (1 - FEE_PCT) - pos['size']
        capital += pos['size'] + pnl_eur
        n_trades += 1
        if close[-1] > pos['entry_price']:
            n_wins += 1
        eoy_closed.append({**pos,
            'exit_idx':   n - 1,
            'exit_time':  ts_last,
            'exit_price': round(close[-1], 4),
            'exit_type':  'EOY',
            'pnl_pct':    round(pnl_pct, 3),
            'pnl_eur':    round(pnl_eur, 4),
            'equity_after': None,  # preenchido abaixo
        })
    for t in eoy_closed:
        trades_log.append({**t, 'equity_after': round(capital, 2)})

    # --- Métricas ---
    total_ret = (capital / initial - 1.0) * 100
    sharpe = sortino = 0.0
    if len(daily_rets) > 1:
        dr = np.array(daily_rets)
        mean_dr = dr.mean()
        std_dr  = dr.std()
        if std_dr > 0:
            sharpe = mean_dr / std_dr * np.sqrt(365)
        downside = dr[dr < 0]
        if len(downside) > 0:
            sortino = mean_dr / downside.std() * np.sqrt(365)

    metrics = {
        'retorno_pct':    round(total_ret, 2),
        'equity_final':   round(capital, 2),
        'sharpe':         round(sharpe, 3),
        'sortino':        round(sortino, 3),
        'max_dd_pct':     round(max_dd * 100, 2),
        'n_trades':       n_trades,
        'n_wins':         n_wins,
        'win_rate_pct':   round(n_wins / n_trades * 100, 1) if n_trades else 0,
        'equity_curve':   equity_curve,
    }
    return metrics, trades_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter',   type=int,   default=1077)
    parser.add_argument('--season', type=int,   default=11)
    parser.add_argument('--sl',     type=float, default=6.5)
    parser.add_argument('--tp',     type=float, default=6.5)
    parser.add_argument('--thr',    type=float, default=0.857)
    parser.add_argument('--years',      type=int,   nargs='+', default=[2022,2023,2024,2025,2026])
    parser.add_argument('--max-pos',    type=int,   default=5, help='Máximo de posições simultâneas (1=full capital)')
    parser.add_argument('--out',        type=str,   default='deployment/results')
    parser.add_argument('--ticker',     type=str,   default='bnb', help='Ticker (btc, bnb, eth...)')
    parser.add_argument('--no-compound', action='store_true', help='Reiniciar €500 em cada ano (desactiva carry-forward)')
    args = parser.parse_args()
    args.compound = not args.no_compound

    # Carregar modelo
    iter_dir  = BASE_DIR / f'best_models/season_{args.season}/iter_{args.iter:04d}'
    meta_path = iter_dir / 'meta.json'
    with open(meta_path) as f:
        meta = json.load(f)

    model, feature_names = load_model(iter_dir / 'model')
    adx_min = meta['params']['ENTRY_ADX_THRESHOLD']

    print(f"=== Backtest Deploy — iter {args.iter} S{args.season} ===")
    print(f"SL={args.sl}%  TP={args.tp}%  Threshold={args.thr}  ADX_min={adx_min}")
    max_pos  = args.max_pos
    mode_str = 'composto (carry-forward)' if args.compound else 'independente (€500/ano)'
    print(f"Capital inicial: €{INITIAL:.0f}  Fee: {FEE_PCT*100:.1f}%  Max pos: {max_pos}  Modo: {mode_str}")
    print(f"Anos: {args.years}")
    print()

    all_trades = []
    summary    = []
    capital    = INITIAL

    for year in args.years:
        data = load_and_prepare(year, model, feature_names, args.ticker)
        if data is None:
            print(f"  {year}: sem dados")
            continue
        data['probs_safe'] = np.where(data['atr_regime'] > ATR_KILL, 0.0, data['probs'])
        timestamps = load_timestamps(year, args.ticker)

        year_capital = capital if args.compound else INITIAL
        metrics, trades = simulate_with_trades(
            data, args.sl, args.tp, args.thr, adx_min, timestamps,
            initial=year_capital, max_pos=max_pos)

        for t in trades:
            t['year'] = year
        all_trades.extend(trades)

        if args.compound:
            capital = metrics['equity_final']

        label = '(val)' if year in [2022,2023,2024] else ('(holdout)' if year==2025 else '(TRUE OOS)')
        inicio_str = f"  início=€{year_capital:.0f}" if args.compound else ''
        print(f"  {year} {label}: Sharpe={metrics['sharpe']:.2f}  Ret={metrics['retorno_pct']:+.1f}%  "
              f"DD={metrics['max_dd_pct']:.1f}%  Trades={metrics['n_trades']}  "
              f"WR={metrics['win_rate_pct']:.0f}%  Equity=€{metrics['equity_final']:.0f}{inicio_str}")
        summary.append({'year': year, 'capital_inicio': year_capital,
                        **{k: v for k, v in metrics.items() if k != 'equity_curve'}})

    # --- CSV de trades ---
    out_dir = BASE_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = out_dir / f'trades_iter{args.iter}_sl{args.sl}_tp{args.tp}_thr{args.thr}_{ts_str}.csv'

    fields = ['year', 'entry_time', 'exit_time', 'exit_type',
              'entry_price', 'exit_price', 'sl_price', 'tp_price',
              'size', 'prob', 'adx', 'pnl_pct', 'pnl_eur', 'equity_after']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_trades)

    print(f"\n  Trades guardadas em: {csv_path}")
    print(f"  Total trades: {len(all_trades)}")

    # --- Resumo por ano ---
    print()
    inicio_col = '  Início' if args.compound else ''
    print(f"{'Ano':>5} {'Sharpe':>7} {'Sortino':>8} {'Ret%':>7} {'DD%':>6} "
          f"{'Trades':>7} {'WR%':>6} {'Equity':>8}{inicio_col}")
    print("-" * (60 + (10 if args.compound else 0)))
    for s in summary:
        label = ' *OOS' if s['year'] >= 2025 else ''
        inicio_str = f"  €{s['capital_inicio']:>7.0f}" if args.compound else ''
        print(f"  {s['year']} {s['sharpe']:>7.2f} {s['sortino']:>8.2f} "
              f"{s['retorno_pct']:>+7.1f} {s['max_dd_pct']:>6.1f} "
              f"{s['n_trades']:>7} {s['win_rate_pct']:>6.1f} "
              f"€{s['equity_final']:>7.0f}{label}{inicio_str}")

    # Total composto ou acumulado OOS
    if args.compound and summary:
        final = summary[-1]['equity_final']
        total_pct = (final / INITIAL - 1) * 100
        print()
        print(f"  Capital final ({summary[0]['year']}→{summary[-1]['year']}): "
              f"€{final:.2f}  ({total_pct:+.1f}%  +€{final-INITIAL:.2f} sobre €{INITIAL:.0f})")
    else:
        oos = [s for s in summary if s['year'] >= 2025]
        if oos:
            print()
            total_ret_oos = sum(s['retorno_pct'] for s in oos)
            print(f"  Retorno acumulado 2025+2026: {total_ret_oos:+.1f}%  "
                  f"(€{INITIAL*(1+total_ret_oos/100):.0f} a partir de €{INITIAL:.0f})")


if __name__ == '__main__':
    main()
