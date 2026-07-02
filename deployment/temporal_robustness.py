"""
deployment/temporal_robustness.py

Computa métricas de robustez temporal para modelos seleccionados em
janelas alternativas (trimestral, semestral, anual civil e rolling
12-meses). Os resultados complementam a Secção 4.7 do manuscrito.

Uso:
    python deployment/temporal_robustness.py
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import ml_sessions_compat.config as ml_config
from pipeline.backtest import load_model, simulate_numba
from ml_sessions_compat.features.technical import merge_timeframes


# --- configuração do backtest (igual ao evaluate_models.py) ---
INITIAL_CAPITAL = 500.0
FEE_PCT         = 0.002
SLIPPAGE        = 0.001
ATR_KILL        = 3.0
MAX_POS         = 5

# Cache de parquets (evita re-leituras desnecessárias)
_PARQUET_CACHE: dict = {}


def _find_parquet(data_dir: str, ticker: str, tf: str, exchange: str = 'binance') -> str | None:
    candidates = []
    if exchange and exchange != 'binance':
        candidates.append(f'{ticker}_{tf}_usdt_{exchange}.parquet')
    candidates.append(f'{ticker}_{tf}_usdt_binance.parquet')
    candidates.append(f'{ticker}_{tf}_usdt.parquet')
    for name in candidates:
        path = Path(data_dir) / name
        if path.exists():
            return str(path)
    return None


def _load_parquet_cached(ticker: str, tf: str, exchange: str = 'binance') -> pd.DataFrame | None:
    key = (ticker, exchange, tf)
    if key not in _PARQUET_CACHE:
        fpath = _find_parquet(ml_config.DATA_DIR, ticker, tf, exchange)
        if fpath is None:
            _PARQUET_CACHE[key] = None
        else:
            df = pd.read_parquet(fpath)
            df['timestamp'] = pd.to_datetime(df['open_time']).astype('datetime64[ns]')
            _PARQUET_CACHE[key] = df
    return _PARQUET_CACHE[key]


def load_and_prepare_range(
    model,
    feature_names: list,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange: str = 'binance',
) -> dict | None:
    """Carrega dados num intervalo de datas arbitrário e prepara arrays."""
    df_15m_full = _load_parquet_cached(ticker, '15m', exchange)
    if df_15m_full is None:
        return None

    df_15m = df_15m_full[
        (df_15m_full['timestamp'] >= start) & (df_15m_full['timestamp'] < end)
    ].sort_values('timestamp').reset_index(drop=True)

    if len(df_15m) < 100:
        return None

    # Dados higher-TF: precisamos de alguma história anterior para cálculo de indicadores
    higher_tf = {}
    for tf_key, tf_code in [('04h', '04h'), ('01d', '01d')]:
        df_full = _load_parquet_cached(ticker, tf_code, exchange)
        if df_full is not None:
            # Usar desde o início do ano anterior ao start para garantir contexto
            cutoff = start - pd.DateOffset(years=1)
            df = df_full[df_full['timestamp'] >= cutoff].sort_values('timestamp').reset_index(drop=True)
            higher_tf[tf_key] = df

    df = merge_timeframes(df_15m, higher_tf)
    if df is None or len(df) < 100:
        return None

    df = df.sort_values('timestamp').reset_index(drop=True)

    X = np.zeros((len(df), len(feature_names)), dtype=np.float64)
    for j, f in enumerate(feature_names):
        if f in df.columns:
            X[:, j] = df[f].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(X)[:, 1].astype(np.float64)

    n = len(df)
    return {
        'timestamp':  df['timestamp'],
        'high':       df['high'].to_numpy(dtype=np.float64),
        'low':        df['low'].to_numpy(dtype=np.float64),
        'close':      df['close'].to_numpy(dtype=np.float64),
        'adx':        df['adx_15m'].to_numpy(dtype=np.float64) if 'adx_15m' in df.columns else np.zeros(n),
        'atr_regime': df['atr_regime_15m'].to_numpy(dtype=np.float64) if 'atr_regime_15m' in df.columns else np.zeros(n),
        'probs':      probs,
        'n':          n,
    }


def _run_window(
    model, feature_names: list, ticker: str,
    start: pd.Timestamp, end: pd.Timestamp,
    sl: float, tp: float, thr: float, adx_min: float,
) -> dict | None:
    data = load_and_prepare_range(model, feature_names, ticker, start, end)
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
        'start':        start.strftime('%Y-%m-%d'),
        'end':          (end - pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
        'n_bars':       data['n'],
        'retorno_pct':  round(ret, 2),
        'sharpe':       round(sharpe, 3),
        'sortino':      round(sortino, 3),
        'max_dd_pct':   round(dd, 2),
        'n_trades':     int(trades),
        'win_rate_pct': round(wr, 1),
    }


def _year_bounds(year: int):
    return pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year + 1}-01-01')


def _quarter_windows(years: list):
    windows = []
    for y in years:
        for q in range(4):
            s = pd.Timestamp(f'{y}-{3*q + 1:02d}-01')
            e = pd.Timestamp(f'{y}-{3*(q + 1) + 1:02d}-01') if q < 3 else pd.Timestamp(f'{y + 1}-01-01')
            windows.append((s, e, f'{y}Q{q + 1}'))
    return windows


def _semester_windows(years: list):
    windows = []
    for y in years:
        windows.append((pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y}-07-01'), f'{y}H1'))
        windows.append((pd.Timestamp(f'{y}-07-01'), pd.Timestamp(f'{y + 1}-01-01'), f'{y}H2'))
    return windows


def _rolling_12m_windows(years: list):
    """Janelas rolling de 12 meses começando em cada ano disponível."""
    windows = []
    for y in years:
        s = pd.Timestamp(f'{y}-01-01')
        e = pd.Timestamp(f'{y + 1}-01-01')
        windows.append((s, e, f'{y}-{y + 1}'))
    return windows


def temporal_robustness_for_model(
    iter_dir: Path,
    ticker: str,
    years: list,
) -> dict:
    model_dir = iter_dir / 'model'
    model, feature_names = load_model(model_dir)

    meta_path = iter_dir / 'meta.json'
    with open(meta_path) as f:
        meta = json.load(f)
    m = meta['metricas']
    sl, tp, thr = m['sl_pct'], m['tp_pct'], m['threshold']
    adx_min = meta['params'].get('ENTRY_ADX_THRESHOLD', 20)

    all_windows = []
    all_windows.extend([(s, e, 'quarterly', lbl) for s, e, lbl in _quarter_windows(years)])
    all_windows.extend([(s, e, 'semester', lbl) for s, e, lbl in _semester_windows(years)])
    all_windows.extend([(*_year_bounds(y), 'annual', str(y)) for y in years])
    all_windows.extend([(s, e, 'rolling_12m', lbl) for s, e, lbl in _rolling_12m_windows(years)])

    records = []
    for s, e, kind, label in all_windows:
        r = _run_window(model, feature_names, ticker, s, e, sl, tp, thr, adx_min)
        if r is None:
            continue
        r['window_kind'] = kind
        r['window_label'] = label
        records.append(r)

    return {
        'iter': meta['iteracao'],
        'season': meta.get('season', '?'),
        'ticker': ticker,
        'params': {'sl': sl, 'tp': tp, 'threshold': thr, 'adx_min': adx_min},
        'records': records,
    }


def _summarise(records: list, kind: str) -> dict | None:
    subset = [r for r in records if r['window_kind'] == kind]
    if not subset:
        return None
    sharpes = [r['sharpe'] for r in subset]
    rets = [r['retorno_pct'] for r in subset]
    dds = [r['max_dd_pct'] for r in subset]
    trades = [r['n_trades'] for r in subset]
    return {
        'window_kind': kind,
        'n_windows': len(subset),
        'sharpe_mean': round(float(np.mean(sharpes)), 2),
        'sharpe_std': round(float(np.std(sharpes)), 2),
        'sharpe_min': round(float(np.min(sharpes)), 2),
        'sharpe_max': round(float(np.max(sharpes)), 2),
        'sharpe_pos_frac': round(sum(1 for s in sharpes if s > 0) / len(sharpes), 2),
        'ret_mean': round(float(np.mean(rets)), 1),
        'ret_std': round(float(np.std(rets)), 1),
        'dd_mean': round(float(np.mean(dds)), 1),
        'trades_total': int(sum(trades)),
    }


def main():
    parser = argparse.ArgumentParser(description='Análise de robustez temporal')
    parser.add_argument('--out', type=str, default='deployment/results',
                        help='Diretório de saída')
    args = parser.parse_args()

    out_dir = BASE_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modelos dos case studies principais
    cases = [
        ('season_11', 'iter_1077', 'bnb', list(range(2022, 2027))),
        ('season_12', 'iter_5502', 'btc', list(range(2021, 2026))),
    ]

    all_results = []
    for season, iter_name, ticker, years in cases:
        iter_dir = BASE_DIR / 'best_models' / season / iter_name
        if not iter_dir.exists():
            print(f'[AVISO] {iter_dir} não encontrado; a ignorar.')
            continue

        print(f'=== Robustez temporal: {season}/{iter_name} ({ticker}) ===')
        result = temporal_robustness_for_model(iter_dir, ticker, years)
        all_results.append(result)

        # Resumo por tipo de janela
        print('\nResumo por tipo de janela:')
        print(f"{'kind':<15} {'n':>4} {'sh_mean':>8} {'sh_std':>8} {'sh_min':>8} "
              f"{'sh_max':>8} {'pos_frac':>8} {'ret_mean':>9} {'dd_mean':>8} {'trades':>8}")
        print('-' * 100)
        for kind in ['quarterly', 'semester', 'annual', 'rolling_12m']:
            s = _summarise(result['records'], kind)
            if s is None:
                continue
            print(f"{s['window_kind']:<15} {s['n_windows']:>4} {s['sharpe_mean']:>8.2f} "
                  f"{s['sharpe_std']:>8.2f} {s['sharpe_min']:>8.2f} {s['sharpe_max']:>8.2f} "
                  f"{s['sharpe_pos_frac']:>8.2f} {s['ret_mean']:>9.1f} {s['dd_mean']:>8.1f} "
                  f"{s['trades_total']:>8}")

        # Guardar CSV com todos os detalhes
        df = pd.DataFrame(result['records'])
        csv_path = out_dir / f'temporal_robustness_{season}_{iter_name}.csv'
        df.to_csv(csv_path, index=False)
        print(f'\nDetalhes guardados em: {csv_path}')

    # Guardar JSON agregado
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    json_path = out_dir / f'temporal_robustness_summary_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nResumo JSON guardado em: {json_path}')


if __name__ == '__main__':
    main()
