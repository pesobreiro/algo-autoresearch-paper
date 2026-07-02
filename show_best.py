"""
Show the best setups found by algo_autoresearch.

Usage:
  python show_best.py                    # top 10 accepted (season from config.yaml)
  python show_best.py --season 2         # season 2
  python show_best.py --season 1 --top 5
  python show_best.py --all
  python show_best.py --full 32
"""
import json
import math
import argparse
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _experiments_dir(season: int) -> Path:
    base = Path(__file__).parent
    if season <= 1:
        return base / 'experiments'
    return base / f'experiments_s{season}'


def _season_from_config() -> int:
    """Reads the current season from config.yaml (fallback: 1)."""
    try:
        import yaml
        cfg = Path(__file__).parent / 'config.yaml'
        if not cfg.exists():
            cfg = Path(__file__).parent / 'config.yaml.example'
        data = yaml.safe_load(cfg.read_text())
        return data.get('agent', {}).get('season', 1)
    except Exception:
        return 1


def load_accepted(experiments_dir: Path) -> list[dict]:
    exps = []
    for f in sorted(experiments_dir.glob('iter_*.json')):
        try:
            d = json.loads(f.read_text())
            if d.get('status') == 'aceite':
                exps.append(d)
        except Exception:
            pass
    exps.sort(key=lambda x: x.get('metricas', {}).get('score_composto', -999), reverse=True)
    return exps


def score_breakdown(m: dict) -> str:
    S  = m.get('sharpe_raw', 0)
    R  = m.get('retorno_anual_pct', 0)
    DD = abs(m.get('max_drawdown_pct', 0))
    s  = math.tanh(S / 2) * 0.50
    r  = math.tanh(R / 100) * 0.30
    d  = (DD / 100) * 0.20
    return f"Sharpe {s:+.3f}/0.50  Return {r:+.3f}/0.30  DD {-d:.3f}/-0.20"


def equity_str(m: dict) -> str:
    eq = m.get('equity_500_final')
    if eq:
        profit = eq - 500
        per_year = m.get('equity_500_por_ano', {})
        s = f"€{eq:.0f} ({profit:+.0f}€)"
        if per_year:
            s += "  [" + "  ".join(f"{yr}→€{v:.0f}" for yr, v in sorted(per_year.items())) + "]"
        return s
    ret = m.get('retorno_total_oos_pct') or m.get('retorno_anual_pct', 0)
    if ret:
        eq = 500 * (1 + ret / 100)
        return f"€{eq:.0f} ({eq-500:+.0f}€)  [total return {ret:+.1f}%]"
    return "n/a"


def show_table(exps: list[dict], top: int, season: int):
    if not HAS_RICH:
        _show_table_simple(exps, top, season)
        return

    console = Console()
    exps_show = exps[:top]

    table = Table(
        title=f"Best Setups — algo_autoresearch S{season}  ({len(exps)} accepted in total)",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("#",         justify="center", style="dim", width=3)
    table.add_column("Iter",      justify="right",  style="cyan", width=5)
    table.add_column("Score",     justify="right",  style="bold green", width=8)
    table.add_column("Sharpe",    justify="right",  width=7)
    table.add_column("Return",    justify="right",  style="green", width=8)
    table.add_column("DD",        justify="right",  style="red", width=6)
    table.add_column("Win%",      justify="right",  width=6)
    table.add_column("Trades",    justify="right",  width=7)
    table.add_column("AUC",       justify="right",  width=7)
    table.add_column("Equity OOS (500€)", style="magenta", width=28)
    table.add_column("Timestamp", style="dim", width=17)

    for rank, e in enumerate(exps_show, 1):
        m   = e.get('metricas', {})
        auc = m.get('cv_auc_mean')
        auc_s = f"{auc:.4f}" if auc else "—"
        table.add_row(
            str(rank),
            str(e['iteracao']),
            f"{m.get('score_composto', 0):.4f}",
            f"{m.get('sharpe_raw', 0):.2f}",
            f"{m.get('retorno_anual_pct', 0):+.1f}%",
            f"{abs(m.get('max_drawdown_pct', 0)):.1f}%",
            f"{m.get('win_rate_pct', 0):.1f}%",
            str(m.get('n_trades', 0)),
            auc_s,
            equity_str(m),
            e.get('timestamp_iso', '')[:16],
        )

    console.print()
    console.print(table)

    # Score breakdown of the best
    if exps_show:
        best = exps_show[0]
        m    = best.get('metricas', {})
        console.print(f"\n[bold]Score breakdown of the best (iter {best['iteracao']}):[/bold]")
        console.print(f"  {score_breakdown(m)}")

        ps = best.get('params_snapshot', {})
        if ps:
            console.print(f"\n[bold]Params of the best:[/bold]")
            console.print(f"  FEATURES     = {ps.get('FEATURES', '?')}")
            console.print(f"  TIMEFRAMES   = {ps.get('TIMEFRAMES', '?')}")
            console.print(f"  Entry signal : stoch_rsi_k < {ps.get('ENTRY_STOCH_THRESHOLD','?')} AND adx > {ps.get('ENTRY_ADX_THRESHOLD','?')} AND ema_diff > 0")
            console.print(f"  XGBoost      : n_est={ps.get('N_ESTIMATORS','?')}  depth={ps.get('MAX_DEPTH','?')}  lr={ps.get('LEARNING_RATE','?')}  alpha={ps.get('REG_ALPHA','?')}  lambda={ps.get('REG_LAMBDA','?')}")
            sl_r = ps.get('SL_RANGE') or ps.get('SL_GRID')
            tp_r = ps.get('TP_RANGE') or ps.get('TP_GRID')
            thr  = ps.get('THRESHOLD_RANGE', ps.get('threshold', '?'))
            sl_b = m.get('sl_pct')
            tp_b = m.get('tp_pct')
            thr_b = m.get('threshold')
            if sl_b:
                console.print(f"  Optuna best  : SL={sl_b:.2f}%  TP={tp_b:.2f}%  Threshold={thr_b:.3f}")
            console.print(f"  Bounds       : SL={sl_r}  TP={tp_r}  Thr={thr}")

        # Feature importance if available
        top_f = m.get('top_features', [])
        bot_f = m.get('bottom_features', [])
        if top_f:
            console.print(f"\n[bold]Feature importance of the best:[/bold]")
            console.print(f"  TOP  : {', '.join(f'{f}={v:.3f}' for f,v in top_f[:6])}")
            if bot_f:
                console.print(f"  WEAK: {', '.join(f for f,_ in bot_f)}")

        # Optuna param importance
        oi = m.get('optuna_param_importance', {})
        if oi:
            sorted_oi = sorted(oi.items(), key=lambda x: -x[1])
            console.print(f"\n[bold]Optuna param importance of the best:[/bold]")
            console.print(f"  {' | '.join(f'{k}={v:.3f}' for k,v in sorted_oi)}")

    console.print()


def _show_table_simple(exps: list[dict], top: int, season: int):
    print(f"\n{'='*90}")
    print(f"  BEST SETUPS — algo_autoresearch S{season}  ({len(exps)} accepted in total)")
    print(f"{'='*90}")
    print(f"  {'#':>3}  {'Iter':>5}  {'Score':>8}  {'Sharpe':>7}  {'Return':>8}  {'DD':>6}  {'Win%':>6}  {'Trades':>7}  Timestamp")
    print(f"  {'-'*80}")
    for rank, e in enumerate(exps[:top], 1):
        m = e.get('metricas', {})
        print(
            f"  {rank:>3}  {e['iteracao']:>5}  {m.get('score_composto',0):>8.4f}  "
            f"{m.get('sharpe_raw',0):>7.2f}  {m.get('retorno_anual_pct',0):>+8.1f}%  "
            f"{abs(m.get('max_drawdown_pct',0)):>5.1f}%  {m.get('win_rate_pct',0):>5.1f}%  "
            f"{m.get('n_trades',0):>7}  {e.get('timestamp_iso','')[:16]}"
        )
    print(f"{'='*90}\n")


def show_detail(iteration: int, experiments_dir: Path):
    f = experiments_dir / f'iter_{iteration:04d}.json'
    if not f.exists():
        print(f"Iteration {iteration} not found.")
        return

    e  = json.loads(f.read_text())
    m  = e.get('metricas', {})
    ps = e.get('params_snapshot', {})

    if HAS_RICH:
        console = Console()
        status_color = {'aceite': 'green', 'rejeitado': 'red', 'erro': 'orange1'}.get(e.get('status'), 'white')

        lines = [
            f"[bold]Status:[/bold] [{status_color}]{e.get('status','?')}[/{status_color}]   "
            f"[bold]Timestamp:[/bold] {e.get('timestamp_iso','')}   "
            f"[bold]Commit:[/bold] {e.get('git_commit','')}",
            "",
            f"[bold green]Composite score:[/bold green] {m.get('score_composto',0):.4f}",
            f"  {score_breakdown(m)}",
            "",
            f"[bold]Metrics:[/bold]",
            f"  Sharpe    : {m.get('sharpe_raw',0):.4f}",
            f"  Return    : {m.get('retorno_anual_pct',0):+.2f}%",
            f"  Drawdown  : {abs(m.get('max_drawdown_pct',0)):.2f}%",
            f"  Win Rate  : {m.get('win_rate_pct',0):.1f}%",
            f"  Trades    : {m.get('n_trades',0)}",
            f"  AUC       : {m.get('cv_auc_mean', 'n/a')}",
            f"  Equity OOS: {equity_str(m)}",
        ]

        sl_b  = m.get('sl_pct')
        tp_b  = m.get('tp_pct')
        thr_b = m.get('threshold')
        if sl_b:
            lines += ["", f"[bold]Optuna best config:[/bold]",
                       f"  SL={sl_b:.2f}%  TP={tp_b:.2f}%  Threshold={thr_b:.3f}"]

        oi = m.get('optuna_param_importance', {})
        if oi:
            sorted_oi = sorted(oi.items(), key=lambda x: -x[1])
            lines += ["", f"[bold]Optuna param importance:[/bold]",
                       f"  {' | '.join(f'{k}={v:.3f}' for k,v in sorted_oi)}"]

        top_f = m.get('top_features', [])
        bot_f = m.get('bottom_features', [])
        if top_f:
            lines += ["", "[bold]Feature importance (XGBoost):[/bold]",
                       f"  TOP   : {', '.join(f'{f}={v:.3f}' for f,v in top_f)}"]
            if bot_f:
                lines.append(f"  WEAK: {', '.join(f for f,_ in bot_f)}")

        if ps:
            lines += ["", "[bold]Params snapshot:[/bold]",
                       f"  FEATURES         = {ps.get('FEATURES',[])}",
                       f"  TIMEFRAMES       = {ps.get('TIMEFRAMES',[])}",
                       f"  STOCH_RSI_PERIOD = {ps.get('STOCH_RSI_PERIOD','?')}  ADX_PERIOD={ps.get('ADX_PERIOD','?')}  BB_PERIOD={ps.get('BB_PERIOD','?')}",
                       f"  Entry signal     : stoch_rsi_k < {ps.get('ENTRY_STOCH_THRESHOLD','?')} AND adx > {ps.get('ENTRY_ADX_THRESHOLD','?')}",
                       f"  XGBoost          : n_est={ps.get('N_ESTIMATORS','?')}  depth={ps.get('MAX_DEPTH','?')}  lr={ps.get('LEARNING_RATE','?')}",
                       f"                     alpha={ps.get('REG_ALPHA','?')}  lambda={ps.get('REG_LAMBDA','?')}  gamma={ps.get('GAMMA','?')}",
                       f"  SL_RANGE         = {ps.get('SL_RANGE', ps.get('SL_GRID','?'))}",
                       f"  TP_RANGE         = {ps.get('TP_RANGE', ps.get('TP_GRID','?'))}",
                       f"  THRESHOLD_RANGE  = {ps.get('THRESHOLD_RANGE','?')}"]

        if e.get('alteracoes_vs_anterior'):
            lines += ["", f"[bold]Changes vs previous:[/bold]",
                       f"  {e['alteracoes_vs_anterior']}"]

        if e.get('nota_humana'):
            lines += ["", f"[bold]Human note:[/bold] {e['nota_humana']}"]

        if e.get('tags'):
            lines += [f"[bold]Tags:[/bold] {', '.join(e['tags'])}"]

        console.print(Panel(
            "\n".join(lines),
            title=f"[bold cyan]Iteration {iteration} — Full Detail[/bold cyan]",
            border_style="cyan",
        ))
    else:
        print(json.dumps(e, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description='Show best setups of algo_autoresearch')
    parser.add_argument('--top',    type=int, default=10,  help='Number of best to show (default: 10)')
    parser.add_argument('--all',    action='store_true',   help='Show all accepted')
    parser.add_argument('--full',   type=int, metavar='N', help='Full detail of iteration N')
    parser.add_argument('--season', '-s', type=int, default=None,
                        help='Season (1=experiments/, 2=experiments_s2/, ...). Default: read from config.yaml')
    args = parser.parse_args()

    season = args.season if args.season is not None else _season_from_config()
    experiments_dir = _experiments_dir(season)

    if not experiments_dir.exists():
        print(f"Directory S{season} not found: {experiments_dir}")
        return

    if args.full is not None:
        show_detail(args.full, experiments_dir)
        return

    exps = load_accepted(experiments_dir)
    if not exps:
        print(f"No accepted iteration found in S{season}.")
        return

    top = len(exps) if args.all else args.top
    show_table(exps, top, season)


if __name__ == '__main__':
    main()
