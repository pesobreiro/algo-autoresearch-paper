#!/usr/bin/env python3
"""
algo_autoresearch — Main CLI

Commands:
  run       — start autonomous research loop
  review    — interactive history review
  tag       — add tag to an iteration
  analysis  — analysis by tag + score trend
  setup     — verify system prerequisites
"""
import sys
import os
from pathlib import Path

# Use local compatibility layer instead of external ~/git/ml_sessions
sys.path.insert(0, str(Path(__file__).parent))

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def load_config(config_path: Path = None) -> dict:
    """Loads config.yaml (or config.yaml.example if it does not exist)."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    if not config_path.exists():
        example = config_path.parent / 'config.yaml.example'
        if example.exists():
            console.print(f"[yellow]config.yaml not found. Using config.yaml.example[/yellow]")
            config_path = example
        else:
            console.print("[red]No configuration file found![/red]")
            sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """algo_autoresearch — Autonomous algorithmic trading research loop."""
    pass


def _experiments_dir(base_dir: Path, season: int) -> Path:
    """Returns the experiments directory for the given season."""
    if season <= 1:
        return base_dir / 'experiments'
    return base_dir / f'experiments_s{season}'


@cli.command()
@click.option('--iters', '-n', default=0, help='Maximum number of iterations (0=infinite)')
@click.option('--review-interval', default=5, help='Ask for review every N iterations (0=disabled)')
@click.option('--season', '-s', default=None, type=int,
              help='Research season (1=experiments/, 2=experiments_s2/, ...)')
@click.option('--config', 'config_path', default=None, type=click.Path(), help='Path to config.yaml')
def run(iters, review_interval, season, config_path):
    """Start the autonomous research loop."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)

    # Season: CLI > config.yaml > 1
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    # Verify prerequisites
    from autoresearch.runner import check_prerequisites
    ok, errors = check_prerequisites(config)

    if not ok:
        console.print(Panel(
            '\n'.join(f"[red]✗[/red] {e}" for e in errors),
            title="Missing prerequisites",
            border_style="red",
        ))
        console.print("\n[yellow]Run 'python main.py setup' for full diagnosis.[/yellow]")
        sys.exit(1)

    console.print(Panel(
        f"[bold green]Prerequisites verified — starting loop[/bold green]\n"
        f"Season: S{season} → {experiments_dir}",
        border_style="green",
    ))

    from autoresearch.runner import run_loop
    run_loop(
        config=config,
        max_iterations=iters,
        human_review_interval=review_interval,
        experiments_dir=experiments_dir,
    )


@cli.command()
@click.option('--season', '-s', default=None, type=int, help='Research season')
@click.option('--config', 'config_path', default=None, type=click.Path())
def review(season, config_path):
    """Interactive review of the experiments history."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    tracker.generate_analysis_report()


@cli.command()
@click.option('--iter', 'iteration', required=True, type=int, help='Iteration number')
@click.option('--label', required=True, type=click.Choice([
    'promising', 'baseline', 'explorado', 'rejeitado', 'interessante', 'bug'
]), help='Tag to add')
@click.option('--note', default='', help='Optional note')
@click.option('--season', '-s', default=None, type=int, help='Research season')
@click.option('--config', 'config_path', default=None, type=click.Path())
def tag(iteration, label, note, season, config_path):
    """Add a tag to a specific iteration."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    ok = tracker.add_tag(iteration, label, note)
    if ok:
        console.print(f"[green]✓ Tag '{label}' added to iteration {iteration} (S{season})[/green]")


@cli.command()
@click.option('--season', '-s', default=None, type=int, help='Research season')
@click.option('--config', 'config_path', default=None, type=click.Path())
def analysis(season, config_path):
    """Full analysis: table by tag + score trend."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    tracker.generate_analysis_report()


@cli.command('new-season')
@click.option('--config', 'config_path', default=None, type=click.Path())
@click.option('--dry-run', is_flag=True, help='Show what would be done without changing anything')
def new_season(config_path, dry_run):
    """Transition to the next season: updates config.yaml and research_params.py."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)
    config_file = Path(config_path) if config_path else base_dir / 'config.yaml'

    current_season = config.get('agent', {}).get('season', 1)
    new_season  = current_season + 1
    experiments_dir = _experiments_dir(base_dir, current_season)

    console.print(Panel(
        f"[bold]Season transition[/bold]\n"
        f"S{current_season} → S{new_season}\n"
        f"Experiments: {experiments_dir}",
        border_style="cyan",
    ))

    # Load best result of current season
    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    best = tracker.best_score()

    if best is None:
        console.print(f"[yellow]No accepted results in S{current_season}. "
                      f"Incrementing season without changing baseline or research_params.py.[/yellow]")
        best_score  = config.get('agent', {}).get('baseline_override', 0.0)
        best_params = None
    else:
        best_score  = best.metricas.get('score_composto', 0.0)
        best_params = best.params_snapshot
        m = best.metricas
        console.print(f"\n[green]Best result S{current_season}:[/green]")
        console.print(f"  Iter {best.iteracao}  Score={best_score:.4f}  "
                      f"Sharpe={m.get('sharpe_raw',0):.2f}  "
                      f"Return={m.get('retorno_anual_pct',0):+.1f}%  "
                      f"DD={abs(m.get('max_drawdown_pct',0)):.1f}%")

    # --- 1. Update config.yaml ---
    console.print(f"\n[bold]config.yaml:[/bold]")
    console.print(f"  agent.season:            {current_season} → {new_season}")
    console.print(f"  agent.baseline_override: → {best_score:.4f}")

    if not dry_run and config_file.exists():
        import yaml
        with open(config_file) as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw.setdefault('agent', {})
        cfg_raw['agent']['season']            = new_season
        cfg_raw['agent']['baseline_override'] = round(best_score, 6)
        with open(config_file, 'w') as f:
            yaml.dump(cfg_raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        console.print(f"  [green]✓ config.yaml updated[/green]")

    # --- 2. Update research_params.py ---
    params_path = base_dir / 'pipeline' / 'research_params.py'

    if best_params:
        new_params = _generate_research_params(best_params, new_season, current_season,
                                                best.iteracao, best_score)
        console.print(f"\n[bold]research_params.py:[/bold]")
        console.print(f"  Starting point: iter {best.iteracao} S{current_season} (score={best_score:.4f})")
        console.print(f"  FEATURES = {best_params.get('FEATURES', [])}")
        console.print(f"  TIMEFRAMES = {best_params.get('TIMEFRAMES', [])}")

        if not dry_run:
            import shutil
            shutil.copy2(params_path, params_path.with_suffix('.py.backup'))
            params_path.write_text(new_params)
            console.print(f"  [green]✓ research_params.py updated (backup in .py.backup)[/green]")
    else:
        console.print(f"\n[dim]research_params.py: no changes (no accepted result in S{current_season})[/dim]")

    if dry_run:
        console.print(f"\n[yellow]--dry-run: no file was changed.[/yellow]")
    else:
        console.print(Panel(
            f"[bold green]S{new_season} ready![/bold green]\n\n"
            f"To start:\n"
            f"  python main.py run",
            border_style="green",
        ))


def _generate_research_params(ps: dict, new_season: int, previous_season: int,
                            source_iteration: int, source_score: float) -> str:
    """Generates research_params.py content from a params_snapshot."""

    def fmt(v):
        if isinstance(v, list):
            return '[' + ', '.join(f'"{x}"' if isinstance(x, str) else str(x) for x in v) + ']'
        if isinstance(v, tuple):
            return f'({v[0]}, {v[1]})'
        return repr(v)

    features  = ps.get('FEATURES', [])
    timeframes = ps.get('TIMEFRAMES', ['15m', '4h', '1d'])

    lines = [
        f"# pipeline/research_params.py",
        f"# ONLY FILE MODIFIED BY THE AGENT",
        f"# CRITICAL RULE: ONLY RELATIVE INDICATORS (no absolute prices)",
        f"#",
        f"# SEASON {new_season} — starting point: best result S{previous_season}",
        f"# (iter {source_iteration}, score={source_score:.4f})",
        f"",
        f"# --- Features to use in training ---",
        f"FEATURES = {fmt(features)}",
        f"",
        f"# --- Timeframes to include ---",
        f"TIMEFRAMES = {fmt(timeframes)}",
        f"",
        f"# --- Technical indicator parameters ---",
        f"STOCH_RSI_PERIOD = {ps.get('STOCH_RSI_PERIOD', 14)}",
        f"ADX_PERIOD = {ps.get('ADX_PERIOD', 14)}",
        f"EMA_FAST = {ps.get('EMA_FAST', 12)}",
        f"EMA_SLOW = {ps.get('EMA_SLOW', 26)}",
        f"BB_PERIOD = {ps.get('BB_PERIOD', 20)}",
        f"",
        f"# --- Entry signal ---",
        f"ENTRY_STOCH_THRESHOLD = {ps.get('ENTRY_STOCH_THRESHOLD', 20)}",
        f"ENTRY_ADX_THRESHOLD = {ps.get('ENTRY_ADX_THRESHOLD', 25)}",
        f"",
        f"# --- XGBoost hyperparameters ---",
        f"N_ESTIMATORS = {ps.get('N_ESTIMATORS', 300)}",
        f"MAX_DEPTH = {ps.get('MAX_DEPTH', 6)}",
        f"LEARNING_RATE = {ps.get('LEARNING_RATE', 0.05)}",
        f"MIN_CHILD_WEIGHT = {ps.get('MIN_CHILD_WEIGHT', 5)}",
        f"GAMMA = {ps.get('GAMMA', 0.5)}",
        f"SUBSAMPLE = {ps.get('SUBSAMPLE', 0.8)}",
        f"COLSAMPLE_BYTREE = {ps.get('COLSAMPLE_BYTREE', 0.8)}",
        f"REG_ALPHA = {ps.get('REG_ALPHA', 1.0)}",
        f"REG_LAMBDA = {ps.get('REG_LAMBDA', 1.5)}",
        f"",
        f"# --- XGBoost Bayesian optimization (Optuna) ---",
    ]

    # XGBoost Optuna ranges — keep if they existed, otherwise disable
    depth_r = ps.get('DEPTH_RANGE')
    lr_r    = ps.get('LR_RANGE')
    est_r   = ps.get('ESTIMATORS_RANGE')
    a_r     = ps.get('ALPHA_RANGE')
    l_r     = ps.get('LAMBDA_RANGE')
    n_xgb   = ps.get('N_TRIALS_XGB', 0)

    if depth_r:
        lines += [
            f"DEPTH_RANGE = {fmt(depth_r)}",
            f"LR_RANGE = {fmt(lr_r) if lr_r else '(0.01, 0.12)'}",
            f"ESTIMATORS_RANGE = {fmt(est_r) if est_r else '(200, 600)'}",
            f"ALPHA_RANGE = {fmt(a_r) if a_r else '(0.1, 2.5)'}",
            f"LAMBDA_RANGE = {fmt(l_r) if l_r else '(0.5, 3.5)'}",
            f"N_TRIALS_XGB = {n_xgb}",
        ]
    else:
        lines.append(f"N_TRIALS_XGB = 0  # enable with >= 20 and define DEPTH_RANGE, LR_RANGE, etc.")

    sl_r  = ps.get('SL_RANGE', (0.5, 10.0))
    tp_r  = ps.get('TP_RANGE', (1.5, 25.0))
    thr_r = ps.get('THRESHOLD_RANGE', (0.1, 0.85))
    n_t   = ps.get('N_TRIALS', 140)

    lines += [
        f"",
        f"# --- SL/TP/Threshold Bayesian optimization ---",
        f"SL_RANGE = {fmt(sl_r)}",
        f"TP_RANGE = {fmt(tp_r)}",
        f"THRESHOLD_RANGE = {fmt(thr_r)}",
        f"N_TRIALS = {n_t}",
        f"",
        f"# --- Objective mode ---",
        f"OBJECTIVE_MODE = \"{ps.get('OBJECTIVE_MODE', 'score')}\"",
        f"",
        f"# --- Notes S{new_season} ---",
        f"# The agent should explore new dimensions from this starting point.",
    ]

    return '\n'.join(lines) + '\n'


@cli.command()
@click.option('--iter', 'iteration', required=True, type=int,
              help='Reference iteration for exploit mode')
@click.option('--margin', default=0.35, show_default=True,
              help='Relative margin to tighten bounds (e.g., 0.35 = ±35% around the optimum)')
@click.option('--xgb-trials', default=0, type=int, show_default=True,
              help='Activate XGBoost Optuna with N trials (0=disabled)')
@click.option('--season', '-s', default=None, type=int)
@click.option('--config', 'config_path', default=None, type=click.Path())
@click.option('--dry-run', is_flag=True, help='Show what would be done without changing anything')
def focus(iteration, margin, xgb_trials, season, config_path, dry_run):
    """Exploit mode: tighten bounds around a promising result."""
    base_dir = Path(__file__).parent
    config = load_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    # Load reference iteration
    import json
    iter_file = experiments_dir / f'iter_{iteration:04d}.json'
    if not iter_file.exists():
        console.print(f"[red]Iteration {iteration} not found in S{season}[/red]")
        return

    e  = json.loads(iter_file.read_text())
    metrics = e.get('metricas', {})
    ps = e.get('params_snapshot', {})

    sl_best  = metrics.get('sl_pct')
    tp_best  = metrics.get('tp_pct')
    thr_best = metrics.get('threshold')
    score    = metrics.get('score_composto', 0)

    console.print(Panel(
        f"[bold]EXPLOIT mode — iter {iteration} S{season}[/bold]\n"
        f"Score={score:.4f} | Sharpe={metrics.get('sharpe_raw',0):.2f} | "
        f"Return={metrics.get('retorno_anual_pct',0):+.1f}% | DD={abs(metrics.get('max_drawdown_pct',0)):.1f}%\n"
        f"Optuna best: SL={sl_best:.2f}% TP={tp_best:.2f}% Thr={thr_best:.3f}" if sl_best else
        f"Score={score:.4f} (no Optuna best available)",
        border_style="yellow",
    ))

    # Calculate new tight bounds around the optimum
    params_path = base_dir / 'pipeline' / 'research_params.py'
    current = ps if ps else {}

    sl_range_orig  = current.get('SL_RANGE', (0.5, 10.0))
    tp_range_orig  = current.get('TP_RANGE', (1.5, 25.0))
    thr_range_orig = current.get('THRESHOLD_RANGE', (0.1, 0.85))

    if sl_best and tp_best and thr_best:
        # Tighten bounds around the optimum with configurable margin
        sl_lo  = max(0.05, sl_best * (1 - margin))
        sl_hi  = sl_best * (1 + margin)
        tp_lo  = max(0.1, tp_best * (1 - margin))
        tp_hi  = tp_best * (1 + margin)
        thr_lo = max(0.05, thr_best * (1 - margin))
        thr_hi = min(0.95, thr_best * (1 + margin))

        # Ensure min < max (TP can be < SL in scalping regime)
        sl_range_new  = (round(min(sl_lo, sl_hi), 3), round(max(sl_lo, sl_hi), 3))
        tp_range_new  = (round(min(tp_lo, tp_hi), 3), round(max(tp_lo, tp_hi), 3))
        thr_range_new = (round(thr_lo, 3), round(thr_hi, 3))
    else:
        sl_range_new  = sl_range_orig
        tp_range_new  = tp_range_orig
        thr_range_new = thr_range_orig
        console.print("[yellow]No Optuna best in this iteration — bounds not changed[/yellow]")

    console.print(f"\n[bold]New bounds (margin ±{margin*100:.0f}%):[/bold]")
    console.print(f"  SL_RANGE:        {sl_range_orig} → {sl_range_new}")
    console.print(f"  TP_RANGE:        {tp_range_orig} → {tp_range_new}")
    console.print(f"  THRESHOLD_RANGE: {thr_range_orig} → {thr_range_new}")
    if xgb_trials > 0:
        console.print(f"  N_TRIALS_XGB:    0 → {xgb_trials} (tune XGBoost)")

    if not dry_run:
        import shutil
        shutil.copy2(params_path, params_path.with_suffix('.py.backup'))

        # Generate research_params.py with tight bounds
        ps_focus = dict(ps)
        ps_focus['SL_RANGE']        = sl_range_new
        ps_focus['TP_RANGE']        = tp_range_new
        ps_focus['THRESHOLD_RANGE'] = thr_range_new
        if xgb_trials > 0:
            ps_focus['N_TRIALS_XGB'] = xgb_trials
            ps_focus['DEPTH_RANGE']  = ps_focus.get('DEPTH_RANGE', (4, 9))
            ps_focus['LR_RANGE']     = ps_focus.get('LR_RANGE', (0.01, 0.12))
            ps_focus['ESTIMATORS_RANGE'] = ps_focus.get('ESTIMATORS_RANGE', (200, 600))
            ps_focus['ALPHA_RANGE']  = ps_focus.get('ALPHA_RANGE', (0.1, 2.5))
            ps_focus['LAMBDA_RANGE'] = ps_focus.get('LAMBDA_RANGE', (0.5, 3.5))

        new_code = _generate_research_params(
            ps_focus, season, season, iteration, score
        )
        # Replace header to indicate exploit mode
        new_code = new_code.replace(
            f"# SEASON {season} — starting point: best result S{season}",
            f"# SEASON {season} — EXPLOIT MODE: from iter {iteration} (score={score:.4f})",
        )
        params_path.write_text(new_code)
        console.print(f"\n[green]✓ research_params.py updated[/green]")

        # Update program.md with exploit instruction
        _update_program_md_exploit(
            base_dir / 'program.md', iteration, score, sl_range_new, tp_range_new,
            thr_range_new, metrics, xgb_trials
        )
        console.print(f"[green]✓ program.md updated for exploit mode[/green]")
        console.print(Panel(
            f"[bold green]Ready to explore around iter {iteration}[/bold green]\n"
            f"Start: python main.py run",
            border_style="green",
        ))
    else:
        console.print(f"\n[yellow]--dry-run: no file changed.[/yellow]")


def _update_program_md_exploit(program_path: Path, iter_ref: int, score_ref: float,
                                sl_r, tp_r, thr_r, metrics: dict, xgb_trials: int):
    """Rewrites the priorities section of program.md for exploit mode."""
    exploit_section = f"""
---

## ⚡ ACTIVE EXPLOIT MODE (iter {iter_ref}, score={score_ref:.4f})

The system is in **localized exploration** mode. Focus on:

1. **Fine-tune the SL/TP/threshold bounds** around the identified optimum region:
   - SL_RANGE = {sl_r} — explore small variations
   - TP_RANGE = {tp_r} — explore small variations
   - THRESHOLD_RANGE = {thr_r} — the threshold is the dominant parameter (Optuna importance)

2. **Keep the features** that generated this result — do not add or remove

3. **Adjust XGBoost regularization** if there is overfitting (REG_ALPHA, REG_LAMBDA)
{"4. **XGBoost Optuna active** (" + str(xgb_trials) + " trials) — fine-tune hyperparameters" if xgb_trials > 0 else "4. **N_TRIALS_XGB = 0** — keep disabled for speed"}

Reference: Sharpe={metrics.get('sharpe_raw',0):.2f} | Return={metrics.get('retorno_anual_pct',0):+.1f}% | DD={abs(metrics.get('max_drawdown_pct',0)):.1f}%

---
"""
    content = program_path.read_text()
    # Remove previous exploit section if it exists
    import re
    content = re.sub(r'\n---\n\n## ⚡ ACTIVE EXPLOIT MODE.*?---\n', '', content, flags=re.DOTALL)
    # Insert at the beginning, after the main header (after the first ---)
    parts = content.split('---', 1)
    if len(parts) == 2:
        content = parts[0] + '---' + exploit_section + parts[1]
    else:
        content = content + exploit_section
    program_path.write_text(content)


@cli.command()
@click.option('--config', 'config_path', default=None, type=click.Path())
def setup(config_path):
    """Verify all system prerequisites."""
    console.print(Panel("[bold]algo_autoresearch — System Check[/bold]", border_style="blue"))

    config = load_config(Path(config_path) if config_path else None)

    checks = []

    # 1. LLM Server
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    from autoresearch.agent import check_llm_server
    llm_ok = check_llm_server(server_url)
    checks.append(('LLM Server', llm_ok,
                   f"Accessible at {server_url}" if llm_ok
                   else f"NOT accessible at {server_url}"))

    # 2. research_params.py
    params_path = Path(__file__).parent / 'pipeline' / 'research_params.py'
    from autoresearch.agent import validate_code
    if params_path.exists():
        ok, msg = validate_code(params_path.read_text())
        checks.append(('research_params.py', ok, msg if ok else f"Invalid: {msg}"))
    else:
        checks.append(('research_params.py', False, 'File not found'))

    # 3. Data
    try:
        import ml_sessions_compat.config as ml_config
        ticker = config['pipeline']['ticker']
        exchange = config['pipeline'].get('exchange', 'binance')
        data_dir = Path(ml_config.DATA_DIR)
        found = False
        for cand in [f'{ticker}_15m_usdt_{exchange}.parquet',
                     f'{ticker}_15m_usdt_binance.parquet',
                     f'{ticker}_15m_usdt.parquet']:
            if (data_dir / cand).exists():
                found = True
                checks.append(('15m Data', True, str(data_dir / cand)))
                break
        if not found:
            checks.append(('15m Data', False,
                           f"Not found for {ticker} in {data_dir}"))
    except Exception as e:
        checks.append(('Data', False, str(e)))

    # 4. Git
    import subprocess
    try:
        r = subprocess.run(['git', 'rev-parse', '--git-dir'],
                          capture_output=True, cwd=Path(__file__).parent)
        git_ok = r.returncode == 0
        checks.append(('Git repo', git_ok, 'Initialized' if git_ok else 'NOT initialized'))
    except Exception:
        checks.append(('Git repo', False, 'git not available'))

    # 5. Experiments dir
    exp_dir = Path(__file__).parent / 'experiments'
    exp_dir.mkdir(parents=True, exist_ok=True)
    checks.append(('Experiments dir', True, str(exp_dir)))

    # 6. Dependencies
    deps_ok = True
    missing = []
    for pkg in ['pandas', 'numpy', 'numba', 'xgboost', 'rich', 'click', 'yaml', 'requests', 'joblib']:
        try:
            __import__(pkg)
        except ImportError:
            deps_ok = False
            missing.append(pkg)
    checks.append(('Python Dependencies', deps_ok,
                   'All installed' if deps_ok else f"Missing: {missing}"))

    # Show results
    console.print()
    for name, ok, msg in checks:
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {icon} {name:<25} {msg}")

    all_ok = all(ok for _, ok, _ in checks if _ != 'LLM Server')
    llm_check = next((ok for name, ok, _ in checks if name == 'LLM Server'), False)

    console.print()
    if all_ok and llm_check:
        console.print(Panel("[bold green]System ready to start![/bold green]\n"
                           "  python main.py run", border_style="green"))
    elif all_ok:
        console.print(Panel(
            "[yellow]System ready but LLM server is not running.[/yellow]\n"
            "Start the LLM server:\n"
            "  ./llm/llama.cpp/build/bin/llama-server \\\n"
            "      --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\\n"
            "      --port 8080 --n-gpu-layers 32 --ctx-size 8192\n\n"
            "Then: python main.py run",
            border_style="yellow",
        ))
    else:
        console.print(Panel("[red]Problems detected — fix before starting.[/red]",
                           border_style="red"))


if __name__ == '__main__':
    cli()
