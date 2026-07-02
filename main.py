#!/usr/bin/env python3
"""
algo_autoresearch — CLI principal

Comandos:
  run       — iniciar loop de pesquisa autónoma
  review    — revisão interativa do histórico
  tag       — adicionar tag a uma iteração
  analysis  — análise por tag + trend de score
  setup     — verificar pré-requisitos do sistema
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


def carregar_config(config_path: Path = None) -> dict:
    """Carrega config.yaml (ou config.yaml.example se não existir)."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    if not config_path.exists():
        example = config_path.parent / 'config.yaml.example'
        if example.exists():
            console.print(f"[yellow]config.yaml não encontrado. A usar config.yaml.example[/yellow]")
            config_path = example
        else:
            console.print("[red]Nenhum ficheiro de configuração encontrado![/red]")
            sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """algo_autoresearch — Loop de pesquisa autónomo de trading algorítmico."""
    pass


def _experiments_dir(base_dir: Path, season: int) -> Path:
    """Retorna o directório de experiências para a temporada indicada."""
    if season <= 1:
        return base_dir / 'experiments'
    return base_dir / f'experiments_s{season}'


@cli.command()
@click.option('--iters', '-n', default=0, help='Número máximo de iterações (0=infinito)')
@click.option('--review-interval', default=5, help='Pedir revisão a cada N iterações (0=desactivado)')
@click.option('--season', '-s', default=None, type=int,
              help='Temporada de pesquisa (1=experiments/, 2=experiments_s2/, ...)')
@click.option('--config', 'config_path', default=None, type=click.Path(), help='Path ao config.yaml')
def run(iters, review_interval, season, config_path):
    """Iniciar o loop de pesquisa autónoma."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)

    # Temporada: CLI > config.yaml > 1
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    # Verificar pré-requisitos
    from autoresearch.runner import verificar_pre_requisitos
    ok, erros = verificar_pre_requisitos(config)

    if not ok:
        console.print(Panel(
            '\n'.join(f"[red]✗[/red] {e}" for e in erros),
            title="Pré-requisitos em falta",
            border_style="red",
        ))
        console.print("\n[yellow]Correr 'python main.py setup' para diagnóstico completo.[/yellow]")
        sys.exit(1)

    console.print(Panel(
        f"[bold green]Pré-requisitos verificados — a iniciar loop[/bold green]\n"
        f"Temporada: S{season} → {experiments_dir}",
        border_style="green",
    ))

    from autoresearch.runner import executar_loop
    executar_loop(
        config=config,
        max_iteracoes=iters,
        human_review_interval=review_interval,
        experiments_dir=experiments_dir,
    )


@cli.command()
@click.option('--season', '-s', default=None, type=int, help='Temporada de pesquisa')
@click.option('--config', 'config_path', default=None, type=click.Path())
def review(season, config_path):
    """Revisão interativa do histórico de experiências."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    tracker.gerar_relatorio_analise()


@cli.command()
@click.option('--iter', 'iteracao', required=True, type=int, help='Número da iteração')
@click.option('--label', required=True, type=click.Choice([
    'promising', 'baseline', 'explorado', 'rejeitado', 'interessante', 'bug'
]), help='Tag a adicionar')
@click.option('--note', default='', help='Nota opcional')
@click.option('--season', '-s', default=None, type=int, help='Temporada de pesquisa')
@click.option('--config', 'config_path', default=None, type=click.Path())
def tag(iteracao, label, note, season, config_path):
    """Adicionar tag a uma iteração específica."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    ok = tracker.adicionar_tag(iteracao, label, note)
    if ok:
        console.print(f"[green]✓ Tag '{label}' adicionada à iteração {iteracao} (S{season})[/green]")


@cli.command()
@click.option('--season', '-s', default=None, type=int, help='Temporada de pesquisa')
@click.option('--config', 'config_path', default=None, type=click.Path())
def analysis(season, config_path):
    """Análise completa: tabela por tag + trend de score."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    tracker.gerar_relatorio_analise()


@cli.command('new-season')
@click.option('--config', 'config_path', default=None, type=click.Path())
@click.option('--dry-run', is_flag=True, help='Mostrar o que seria feito sem alterar nada')
def new_season(config_path, dry_run):
    """Transição para a próxima temporada: actualiza config.yaml e research_params.py."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)
    config_file = Path(config_path) if config_path else base_dir / 'config.yaml'

    season_atual = config.get('agent', {}).get('season', 1)
    season_nova  = season_atual + 1
    experiments_dir = _experiments_dir(base_dir, season_atual)

    console.print(Panel(
        f"[bold]Transição de temporada[/bold]\n"
        f"S{season_atual} → S{season_nova}\n"
        f"Experiências: {experiments_dir}",
        border_style="cyan",
    ))

    # Carregar melhor resultado da temporada actual
    from autoresearch.tracker import Tracker
    tracker = Tracker(experiments_dir)
    melhor = tracker.melhor_score()

    if melhor is None:
        console.print(f"[yellow]Sem resultados aceites em S{season_atual}. "
                      f"A incrementar temporada sem alterar baseline nem research_params.py.[/yellow]")
        best_score  = config.get('agent', {}).get('baseline_override', 0.0)
        best_params = None
    else:
        best_score  = melhor.metricas.get('score_composto', 0.0)
        best_params = melhor.params_snapshot
        m = melhor.metricas
        console.print(f"\n[green]Melhor resultado S{season_atual}:[/green]")
        console.print(f"  Iter {melhor.iteracao}  Score={best_score:.4f}  "
                      f"Sharpe={m.get('sharpe_raw',0):.2f}  "
                      f"Return={m.get('retorno_anual_pct',0):+.1f}%  "
                      f"DD={abs(m.get('max_drawdown_pct',0)):.1f}%")

    # --- 1. Actualizar config.yaml ---
    console.print(f"\n[bold]config.yaml:[/bold]")
    console.print(f"  agent.season:            {season_atual} → {season_nova}")
    console.print(f"  agent.baseline_override: → {best_score:.4f}")

    if not dry_run and config_file.exists():
        import yaml
        with open(config_file) as f:
            cfg_raw = yaml.safe_load(f)
        cfg_raw.setdefault('agent', {})
        cfg_raw['agent']['season']            = season_nova
        cfg_raw['agent']['baseline_override'] = round(best_score, 6)
        with open(config_file, 'w') as f:
            yaml.dump(cfg_raw, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        console.print(f"  [green]✓ config.yaml actualizado[/green]")

    # --- 2. Actualizar research_params.py ---
    params_path = base_dir / 'pipeline' / 'research_params.py'

    if best_params:
        novo_params = _gerar_research_params(best_params, season_nova, season_atual,
                                             melhor.iteracao, best_score)
        console.print(f"\n[bold]research_params.py:[/bold]")
        console.print(f"  Ponto de partida: iter {melhor.iteracao} S{season_atual} (score={best_score:.4f})")
        console.print(f"  FEATURES = {best_params.get('FEATURES', [])}")
        console.print(f"  TIMEFRAMES = {best_params.get('TIMEFRAMES', [])}")

        if not dry_run:
            import shutil
            shutil.copy2(params_path, params_path.with_suffix('.py.backup'))
            params_path.write_text(novo_params)
            console.print(f"  [green]✓ research_params.py actualizado (backup em .py.backup)[/green]")
    else:
        console.print(f"\n[dim]research_params.py: sem alterações (nenhum resultado aceite em S{season_atual})[/dim]")

    if dry_run:
        console.print(f"\n[yellow]--dry-run: nenhum ficheiro foi alterado.[/yellow]")
    else:
        console.print(Panel(
            f"[bold green]S{season_nova} pronta![/bold green]\n\n"
            f"Para iniciar:\n"
            f"  python main.py run",
            border_style="green",
        ))


def _gerar_research_params(ps: dict, season_nova: int, season_anterior: int,
                            iter_origem: int, score_origem: float) -> str:
    """Gera conteúdo de research_params.py a partir de um params_snapshot."""

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
        f"# ÚNICO FICHEIRO MODIFICADO PELO AGENTE",
        f"# REGRA CRÍTICA: APENAS INDICADORES RELATIVOS (sem preços absolutos)",
        f"#",
        f"# TEMPORADA {season_nova} — ponto de partida: melhor resultado S{season_anterior}",
        f"# (iter {iter_origem}, score={score_origem:.4f})",
        f"",
        f"# --- Features a usar no treino ---",
        f"FEATURES = {fmt(features)}",
        f"",
        f"# --- Timeframes a incluir ---",
        f"TIMEFRAMES = {fmt(timeframes)}",
        f"",
        f"# --- Parâmetros dos indicadores técnicos ---",
        f"STOCH_RSI_PERIOD = {ps.get('STOCH_RSI_PERIOD', 14)}",
        f"ADX_PERIOD = {ps.get('ADX_PERIOD', 14)}",
        f"EMA_FAST = {ps.get('EMA_FAST', 12)}",
        f"EMA_SLOW = {ps.get('EMA_SLOW', 26)}",
        f"BB_PERIOD = {ps.get('BB_PERIOD', 20)}",
        f"",
        f"# --- Sinal de entrada ---",
        f"ENTRY_STOCH_THRESHOLD = {ps.get('ENTRY_STOCH_THRESHOLD', 20)}",
        f"ENTRY_ADX_THRESHOLD = {ps.get('ENTRY_ADX_THRESHOLD', 25)}",
        f"",
        f"# --- Hiperparâmetros XGBoost ---",
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
        f"# --- Otimização Bayesiana XGBoost (Optuna) ---",
    ]

    # XGBoost Optuna ranges — manter se existiam, senão desactivar
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
        lines.append(f"N_TRIALS_XGB = 0  # activar com >= 20 e definir DEPTH_RANGE, LR_RANGE, etc.")

    sl_r  = ps.get('SL_RANGE', (0.5, 10.0))
    tp_r  = ps.get('TP_RANGE', (1.5, 25.0))
    thr_r = ps.get('THRESHOLD_RANGE', (0.1, 0.85))
    n_t   = ps.get('N_TRIALS', 140)

    lines += [
        f"",
        f"# --- Otimização Bayesiana SL/TP/Threshold ---",
        f"SL_RANGE = {fmt(sl_r)}",
        f"TP_RANGE = {fmt(tp_r)}",
        f"THRESHOLD_RANGE = {fmt(thr_r)}",
        f"N_TRIALS = {n_t}",
        f"",
        f"# --- Modo objetivo ---",
        f"OBJECTIVE_MODE = \"{ps.get('OBJECTIVE_MODE', 'score')}\"",
        f"",
        f"# --- Notas S{season_nova} ---",
        f"# O agente deve explorar novas dimensões a partir deste ponto de partida.",
    ]

    return '\n'.join(lines) + '\n'


@cli.command()
@click.option('--iter', 'iteracao', required=True, type=int,
              help='Iteração de referência para o modo exploit')
@click.option('--margin', default=0.35, show_default=True,
              help='Margem relativa para estreitar bounds (ex: 0.35 = ±35% em torno do ótimo)')
@click.option('--xgb-trials', default=0, type=int, show_default=True,
              help='Activar XGBoost Optuna com N trials (0=desactivado)')
@click.option('--season', '-s', default=None, type=int)
@click.option('--config', 'config_path', default=None, type=click.Path())
@click.option('--dry-run', is_flag=True, help='Mostrar o que seria feito sem alterar nada')
def focus(iteracao, margin, xgb_trials, season, config_path, dry_run):
    """Modo exploit: estreitar bounds em torno de um resultado promissor."""
    base_dir = Path(__file__).parent
    config = carregar_config(Path(config_path) if config_path else None)
    if season is None:
        season = config.get('agent', {}).get('season', 1)
    experiments_dir = _experiments_dir(base_dir, season)

    # Carregar iteração de referência
    import json
    iter_file = experiments_dir / f'iter_{iteracao:04d}.json'
    if not iter_file.exists():
        console.print(f"[red]Iteração {iteracao} não encontrada em S{season}[/red]")
        return

    e  = json.loads(iter_file.read_text())
    m  = e.get('metricas', {})
    ps = e.get('params_snapshot', {})

    sl_best  = m.get('sl_pct')
    tp_best  = m.get('tp_pct')
    thr_best = m.get('threshold')
    score    = m.get('score_composto', 0)

    console.print(Panel(
        f"[bold]Modo EXPLOIT — iter {iteracao} S{season}[/bold]\n"
        f"Score={score:.4f} | Sharpe={m.get('sharpe_raw',0):.2f} | "
        f"Return={m.get('retorno_anual_pct',0):+.1f}% | DD={abs(m.get('max_drawdown_pct',0)):.1f}%\n"
        f"Optuna best: SL={sl_best:.2f}% TP={tp_best:.2f}% Thr={thr_best:.3f}" if sl_best else
        f"Score={score:.4f} (sem Optuna best disponível)",
        border_style="yellow",
    ))

    # Calcular novos bounds estreitos em torno do ótimo
    params_path = base_dir / 'pipeline' / 'research_params.py'
    current = ps if ps else {}

    sl_range_orig  = current.get('SL_RANGE', (0.5, 10.0))
    tp_range_orig  = current.get('TP_RANGE', (1.5, 25.0))
    thr_range_orig = current.get('THRESHOLD_RANGE', (0.1, 0.85))

    if sl_best and tp_best and thr_best:
        # Estreitar bounds em torno do ótimo com margem configurável
        sl_lo  = max(0.05, sl_best * (1 - margin))
        sl_hi  = sl_best * (1 + margin)
        tp_lo  = max(0.1, tp_best * (1 - margin))
        tp_hi  = tp_best * (1 + margin)
        thr_lo = max(0.05, thr_best * (1 - margin))
        thr_hi = min(0.95, thr_best * (1 + margin))

        # Garantir que min < max (TP pode ser < SL em regime scalping)
        sl_range_novo  = (round(min(sl_lo, sl_hi), 3), round(max(sl_lo, sl_hi), 3))
        tp_range_novo  = (round(min(tp_lo, tp_hi), 3), round(max(tp_lo, tp_hi), 3))
        thr_range_novo = (round(thr_lo, 3), round(thr_hi, 3))
    else:
        sl_range_novo  = sl_range_orig
        tp_range_novo  = tp_range_orig
        thr_range_novo = thr_range_orig
        console.print("[yellow]Sem Optuna best nesta iteração — bounds não alterados[/yellow]")

    console.print(f"\n[bold]Novos bounds (margem ±{margin*100:.0f}%):[/bold]")
    console.print(f"  SL_RANGE:        {sl_range_orig} → {sl_range_novo}")
    console.print(f"  TP_RANGE:        {tp_range_orig} → {tp_range_novo}")
    console.print(f"  THRESHOLD_RANGE: {thr_range_orig} → {thr_range_novo}")
    if xgb_trials > 0:
        console.print(f"  N_TRIALS_XGB:    0 → {xgb_trials} (afinar XGBoost)")

    if not dry_run:
        import shutil
        shutil.copy2(params_path, params_path.with_suffix('.py.backup'))

        # Gerar research_params.py com bounds estreitos
        ps_focus = dict(ps)
        ps_focus['SL_RANGE']        = sl_range_novo
        ps_focus['TP_RANGE']        = tp_range_novo
        ps_focus['THRESHOLD_RANGE'] = thr_range_novo
        if xgb_trials > 0:
            ps_focus['N_TRIALS_XGB'] = xgb_trials
            ps_focus['DEPTH_RANGE']  = ps_focus.get('DEPTH_RANGE', (4, 9))
            ps_focus['LR_RANGE']     = ps_focus.get('LR_RANGE', (0.01, 0.12))
            ps_focus['ESTIMATORS_RANGE'] = ps_focus.get('ESTIMATORS_RANGE', (200, 600))
            ps_focus['ALPHA_RANGE']  = ps_focus.get('ALPHA_RANGE', (0.1, 2.5))
            ps_focus['LAMBDA_RANGE'] = ps_focus.get('LAMBDA_RANGE', (0.5, 3.5))

        novo_codigo = _gerar_research_params(
            ps_focus, season, season, iteracao, score
        )
        # Substituir o cabeçalho para indicar modo exploit
        novo_codigo = novo_codigo.replace(
            f"# TEMPORADA {season} — ponto de partida: melhor resultado S{season}",
            f"# TEMPORADA {season} — MODO EXPLOIT: a partir de iter {iteracao} (score={score:.4f})",
        )
        params_path.write_text(novo_codigo)
        console.print(f"\n[green]✓ research_params.py actualizado[/green]")

        # Actualizar program.md com instrução de exploit
        _actualizar_program_md_exploit(
            base_dir / 'program.md', iteracao, score, sl_range_novo, tp_range_novo,
            thr_range_novo, m, xgb_trials
        )
        console.print(f"[green]✓ program.md actualizado para modo exploit[/green]")
        console.print(Panel(
            f"[bold green]Pronto para explorar em torno de iter {iteracao}[/bold green]\n"
            f"Iniciar: python main.py run",
            border_style="green",
        ))
    else:
        console.print(f"\n[yellow]--dry-run: nenhum ficheiro alterado.[/yellow]")


def _actualizar_program_md_exploit(program_path: Path, iter_ref: int, score_ref: float,
                                    sl_r, tp_r, thr_r, metricas: dict, xgb_trials: int):
    """Reescreve a secção de prioridades do program.md para modo exploit."""
    exploit_section = f"""
---

## ⚡ MODO EXPLOIT ATIVO (iter {iter_ref}, score={score_ref:.4f})

O sistema está em modo de **exploração localizada**. Foca-te em:

1. **Afinar os bounds SL/TP/threshold** em torno da região ótima identificada:
   - SL_RANGE = {sl_r} — explorar variações pequenas
   - TP_RANGE = {tp_r} — explorar variações pequenas
   - THRESHOLD_RANGE = {thr_r} — o threshold é o parâmetro dominante (importância Optuna)

2. **Manter as features** que geraram este resultado — não adicionar nem remover

3. **Ajustar regularização XGBoost** se houver overfitting (REG_ALPHA, REG_LAMBDA)
{"4. **XGBoost Optuna activo** (" + str(xgb_trials) + " trials) — afinar hiperparâmetros" if xgb_trials > 0 else "4. **N_TRIALS_XGB = 0** — manter desactivado para velocidade"}

Referência: Sharpe={metricas.get('sharpe_raw',0):.2f} | Return={metricas.get('retorno_anual_pct',0):+.1f}% | DD={abs(metricas.get('max_drawdown_pct',0)):.1f}%

---
"""
    content = program_path.read_text()
    # Remover secção de exploit anterior se existir
    import re
    content = re.sub(r'\n---\n\n## ⚡ MODO EXPLOIT.*?---\n', '', content, flags=re.DOTALL)
    # Inserir no início, depois do cabeçalho principal (após o primeiro ---)
    parts = content.split('---', 1)
    if len(parts) == 2:
        content = parts[0] + '---' + exploit_section + parts[1]
    else:
        content = content + exploit_section
    program_path.write_text(content)


@cli.command()
@click.option('--config', 'config_path', default=None, type=click.Path())
def setup(config_path):
    """Verificar todos os pré-requisitos do sistema."""
    console.print(Panel("[bold]algo_autoresearch — Verificação do Sistema[/bold]", border_style="blue"))

    config = carregar_config(Path(config_path) if config_path else None)

    checks = []

    # 1. LLM Server
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    from autoresearch.agent import verificar_servidor_llm
    llm_ok = verificar_servidor_llm(server_url)
    checks.append(('LLM Server', llm_ok,
                   f"Acessível em {server_url}" if llm_ok
                   else f"NÃO acessível em {server_url}"))

    # 2. research_params.py
    params_path = Path(__file__).parent / 'pipeline' / 'research_params.py'
    from autoresearch.agent import validar_codigo
    if params_path.exists():
        ok, msg = validar_codigo(params_path.read_text())
        checks.append(('research_params.py', ok, msg if ok else f"Inválido: {msg}"))
    else:
        checks.append(('research_params.py', False, 'Ficheiro não encontrado'))

    # 3. Dados
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
                checks.append(('Dados 15m', True, str(data_dir / cand)))
                break
        if not found:
            checks.append(('Dados 15m', False,
                           f"Não encontrado para {ticker} em {data_dir}"))
    except Exception as e:
        checks.append(('Dados', False, str(e)))

    # 4. Git
    import subprocess
    try:
        r = subprocess.run(['git', 'rev-parse', '--git-dir'],
                          capture_output=True, cwd=Path(__file__).parent)
        git_ok = r.returncode == 0
        checks.append(('Git repo', git_ok, 'Inicializado' if git_ok else 'NÃO inicializado'))
    except Exception:
        checks.append(('Git repo', False, 'git não disponível'))

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
    checks.append(('Dependências Python', deps_ok,
                   'Todas instaladas' if deps_ok else f"Em falta: {missing}"))

    # Mostrar resultados
    console.print()
    for nome, ok, msg in checks:
        icone = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {icone} {nome:<25} {msg}")

    tudo_ok = all(ok for _, ok, _ in checks if _ != 'LLM Server')
    llm_check = next((ok for nome, ok, _ in checks if nome == 'LLM Server'), False)

    console.print()
    if tudo_ok and llm_check:
        console.print(Panel("[bold green]Sistema pronto para iniciar![/bold green]\n"
                           "  python main.py run", border_style="green"))
    elif tudo_ok:
        console.print(Panel(
            "[yellow]Sistema pronto mas LLM server não está a correr.[/yellow]\n"
            "Iniciar o servidor LLM:\n"
            "  ./llm/llama.cpp/build/bin/llama-server \\\n"
            "      --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\\n"
            "      --port 8080 --n-gpu-layers 32 --ctx-size 8192\n\n"
            "Depois: python main.py run",
            border_style="yellow",
        ))
    else:
        console.print(Panel("[red]Problemas detectados — corrigir antes de iniciar.[/red]",
                           border_style="red"))


if __name__ == '__main__':
    cli()
