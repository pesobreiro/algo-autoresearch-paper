"""
Runner principal — loop de pesquisa autónoma.

Fluxo por iteração:
  1. Carregar research_params.py atual
  2. LLM propor novo research_params.py
  3. Validar sintaxe + indicadores relativos
  4. Correr pipeline (labels → treino → backtest)
  5. Calcular score e comparar com baseline
  6. Aceitar ou reverter
  7. Registar experiência
  8. [Opcional] Revisão humana
"""
import gc
import shutil
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from autoresearch.agent import (
    propor_novos_params, validar_codigo, verificar_servidor_llm
)
from autoresearch.tracker import Tracker, RegistoExperiencia
from autoresearch.human_loop import (
    mostrar_resultado_iteracao, mostrar_params_propostos, solicitar_revisao_humana
)
from pipeline.run_pipeline import executar_pipeline, carregar_params, hash_entry_params, hash_params_completo

console = Console()


def verificar_pre_requisitos(config: dict) -> tuple[bool, list[str]]:
    """
    Verifica pré-requisitos antes de iniciar o loop.

    Returns:
        (ok, lista_de_erros)
    """
    erros = []

    # Servidor LLM
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    if not verificar_servidor_llm(server_url):
        erros.append(f"LLM server não acessível em {server_url}\n"
                     f"  Iniciar com: ./llm/llama.cpp/build/bin/llama-server "
                     f"--model models/*.gguf --port 8080 --n-gpu-layers 32")

    # Dados
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        import ml_sessions_compat.config as ml_config
        data_dir = ml_config.DATA_DIR
        ticker = config['pipeline']['ticker']
        exchange = config['pipeline'].get('exchange', 'binance')

        f15m = None
        for cand in [f'{ticker}_15m_usdt_{exchange}.parquet',
                     f'{ticker}_15m_usdt_binance.parquet',
                     f'{ticker}_15m_usdt.parquet']:
            p = Path(data_dir) / cand
            if p.exists():
                f15m = p
                break
        if f15m is None:
            erros.append(f"Dados 15m não encontrados para {ticker} em {data_dir}")
    except Exception as e:
        erros.append(f"Erro ao verificar dados: {e}")

    # research_params.py
    params_path = Path(__file__).parent.parent / 'pipeline' / 'research_params.py'
    if not params_path.exists():
        erros.append(f"research_params.py não encontrado: {params_path}")
    else:
        try:
            params = carregar_params(params_path)
            ok, msg = validar_codigo(params_path.read_text())
            if not ok:
                erros.append(f"research_params.py inválido: {msg}")
        except Exception as e:
            erros.append(f"Erro ao carregar research_params.py: {e}")

    return len(erros) == 0, erros


def limpar_cache(cache_dir: Path, params_hash_atual: str,
                 modelo_hash_atual: str, keep_labels: int = 3, keep_models: int = 5):
    """
    Remove labels e modelos antigos do cache.

    Mantém:
      - Os últimos `keep_labels` ficheiros de labels por data de modificação
      - Os últimos `keep_models` dirs de modelos por data de modificação
      - Sempre preserva o hash actual (labels + modelo)
    """
    labels_dir = cache_dir / 'labels'
    models_dir = cache_dir / 'models'

    # --- Labels ---
    if labels_dir.exists():
        parquets = sorted(labels_dir.glob('*.parquet'), key=lambda f: f.stat().st_mtime)
        # preservar o actual e os mais recentes
        para_apagar = [f for f in parquets
                       if params_hash_atual not in f.name][:-keep_labels]
        for f in para_apagar:
            f.unlink(missing_ok=True)
        if para_apagar:
            console.print(f"  [dim]Cache: apagados {len(para_apagar)} labels antigos[/dim]")

    # --- Modelos ---
    if models_dir.exists():
        model_dirs = sorted(
            [d for d in models_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        para_apagar = [d for d in model_dirs
                       if d.name != modelo_hash_atual][:-keep_models]
        for d in para_apagar:
            import shutil as _shutil
            _shutil.rmtree(d, ignore_errors=True)
        if para_apagar:
            console.print(f"  [dim]Cache: apagados {len(para_apagar)} modelos antigos[/dim]")


def _guardar_melhor_modelo(cache_dir: Path, model_hash: str, iteracao: int,
                           season: int, metricas: dict, params: dict):
    """Copia o modelo aceite para best_models/season_N/ para não ser apagado pelo cleanup."""
    src = cache_dir / 'models' / model_hash
    if not src.exists():
        return

    base_dir = cache_dir.parent
    dest = base_dir / 'best_models' / f'season_{season}' / f'iter_{iteracao:04d}'
    dest.mkdir(parents=True, exist_ok=True)

    import shutil as _shutil
    _shutil.copytree(src, dest / 'model', dirs_exist_ok=True)

    # Guardar metadados junto ao modelo
    import json as _json
    meta = {
        'iteracao': iteracao,
        'season': season,
        'model_hash': model_hash,
        'metricas': metricas,
        'params': {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in params.items()},
    }
    (dest / 'meta.json').write_text(_json.dumps(meta, indent=2, ensure_ascii=False))
    console.print(f"  [dim]Modelo guardado em best_models/season_{season}/iter_{iteracao:04d}/[/dim]")


def _actualizar_program_md_top(program_path: Path, top_registos: list[dict]):
    """Actualiza a secção TOP RESULTADOS no program.md após cada iteração."""
    if not top_registos:
        return

    linhas = ["## 📊 Melhores Resultados Actuais (auto-actualizado)\n\n"]
    sl_vals, tp_vals, thr_vals = [], [], []
    tfs_counter: dict = {}

    for i, h in enumerate(top_registos):
        m = h.get('metricas', {})
        ps = h.get('params_snapshot', {})
        sl  = m.get('sl_pct')
        tp  = m.get('tp_pct')
        thr = m.get('threshold')
        tfs = ps.get('TIMEFRAMES', '?')
        score = m.get('score_composto', 0)
        sv    = m.get('sharpe_validation')
        sh    = m.get('sharpe_holdout')
        auc   = m.get('cv_auc_mean', 0)
        if sv is not None:
            score_fmt = (f"AUC={auc:.3f} | Sharpe(val)={sv:.2f} | "
                         f"Sharpe(holdout)={sh:.2f}" if sh is not None else f"AUC={auc:.3f} | Sharpe(val)={sv:.2f}")
        else:
            score_fmt = f"score={score:.4f}"
        linhas.append(
            f"**#{i+1} iter={h.get('iteracao','?')}** {score_fmt} | "
            f"DD={abs(m.get('max_drawdown_pct',0)):.1f}% | Trades={m.get('n_trades',0)} | "
            f"WR={m.get('win_rate_pct',0):.1f}%  \n"
        )
        if sl and tp and thr:
            linhas.append(f"→ SL={sl:.2f}% TP={tp:.2f}% Thr={thr:.3f} | TFs={tfs} | "
                          f"Entry: stoch<{ps.get('ENTRY_STOCH_THRESHOLD','?')} adx>{ps.get('ENTRY_ADX_THRESHOLD','?')}  \n\n")
            sl_vals.append(sl); tp_vals.append(tp); thr_vals.append(thr)
        tfs_key = str(tfs)
        tfs_counter[tfs_key] = tfs_counter.get(tfs_key, 0) + 1

    if sl_vals:
        tfs_dom = max(tfs_counter, key=tfs_counter.get)
        linhas.append(
            f"**Padrão dominante:** SL={sum(sl_vals)/len(sl_vals):.1f}% "
            f"(±{(max(sl_vals)-min(sl_vals))/2:.1f}) | "
            f"TP={sum(tp_vals)/len(tp_vals):.1f}% (±{(max(tp_vals)-min(tp_vals))/2:.1f}) | "
            f"Thr={sum(thr_vals)/len(thr_vals):.2f} | TFs={tfs_dom}  \n"
            f"→ Explora variações de features e entry signal nesta zona. Não repitas params iguais.\n"
        )

    secao = "\n---\n\n" + "".join(linhas) + "\n---\n"

    # Substituir secção existente ou inserir no fim
    texto = program_path.read_text() if program_path.exists() else ""
    import re as _re
    if "## 📊 Melhores Resultados Actuais" in texto:
        texto = _re.sub(
            r'\n---\n\n## 📊 Melhores Resultados Actuais.*?\n---\n',
            secao, texto, flags=_re.DOTALL
        )
    else:
        texto = texto.rstrip() + "\n" + secao
    program_path.write_text(texto)


def executar_loop(config: dict, max_iteracoes: int = 0,
                  human_review_interval: int = 5,
                  experiments_dir: Path = None,
                  cache_dir: Path = None):
    """
    Loop principal de pesquisa autónoma.

    Args:
        config: configuração do sistema
        max_iteracoes: 0 = infinito
        human_review_interval: pedir revisão a cada N iterações (0 = desactivado)
        experiments_dir: onde guardar experiências
        cache_dir: cache de labels e modelos
    """
    import signal

    _parar = False

    def _handler_sigint(sig, frame):
        nonlocal _parar
        if not _parar:
            console.print("\n[bold yellow]Ctrl+C recebido — a terminar após a iteração atual...[/bold yellow]")
            _parar = True
        else:
            console.print("\n[bold red]Ctrl+C forçado — a sair imediatamente.[/bold red]")
            raise SystemExit(1)

    signal.signal(signal.SIGINT, _handler_sigint)

    base_dir = Path(__file__).parent.parent
    params_path = base_dir / 'pipeline' / 'research_params.py'
    backup_path = base_dir / 'pipeline' / 'research_params.py.backup'
    program_path = base_dir / 'program.md'

    if experiments_dir is None:
        experiments_dir = base_dir / 'experiments'
    if cache_dir is None:
        cache_dir = base_dir / 'cache'

    tracker = Tracker(experiments_dir)
    iteracao = tracker.proximo_numero_iteracao()
    params_anteriores = None

    # Carregar score do último aceite como baseline, ou usar baseline_override do config
    baseline_override  = config.get('agent', {}).get('baseline_override', 0.0)
    accept_auc_min          = config.get('agent', {}).get('accept_auc_min', 0.0)
    accept_sharpe_min       = config.get('agent', {}).get('accept_sharpe_min', 0.0)
    accept_sharpe_holdout_min = config.get('agent', {}).get('accept_sharpe_holdout_min', 0.0)
    ultimo = tracker.ultimo_aceite()
    if ultimo:
        ultimo_mode = (ultimo.params_snapshot or {}).get('OBJECTIVE_MODE', 'score')
        if ultimo_mode == 'profit':
            score_baseline = ultimo.metricas.get('retorno_total_oos_pct', baseline_override)
        else:
            sv = ultimo.metricas.get('sharpe_validation')
            score_baseline = float(sv) if sv is not None else ultimo.metricas.get('score_composto', 0.0)
        params_anteriores = ultimo.params_snapshot
        console.print(f"[cyan]Retomando pesquisa. Última aceite: iter {ultimo.iteracao}, "
                      f"score={score_baseline:.4f}[/cyan]")
    else:
        score_baseline = baseline_override
        if baseline_override > 0:
            console.print(f"[cyan]Nova temporada. Baseline mínimo definido: {baseline_override:.4f}[/cyan]")
        else:
            score_baseline = 0.0

    console.print(Panel(
        f"[bold green]algo_autoresearch — Loop de Pesquisa[/bold green]\n"
        f"Iteração inicial: {iteracao}\n"
        f"Score baseline: {score_baseline:.4f}\n"
        f"Max iterações: {'∞' if max_iteracoes == 0 else max_iteracoes}",
        border_style="green",
    ))

    accept_threshold = config.get('agent', {}).get('accept_threshold', 0.01)
    revert_on_worse  = config.get('agent', {}).get('revert_on_worse', True)

    # --- Temperature curriculum ---
    t_base  = config.get('llm', {}).get('temperature', 0.7)
    t_min   = config.get('llm', {}).get('t_min', 0.3)
    t_max   = config.get('llm', {}).get('t_max', 1.2)
    t_decay = config.get('llm', {}).get('t_decay', 0.92)   # quando melhora → exploit
    t_grow  = config.get('llm', {}).get('t_grow', 1.08)    # quando estagna → explore
    stagnation_threshold = config.get('llm', {}).get('stagnation_threshold', 5)
    temp_atual         = t_base
    iters_sem_melhoria = 0

    rejeicoes_recentes: list[str] = []  # últimas rejeições de validação

    while True:
        if _parar:
            console.print("[bold yellow]Loop terminado por Ctrl+C.[/bold yellow]")
            break

        if max_iteracoes > 0 and iteracao > max_iteracoes:
            console.print("[bold]Número máximo de iterações atingido.[/bold]")
            break

        console.print(f"\n[bold cyan]══ Iteração {iteracao} ══[/bold cyan]  "
                      f"[dim]temp={temp_atual:.2f} | sem_melhoria={iters_sem_melhoria}[/dim]")

        # --- 1. Carregar params atuais ---
        codigo_atual = params_path.read_text()
        params_atuais = carregar_params(params_path)

        # --- 2. LLM propor novos params ---
        console.print("  [dim]A consultar LLM...[/dim]")
        program_md = program_path.read_text() if program_path.exists() else ""
        historico  = tracker.listar_historico(30)
        melhor     = tracker.melhor_score()
        melhor_dict = melhor.to_dict() if melhor else None
        top5        = tracker.top_n_scores(10)

        codigo_proposto = propor_novos_params(
            codigo_atual, program_md, historico, config,
            melhor_registo=melhor_dict,
            top_registos=top5,
            rejeicoes_recentes=rejeicoes_recentes,
            temperature=temp_atual,
        )

        if codigo_proposto is None:
            console.print("  [red]LLM não retornou código válido. A manter params atuais.[/red]")
            # Correr com params atuais mesmo assim
            codigo_proposto = codigo_atual

        # --- 3. Validar ---
        ok, msg = validar_codigo(codigo_proposto)
        if not ok:
            console.print(f"  [red]Código rejeitado: {msg}[/red]")
            rejeicoes_recentes.append(msg)
            rejeicoes_recentes = rejeicoes_recentes[-5:]  # manter apenas últimas 5
            registo = tracker.criar_registo(
                iteracao=iteracao,
                status='rejeitado',
                metricas={'score_composto': 0.0},
                params_hash='invalid',
                labels_reutilizados=False,
                duracao=0.0,
                alteracoes=f"REJEITADO: {msg}",
            )
            tracker.guardar_experiencia(registo)
            iteracao += 1
            continue

        # Validação passou — limpar rejeições pendentes
        rejeicoes_recentes.clear()

        # Auto-corrigir N_TRIALS_XGB: forçar 0 salvo em modo exploit ou opção B
        exploit_mode = "MODO EXPLOIT ATIVO" in program_md or "OPÇÃO B" in program_md
        if not exploit_mode:
            import re as _re
            codigo_corrigido = _re.sub(
                r'(N_TRIALS_XGB\s*=\s*)\d+',
                r'\g<1>0  # forçado a 0 pelo runner (modo exploração)',
                codigo_proposto
            )
            if codigo_corrigido != codigo_proposto:
                console.print("  [dim]Auto-correcção: N_TRIALS_XGB → 0 (modo exploração)[/dim]")
                codigo_proposto = codigo_corrigido

        # Mostrar o que o agente propõe
        mostrar_params_propostos(codigo_atual, codigo_proposto)

        # --- 4. Guardar backup e aplicar ---
        shutil.copy2(params_path, backup_path)
        params_path.write_text(codigo_proposto)

        params_novos = carregar_params(params_path)
        hash_novo = hash_params_completo(params_novos)

        # --- 4b. Early stopping: rejeitar configuração já explorada ---
        if tracker.hash_ja_explorado(hash_novo):
            console.print(f"  [yellow]✗ DUPLICADO: configuração já testada (hash={hash_novo[:8]}) — a saltar pipeline[/yellow]")
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)
            msg_dup = f"Configuração duplicada (hash={hash_novo[:8]}) — propor variação diferente"
            rejeicoes_recentes.append(msg_dup)
            rejeicoes_recentes = rejeicoes_recentes[-5:]
            iters_sem_melhoria += 1
            if iters_sem_melhoria >= stagnation_threshold:
                temp_anterior = temp_atual
                temp_atual = min(t_max, temp_atual * t_grow)
                console.print(f"  [dim]temp {temp_anterior:.2f}→{temp_atual:.2f} (explore — stagnação)[/dim]")
            registo = tracker.criar_registo(
                iteracao=iteracao,
                status='rejeitado',
                metricas={'score_composto': 0.0},
                params_hash=hash_novo,
                labels_reutilizados=False,
                duracao=0.0,
                alteracoes=f"DUPLICADO: {hash_novo[:8]}",
            )
            tracker.guardar_experiencia(registo)
            iteracao += 1
            continue

        alteracoes = tracker.calcular_alteracoes(
            params_anteriores or params_atuais,
            params_novos
        )
        console.print(f"  Alterações: [yellow]{alteracoes}[/yellow]")

        # --- 5. Correr pipeline ---
        t_inicio = time.time()
        resultado = executar_pipeline(config, params_path, cache_dir)
        duracao = time.time() - t_inicio

        # --- 6. Calcular score e decidir ---
        objective_mode    = params_novos.get('OBJECTIVE_MODE', 'score')
        auc_atual         = resultado.metricas.get('cv_auc_mean', 0.0) if resultado.sucesso else 0.0
        sharpe_validation = resultado.metricas.get('sharpe_validation', 0.0) if resultado.sucesso else 0.0
        sharpe_holdout_v  = resultado.metricas.get('sharpe_holdout', 0.0) if resultado.sucesso else 0.0

        if objective_mode == 'profit':
            score_atual = resultado.metricas.get('retorno_total_oos_pct', -999.0) if resultado.sucesso else -999.0
            melhorou = resultado.sucesso and (score_atual > score_baseline + accept_threshold)
        else:
            score_atual   = sharpe_validation
            gate_auc      = auc_atual >= accept_auc_min
            gate_sharpe   = sharpe_validation >= accept_sharpe_min
            gate_holdout  = sharpe_holdout_v >= accept_sharpe_holdout_min
            melhorou      = resultado.sucesso and gate_auc and gate_sharpe and gate_holdout

        # Detetar resultado Optuna já encontrado (mesmo ótimo local, params diferentes)
        if resultado.sucesso and tracker.resultado_ja_encontrado(resultado.metricas):
            sl  = resultado.metricas.get('sl_pct', '?')
            tp  = resultado.metricas.get('tp_pct', '?')
            thr = resultado.metricas.get('threshold', '?')
            console.print(f"  [yellow]✗ RESULTADO DUPLICADO: Optuna convergiu para o mesmo ótimo "
                          f"(SL={sl:.2f}% TP={tp:.2f}% T={thr:.2f}) — escapar desta zona[/yellow]")
            msg_res = f"Resultado duplicado: Optuna encontrou SL={sl:.2f}% TP={tp:.2f}% T={thr:.2f} — mudar features ou TIMEFRAMES para escapar"
            rejeicoes_recentes.append(msg_res)
            rejeicoes_recentes = rejeicoes_recentes[-5:]
            iters_sem_melhoria += 1
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)
            registo = tracker.criar_registo(
                iteracao=iteracao,
                status='rejeitado',
                metricas=resultado.metricas,
                params_hash=hash_params_completo(params_novos),
                labels_reutilizados=resultado.labels_reutilizados,
                duracao=duracao,
                alteracoes=f"RESULTADO DUPLICADO: SL={sl:.2f}% TP={tp:.2f}%",
                params_snapshot=params_novos,
            )
            tracker.guardar_experiencia(registo)
            iteracao += 1
            continue

        mostrar_resultado_iteracao(iteracao, resultado, score_baseline)

        if resultado.sucesso and melhorou:
            status = 'aceite'
            params_anteriores = params_novos
            # Melhoria → reduzir temperatura (exploit zona boa)
            temp_anterior = temp_atual
            temp_atual = max(t_min, temp_atual * t_decay)
            iters_sem_melhoria = 0
            if objective_mode == 'profit':
                console.print(f"  [bold green]✓ ACEITE (retorno {score_baseline:+.1f}%)[/bold green]  "
                              f"[dim]temp {temp_anterior:.2f}→{temp_atual:.2f} (decay)[/dim]")
            else:
                console.print(f"  [bold green]✓ ACEITE | AUC={auc_atual:.3f} | "
                              f"Sharpe(val)={sharpe_validation:.2f} | "
                              f"Sharpe(holdout/passivo)={sharpe_holdout_v:.2f}[/bold green]  "
                              f"[dim]temp {temp_anterior:.2f}→{temp_atual:.2f} (decay)[/dim]")
            # Preservar modelo aceite de limpeza pelo cleanup
            season = config.get('agent', {}).get('season', 0)
            _guardar_melhor_modelo(cache_dir, hash_novo, iteracao, season,
                                   resultado.metricas, params_novos)
        else:
            status = 'rejeitado' if resultado.sucesso else 'erro'
            if objective_mode == 'profit':
                razao = (f"retorno {score_atual:+.1f}% ≤ baseline {score_baseline:+.1f}% + {accept_threshold}%"
                         if resultado.sucesso else resultado.erro)
            else:
                if resultado.sucesso:
                    razao = (f"AUC={auc_atual:.3f}(≥{accept_auc_min}) | "
                             f"Sharpe(val)={sharpe_validation:.2f}(≥{accept_sharpe_min}) | "
                             f"Sharpe(holdout)={sharpe_holdout_v:.2f}(≥{accept_sharpe_holdout_min})")
                else:
                    razao = resultado.erro
            # Sem melhoria → contar; se ultrapassar threshold, aumentar temperatura (explore)
            iters_sem_melhoria += 1
            if iters_sem_melhoria >= stagnation_threshold:
                temp_anterior = temp_atual
                temp_atual = min(t_max, temp_atual * t_grow)
                console.print(f"  [yellow]✗ REVERTIDO ({razao})[/yellow]  "
                              f"[dim]temp {temp_anterior:.2f}→{temp_atual:.2f} (explore)[/dim]")
            else:
                console.print(f"  [yellow]✗ REVERTIDO ({razao})[/yellow]  "
                              f"[dim]{iters_sem_melhoria}/{stagnation_threshold} sem melhoria[/dim]")
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)

        # --- 7. Limpar cache a cada 10 iterações ---
        if iteracao % 10 == 0:
            _train_start = config['pipeline'].get('train_start', 2017)
            _train_end   = config['pipeline'].get('train_end', 2024)
            limpar_cache(
                cache_dir,
                params_hash_atual=hash_entry_params(params_novos, train_start=_train_start, train_end=_train_end),
                modelo_hash_atual=hash_params_completo(params_novos),
            )

        # --- 8. Registar ---
        registo = tracker.criar_registo(
            iteracao=iteracao,
            status=status,
            metricas=resultado.metricas if resultado.sucesso else {'score_composto': score_atual},
            params_hash=hash_params_completo(params_novos),
            labels_reutilizados=resultado.labels_reutilizados,
            duracao=duracao,
            alteracoes=alteracoes,
            params_snapshot=params_novos,
        )
        tracker.guardar_experiencia(registo)

        # --- 8c. Libertar memória acumulada (mitigação OOM S7) ---
        gc.collect()

        # --- 8b. Actualizar program.md com top resultados (relido em cada iteração) ---
        _actualizar_program_md_top(program_path, tracker.top_n_scores(10))

        # --- 9. Revisão humana periódica ---
        if human_review_interval > 0 and iteracao % human_review_interval == 0:
            acao = solicitar_revisao_humana(tracker, params_path, iteracao, config)
            if acao.get('acao') == 'sair':
                console.print("[bold]A terminar por pedido do utilizador.[/bold]")
                break
            elif acao.get('acao') == 'injetar':
                # Usar params injetados manualmente
                console.print("  [cyan]Usando params injetados manualmente.[/cyan]")

        iteracao += 1
