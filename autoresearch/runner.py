"""
Main runner — autonomous research loop.

Flow per iteration:
  1. Load current research_params.py
  2. LLM proposes new research_params.py
  3. Validate syntax + relative indicators
  4. Run pipeline (labels → train → backtest)
  5. Calculate score and compare with baseline
  6. Accept or revert
  7. Register experience
  8. [Optional] Human review
"""
import gc
import shutil
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from autoresearch.agent import (
    propose_new_params, validate_code, check_llm_server
)
from autoresearch.tracker import Tracker, ExperimentRecord
from autoresearch.human_loop import (
    show_iteration_result, show_proposed_params, request_human_review
)
from pipeline.run_pipeline import executar_pipeline, carregar_params, hash_entry_params, hash_params_completo

console = Console()


def _fmt_num(v, decimals=2, sign=False):
    """Formats a number with `decimals` decimal places; non-numeric values are returned as string."""
    if isinstance(v, (int, float)):
        fmt = f"{{:{'+' if sign else ''}.{decimals}f}}"
        return fmt.format(v)
    return str(v)


def check_prerequisites(config: dict) -> tuple[bool, list[str]]:
    """
    Checks prerequisites before starting the loop.

    Returns:
        (ok, error_list)
    """
    errors = []

    # LLM Server
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    if not check_llm_server(server_url):
        errors.append(f"LLM server not accessible at {server_url}\n"
                     f"  Start with: ./llm/llama.cpp/build/bin/llama-server "
                     f"--model models/*.gguf --port 8080 --n-gpu-layers 32")

    # Data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        import ml_sessions_compat.config as ml_config
        data_dir = ml_config.DATA_DIR
        ticker = config['pipeline']['ticker']
        exchange = config['pipeline'].get('exchange', 'binance')

        file_15m = None
        for cand in [f'{ticker}_15m_usdt_{exchange}.parquet',
                     f'{ticker}_15m_usdt_binance.parquet',
                     f'{ticker}_15m_usdt.parquet']:
            p = Path(data_dir) / cand
            if p.exists():
                file_15m = p
                break
        if file_15m is None:
            errors.append(f"15m data not found for {ticker} in {data_dir}")
    except Exception as e:
        errors.append(f"Error checking data: {e}")

    # research_params.py
    params_path = Path(__file__).parent.parent / 'pipeline' / 'research_params.py'
    if not params_path.exists():
        errors.append(f"research_params.py not found: {params_path}")
    else:
        try:
            params = carregar_params(params_path)
            ok, message = validate_code(params_path.read_text())
            if not ok:
                errors.append(f"research_params.py invalid: {message}")
        except Exception as e:
            errors.append(f"Error loading research_params.py: {e}")

    return len(errors) == 0, errors


# Backward-compatible alias used by main.py
verificar_pre_requisitos = check_prerequisites


def clean_cache(cache_dir: Path, current_params_hash: str,
                current_model_hash: str, keep_labels: int = 3, keep_models: int = 5):
    """
    Removes old labels and models from cache.

    Keeps:
      - The latest `keep_labels` label files by modification time
      - The latest `keep_models` model dirs by modification time
      - Always preserves the current hash (labels + model)
    """
    labels_dir = cache_dir / 'labels'
    models_dir = cache_dir / 'models'

    # --- Labels ---
    if labels_dir.exists():
        parquets = sorted(labels_dir.glob('*.parquet'), key=lambda f: f.stat().st_mtime)
        # preserve current and most recent
        to_delete = [f for f in parquets
                       if current_params_hash not in f.name][:-keep_labels]
        for f in to_delete:
            f.unlink(missing_ok=True)
        if to_delete:
            console.print(f"  [dim]Cache: deleted {len(to_delete)} old labels[/dim]")

    # --- Models ---
    if models_dir.exists():
        model_dirs = sorted(
            [d for d in models_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        to_delete = [d for d in model_dirs
                       if d.name != current_model_hash][:-keep_models]
        for d in to_delete:
            import shutil as _shutil
            _shutil.rmtree(d, ignore_errors=True)
        if to_delete:
            console.print(f"  [dim]Cache: deleted {len(to_delete)} old models[/dim]")


def _save_best_model(cache_dir: Path, model_hash: str, iteration: int,
                     season: int, metrics: dict, params: dict):
    """Copies the accepted model to best_models/season_N/ so it is not deleted by cleanup."""
    src = cache_dir / 'models' / model_hash
    if not src.exists():
        return

    base_dir = cache_dir.parent
    dest = base_dir / 'best_models' / f'season_{season}' / f'iter_{iteration:04d}'
    dest.mkdir(parents=True, exist_ok=True)

    import shutil as _shutil
    _shutil.copytree(src, dest / 'model', dirs_exist_ok=True)

    # Save metadata alongside the model
    import json as _json
    meta = {
        'iteracao': iteration,
        'season': season,
        'model_hash': model_hash,
        'metricas': metrics,
        'params': {k: list(v) if isinstance(v, (list, tuple)) else v for k, v in params.items()},
    }
    (dest / 'meta.json').write_text(_json.dumps(meta, indent=2, ensure_ascii=False))
    console.print(f"  [dim]Model saved to best_models/season_{season}/iter_{iteration:04d}/[/dim]")


def _update_program_md_top(program_path: Path, top_records: list[dict]):
    """Updates the TOP RESULTS section in program.md after each iteration."""
    if not top_records:
        return

    lines = ["## 📊 Melhores Resultados Actuais (auto-actualizado)\n\n"]
    sl_vals, tp_vals, thr_vals = [], [], []
    tfs_counter: dict = {}

    for i, h in enumerate(top_records):
        metrics = h.get('metricas', {})
        ps = h.get('params_snapshot', {})
        sl  = metrics.get('sl_pct')
        tp  = metrics.get('tp_pct')
        thr = metrics.get('threshold')
        tfs = ps.get('TIMEFRAMES', '?')
        score = metrics.get('score_composto', 0)
        sv    = metrics.get('sharpe_validation')
        sh    = metrics.get('sharpe_holdout')
        auc   = metrics.get('cv_auc_mean', 0)
        if sv is not None:
            score_fmt = (f"AUC={_fmt_num(auc, 3)} | Sharpe(val)={_fmt_num(sv, 2)} | "
                         f"Sharpe(holdout)={_fmt_num(sh, 2)}" if sh is not None else f"AUC={_fmt_num(auc, 3)} | Sharpe(val)={_fmt_num(sv, 2)}")
        else:
            score_fmt = f"score={_fmt_num(score, 4)}"
        lines.append(
            f"**#{i+1} iter={h.get('iteracao','?')}** {score_fmt} | "
            f"DD={_fmt_num(abs(metrics.get('max_drawdown_pct', 0)), 1)}% | Trades={metrics.get('n_trades', 0)} | "
            f"WR={_fmt_num(metrics.get('win_rate_pct', 0), 1)}%  \n"
        )
        if isinstance(sl, (int, float)) and isinstance(tp, (int, float)) and isinstance(thr, (int, float)):
            lines.append(f"→ SL={_fmt_num(sl, 2)}% TP={_fmt_num(tp, 2)}% Thr={_fmt_num(thr, 3)} | TFs={tfs} | "
                          f"Entry: stoch<{ps.get('ENTRY_STOCH_THRESHOLD','?')} adx>{ps.get('ENTRY_ADX_THRESHOLD','?')}  \n\n")
            sl_vals.append(sl); tp_vals.append(tp); thr_vals.append(thr)
        tfs_key = str(tfs)
        tfs_counter[tfs_key] = tfs_counter.get(tfs_key, 0) + 1

    if sl_vals:
        tfs_dom = max(tfs_counter, key=tfs_counter.get)
        lines.append(
            f"**Padrão dominante:** SL={_fmt_num(sum(sl_vals)/len(sl_vals), 1)}% "
            f"(±{_fmt_num((max(sl_vals)-min(sl_vals))/2, 1)}) | "
            f"TP={_fmt_num(sum(tp_vals)/len(tp_vals), 1)}% (±{_fmt_num((max(tp_vals)-min(tp_vals))/2, 1)}) | "
            f"Thr={_fmt_num(sum(thr_vals)/len(thr_vals), 2)} | TFs={tfs_dom}  \n"
            f"→ Explora variações de features e entry signal nesta zona. Não repitas params iguais.\n"
        )

    section = "\n---\n\n" + "".join(lines) + "\n---\n"

    # Replace existing section or append at the end
    text = program_path.read_text() if program_path.exists() else ""
    import re as _re
    if "## 📊 Melhores Resultados Actuais" in text:
        text = _re.sub(
            r'\n---\n\n## 📊 Melhores Resultados Actuais.*?\n---\n',
            section, text, flags=_re.DOTALL
        )
    else:
        text = text.rstrip() + "\n" + section
    program_path.write_text(text)


def run_loop(config: dict, max_iterations: int = 0,
             human_review_interval: int = 5,
             experiments_dir: Path = None,
             cache_dir: Path = None):
    """
    Main autonomous research loop.

    Args:
        config: system configuration
        max_iterations: 0 = infinite
        human_review_interval: ask for review every N iterations (0 = disabled)
        experiments_dir: where to save experiences
        cache_dir: labels and models cache
    """
    import signal

    _stop = False

    def _handler_sigint(sig, frame):
        nonlocal _stop
        if not _stop:
            console.print("\n[bold yellow]Ctrl+C received — finishing after current iteration...[/bold yellow]")
            _stop = True
        else:
            console.print("\n[bold red]Ctrl+C forced — exiting immediately.[/bold red]")
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
    iteration = tracker.next_iteration_number()
    previous_params = None

    # Load score from last accepted as baseline, or use baseline_override from config
    baseline_override  = config.get('agent', {}).get('baseline_override', 0.0)
    accept_auc_min          = config.get('agent', {}).get('accept_auc_min', 0.0)
    accept_sharpe_min       = config.get('agent', {}).get('accept_sharpe_min', 0.0)
    accept_sharpe_holdout_min = config.get('agent', {}).get('accept_sharpe_holdout_min', 0.0)
    last = tracker.last_accepted()
    if last:
        last_mode = (last.params_snapshot or {}).get('OBJECTIVE_MODE', 'score')
        if last_mode == 'profit':
            score_baseline = last.metricas.get('retorno_total_oos_pct', baseline_override)
        else:
            sv = last.metricas.get('sharpe_validation')
            score_baseline = float(sv) if sv is not None else last.metricas.get('score_composto', 0.0)
        previous_params = last.params_snapshot
        console.print(f"[cyan]Resuming research. Last accepted: iter {last.iteracao}, "
                      f"score={_fmt_num(score_baseline, 4)}[/cyan]")
    else:
        score_baseline = baseline_override
        if baseline_override > 0:
            console.print(f"[cyan]New season. Minimum baseline set: {_fmt_num(baseline_override, 4)}[/cyan]")
        else:
            score_baseline = 0.0

    console.print(Panel(
        f"[bold green]algo_autoresearch — Research Loop[/bold green]\n"
        f"Initial iteration: {iteration}\n"
        f"Score baseline: {_fmt_num(score_baseline, 4)}\n"
        f"Max iterations: {'∞' if max_iterations == 0 else max_iterations}",
        border_style="green",
    ))

    accept_threshold = config.get('agent', {}).get('accept_threshold', 0.01)
    revert_on_worse  = config.get('agent', {}).get('revert_on_worse', True)

    # --- Temperature curriculum ---
    t_base  = config.get('llm', {}).get('temperature', 0.7)
    t_min   = config.get('llm', {}).get('t_min', 0.3)
    t_max   = config.get('llm', {}).get('t_max', 1.2)
    t_decay = config.get('llm', {}).get('t_decay', 0.92)   # when improving → exploit
    t_grow  = config.get('llm', {}).get('t_grow', 1.08)    # when stagnating → explore
    stagnation_threshold = config.get('llm', {}).get('stagnation_threshold', 5)
    current_temp         = t_base
    iters_without_improvement = 0

    recent_rejections: list[str] = []  # latest validation rejections

    while True:
        if _stop:
            console.print("[bold yellow]Loop finished by Ctrl+C.[/bold yellow]")
            break

        if max_iterations > 0 and iteration > max_iterations:
            console.print("[bold]Maximum number of iterations reached.[/bold]")
            break

        console.print(f"\n[bold cyan]══ Iteration {iteration} ══[/bold cyan]  "
                      f"[dim]temp={_fmt_num(current_temp, 2)} | no_improvement={iters_without_improvement}[/dim]")

        # --- 1. Load current params ---
        current_code = params_path.read_text()
        current_params = carregar_params(params_path)

        # --- 2. LLM propose new params ---
        console.print("  [dim]Querying LLM...[/dim]")
        program_md = program_path.read_text() if program_path.exists() else ""
        history  = tracker.list_history(30)
        best     = tracker.best_score()
        best_dict = best.to_dict() if best else None
        top5        = tracker.top_n_scores(10)

        proposed_code = propose_new_params(
            current_code, program_md, history, config,
            best_record=best_dict,
            top_records=top5,
            recent_rejections=recent_rejections,
            temperature=current_temp,
        )

        if proposed_code is None:
            console.print("  [red]LLM did not return valid code. Keeping current params.[/red]")
            # Run with current params anyway
            proposed_code = current_code

        # --- 3. Validate ---
        ok, message = validate_code(proposed_code)
        if not ok:
            console.print(f"  [red]Code rejected: {message}[/red]")
            recent_rejections.append(message)
            recent_rejections = recent_rejections[-5:]  # keep only latest 5
            record = tracker.create_record(
                iteration=iteration,
                status='rejeitado',
                metricas={'score_composto': 0.0},
                params_hash='invalid',
                labels_reutilizados=False,
                duracao=0.0,
                alteracoes=f"REJECTED: {message}",
            )
            tracker.save_experience(record)
            iteration += 1
            continue

        # Validation passed — clear pending rejections
        recent_rejections.clear()

        # Auto-correct N_TRIALS_XGB: force 0 unless in exploit mode or option B
        exploit_mode = "MODO EXPLOIT ATIVO" in program_md or "OPÇÃO B" in program_md
        if not exploit_mode:
            import re as _re
            corrected_code = _re.sub(
                r'(N_TRIALS_XGB\s*=\s*)\d+',
                r'\g<1>0  # forced to 0 by runner (exploration mode)',
                proposed_code
            )
            if corrected_code != proposed_code:
                console.print("  [dim]Auto-correction: N_TRIALS_XGB → 0 (exploration mode)[/dim]")
                proposed_code = corrected_code

        # Show what the agent proposes
        show_proposed_params(current_code, proposed_code)

        # --- 4. Save backup and apply ---
        shutil.copy2(params_path, backup_path)
        params_path.write_text(proposed_code)

        new_params = carregar_params(params_path)
        new_hash = hash_params_completo(new_params)

        # --- 4b. Early stopping: reject already explored configuration ---
        if tracker.hash_already_explored(new_hash):
            console.print(f"  [yellow]✗ DUPLICATE: configuration already tested (hash={new_hash[:8]}) — skipping pipeline[/yellow]")
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)
            msg_dup = f"Duplicate configuration (hash={new_hash[:8]}) — propose a different variation"
            recent_rejections.append(msg_dup)
            recent_rejections = recent_rejections[-5:]
            iters_without_improvement += 1
            if iters_without_improvement >= stagnation_threshold:
                previous_temp = current_temp
                current_temp = min(t_max, current_temp * t_grow)
                console.print(f"  [dim]temp {_fmt_num(previous_temp, 2)}→{_fmt_num(current_temp, 2)} (explore — stagnation)[/dim]")
            record = tracker.create_record(
                iteration=iteration,
                status='rejeitado',
                metricas={'score_composto': 0.0},
                params_hash=new_hash,
                labels_reutilizados=False,
                duracao=0.0,
                alteracoes=f"DUPLICATE: {new_hash[:8]}",
            )
            tracker.save_experience(record)
            iteration += 1
            continue

        changes = tracker.compute_changes(
            previous_params or current_params,
            new_params
        )
        console.print(f"  Changes: [yellow]{changes}[/yellow]")

        # --- 5. Run pipeline ---
        start_time = time.time()
        result = executar_pipeline(config, params_path, cache_dir)
        duration = time.time() - start_time

        # --- 6. Calculate score and decide ---
        objective_mode    = new_params.get('OBJECTIVE_MODE', 'score')
        current_auc         = result.metricas.get('cv_auc_mean', 0.0) if result.sucesso else 0.0
        sharpe_validation = result.metricas.get('sharpe_validation', 0.0) if result.sucesso else 0.0
        sharpe_holdout_value  = result.metricas.get('sharpe_holdout', 0.0) if result.sucesso else 0.0

        if objective_mode == 'profit':
            current_score = result.metricas.get('retorno_total_oos_pct', -999.0) if result.sucesso else -999.0
            improved = result.sucesso and (current_score > score_baseline + accept_threshold)
        else:
            current_score   = sharpe_validation
            gate_auc      = current_auc >= accept_auc_min
            gate_sharpe   = sharpe_validation >= accept_sharpe_min
            gate_holdout  = sharpe_holdout_value >= accept_sharpe_holdout_min
            improved      = result.sucesso and gate_auc and gate_sharpe and gate_holdout

        # Detect Optuna result already found (same local optimum, different params)
        if result.sucesso and tracker.result_already_found(result.metricas):
            sl  = result.metricas.get('sl_pct', '?')
            tp  = result.metricas.get('tp_pct', '?')
            thr = result.metricas.get('threshold', '?')
            console.print(f"  [yellow]✗ DUPLICATE RESULT: Optuna converged to the same optimum "
                          f"(SL={_fmt_num(sl)}% TP={_fmt_num(tp)}% T={_fmt_num(thr)}) — escape this region[/yellow]")
            msg_res = f"Duplicate result: Optuna found SL={_fmt_num(sl)}% TP={_fmt_num(tp)}% T={_fmt_num(thr)} — change features or TIMEFRAMES to escape"
            recent_rejections.append(msg_res)
            recent_rejections = recent_rejections[-5:]
            iters_without_improvement += 1
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)
            record = tracker.create_record(
                iteration=iteration,
                status='rejeitado',
                metricas=result.metricas,
                params_hash=hash_params_completo(new_params),
                labels_reutilizados=result.labels_reutilizados,
                duracao=duration,
                alteracoes=f"DUPLICATE RESULT: SL={_fmt_num(sl)}% TP={_fmt_num(tp)}%",
                params_snapshot=new_params,
            )
            tracker.save_experience(record)
            iteration += 1
            continue

        show_iteration_result(iteration, result, score_baseline)

        if result.sucesso and improved:
            status = 'aceite'
            previous_params = new_params
            # Improvement → lower temperature (exploit good region)
            previous_temp = current_temp
            current_temp = max(t_min, current_temp * t_decay)
            iters_without_improvement = 0
            if objective_mode == 'profit':
                console.print(f"  [bold green]✓ ACCEPTED (return {_fmt_num(score_baseline, 1, sign=True)}%)[/bold green]  "
                              f"[dim]temp {_fmt_num(previous_temp, 2)}→{_fmt_num(current_temp, 2)} (decay)[/dim]")
            else:
                console.print(f"  [bold green]✓ ACCEPTED | AUC={_fmt_num(current_auc, 3)} | "
                              f"Sharpe(val)={_fmt_num(sharpe_validation, 2)} | "
                              f"Sharpe(holdout/passive)={_fmt_num(sharpe_holdout_value, 2)}[/bold green]  "
                              f"[dim]temp {_fmt_num(previous_temp, 2)}→{_fmt_num(current_temp, 2)} (decay)[/dim]")
            # Preserve accepted model from cleanup
            season = config.get('agent', {}).get('season', 0)
            _save_best_model(cache_dir, new_hash, iteration, season,
                             result.metricas, new_params)
        else:
            status = 'rejeitado' if result.sucesso else 'erro'
            if objective_mode == 'profit':
                reason = (f"return {_fmt_num(current_score, 1, sign=True)}% ≤ baseline {_fmt_num(score_baseline, 1, sign=True)}% + {accept_threshold}%"
                         if result.sucesso else result.erro)
            else:
                if result.sucesso:
                    reason = (f"AUC={_fmt_num(current_auc, 3)}(≥{accept_auc_min}) | "
                             f"Sharpe(val)={_fmt_num(sharpe_validation, 2)}(≥{accept_sharpe_min}) | "
                             f"Sharpe(holdout)={_fmt_num(sharpe_holdout_value, 2)}(≥{accept_sharpe_holdout_min})")
                else:
                    reason = result.erro
            # No improvement → count; if threshold exceeded, increase temperature (explore)
            iters_without_improvement += 1
            if iters_without_improvement >= stagnation_threshold:
                previous_temp = current_temp
                current_temp = min(t_max, current_temp * t_grow)
                console.print(f"  [yellow]✗ REVERTED ({reason})[/yellow]  "
                              f"[dim]temp {_fmt_num(previous_temp, 2)}→{_fmt_num(current_temp, 2)} (explore)[/dim]")
            else:
                console.print(f"  [yellow]✗ REVERTED ({reason})[/yellow]  "
                              f"[dim]{iters_without_improvement}/{stagnation_threshold} without improvement[/dim]")
            if revert_on_worse:
                shutil.copy2(backup_path, params_path)

        # --- 7. Clean cache every 10 iterations ---
        if iteration % 10 == 0:
            _train_start = config['pipeline'].get('train_start', 2017)
            _train_end   = config['pipeline'].get('train_end', 2024)
            clean_cache(
                cache_dir,
                current_params_hash=hash_entry_params(new_params, train_start=_train_start, train_end=_train_end),
                current_model_hash=hash_params_completo(new_params),
            )

        # --- 8. Register ---
        record = tracker.create_record(
            iteration=iteration,
            status=status,
            metricas=result.metricas if result.sucesso else {'score_composto': current_score},
            params_hash=hash_params_completo(new_params),
            labels_reutilizados=result.labels_reutilizados,
            duracao=duration,
            alteracoes=changes,
            params_snapshot=new_params,
        )
        tracker.save_experience(record)

        # --- 8c. Release accumulated memory (OOM mitigation S7) ---
        gc.collect()

        # --- 8b. Update program.md with top results (re-read each iteration) ---
        _update_program_md_top(program_path, tracker.top_n_scores(10))

        # --- 9. Periodic human review ---
        if human_review_interval > 0 and iteration % human_review_interval == 0:
            action = request_human_review(tracker, params_path, iteration, config)
            if action.get('action') == 'exit':
                console.print("[bold]Exiting by user request.[/bold]")
                break
            elif action.get('action') == 'inject':
                # Use manually injected params
                console.print("  [cyan]Using manually injected params.[/cyan]")

        iteration += 1


# Backward-compatible wrapper used by main.py (keeps Portuguese parameter names)
def executar_loop(config: dict, max_iteracoes: int = 0,
                  human_review_interval: int = 5,
                  experiments_dir: Path = None,
                  cache_dir: Path = None):
    """Backward-compatible wrapper for run_loop."""
    return run_loop(config, max_iteracoes, human_review_interval, experiments_dir, cache_dir)
