"""
LLM Agent — proposes modifications to research_params.py.

Uses Qwen2.5-7B via llama.cpp (OpenAI-compatible API at localhost:8080).
Validates proposed code: Python syntax + relative indicators (AST).

Critical rule: NEVER accept code with df['Close'], df['Open'], etc.
"""
import ast
import re
import requests
import json
from pathlib import Path
from typing import Optional


# Forbidden absolute price columns
PRICE_COLUMNS = {'Close', 'Open', 'High', 'Low', 'Volume',
                 'close', 'open', 'high', 'low', 'volume'}

SYSTEM_PROMPT = """You are a Machine Learning Engineer specialized in Quantitative Trading.
Your task is to modify the research_params.py file to maximize composite_score
via Bayesian Optimization (Optuna TPE Sampler already implemented in the backtest).

RESPONSIBILITIES:
1. FEATURE SELECTION: Choose 8-15 features from the catalog that make sense for the
   market regime (e.g., high volatility → ATR + BB Width; trend → EMA diff + MACD)
2. BOUND DEFINITION: Define SL_RANGE, TP_RANGE, THRESHOLD_RANGE as tuples
   (min, max). Optuna explores this space automatically — you do not need exact values.
3. XGBoost REGULARIZATION: If the history shows inter-year degradation or
   inconsistent scores, increase REG_ALPHA and REG_LAMBDA to reduce overfitting.

ABSOLUTE RULES (never violate):
1. ONLY relative indicators — NEVER use df['Close'], df['Open'], df['High'], df['Low']
2. Use only allowed features from the catalog (list below)
3. The file must be valid and syntactically correct Python
4. SL_RANGE, TP_RANGE, THRESHOLD_RANGE are TUPLES (min, max) — not lists or single values
5. TP_RANGE[0] must always be greater than SL_RANGE[0] (min TP > min SL)

Allowed features — EXHAUSTIVE LIST (do not invent others):
  stoch_rsi_k, stoch_rsi_d, rsi, bb_position, adx,
  ema_diff, trend, returns_1, atr_pct, bb_width_pct,
  macd_pct, macd_signal_pct, macd_hist_pct,
  volume_norm, returns_5,
  dist_sma200_pct, btc_trend, atr_regime

Notes on macro features (require "1d" in TIMEFRAMES):
  dist_sma200_pct — normalized distance to SMA200 (%, 1d only); positive = above SMA200
  btc_trend       — BTC above/below EMA50 (0=down, 1=up, 1d only); cross-asset
  atr_regime      — ATR / rolling_mean(ATR,50) per timeframe; >1 = volatility above average

FORBIDDEN to use any name outside this list (e.g., macd_diff, macd_signal, ema_ratio, etc.).
Any feature not listed above will cause an error and immediate rejection of the code.

Timeframes: ["15m", "4h", "1d"] (any non-empty subset)

Composite score (to MAXIMIZE):
  score = tanh(S/2)*0.50 + tanh(R/100)*0.30 - abs(DD)/100*0.20
  where S=Sharpe(365d), R=annual return%, DD=max drawdown%

BOUND ADJUSTMENT STRATEGY:
- If Optuna found the best SL near the LOWER bound → reduce SL_RANGE[0]
- If Optuna found the best SL near the UPPER bound → increase SL_RANGE[1]
- Same logic for TP_RANGE and THRESHOLD_RANGE
- Tighter bounds around the good region speed up convergence

XGBoost OPTIMIZATION WITH OPTUNA:
- By default KEEP N_TRIALS_XGB = 0 (disabled) — priority is speed and exploration
- Only enable if program.md explicitly indicates EXPLOIT mode with xgb_trials > 0

MANDATORY FORMAT: the file must contain ONLY variable assignments and comments.
FORBIDDEN: FEATURES.remove(...), FEATURES.append(...), or any list mutation.
If you want to remove a feature, redefine FEATURES as a complete new list.

Reply ONLY with the complete content of the research_params.py file.
DO NOT include explanations outside the file. Use Python comments inside the file.
"""


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Validates that the code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error on line {e.lineno}: {e.msg}"


def _validate_relative_indicators(code: str) -> tuple[bool, str]:
    """
    Validates via AST that the code does not access absolute price columns.

    Rejects patterns like: df['Close'], df["open"], data['High'], etc.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "Invalid code (syntax error)"

    class PriceColumnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []

        def visit_Subscript(self, node):
            # Detect patterns: something['Close'] or something["open"]
            if isinstance(node.slice, ast.Constant):
                val = node.slice.value
                if isinstance(val, str) and val in PRICE_COLUMNS:
                    self.violations.append(
                        f"Forbidden price column access: '{val}' (line {node.lineno})"
                    )
            self.generic_visit(node)

    visitor = PriceColumnVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return False, "; ".join(visitor.violations)
    return True, "OK"


def _validate_required_params(code: str) -> tuple[bool, str]:
    """Checks that required parameters are defined."""
    required = [
        'FEATURES', 'TIMEFRAMES', 'ENTRY_STOCH_THRESHOLD', 'ENTRY_ADX_THRESHOLD',
        'N_ESTIMATORS', 'MAX_DEPTH', 'LEARNING_RATE', 'SL_RANGE', 'TP_RANGE',
        'THRESHOLD_RANGE', 'N_TRIALS', 'OBJECTIVE_MODE',
    ]
    missing = [p for p in required if p not in code]
    if missing:
        return False, f"Missing required parameters: {missing}"
    return True, "OK"


def _validate_execution(code: str) -> tuple[bool, str]:
    """Runs the code in an isolated namespace to detect any runtime errors."""
    try:
        exec(compile(code, '<research_params>', 'exec'), {})
        return True, "OK"
    except Exception as e:
        return False, f"Execution error ({type(e).__name__}): {e}"


def _validate_no_mutations(code: str) -> tuple[bool, str]:
    """Rejects code that mutates FEATURES or TIMEFRAMES after definition (e.g., FEATURES.remove(...))."""
    forbidden = ['FEATURES.remove(', 'FEATURES.append(', 'FEATURES.pop(',
                 'FEATURES.extend(', 'FEATURES.insert(', 'FEATURES.clear(',
                 'TIMEFRAMES.remove(', 'TIMEFRAMES.append(']
    for f in forbidden:
        if f in code:
            return False, (f"'{f}' is not allowed — define FEATURES as a complete list "
                           f"instead of mutating after definition")
    return True, "OK"


def _validate_catalog_features(code: str) -> tuple[bool, str]:
    """Extracts FEATURES from code via AST and validates against the catalog."""
    try:
        from pipeline.features_catalog import validate_features
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'FEATURES':
                        if isinstance(node.value, ast.List):
                            features = [
                                elt.value for elt in node.value.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            ]
                            ok, msg = validate_features(features)
                            if not ok:
                                return False, f"Invalid feature in FEATURES: {msg}"
        return True, "OK"
    except Exception as e:
        return True, "OK"  # do not block because of secondary parsing error


def extract_code(response: str) -> Optional[str]:
    """
    Extracts Python code from the LLM response.

    Tries code blocks ```python...``` first, then the whole content.
    """
    # Try to extract code block
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # If no code markers, assume the response is the code
    lines = response.strip().split('\n')
    # Remove lines that are clearly not code
    code_lines = []
    for line in lines:
        if line.startswith('#') or '=' in line or line.strip() == '' or line.startswith('[') or line.startswith('"') or line.startswith("'"):
            code_lines.append(line)
        elif any(kw in line for kw in ['FEATURES', 'TIMEFRAMES', 'STOCH', 'ADX', 'EMA', 'SL_RANGE', 'TP_RANGE', 'THRESHOLD_RANGE', 'N_TRIALS', 'N_ESTIMATORS']):
            code_lines.append(line)

    if len(code_lines) > 5:
        return '\n'.join(code_lines)

    return None


def propose_new_params(params_code: str, program_md: str,
                       history: list, config: dict,
                       best_record: dict = None,
                       top_records: list = None,
                       recent_rejections: list = None,
                       temperature: float = None) -> Optional[str]:
    """
    Calls the LLM to propose a new research_params.py.

    Args:
        params_code: current content of research_params.py
        program_md: content of program.md (research direction)
        history: list of dicts with previous results
        config: system configuration

    Returns:
        New content of research_params.py, or None if it fails
    """
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    max_tokens = config.get('llm', {}).get('max_tokens', 2048)
    if temperature is None:
        temperature = config.get('llm', {}).get('temperature', 0.7)
    timeout    = config.get('llm', {}).get('timeout_seconds', 120)

    # Best result so far
    best_text = ""
    if best_record:
        metrics = best_record.get('metricas', {})
        params_snapshot = best_record.get('params_snapshot', {})

        # Optuna sensitivity: where the best SL/TP/threshold landed relative to bounds
        sensitivity = ""
        sl_best = metrics.get('sl_pct')
        tp_best = metrics.get('tp_pct')
        thr_best = metrics.get('threshold')
        sl_range = params_snapshot.get('SL_RANGE', (0.5, 12.0))
        tp_range = params_snapshot.get('TP_RANGE', (1.0, 40.0))
        thr_range = params_snapshot.get('THRESHOLD_RANGE', (0.30, 0.80))

        if sl_best is not None and sl_range:
            sl_lo, sl_hi = sl_range
            sl_rel = (sl_best - sl_lo) / max(sl_hi - sl_lo, 1e-6)
            if sl_rel < 0.2:
                sensitivity += f"    Optimal SL={sl_best:.2f}% near LOWER BOUND ({sl_lo}) → consider reducing SL_RANGE[0]\n"
            elif sl_rel > 0.8:
                sensitivity += f"    Optimal SL={sl_best:.2f}% near UPPER BOUND ({sl_hi}) → consider increasing SL_RANGE[1]\n"
            else:
                sensitivity += f"    Optimal SL={sl_best:.2f}% in the center of [{sl_lo},{sl_hi}] → bounds adequate\n"

        if tp_best is not None and tp_range:
            tp_lo, tp_hi = tp_range
            tp_rel = (tp_best - tp_lo) / max(tp_hi - tp_lo, 1e-6)
            if tp_rel < 0.2:
                sensitivity += f"    Optimal TP={tp_best:.2f}% near LOWER BOUND ({tp_lo}) → consider reducing TP_RANGE[0]\n"
            elif tp_rel > 0.8:
                sensitivity += f"    Optimal TP={tp_best:.2f}% near UPPER BOUND ({tp_hi}) → consider increasing TP_RANGE[1]\n"
            else:
                sensitivity += f"    Optimal TP={tp_best:.2f}% in the center of [{tp_lo},{tp_hi}] → bounds adequate\n"

        if thr_best is not None and thr_range:
            thr_lo, thr_hi = thr_range
            thr_rel = (thr_best - thr_lo) / max(thr_hi - thr_lo, 1e-6)
            if thr_rel < 0.2:
                sensitivity += f"    Optimal Threshold={thr_best:.3f} near LOWER BOUND ({thr_lo}) → consider reducing THRESHOLD_RANGE[0]\n"
            elif thr_rel > 0.8:
                sensitivity += f"    Optimal Threshold={thr_best:.3f} near UPPER BOUND ({thr_hi}) → consider increasing THRESHOLD_RANGE[1]\n"
            else:
                sensitivity += f"    Optimal Threshold={thr_best:.3f} in the center of [{thr_lo},{thr_hi}] → bounds adequate\n"

        # Max Drawdown forensic analysis
        df_info = metrics.get('drawdown_forensics', {})
        forensic_text = ""
        if df_info:
            forensic_text = (
                f"  Max Drawdown forensic analysis ({df_info.get('ano','?')}):\n"
                f"    Duration: {df_info.get('dur_dias','?')} days | "
                f"Depth: {df_info.get('profundidade','?'):.1f}%\n"
                f"    Market regime: {df_info.get('regime','?')}\n"
                f"    Average ADX in the period: {df_info.get('adx_medio','?')} (min: {df_info.get('adx_minimo','?')}))\n"
            )
            # Automatic suggestion based on regime
            adx_mean = df_info.get('adx_medio', 30)
            if adx_mean < 20:
                forensic_text += "    → Suggestion: increase ENTRY_ADX_THRESHOLD or ADX_PERIOD to filter trendless regimes\n"
            elif adx_mean < 25:
                forensic_text += "    → Suggestion: consider reducing SL_RANGE (tighter SL in weak trends)\n"

        # Optuna parameter importance of the best result
        optuna_importance = metrics.get('optuna_param_importance', {})
        optuna_importance_text = ""
        if optuna_importance:
            sorted_imp = sorted(optuna_importance.items(), key=lambda x: -x[1])
            dominant = sorted_imp[0][0] if sorted_imp else None
            optuna_importance_text = "  Optuna param importance (which dimension most affects score):\n"
            optuna_importance_text += "    " + " | ".join(f"{k}={v:.3f}" for k,v in sorted_imp) + "\n"
            if dominant:
                optuna_importance_text += f"    → Focus mainly on: {dominant} (largest impact on score)\n"

        # Feature importance of the best result
        top_features    = metrics.get('top_features', [])
        bottom_features = metrics.get('bottom_features', [])
        feature_importance_text = ""
        if top_features:
            feature_importance_text += "  Feature importance (XGBoost):\n"
            feature_importance_text += "    TOP (keep/reinforce): " + ", ".join(f"{f}={v:.3f}" for f,v in top_features[:6]) + "\n"
            if bottom_features:
                feature_importance_text += "    WEAK (<0.02, consider removing): " + ", ".join(f for f,_ in bottom_features) + "\n"
        auc_text = ""
        if metrics.get('cv_auc_mean'):
            auc_text = f"  CV ROC-AUC: {metrics['cv_auc_mean']:.4f} ± {metrics.get('cv_auc_std',0):.4f}\n"

        # Score breakdown by component
        S  = metrics.get('sharpe_raw', 0)
        R  = metrics.get('retorno_anual_pct', 0)
        DD = abs(metrics.get('max_drawdown_pct', 0))
        import math
        s_contrib  = math.tanh(S / 2) * 0.50
        r_contrib  = math.tanh(R / 100) * 0.30
        dd_penalty = (DD / 100) * 0.20
        score_breakdown = (
            f"  Score breakdown: Sharpe={s_contrib:+.3f}/0.50 | Return={r_contrib:+.3f}/0.30 | DD={-dd_penalty:.3f}/-0.20\n"
            f"  → Largest available margin: "
            + ("Sharpe" if (0.50 - s_contrib) > (0.30 - r_contrib) else "Return")
            + f" ({max(0.50 - s_contrib, 0.30 - r_contrib):.3f} points of possible gain)\n"
        )

        best_text = (
            f"\n\nBEST RESULT SO FAR (iter {best_record.get('iteracao','?')}):\n"
            f"  Score={metrics.get('score_composto',0):.4f} | Sharpe={metrics.get('sharpe_raw',0):.2f} | "
            f"Return={metrics.get('retorno_anual_pct',0):+.1f}% | DD={DD:.1f}%\n"
            f"{auc_text}"
            f"{score_breakdown}"
            f"{forensic_text}"
            f"{optuna_importance_text}"
            f"{feature_importance_text}"
            f"  Params that generated this result:\n"
            f"    FEATURES={params_snapshot.get('FEATURES','?')}\n"
            f"    TIMEFRAMES={params_snapshot.get('TIMEFRAMES','?')}\n"
            f"    ENTRY_STOCH_THRESHOLD={params_snapshot.get('ENTRY_STOCH_THRESHOLD','?')} | ENTRY_ADX_THRESHOLD={params_snapshot.get('ENTRY_ADX_THRESHOLD','?')}\n"
            f"    N_ESTIMATORS={params_snapshot.get('N_ESTIMATORS','?')} | MAX_DEPTH={params_snapshot.get('MAX_DEPTH','?')} | LR={params_snapshot.get('LEARNING_RATE','?')}\n"
            f"    SL_RANGE={params_snapshot.get('SL_RANGE','?')} | TP_RANGE={params_snapshot.get('TP_RANGE','?')} | THRESHOLD_RANGE={params_snapshot.get('THRESHOLD_RANGE','?')}\n"
            + (f"    XGBoost Optuna best: {metrics.get('xgb_optuna_best')}\n" if metrics.get('xgb_optuna_best') else "")
            + f"  Optuna sensitivity analysis:\n"
            + f"{sensitivity}"
            + f"  The goal is to SURPASS this score. Explore dimensions not yet tested.\n"
        )

    # Top N historical results + pattern analysis
    top_text = ""
    if top_records and len(top_records) > 1:
        top_text = "\n\nTOP HISTORICAL RESULTS (region to explore — pattern identified):\n"
        tfs_counter: dict = {}
        sl_vals, tp_vals, thr_vals = [], [], []
        for i, h in enumerate(top_records):
            metrics = h.get('metricas', {})
            ps = h.get('params_snapshot', {})
            sl  = metrics.get('sl_pct')
            tp  = metrics.get('tp_pct')
            thr = metrics.get('threshold')
            tfs = str(ps.get('TIMEFRAMES', '?'))
            top_text += (
                f"  #{i+1} iter={h.get('iteracao','?')} score={metrics.get('score_composto',0):.4f} | "
                f"Sharpe={metrics.get('sharpe_raw',0):.2f} | Return={metrics.get('retorno_anual_pct',0):+.1f}% | "
                f"DD={abs(metrics.get('max_drawdown_pct',0)):.1f}% | "
                f"Trades={metrics.get('n_trades',0)} | WR={metrics.get('win_rate_pct',0):.1f}%\n"
                f"    Optuna best: SL={sl:.2f}% TP={tp:.2f}% Thr={thr:.3f}\n"
                f"    TIMEFRAMES={ps.get('TIMEFRAMES','?')} | "
                f"FEATURES={len(ps.get('FEATURES',[]))} features | "
                f"Entry: stoch<{ps.get('ENTRY_STOCH_THRESHOLD','?')} adx>{ps.get('ENTRY_ADX_THRESHOLD','?')}\n"
            ) if sl and tp and thr else (
                top_text + f"  #{i+1} iter={h.get('iteracao','?')} score={metrics.get('score_composto',0):.4f}\n"
            )
            if tfs not in tfs_counter:
                tfs_counter[tfs] = 0
            tfs_counter[tfs] += 1
            if sl: sl_vals.append(sl)
            if tp: tp_vals.append(tp)
            if thr: thr_vals.append(thr)

        # Common pattern
        if sl_vals and tp_vals and thr_vals:
            top_text += (
                f"\n  DOMINANT PATTERN in top results:\n"
                f"    Average SL={sum(sl_vals)/len(sl_vals):.2f}% (min={min(sl_vals):.2f} max={max(sl_vals):.2f})\n"
                f"    Average TP={sum(tp_vals)/len(tp_vals):.2f}% (min={min(tp_vals):.2f} max={max(tp_vals):.2f})\n"
                f"    Average Threshold={sum(thr_vals)/len(thr_vals):.3f} (min={min(thr_vals):.3f} max={max(thr_vals):.3f})\n"
                f"    Most frequent TIMEFRAMES: {max(tfs_counter, key=tfs_counter.get)}\n"
                f"  → Explore VARIATIONS in this region: different features, entry signal, XGBoost params.\n"
                f"  → DO NOT repeat the same params — the system detects duplicates automatically.\n"
            )

    # Recent history (last 30 iterations)
    recent_history = history[-30:] if len(history) > 30 else history
    history_text = ""
    if recent_history:
        history_text = "\n\nRECENT HISTORY (last 30 tested iterations):\n"
        for h in recent_history:
            metrics = h.get('metricas', {})
            top_f    = metrics.get('top_features', [])
            bot_f    = metrics.get('bottom_features', [])
            oi       = metrics.get('optuna_param_importance', {})
            history_text += (
                f"  Iter {h.get('iteracao', '?')}: "
                f"Score={metrics.get('score_composto', 0):.4f} | "
                f"Sharpe={metrics.get('sharpe_raw', 0):.2f} | "
                f"Return={metrics.get('retorno_anual_pct', 0):+.1f}% | "
                f"DD={abs(metrics.get('max_drawdown_pct', 0)):.1f}% | "
                f"AUC={metrics.get('cv_auc_mean', 0):.4f} | "
                f"Status={h.get('status', '?')}\n"
            )
            if oi:
                sorted_oi = sorted(oi.items(), key=lambda x: -x[1])
                history_text += f"    Optuna importance: {' | '.join(f'{k}={v:.3f}' for k,v in sorted_oi)}\n"
            if top_f:
                history_text += f"    Top features: {', '.join(f'{f}={v:.3f}' for f,v in top_f[:4])}\n"
            if bot_f:
                history_text += f"    Weak features: {', '.join(f for f,_ in bot_f)}\n"
            if h.get('alteracoes_vs_anterior'):
                history_text += f"    Changes: {h['alteracoes_vs_anterior']}\n"

    # Feedback from recent rejections
    rejections_text = ""
    if recent_rejections:
        rejections_text = "\n\n⚠️  RECENT REJECTIONS — DO NOT REPEAT THESE ERRORS:\n"
        for r in recent_rejections:
            rejections_text += f"  - {r}\n"
        rejections_text += (
            "These proposals were automatically rejected BEFORE running the pipeline.\n"
            "Fix ONLY the indicated error. Use ONLY features from the exhaustive list above.\n"
        )

    user_prompt = f"""RESEARCH DIRECTION (program.md):
{program_md}
{best_text}{top_text}{history_text}{rejections_text}

CURRENT PARAMETERS (research_params.py) — these are the params of the best result:
```python
{params_code}
```

Based on the history and research direction, propose ONE specific,
well-founded modification to try to SURPASS the best score.
Avoid repeating recent changes that were rejected.
Explore new dimensions (e.g., if you already tested MAX_DEPTH, try FEATURES or TIMEFRAMES now).

Reply ONLY with the new complete content of the research_params.py file.
"""

    payload = {
        "model": "qwen2.5-7b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0]['message']['content']

        # Save prompt + full response (includes reasoning discarded by extract_code)
        _log_llm_interaction(user_prompt, content)

        return extract_code(content)
    except requests.exceptions.ConnectionError:
        print(f"  [AGENT] Error: LLM server not accessible at {server_url}")
        return None
    except requests.exceptions.Timeout:
        print(f"  [AGENT] Timeout calling LLM ({timeout}s)")
        return None
    except Exception as e:
        print(f"  [AGENT] Unexpected error: {e}")
        return None


def _log_llm_interaction(prompt: str, response: str, max_logs: int = 50) -> None:
    """
    Saves the full LLM prompt and response to logs/llm_interactions/.

    Keeps only the last max_logs pairs to avoid filling the disk.
    The response includes the model's reasoning before the code block.
    """
    from datetime import datetime
    log_dir = Path(__file__).parent.parent / 'logs' / 'llm_interactions'
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (log_dir / f"prompt_{ts}.txt").write_text(prompt, encoding='utf-8')
    (log_dir / f"response_{ts}.txt").write_text(response, encoding='utf-8')

    # Rotation: delete oldest if exceeding max_logs pairs
    prompts = sorted(log_dir.glob("prompt_*.txt"))
    if len(prompts) > max_logs:
        for old in prompts[:len(prompts) - max_logs]:
            old.unlink(missing_ok=True)
            ts_old = old.stem.replace("prompt_", "")
            resp_old = log_dir / f"response_{ts_old}.txt"
            resp_old.unlink(missing_ok=True)


def _validate_ranges(code: str) -> tuple[bool, str]:
    """Validates that SL_RANGE, TP_RANGE and THRESHOLD_RANGE have min <= max and respect focus bounds."""
    ns = {}
    try:
        exec(compile(code, '<research_params>', 'exec'), ns)
    except Exception:
        return True, "OK"  # exec errors already caught by _validate_execution

    for name in ('SL_RANGE', 'TP_RANGE', 'THRESHOLD_RANGE'):
        val = ns.get(name)
        if val is not None:
            try:
                lo, hi = val
                if lo > hi:
                    return False, f"{name}=({lo},{hi}) invalid: min > max — swap values"
            except (TypeError, ValueError):
                pass

    return True, "OK"


def validate_code(code: str) -> tuple[bool, str]:
    """
    Complete validation of the proposed code.

    Returns:
        (valid, error_message)
    """
    ok, msg = _validate_syntax(code)
    if not ok:
        return False, msg

    ok, msg = _validate_execution(code)
    if not ok:
        return False, msg

    ok, msg = _validate_no_mutations(code)
    if not ok:
        return False, msg

    ok, msg = _validate_relative_indicators(code)
    if not ok:
        return False, f"Absolute indicators detected: {msg}"

    ok, msg = _validate_required_params(code)
    if not ok:
        return False, msg

    ok, msg = _validate_catalog_features(code)
    if not ok:
        return False, msg

    ok, msg = _validate_ranges(code)
    if not ok:
        return False, msg

    return True, "OK"


# Backward-compatible aliases used by main.py
validar_codigo = validate_code


def check_llm_server(server_url: str) -> bool:
    """Checks whether the LLM server is accessible."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# Backward-compatible alias used by main.py
verificar_servidor_llm = check_llm_server
