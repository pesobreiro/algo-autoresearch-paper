"""
Agente LLM — propõe modificações ao research_params.py.

Usa Qwen2.5-7B via llama.cpp (API OpenAI-compatível em localhost:8080).
Valida código proposto: sintaxe Python + indicadores relativos (AST).

Regra crítica: NUNCA aceitar código com df['Close'], df['Open'], etc.
"""
import ast
import re
import requests
import json
from pathlib import Path
from typing import Optional


# Colunas de preço absoluto proibidas
PRICE_COLUMNS = {'Close', 'Open', 'High', 'Low', 'Volume',
                 'close', 'open', 'high', 'low', 'volume'}

SYSTEM_PROMPT = """És um Engenheiro de Machine Learning especializado em Trading Quantitativo.
A tua tarefa é modificar o ficheiro research_params.py para maximizar o score_composto
via Otimização Bayesiana (Optuna TPE Sampler já implementado no backtest).

RESPONSABILIDADES:
1. SELEÇÃO DE FEATURES: Escolhe entre 8-15 features do catálogo que façam sentido para o
   regime de mercado (ex: alta volatilidade → ATR + BB Width; tendência → EMA diff + MACD)
2. DEFINIÇÃO DE BOUNDS (intervalos): Define SL_RANGE, TP_RANGE, THRESHOLD_RANGE como tuplos
   (min, max). A Optuna explora este espaço automaticamente — não precisas de valores exatos.
3. AJUSTE DE REGULARIZAÇÃO XGBoost: Se o histórico mostrar degradação inter-anual ou
   scores inconsistentes, aumenta REG_ALPHA e REG_LAMBDA para reduzir overfitting.

REGRAS ABSOLUTAS (nunca violar):
1. APENAS indicadores relativos — NUNCA usar df['Close'], df['Open'], df['High'], df['Low']
2. Usar apenas features do catálogo permitido (lista abaixo)
3. O ficheiro deve ser Python válido e sintaticamente correto
4. SL_RANGE, TP_RANGE, THRESHOLD_RANGE são TUPLOS (min, max) — não listas nem valores únicos
5. TP_RANGE[0] deve ser sempre maior que SL_RANGE[0] (TP mínimo > SL mínimo)

Features permitidas — LISTA EXAUSTIVA (não inventar outras):
  stoch_rsi_k, stoch_rsi_d, rsi, bb_position, adx,
  ema_diff, trend, returns_1, atr_pct, bb_width_pct,
  macd_pct, macd_signal_pct, macd_hist_pct,
  volume_norm, returns_5,
  dist_sma200_pct, btc_trend, atr_regime

Notas sobre features macro (requerem "1d" em TIMEFRAMES):
  dist_sma200_pct — distância à SMA200 normalizada (%, apenas 1d); positivo = acima da SMA200
  btc_trend       — BTC acima/abaixo da EMA50 (0=baixa, 1=alta, apenas 1d); cross-asset
  atr_regime      — ATR / rolling_mean(ATR,50) por timeframe; >1 = volatilidade acima da média

PROIBIDO usar qualquer nome fora desta lista (ex: macd_diff, macd_signal, ema_ratio, etc.).
Qualquer feature não listada acima causará erro e rejeição imediata do código.

Timeframes: ["15m", "4h", "1d"] (qualquer subconjunto não-vazio)

Score composto (a MAXIMIZAR):
  score = tanh(S/2)*0.50 + tanh(R/100)*0.30 - abs(DD)/100*0.20
  onde S=Sharpe(365d), R=retorno anual%, DD=max drawdown%

ESTRATÉGIA DE AJUSTE DE BOUNDS:
- Se a Optuna encontrou o melhor SL perto do limite INFERIOR → reduce SL_RANGE[0]
- Se a Optuna encontrou o melhor SL perto do limite SUPERIOR → aumenta SL_RANGE[1]
- Mesma lógica para TP_RANGE e THRESHOLD_RANGE
- Bounds mais estreitos em torno da região boa aceleram convergência

OTIMIZAÇÃO XGBOOST COM OPTUNA:
- Por defeito MANTER N_TRIALS_XGB = 0 (desativado) — prioridade é velocidade e exploração
- Só ativar se o program.md indicar explicitamente modo EXPLOIT com xgb_trials > 0

FORMATO OBRIGATÓRIO: o ficheiro deve conter APENAS atribuições de variáveis e comentários.
PROIBIDO: FEATURES.remove(...), FEATURES.append(...), ou qualquer mutação de lista.
Se quiseres remover uma feature, redefine FEATURES como uma nova lista completa.

Responde APENAS com o conteúdo completo do ficheiro research_params.py.
NÃO incluir explicações fora do ficheiro. Usa comentários Python dentro do ficheiro.
"""


def _validar_sintaxe(codigo: str) -> tuple[bool, str]:
    """Valida que o código é Python sintaticamente válido."""
    try:
        ast.parse(codigo)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Erro de sintaxe na linha {e.lineno}: {e.msg}"


def _validar_indicadores_relativos(codigo: str) -> tuple[bool, str]:
    """
    Valida via AST que o código não acede a colunas de preço absoluto.

    Rejeita padrões como: df['Close'], df["open"], data['High'], etc.
    """
    try:
        tree = ast.parse(codigo)
    except SyntaxError:
        return False, "Código inválido (erro de sintaxe)"

    class PriceColumnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.violations = []

        def visit_Subscript(self, node):
            # Detecta padrões: algo['Close'] ou algo["open"]
            if isinstance(node.slice, ast.Constant):
                val = node.slice.value
                if isinstance(val, str) and val in PRICE_COLUMNS:
                    self.violations.append(
                        f"Acesso proibido a coluna de preço: '{val}' (linha {node.lineno})"
                    )
            self.generic_visit(node)

    visitor = PriceColumnVisitor()
    visitor.visit(tree)

    if visitor.violations:
        return False, "; ".join(visitor.violations)
    return True, "OK"


def _validar_params_obrigatorios(codigo: str) -> tuple[bool, str]:
    """Verifica que os parâmetros obrigatórios estão definidos."""
    obrigatorios = [
        'FEATURES', 'TIMEFRAMES', 'ENTRY_STOCH_THRESHOLD', 'ENTRY_ADX_THRESHOLD',
        'N_ESTIMATORS', 'MAX_DEPTH', 'LEARNING_RATE', 'SL_RANGE', 'TP_RANGE',
        'THRESHOLD_RANGE', 'N_TRIALS', 'OBJECTIVE_MODE',
    ]
    em_falta = [p for p in obrigatorios if p not in codigo]
    if em_falta:
        return False, f"Parâmetros obrigatórios em falta: {em_falta}"
    return True, "OK"


def _validar_execucao(codigo: str) -> tuple[bool, str]:
    """Executa o código num namespace isolado para detectar qualquer erro de runtime."""
    try:
        exec(compile(codigo, '<research_params>', 'exec'), {})
        return True, "OK"
    except Exception as e:
        return False, f"Erro de execução ({type(e).__name__}): {e}"


def _validar_sem_mutacoes(codigo: str) -> tuple[bool, str]:
    """Rejeita código que muta FEATURES ou TIMEFRAMES após definição (ex: FEATURES.remove(...))."""
    proibidos = ['FEATURES.remove(', 'FEATURES.append(', 'FEATURES.pop(',
                 'FEATURES.extend(', 'FEATURES.insert(', 'FEATURES.clear(',
                 'TIMEFRAMES.remove(', 'TIMEFRAMES.append(']
    for p in proibidos:
        if p in codigo:
            return False, (f"'{p}' não é permitido — define FEATURES como lista completa "
                           f"em vez de mutar após definição")
    return True, "OK"


def _validar_features_catalogo(codigo: str) -> tuple[bool, str]:
    """Extrai FEATURES do código via AST e valida contra o catálogo."""
    try:
        from pipeline.features_catalog import validate_features
        tree = ast.parse(codigo)
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
                                return False, f"Feature inválida em FEATURES: {msg}"
        return True, "OK"
    except Exception as e:
        return True, "OK"  # não bloquear por erro de parsing secundário


def extrair_codigo(resposta: str) -> Optional[str]:
    """
    Extrai o código Python da resposta do LLM.

    Tenta primeiro blocos ```python...```, depois todo o conteúdo.
    """
    # Tentar extrair bloco de código
    padroes = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```',
    ]
    for padrao in padroes:
        match = re.search(padrao, resposta, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Se não há marcadores de código, assumir que a resposta é o código
    linhas = resposta.strip().split('\n')
    # Remover linhas que claramente não são código
    codigo_linhas = []
    for linha in linhas:
        if linha.startswith('#') or '=' in linha or linha.strip() == '' or linha.startswith('[') or linha.startswith('"') or linha.startswith("'"):
            codigo_linhas.append(linha)
        elif any(kw in linha for kw in ['FEATURES', 'TIMEFRAMES', 'STOCH', 'ADX', 'EMA', 'SL_RANGE', 'TP_RANGE', 'THRESHOLD_RANGE', 'N_TRIALS', 'N_ESTIMATORS']):
            codigo_linhas.append(linha)

    if len(codigo_linhas) > 5:
        return '\n'.join(codigo_linhas)

    return None


def propor_novos_params(params_code: str, program_md: str,
                        historico: list, config: dict,
                        melhor_registo: dict = None,
                        top_registos: list = None,
                        rejeicoes_recentes: list = None,
                        temperature: float = None) -> Optional[str]:
    """
    Chama o LLM para propor um novo research_params.py.

    Args:
        params_code: conteúdo atual do research_params.py
        program_md: conteúdo do program.md (direção da pesquisa)
        historico: lista de dicts com resultados anteriores
        config: configuração do sistema

    Returns:
        Novo conteúdo do research_params.py, ou None se falhar
    """
    server_url = config.get('llm', {}).get('server_url', 'http://localhost:8080')
    max_tokens = config.get('llm', {}).get('max_tokens', 2048)
    if temperature is None:
        temperature = config.get('llm', {}).get('temperature', 0.7)
    timeout    = config.get('llm', {}).get('timeout_seconds', 120)

    # Melhor resultado até agora
    melhor_texto = ""
    if melhor_registo:
        m = melhor_registo.get('metricas', {})
        ps = melhor_registo.get('params_snapshot', {})

        # Sensibilidade Optuna: onde o melhor SL/TP/threshold ficou em relação aos bounds
        sensitivity = ""
        sl_best = m.get('sl_pct')
        tp_best = m.get('tp_pct')
        thr_best = m.get('threshold')
        sl_range = ps.get('SL_RANGE', (0.5, 12.0))
        tp_range = ps.get('TP_RANGE', (1.0, 40.0))
        thr_range = ps.get('THRESHOLD_RANGE', (0.30, 0.80))

        if sl_best is not None and sl_range:
            sl_lo, sl_hi = sl_range
            sl_rel = (sl_best - sl_lo) / max(sl_hi - sl_lo, 1e-6)
            if sl_rel < 0.2:
                sensitivity += f"    SL ótimo={sl_best:.2f}% perto do LIMITE INFERIOR ({sl_lo}) → considera reduzir SL_RANGE[0]\n"
            elif sl_rel > 0.8:
                sensitivity += f"    SL ótimo={sl_best:.2f}% perto do LIMITE SUPERIOR ({sl_hi}) → considera aumentar SL_RANGE[1]\n"
            else:
                sensitivity += f"    SL ótimo={sl_best:.2f}% no centro do intervalo [{sl_lo},{sl_hi}] → bounds adequados\n"

        if tp_best is not None and tp_range:
            tp_lo, tp_hi = tp_range
            tp_rel = (tp_best - tp_lo) / max(tp_hi - tp_lo, 1e-6)
            if tp_rel < 0.2:
                sensitivity += f"    TP ótimo={tp_best:.2f}% perto do LIMITE INFERIOR ({tp_lo}) → considera reduzir TP_RANGE[0]\n"
            elif tp_rel > 0.8:
                sensitivity += f"    TP ótimo={tp_best:.2f}% perto do LIMITE SUPERIOR ({tp_hi}) → considera aumentar TP_RANGE[1]\n"
            else:
                sensitivity += f"    TP ótimo={tp_best:.2f}% no centro do intervalo [{tp_lo},{tp_hi}] → bounds adequados\n"

        if thr_best is not None and thr_range:
            thr_lo, thr_hi = thr_range
            thr_rel = (thr_best - thr_lo) / max(thr_hi - thr_lo, 1e-6)
            if thr_rel < 0.2:
                sensitivity += f"    Threshold ótimo={thr_best:.3f} perto do LIMITE INFERIOR ({thr_lo}) → considera reduzir THRESHOLD_RANGE[0]\n"
            elif thr_rel > 0.8:
                sensitivity += f"    Threshold ótimo={thr_best:.3f} perto do LIMITE SUPERIOR ({thr_hi}) → considera aumentar THRESHOLD_RANGE[1]\n"
            else:
                sensitivity += f"    Threshold ótimo={thr_best:.3f} no centro do intervalo [{thr_lo},{thr_hi}] → bounds adequados\n"

        # Análise forense do drawdown
        df_info = m.get('drawdown_forensics', {})
        forensic_txt = ""
        if df_info:
            forensic_txt = (
                f"  Análise forense do Max Drawdown ({df_info.get('ano','?')}):\n"
                f"    Duração: {df_info.get('dur_dias','?')} dias | "
                f"Profundidade: {df_info.get('profundidade','?'):.1f}%\n"
                f"    Regime de mercado: {df_info.get('regime','?')}\n"
                f"    ADX médio no período: {df_info.get('adx_medio','?')} (mín: {df_info.get('adx_minimo','?')})\n"
            )
            # Sugestão automática baseada no regime
            adx_m = df_info.get('adx_medio', 30)
            if adx_m < 20:
                forensic_txt += "    → Sugestão: aumentar ENTRY_ADX_THRESHOLD ou ADX_PERIOD para filtrar regimes sem tendência\n"
            elif adx_m < 25:
                forensic_txt += "    → Sugestão: considerar reduzir SL_RANGE (SL mais apertado em tendências fracas)\n"

        # Optuna parameter importance do melhor resultado
        optuna_imp = m.get('optuna_param_importance', {})
        optuna_imp_txt = ""
        if optuna_imp:
            sorted_oi = sorted(optuna_imp.items(), key=lambda x: -x[1])
            dominant = sorted_oi[0][0] if sorted_oi else None
            optuna_imp_txt = "  Optuna param importance (qual dimensão mais afeta o score):\n"
            optuna_imp_txt += "    " + " | ".join(f"{k}={v:.3f}" for k,v in sorted_oi) + "\n"
            if dominant:
                optuna_imp_txt += f"    → Afinar principalmente: {dominant} (maior impacto no score)\n"

        # Feature importance do melhor resultado
        top_feats    = m.get('top_features', [])
        bottom_feats = m.get('bottom_features', [])
        feat_imp_txt = ""
        if top_feats:
            feat_imp_txt += "  Feature importance (XGBoost):\n"
            feat_imp_txt += "    TOP (manter/reforçar): " + ", ".join(f"{f}={v:.3f}" for f,v in top_feats[:6]) + "\n"
            if bottom_feats:
                feat_imp_txt += "    FRACAS (<0.02, considerar remover): " + ", ".join(f for f,_ in bottom_feats) + "\n"
        auc_txt = ""
        if m.get('cv_auc_mean'):
            auc_txt = f"  CV ROC-AUC: {m['cv_auc_mean']:.4f} ± {m.get('cv_auc_std',0):.4f}\n"

        # Score breakdown por componente
        S  = m.get('sharpe_raw', 0)
        R  = m.get('retorno_anual_pct', 0)
        DD = abs(m.get('max_drawdown_pct', 0))
        import math
        s_contrib  = math.tanh(S / 2) * 0.50
        r_contrib  = math.tanh(R / 100) * 0.30
        dd_penalty = (DD / 100) * 0.20
        score_breakdown = (
            f"  Score breakdown: Sharpe={s_contrib:+.3f}/0.50 | Return={r_contrib:+.3f}/0.30 | DD={-dd_penalty:.3f}/-0.20\n"
            f"  → Maior margem disponível: "
            + ("Sharpe" if (0.50 - s_contrib) > (0.30 - r_contrib) else "Return")
            + f" ({max(0.50 - s_contrib, 0.30 - r_contrib):.3f} pontos de ganho possível)\n"
        )

        melhor_texto = (
            f"\n\nMELHOR RESULTADO ATÉ AGORA (iter {melhor_registo.get('iteracao','?')}):\n"
            f"  Score={m.get('score_composto',0):.4f} | Sharpe={m.get('sharpe_raw',0):.2f} | "
            f"Return={m.get('retorno_anual_pct',0):+.1f}% | DD={DD:.1f}%\n"
            f"{auc_txt}"
            f"{score_breakdown}"
            f"{forensic_txt}"
            f"{optuna_imp_txt}"
            f"{feat_imp_txt}"
            f"  Params que geraram este resultado:\n"
            f"    FEATURES={ps.get('FEATURES','?')}\n"
            f"    TIMEFRAMES={ps.get('TIMEFRAMES','?')}\n"
            f"    ENTRY_STOCH_THRESHOLD={ps.get('ENTRY_STOCH_THRESHOLD','?')} | ENTRY_ADX_THRESHOLD={ps.get('ENTRY_ADX_THRESHOLD','?')}\n"
            f"    N_ESTIMATORS={ps.get('N_ESTIMATORS','?')} | MAX_DEPTH={ps.get('MAX_DEPTH','?')} | LR={ps.get('LEARNING_RATE','?')}\n"
            f"    SL_RANGE={ps.get('SL_RANGE','?')} | TP_RANGE={ps.get('TP_RANGE','?')} | THRESHOLD_RANGE={ps.get('THRESHOLD_RANGE','?')}\n"
            + (f"    XGBoost Optuna best: {m.get('xgb_optuna_best')}\n" if m.get('xgb_optuna_best') else "")
            + f"  Análise de sensibilidade Optuna:\n"
            + f"{sensitivity}"
            + f"  O objetivo é SUPERAR este score. Explora dimensões ainda não testadas.\n"
        )

    # Top N resultados históricos + análise de padrão
    top_texto = ""
    if top_registos and len(top_registos) > 1:
        top_texto = "\n\nTOP RESULTADOS HISTÓRICOS (zona a explorar — padrão identificado):\n"
        tfs_counter: dict = {}
        sl_vals, tp_vals, thr_vals = [], [], []
        for i, h in enumerate(top_registos):
            m = h.get('metricas', {})
            ps = h.get('params_snapshot', {})
            sl  = m.get('sl_pct')
            tp  = m.get('tp_pct')
            thr = m.get('threshold')
            tfs = str(ps.get('TIMEFRAMES', '?'))
            top_texto += (
                f"  #{i+1} iter={h.get('iteracao','?')} score={m.get('score_composto',0):.4f} | "
                f"Sharpe={m.get('sharpe_raw',0):.2f} | Return={m.get('retorno_anual_pct',0):+.1f}% | "
                f"DD={abs(m.get('max_drawdown_pct',0)):.1f}% | "
                f"Trades={m.get('n_trades',0)} | WR={m.get('win_rate_pct',0):.1f}%\n"
                f"    Optuna best: SL={sl:.2f}% TP={tp:.2f}% Thr={thr:.3f}\n"
                f"    TIMEFRAMES={ps.get('TIMEFRAMES','?')} | "
                f"FEATURES={len(ps.get('FEATURES',[]))} features | "
                f"Entry: stoch<{ps.get('ENTRY_STOCH_THRESHOLD','?')} adx>{ps.get('ENTRY_ADX_THRESHOLD','?')}\n"
            ) if sl and tp and thr else (
                top_texto + f"  #{i+1} iter={h.get('iteracao','?')} score={m.get('score_composto',0):.4f}\n"
            )
            if tfs not in tfs_counter:
                tfs_counter[tfs] = 0
            tfs_counter[tfs] += 1
            if sl: sl_vals.append(sl)
            if tp: tp_vals.append(tp)
            if thr: thr_vals.append(thr)

        # Padrão comum
        if sl_vals and tp_vals and thr_vals:
            top_texto += (
                f"\n  PADRÃO DOMINANTE nos top resultados:\n"
                f"    SL médio={sum(sl_vals)/len(sl_vals):.2f}% (min={min(sl_vals):.2f} max={max(sl_vals):.2f})\n"
                f"    TP médio={sum(tp_vals)/len(tp_vals):.2f}% (min={min(tp_vals):.2f} max={max(tp_vals):.2f})\n"
                f"    Threshold médio={sum(thr_vals)/len(thr_vals):.3f} (min={min(thr_vals):.3f} max={max(thr_vals):.3f})\n"
                f"    TIMEFRAMES mais frequentes: {max(tfs_counter, key=tfs_counter.get)}\n"
                f"  → Explora VARIAÇÕES nesta zona: features diferentes, entry signal, XGBoost params.\n"
                f"  → NÃO repitas os mesmos params — o sistema deteta duplicados automaticamente.\n"
            )

    # Histórico recente (últimas 30 iterações)
    hist_recente = historico[-30:] if len(historico) > 30 else historico
    hist_texto = ""
    if hist_recente:
        hist_texto = "\n\nHISTÓRICO RECENTE (últimas 30 iterações testadas):\n"
        for h in hist_recente:
            m = h.get('metricas', {})
            top_f    = m.get('top_features', [])
            bot_f    = m.get('bottom_features', [])
            oi       = m.get('optuna_param_importance', {})
            hist_texto += (
                f"  Iter {h.get('iteracao', '?')}: "
                f"Score={m.get('score_composto', 0):.4f} | "
                f"Sharpe={m.get('sharpe_raw', 0):.2f} | "
                f"Return={m.get('retorno_anual_pct', 0):+.1f}% | "
                f"DD={abs(m.get('max_drawdown_pct', 0)):.1f}% | "
                f"AUC={m.get('cv_auc_mean', 0):.4f} | "
                f"Status={h.get('status', '?')}\n"
            )
            if oi:
                sorted_oi = sorted(oi.items(), key=lambda x: -x[1])
                hist_texto += f"    Optuna importance: {' | '.join(f'{k}={v:.3f}' for k,v in sorted_oi)}\n"
            if top_f:
                hist_texto += f"    Top features: {', '.join(f'{f}={v:.3f}' for f,v in top_f[:4])}\n"
            if bot_f:
                hist_texto += f"    Features fracas: {', '.join(f for f,_ in bot_f)}\n"
            if h.get('alteracoes_vs_anterior'):
                hist_texto += f"    Alterações: {h['alteracoes_vs_anterior']}\n"

    # Feedback de rejeições recentes
    rejeicoes_txt = ""
    if rejeicoes_recentes:
        rejeicoes_txt = "\n\n⚠️  REJEIÇÕES RECENTES — NÃO REPETIR ESTES ERROS:\n"
        for r in rejeicoes_recentes:
            rejeicoes_txt += f"  - {r}\n"
        rejeicoes_txt += (
            "Estas propostas foram rejeitadas automaticamente ANTES de correr o pipeline.\n"
            "Corrige APENAS o erro indicado. Usa SOMENTE features da lista exaustiva acima.\n"
        )

    user_prompt = f"""DIREÇÃO DA PESQUISA (program.md):
{program_md}
{melhor_texto}{top_texto}{hist_texto}{rejeicoes_txt}

PARÂMETROS ATUAIS (research_params.py) — estes são os params do melhor resultado:
```python
{params_code}
```

Com base no histórico e na direção da pesquisa, propõe UMA modificação
específica e fundamentada para tentar SUPERAR o melhor score.
Evita repetir alterações recentes que já foram rejeitadas.
Explora dimensões novas (ex: se já testaste MAX_DEPTH, tenta agora FEATURES ou TIMEFRAMES).

Responde APENAS com o novo conteúdo completo do ficheiro research_params.py.
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
        resp = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        conteudo = data['choices'][0]['message']['content']

        # Gravar prompt + resposta completa (inclui raciocínio descartado pelo extrair_codigo)
        _log_llm_interaction(user_prompt, conteudo)

        return extrair_codigo(conteudo)
    except requests.exceptions.ConnectionError:
        print(f"  [AGENTE] Erro: LLM server não acessível em {server_url}")
        return None
    except requests.exceptions.Timeout:
        print(f"  [AGENTE] Timeout ao chamar LLM ({timeout}s)")
        return None
    except Exception as e:
        print(f"  [AGENTE] Erro inesperado: {e}")
        return None


def _log_llm_interaction(prompt: str, resposta: str, max_logs: int = 50) -> None:
    """
    Grava prompt e resposta completa do LLM em logs/llm_interactions/.

    Mantém apenas os últimos max_logs pares para não encher o disco.
    A resposta inclui o raciocínio do modelo antes do bloco de código.
    """
    from datetime import datetime
    log_dir = Path(__file__).parent.parent / 'logs' / 'llm_interactions'
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (log_dir / f"prompt_{ts}.txt").write_text(prompt, encoding='utf-8')
    (log_dir / f"resposta_{ts}.txt").write_text(resposta, encoding='utf-8')

    # Rotação: apagar os mais antigos se ultrapassar max_logs pares
    prompts = sorted(log_dir.glob("prompt_*.txt"))
    if len(prompts) > max_logs:
        for old in prompts[:len(prompts) - max_logs]:
            old.unlink(missing_ok=True)
            ts_old = old.stem.replace("prompt_", "")
            resp_old = log_dir / f"resposta_{ts_old}.txt"
            resp_old.unlink(missing_ok=True)


def _validar_ranges(codigo: str) -> tuple[bool, str]:
    """Valida que SL_RANGE, TP_RANGE e THRESHOLD_RANGE têm min <= max e respeitam focus bounds."""
    ns = {}
    try:
        exec(compile(codigo, '<research_params>', 'exec'), ns)
    except Exception:
        return True, "OK"  # exec errors já apanhados por _validar_execucao

    for nome in ('SL_RANGE', 'TP_RANGE', 'THRESHOLD_RANGE'):
        val = ns.get(nome)
        if val is not None:
            try:
                lo, hi = val
                if lo > hi:
                    return False, f"{nome}=({lo},{hi}) inválido: min > max — trocar os valores"
            except (TypeError, ValueError):
                pass

    return True, "OK"


def validar_codigo(codigo: str) -> tuple[bool, str]:
    """
    Validação completa do código proposto.

    Returns:
        (valido, mensagem_erro)
    """
    ok, msg = _validar_sintaxe(codigo)
    if not ok:
        return False, msg

    ok, msg = _validar_execucao(codigo)
    if not ok:
        return False, msg

    ok, msg = _validar_sem_mutacoes(codigo)
    if not ok:
        return False, msg

    ok, msg = _validar_indicadores_relativos(codigo)
    if not ok:
        return False, f"Indicadores absolutos detectados: {msg}"

    ok, msg = _validar_params_obrigatorios(codigo)
    if not ok:
        return False, msg

    ok, msg = _validar_features_catalogo(codigo)
    if not ok:
        return False, msg

    ok, msg = _validar_ranges(codigo)
    if not ok:
        return False, msg

    return True, "OK"


def verificar_servidor_llm(server_url: str) -> bool:
    """Verifica se o servidor LLM está acessível."""
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
