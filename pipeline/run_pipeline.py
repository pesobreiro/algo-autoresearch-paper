"""
Orquestrador do pipeline algo_autoresearch.

Coordena: generate_labels → train → backtest
Com cache de labels keyed por hash dos parâmetros de entrada.
"""
import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from typing import Optional

from pipeline.generate_labels import gerar_labels
from pipeline.train import treinar_modelo
from pipeline.backtest import correr_backtest


@dataclass
class ResultadoPipeline:
    """Resultado completo de uma iteração do pipeline."""
    sucesso: bool
    metricas: dict = field(default_factory=dict)
    stats_labels: dict = field(default_factory=dict)
    stats_treino: dict = field(default_factory=dict)
    labels_reutilizados: bool = False
    duracao_total_segundos: float = 0.0
    erro: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def carregar_params(params_path: Path) -> dict:
    """Carrega research_params.py como dict."""
    spec = importlib.util.spec_from_file_location("research_params", params_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return {
        'FEATURES':               getattr(mod, 'FEATURES', []),
        'TIMEFRAMES':             getattr(mod, 'TIMEFRAMES', ['15m', '4h']),
        'STOCH_RSI_PERIOD':       getattr(mod, 'STOCH_RSI_PERIOD', 14),
        'ADX_PERIOD':             getattr(mod, 'ADX_PERIOD', 14),
        'EMA_FAST':               getattr(mod, 'EMA_FAST', 12),
        'EMA_SLOW':               getattr(mod, 'EMA_SLOW', 26),
        'BB_PERIOD':              getattr(mod, 'BB_PERIOD', 20),
        'ENTRY_STOCH_THRESHOLD':  getattr(mod, 'ENTRY_STOCH_THRESHOLD', 20),
        'ENTRY_ADX_THRESHOLD':    getattr(mod, 'ENTRY_ADX_THRESHOLD', 25),
        'N_ESTIMATORS':           getattr(mod, 'N_ESTIMATORS', 200),
        'MAX_DEPTH':              getattr(mod, 'MAX_DEPTH', 6),
        'LEARNING_RATE':          getattr(mod, 'LEARNING_RATE', 0.05),
        'MIN_CHILD_WEIGHT':       getattr(mod, 'MIN_CHILD_WEIGHT', 5),
        'GAMMA':                  getattr(mod, 'GAMMA', 1.0),
        'SUBSAMPLE':              getattr(mod, 'SUBSAMPLE', 0.7),
        'COLSAMPLE_BYTREE':       getattr(mod, 'COLSAMPLE_BYTREE', 0.7),
        'REG_ALPHA':              getattr(mod, 'REG_ALPHA', 0.5),
        'REG_LAMBDA':             getattr(mod, 'REG_LAMBDA', 1.0),
        'SL_RANGE':               getattr(mod, 'SL_RANGE', (0.5, 12.0)),
        'TP_RANGE':               getattr(mod, 'TP_RANGE', (1.0, 40.0)),
        'THRESHOLD_RANGE':        getattr(mod, 'THRESHOLD_RANGE', (0.30, 0.80)),
        'N_TRIALS':               getattr(mod, 'N_TRIALS', 120),
        # XGBoost Optuna ranges (opcionais — ativam otimização de hiperparâmetros)
        'DEPTH_RANGE':            getattr(mod, 'DEPTH_RANGE', None),
        'LR_RANGE':               getattr(mod, 'LR_RANGE', None),
        'ESTIMATORS_RANGE':       getattr(mod, 'ESTIMATORS_RANGE', None),
        'ALPHA_RANGE':            getattr(mod, 'ALPHA_RANGE', None),
        'LAMBDA_RANGE':           getattr(mod, 'LAMBDA_RANGE', None),
        'N_TRIALS_XGB':           getattr(mod, 'N_TRIALS_XGB', 0),
        'OBJECTIVE_MODE':         getattr(mod, 'OBJECTIVE_MODE', 'score'),
    }


def hash_entry_params(params: dict, train_start: int = 2019, train_end: int = 2021) -> str:
    """Hash dos parâmetros que afetam a geração de labels."""
    chave = {
        'ENTRY_STOCH_THRESHOLD': params['ENTRY_STOCH_THRESHOLD'],
        'ENTRY_ADX_THRESHOLD':   params['ENTRY_ADX_THRESHOLD'],
        'SL_RANGE':              list(params.get('SL_RANGE', (0.5, 12.0))),
        'TP_RANGE':              list(params.get('TP_RANGE', (1.0, 40.0))),
        'THRESHOLD_RANGE':       list(params.get('THRESHOLD_RANGE', (0.30, 0.80))),
        'train_start':           train_start,
        'train_end':             train_end,
    }
    return hashlib.md5(json.dumps(chave, sort_keys=True).encode()).hexdigest()[:8]


def hash_params_completo(params: dict) -> str:
    """Hash de todos os parâmetros (para identificar a iteração)."""
    serializable = {k: (sorted(v) if isinstance(v, list) else v)
                    for k, v in params.items()}
    return hashlib.md5(json.dumps(serializable, sort_keys=True).encode()).hexdigest()[:8]


def executar_pipeline(config: dict, params_path: Path,
                      cache_dir: Path) -> ResultadoPipeline:
    """
    Executa o pipeline completo: labels → treino → backtest.

    Cache de labels: labels são reutilizados se ENTRY_* params não mudaram.
    Treino e backtest são sempre re-executados.

    Args:
        config: configuração do sistema (config.yaml)
        params_path: path ao research_params.py
        cache_dir: directório para cache de labels e modelos

    Returns:
        ResultadoPipeline com métricas e metadados
    """
    t_inicio = time.time()

    try:
        params = carregar_params(params_path)
    except Exception as e:
        return ResultadoPipeline(sucesso=False, erro=f"Erro ao carregar params: {e}")

    train_start   = config['pipeline'].get('train_start', 2017)
    train_end     = config['pipeline'].get('train_end', 2024)
    params_hash   = hash_entry_params(params, train_start=train_start, train_end=train_end)
    hash_completo = hash_params_completo(params)

    labels_dir  = cache_dir / 'labels'
    model_dir   = cache_dir / 'models' / hash_completo
    ticker      = config['pipeline']['ticker']
    labels_path = labels_dir / f'labels_{ticker}_{params_hash}.parquet'
    hash_file   = labels_dir / f'hash_atual_{ticker}.txt'

    labels_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Passo 1: Labels (com cache) ---
    labels_reutilizados = False
    stats_labels = {}

    hash_anterior = hash_file.read_text().strip() if hash_file.exists() else ""

    if labels_path.exists() and hash_anterior == params_hash:
        print(f"\n  [CACHE] Labels reutilizados (hash={params_hash})")
        labels_reutilizados = True
        # Recuperar feature_cols do parquet
        import pandas as pd
        df_tmp = pd.read_parquet(labels_path)
        exclude = {'ticker', 'entry_idx', 'timestamp', 'label', 'pnl_pct', 'sl_pct', 'tp_pct'}
        feature_cols = [c for c in df_tmp.columns if c not in exclude]
        stats_labels = {'feature_cols': feature_cols, 'n_samples': len(df_tmp)}
    else:
        print(f"\n  [LABELS] Gerando novos labels (hash={params_hash})...")
        try:
            # Construir feature_cols a partir de FEATURES × TIMEFRAMES
            from pipeline.features_catalog import get_feature_columns as cat_get_cols
            feature_cols_target = cat_get_cols(params['FEATURES'], params['TIMEFRAMES'])
            stats_labels = gerar_labels(
                config, params, labels_path,
                feature_cols_override=None  # usar whitelist do get_feature_columns
            )
            feature_cols = stats_labels.get('feature_cols', [])
            hash_file.write_text(params_hash)
        except Exception as e:
            return ResultadoPipeline(
                sucesso=False,
                erro=f"Erro na geração de labels: {e}",
                duracao_total_segundos=time.time() - t_inicio,
            )

    # --- Passo 2: Treino ---
    try:
        stats_treino = treinar_modelo(
            config, params, labels_path, model_dir,
            feature_cols=stats_labels.get('feature_cols')
        )
    except Exception as e:
        return ResultadoPipeline(
            sucesso=False,
            stats_labels=stats_labels,
            labels_reutilizados=labels_reutilizados,
            erro=f"Erro no treino: {e}",
            duracao_total_segundos=time.time() - t_inicio,
        )

    # --- Passo 3: Backtest ---
    try:
        objective_mode = params.get('OBJECTIVE_MODE', 'score')
        metricas = correr_backtest(config, params, model_dir, objective_mode=objective_mode)
    except Exception as e:
        return ResultadoPipeline(
            sucesso=False,
            stats_labels=stats_labels,
            stats_treino=stats_treino,
            labels_reutilizados=labels_reutilizados,
            erro=f"Erro no backtest: {e}",
            duracao_total_segundos=time.time() - t_inicio,
        )

    duracao = time.time() - t_inicio

    # Propagar stats de treino para as métricas (usado pelo agente)
    metricas['top_features']    = stats_treino.get('top_features', [])
    metricas['bottom_features'] = stats_treino.get('bottom_features', [])
    metricas['cv_auc_mean']     = stats_treino.get('cv_auc_mean', 0.0)
    metricas['cv_auc_std']      = stats_treino.get('cv_auc_std', 0.0)
    metricas['xgb_optuna_best'] = stats_treino.get('xgb_optuna_best', {})

    return ResultadoPipeline(
        sucesso=True,
        metricas=metricas,
        stats_labels=stats_labels,
        stats_treino=stats_treino,
        labels_reutilizados=labels_reutilizados,
        duracao_total_segundos=duracao,
    )
