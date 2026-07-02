# pipeline/research_params.py
# ONLY FILE MODIFIED BY THE AGENT
# CRITICAL RULE: ONLY RELATIVE INDICATORS (no absolute prices)
#
# SEASON 12 — EXPLOIT MODE: iter 8942 (score=0.93, sv=1.25, sh=0.67)
# Best OOS equity: 2024=€684, 2025=€630
# Bounds ±35% around SL=7.431, TP=9.748, THR=0.876

# --- Features to use in training ---
FEATURES = ["ema_diff_15m", "macd_pct_4h", "bb_width_pct_15m", "dist_sma200_pct", "btc_trend", "volume_norm", "adx_15m", "rsi_4h", "macd_hist_pct_15m", "stoch_rsi_k_1d", "atr_regime_4h", "macd_signal_pct_1d", "macd_pct_1d", "bb_width_pct_1d", "atr_regime_15m", "dist_sma200_pct_4h", "macd_pct_15m", "stoch_rsi_d_1d", "ema_diff_4h"]

# --- Timeframes to include ---
TIMEFRAMES = ["15m", "4h", "1d"]

# --- Technical indicator parameters ---
STOCH_RSI_PERIOD = 14
ADX_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
BB_PERIOD = 20

# --- Entry signal ---
ENTRY_STOCH_THRESHOLD = 28
ENTRY_ADX_THRESHOLD = 25

# --- XGBoost hyperparameters ---
N_ESTIMATORS = 800
MAX_DEPTH = 8
LEARNING_RATE = 0.05
MIN_CHILD_WEIGHT = 30
GAMMA = 0.0
SUBSAMPLE = 0.9
COLSAMPLE_BYTREE = 0.9
REG_ALPHA = 7.0
REG_LAMBDA = 7.0

# --- Bayesian XGBoost optimization (Optuna) ---
N_TRIALS_XGB = 0

# --- Bayesian SL/TP/Threshold optimization ---
SL_RANGE = (4.830, 10.032)
TP_RANGE = (6.336, 13.160)
THRESHOLD_RANGE = (0.701, 0.95)
N_TRIALS = 150

# --- Objective mode ---
OBJECTIVE_MODE = "score"
