# algo-autoresearch-paper

Code, data specification, and reproducibility artifacts for *A Season-Structured Validation and Governance Architecture for Autonomous Strategy-Proposal Generation in Algorithmic Trading*.

## Repository contents

- `autoresearch/` — autonomous LLM-driven strategy generation and validation loop.
- `pipeline/` — feature engineering, label generation, XGBoost training, and Optuna backtest.
- `deployment/` — scripts to re-run and stress-test the two accepted case studies.
- `best_models/` — final accepted models for S11 iter 1077 (BNB/USDT) and S12 iter 5502 (BTC/USDT).
- `manuscript/submission/` — paper source, figures, and tables.
- `llm/` — instructions to build llama.cpp and serve the local Qwen2.5-7B-Instruct model.

## Quick start

1. Clone the repo:
   ```bash
   git clone https://github.com/pesobreiro/algo-autoresearch-paper.git
   cd algo-autoresearch-paper
   ```

2. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Build and start the local LLM server (optional if you only want to reproduce the case studies):
   ```bash
   ./llm/setup.sh
   ./llm/start_server.sh
   ```

4. Download public OHLCV data from Binance:
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml if you want a different data directory
   python download_data.py
   ```

5. Re-run a case-study backtest:
   ```bash
   python deployment/backtest_deploy.py --season 11 --iter 1077 \
       --sl 7.80 --tp 7.10 --thr 0.857 --ticker bnb
   python deployment/backtest_deploy.py --season 12 --iter 5502 \
       --sl 9.70 --tp 9.12 --thr 0.890 --ticker btc
   ```

## Data

Raw OHLCV data comes from the public Binance API. No API key is required. The downloader fetches the symbols and timeframes used in the paper.

## Citation

Pedro Sobreiro, Domingos Martinho, Pedro Ramos, Antonio Pratas. *A Season-Structured Validation and Governance Architecture for Autonomous Strategy-Proposal Generation in Algorithmic Trading*. 2026.

## License

MIT — see LICENSE.
