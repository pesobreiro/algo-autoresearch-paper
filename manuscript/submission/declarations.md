# Declarations

## Data Availability Statement

The raw OHLCV data used in this study are publicly available from the Binance public API. The code, experiment logs, trade-level CSVs, generated figures, and reproducibility scripts are available in the repository associated with this submission. A persistent archived snapshot will be deposited upon acceptance (DOI to be assigned). In the interim, the repository can be obtained from the corresponding author on reasonable request.

To reproduce the two case studies: (1) install the pinned Python environment listed in `requirements.txt`; (2) download 15-minute, 4-hour, and daily OHLCV bars for BNB/USDT and BTC/USDC from the Binance public API; (3) start the local Qwen2.5-7B-Instruct GGUF server via llama.cpp; (4) run `python main.py` for the desired season; (5) for the selected iterations, run `deployment/evaluate_models.py` and `deployment/backtest_deploy.py` using the saved model and parameters in `best_models/season_{N}/iter_{XXXX}/`. A `download_data.py` helper script that performs the exact Binance API calls used in this study is included in the repository. Byte-identical reproduction across operating systems is not guaranteed because of Numba JIT compilation and BLAS threading; the statistical conclusions, however, are robust to the residual non-determinism.

## Ethics Declaration

This study uses publicly available historical market data and simulation-based backtesting. No human subjects, personally identifiable information, or animal data were involved. Ethical approval was therefore not required.

## Informed Consent Statement

Not applicable. This research did not involve human subjects.

## Author Contributions (CRediT)

Conceptualisation, methodology, software, validation, formal analysis, investigation, data curation, writing — original draft, writing — review and editing, visualisation, supervision: Pedro Sobreiro.

## Conflict of Interest Statement

No conflict of interest is declared. The research was conducted independently of any trading firm, exchange, or asset-management company.

## Funding Acknowledgment

No external funding was received for this research.

## AI Usage Disclosure

Parts of the code-generation and text-revision workflow were assisted by a locally deployed large language model (Qwen2.5-7B-Instruct via llama.cpp). The LLM proposed candidate strategies and drafting suggestions; all methodology, validation, statistical analysis, and final editorial decisions remained under human control.

The table below summarises the LLM involvement by manuscript component. "Generated" means the LLM produced a first draft or code scaffold; "Assisted" means the LLM suggested revisions or alternatives that were then edited and verified by the authors; "Human" means the component was produced without LLM assistance.

| Component | LLM role | Human verification |
|---|---|---|
| Research question and methodological design | Human | — |
| Feature catalog and validation constraints | Human | — |
| LLM agent prompts and JSON schema | Human | — |
| Python implementation of pipeline and deployment scripts | Assisted (scaffolding, refactoring) | Code review, unit tests, backtest reconciliation |
| Candidate strategy proposals (S1--S12) | Generated | Multi-layer validator and human season-review gates |
| Tables and figures | Assisted (scripted generation) | Manual inspection and label verification |
| Manuscript draft | Assisted (structuring, wording, revision) | All claims, numbers, and interpretations verified by authors |
| Statistical analysis and interpretation | Human | — |
| Final editorial decisions | Human | — |

No proprietary, personal, or non-public data were sent to external APIs. All LLM inference ran locally on the authors' workstation.
