# Highlights

- A season-structured validation architecture for prescriptive analytics in which an LLM proposes directional candidates and a multi-layer validation system rejects unsound proposals before they reach a passive holdout.
- Leakage-free feature constraints, AST-based look-ahead detection, layered validation, and a holdout-blind selection protocol reduce the risk of data leakage and overfitting in automated strategy research.
- 34 333 candidate strategies evaluated across twelve experimental seasons yielded 329 accepted strategies; a selected BNB/USDT case study and a cross-asset BTC/USDC replication both show positive passive-holdout Sharpe ratios.
- Lightweight ablations indicate that the pipeline's value lies in systematic validation and rejection of unsound candidates rather than in uniquely superior LLM feature selection.
- The Q1 2026 period was used for production-model selection and is reported only as a noisy post-selection observation; bootstrap p-values and an approximate Deflated Sharpe Ratio are descriptive diagnostics that do not correct for the cross-season meta-search.
