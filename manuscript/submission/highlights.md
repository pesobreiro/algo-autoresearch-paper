# Highlights

- We let a local LLM propose directional crypto strategies inside experimental "seasons," then run every proposal through syntax checks, AST-based look-ahead detection, catalog compliance, and a passive holdout before accepting it.
- The feature catalog and validation layers are deliberately leakage-free, so the search is less likely to discover strategies that only look good because they peek at the future.
- Across 34 333 candidates and twelve seasons, 329 strategies passed all gates. A BNB/USDT case study and a BTC/USDC replication both produced positive Sharpe ratios on holdout data the model never saw during selection.
- A small ablation suggests the pipeline's edge comes more from rejecting bad candidates systematically than from the LLM picking uniquely good features.
- We used Q1 2026 only to pick the production model, so those numbers are reported as a noisy post-selection snapshot. The bootstrap p-values and Deflated Sharpe Ratio are descriptive; they do not correct for the broader meta-search across seasons.
