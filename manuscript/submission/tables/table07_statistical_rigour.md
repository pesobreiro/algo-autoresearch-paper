## Table 7. Bootstrap p-values, approximate Deflated Sharpe Ratios, and 95 % confidence intervals

| Case | Annualized daily Sharpe | 95 % CI for Sharpe | Bootstrap p-value (H₀: μ ≤ 0) | 95 % CI for p-value | Approx. DSR | Active trading days | Skewness | Kurtosis | ACF(1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S11 iter 1077 | 13.75 | [6.72, 25.83] | < 0.0001 | [0.0000, 0.0003] | 11.78 | 45 | −1.46 | 3.44 | −0.22 |
| S11 iter 1066 | 9.97 | [4.48, 17.25] | < 0.0001 | [0.0000, 0.0003] | 8.32 | 63 | −0.91 | 5.25 | −0.24 |
| S12 iter 5502 | 7.80 | [2.59, 13.65] | 0.0012 | [0.0005, 0.0019] | 3.24 | 53 | −0.20 | 1.97 | 0.18 |

*Note: The approximate DSR corrects for the number of independent trials in the respective season (S11: 3 797; S12: 9 957) using the Bailey & López de Prado (2014) formula. Bootstrap uses 10 000 resamples with replacement of daily returns from active trading days with at least one closed position. The 95 % confidence intervals for the Sharpe ratio are percentile bootstrap intervals; they are wide because the sample of active trading days is small. ACF(1) is the first-order autocorrelation of daily returns. Because strategies share features and hyper-parameter ranges, the effective number of independent trials is smaller than the raw iteration count; the reported DSR values are therefore an upper bound on the correction. Jarque-Bera tests reject normality for S11 iter 1077 (p = 0.0003) and S11 iter 1066 (p < 0.0001) but not for S12 iter 5502 (p = 0.26).*
