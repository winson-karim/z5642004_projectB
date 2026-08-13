# AI Log 08 — Sentiment-Fusion Comparison
## What I wanted
Fuse the sector sentiment index into a systematic equity fund and compare the
base fund against the sentiment-fused variant on the full risk-return trade-off
and on average monthly turnover, not on return alone.
## What the assistant produced
An `apply_sentiment()` overlay in `src/fusion.py` that lags sentiment by one
trading day, asof-matches each monthly rebalance date to the sentiment calendar,
multiplies sector weights by `(1 + tilt_strength * sector_sentiment)`, renormalises,
and iteratively caps any asset above the 0.40 ceiling. `scripts/run_part_b.py`
now fuses `equity_equal_weight` (the clean 1/n x 50 baseline), holds the fused
weights between rebalance dates, and writes:
- results/data/sentiment_fusion_returns.csv
- results/data/sentiment_fusion_weights.csv
- results/tables/sentiment_fusion_comparison.csv
## What was wrong or risky
- Two intermediate fusion tests "failed" but were flawed test designs, not code
  bugs: a same-sector test whose sector factor cancelled out, and a two-asset
  test where both assets exceeded the 0.40 cap and both were clamped to 0.40.
- A name collision: a local `performance_metrics` DataFrame inside `main()`
  shadowed the imported `performance_metrics()` function and raised a TypeError.
- Look-ahead risk: sentiment must be used one trading day after it is observed.
## What I changed and why
I renamed the local DataFrame to `performance_table` so it no longer shadowed
the imported function. I verified the lag with a synthetic four-asset test: the
2023-01-02 rebalance tilts toward Tech using the prior trading day's positive
sentiment (+0.8) and ignores the same-day flip to -0.8, confirming the overlay
uses only past information.
## Verification
The fused weights sum to 1.0 on every rebalance date, are non-negative, and no
weight exceeds 0.40. The two-row comparison shows:
- equity_equal_weight: return 0.1264, vol 0.1617, Sharpe 0.8174, maxDD -0.2032,
  turnover 0.0000
- fused variant: return 0.1244, vol 0.1618, Sharpe 0.8061, maxDD -0.2057,
  turnover 0.0174
## Final decision
Keep the fusion as the baseline overlay and report it as an honest negative
result: the naive tilt does not improve the risk-adjusted trade-off and adds
~1.7% monthly one-way turnover, which motivates a stronger or conditional signal
(e.g. trade only on extreme sentiment) as a future recommendation.
