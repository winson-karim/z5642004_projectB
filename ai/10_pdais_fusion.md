# AI Log 10 — PDAIS 2.0 Fusion (apply_pdais)(Price Dynamic Asset Investability Score)
## What I wanted
A look-ahead-safe way to tilt fund weights toward assets with higher PDAIS,
with a clean unit test proving the monthly lag and the score reversal.
## What the assistant produced
`apply_pdais()` in `src/fusion.py`: for each monthly rebalance it reads each
asset's PDAIS for the PREVIOUS calendar month and scales the weight by
`(1 + tilt_strength * (score/50 - 1))`, so a score of 50 is neutral. Rows are
renormalised to sum to 1 and capped at `max_weight` via the existing
`_cap_weights()`.
## What was wrong or risky
- The function did not exist; the first import failed with ImportError.
- Look-ahead risk: using the current month's score at that month's rebalance
  would leak information. The key design check is that a rebalance in month M
  must use month M-1 scores.
- PDAIS is 0-100, unlike sentiment (-1, 1), so the raw score cannot be used
  directly as a factor; it is rescaled to a symmetric signal around 50.
## What I changed and why
I implemented the one-month lag as `date - pd.DateOffset(months=1)` mapped to a
`YYYY-MM` key, looked up per asset through the ticker map, rescaled the 0-100
score to a [-1, 1] signal, and renormalised and capped each row. I kept the
design and helpers consistent with the existing `apply_sentiment()`.
## Verification
Unit test with four assets and scores that flip between months:
- 2021-02-01 rebalance uses 2021-01 scores: A (90) 0.30 > D (10) 0.20
- 2021-03-01 rebalance uses 2021-02 scores: D (90) 0.30 > A (10) 0.20
- row sums 1.0, all weights non-negative and below 0.40
## Final decision
Keep `apply_pdais()` as the PDAIS fusion extension. It will be compared against
the equal-weight baseline and the sentiment-fused variant on the full
risk-return trade-off and turnover before it is added to the report.
