# AI Log 09 — PDAIS Port and Project A Reproduction Check
## What I wanted
Bring Project A's Vestra Dynamic Asset Investability Score (PDAIS) into Project
B and verify the recomputed monthly scores match Project A's
`investability_monthly_scores.csv` exactly.
## What the assistant produced
The assistant found the Project A implementation at
`src/features.py` (`compute_investability_scores` plus helpers `_rolling_semi_dev`,
`_min_max_norm`, `_data_reliability_gate`) and ported it into Project B's
`src/features.py`.
## What was wrong or risky
- The edit claimed to have been applied to `src/features.py` did not land twice;
  the import failed with ImportError both times.
- Without capping, the crypto input includes stray 2024-01-01 rows, so the
  monthly grid has 49 months (2,890 rows) instead of the 48-month Project A grid
  (2,880 rows).
- A 21-day rolling window cannot be filled during the first calendar month, so
  the 50 Equity 2020-01 rows are NaN; these are expected artifacts also present
  in Project A.
## What I changed and why
I reviewed the Project A code for compatibility (same data bundle, `date` already
datetime, `volume`/`adjClose`/`sector` present) and then ported the function and
its three helpers verbatim, adding the `numpy` import. Consistent with the
existing Project B sample convention, crypto inputs are capped at 2023-12-31
before scoring. The 50 Equity 2020-01 NaNs are intentionally left unfilled.
## Verification
- Rows match: 2,880
- Months match: 48
- Ticker/month keys match: 100% (0 mismatches)
- PDAIS values match: max abs diff 1.4e-14 across all raw/normalised/composite
  columns and the sector map, i.e. floating-point equality.
## Final decision
Keep the ported PDAIS as the Project B investability score. It is a descriptive
score and is not used as a trading signal, so the one-trading-day sentiment lag
and the look-ahead rules for portfolio weights do not apply to it.
