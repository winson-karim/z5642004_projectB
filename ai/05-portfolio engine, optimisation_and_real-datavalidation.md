# AI Log 05 — Portfolio Engine, Optimisation and Real-Data Validation
## What I wanted
Build and validate the Project B systematic fund engine using walk-forward out-of-sample portfolio construction.
## What the assistant produced
The AI assisted with performance metrics, monthly walk-forward backtesting, Equal Weight, Minimum Variance, Risk Parity 
and Maximum Sharpe methods, followed by integration with the real equity and crypto datasets.
## What was wrong or risky
Several issues required checking rather than blindly accepting the AI output:
- SLSQP initially treated Equal Weight as the Minimum Variance solution because the daily covariance objective was extremely small.
- After rescaling, SLSQP found valid improved weights but sometimes returned `success=False`, causing good solutions to be incorrectly discarded.
- The AI initially referenced a non-existent `load_prices()` function rather than the separate starter data-access functions.
- Calendar alignment could have produced distorted crypto returns if prices had been merged before calculating returns.
## What I changed and why
I rescaled the Minimum Variance objective so SLSQP could detect meaningful differences.
I changed the solver validation so a feasible finite solution satisfying the constraints is accepted even when SLSQP reports a
false-negative convergence flag.
I checked `src/data_access.py` and replaced the invented loader with the actual `load_equity_prices()` and `load_crypto_prices()` functions.
I calculated equity and crypto returns separately on their native calendars before aligning crypto returns to equity dates for the combined fund.

## Verification
I first tested all four methods using controlled synthetic data.
I then tested them using the real 60-asset combined panel and confirmed:
- weights sum to one
- long-only constraints hold
- optimisation caps hold
- the first live OOS date occurs after the 252-observation estimation window
- 36 monthly rebalances occur
- all four methods produce finite performance metrics

The full pipeline then produced 12 funds across Equity, Crypto and Combined families and saved:
- results/data/fund_returns.csv
- results/data/fund_weights.csv
- results/tables/performance_metrics.csv

## Final decision
The validated 12-fund engine is retained as the baseline systematic investment platform for Vestra.
