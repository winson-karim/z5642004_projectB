# Prompt log 03 - Portfolio and Backtest Design
## What I wanted
Design the systematic fund families, portfolio methods and out-of-sample 
backtest assumptions before implementing the portfolio code.

## Prompt(s)
I asked the AI assistant to review the Project B portfolio starter code and recommend a 
portfolio framework aligned with PROJECT_BRIEF.md and the HD rubric.
## What the assistant produced
The AI proposed a 3 fund families (equity, crypto, and combined), 4 methods (EW, MV, MS, RP), monthly 
walk-forward rebalancing, a 252-day rolling window, long-only weights, asset caps, and separate 252/365 annualisation rules.
## What was wrong or risky
Maximum Sharpe and covariance estimates may be unstable, fixed weight caps may be arbitrary, and calendar alignment or rebalance 
timing could still create look-ahead or distorted crypto returns if implemented incorrectly.
## What I changed and why
I kept the design as a baseline but it required validation of solver success, weight constraints, calendar
allignment, fist live date, and comparison against Equal Weight before the final implementation.

