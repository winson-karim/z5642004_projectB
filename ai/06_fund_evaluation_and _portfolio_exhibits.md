# AI Log 06 — Fund Evaluation and Portfolio Exhibits
## What I wanted
Turn the validated 12-fund portfolio results into investor-facing exhibits for the report and Streamlit application.
## What the assistant produced
The AI assisted with generating:
- Combined fund growth-of-$1 comparison
- Combined fund drawdown comparison
- Combined fund crypto allocation over time
- Risk-return comparison across all 12 funds
## What was wrong or risky
Plotting all 60 individual Combined fund asset weights would have been difficult to interpret and visually cluttered.
There was also a risk of treating raw optimisation outputs as economically meaningful without first presenting them in a comparable investor-facing format.
## What I changed and why
I used the underlying Combined fund weights to aggregate crypto exposure instead of plotting every individual asset.
This produces a clearer comparison of how Equal Weight, Minimum Variance, Risk Parity and Maximum Sharpe change the portfolio's crypto allocation through time.
I retained the full 12-fund risk-return comparison to show how the Equity, Crypto and Combined families differ in expected investor trade-offs.
## Verify all the outputs
I checked that all four figures were generated successfully and that:
- all Combined methods appear in the growth and drawdown figures
- drawdowns remain at or below zero
- Equal Weight crypto exposure is approximately constant at 10/60 of the Combined portfolio
- optimised crypto allocations vary over time
- all 12 funds appear in the risk-return comparison
## Final decision
Retain these four exhibits as the core portfolio evaluation outputs.
