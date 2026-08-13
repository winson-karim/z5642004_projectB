# AI Log 11 — PDAIS Investability Guardrail
## What I wanted
Extend Vestra's Project A Dynamic Asset Investability Score (PDAIS) into a Project B investment methodology without introducing look-ahead bias or tuning the model
to maximise out-of-sample performance.
The objective was to test whether PDAIS could improve the Combined Risk Parity portfolio while preserving Vestra's long-only portfolio constraints.

## What the assistant produced
The initial extension used a continuous PDAIS overlay. Previous-month PDAIS scores modified the baseline Combined Risk Parity weights through a bounded multiplicative tilt.
A second extension was then proposed after the first result was negative: the PDAIS Investability Guardrail.

The Guardrail:
- uses only the previous month's PDAIS information;
- ranks Return Stability, Downside Resilience and Liquidity Quality within each asset class;
- flags an asset when at least two of the three components fall within the bottom quartile;
- reduces the baseline weight of a flagged asset by 30%;
- gives no direct positive boost to high-scoring assets;
- redistributes removed weight across the remaining portfolio;
- preserves long-only weights, a 20% individual-asset cap and weights summing to one.

## What was wrong or risky
The first PDAIS overlay implicitly treated higher investability as if it predicted higher future returns.
this assumption was not supported by the construction of PDAIS, which measures asset stability, downside resilience and liquidity rather than expected return.
The naive overlay produced a lower Sharpe ratio than the baseline and increased turnover.
There was also a risk of repeatedly tuning PDAIS thresholds and tilt strengths after observing out-of-sample performance. Doing so could create data-mining and weaken the validity of the results.

## What I changed and why
I rejected performance-based tuning of the naive overlay.
Instead, I reframed PDAIS as a defensive investability measure and implemented one pre-specified Guardrail specification.

The rule used:
- previous-month information only;
- within-asset-class bottom-quartile rankings;
- agreement from at least two of three PDAIS components;
- a fixed 30% defensive weight penalty.

The 25% threshold, two-of-three rule and 30% penalty were fixed before the final test and were not tuned after observing performance.

## What I have verify

Seven dedicated guardrail unit tests passed, including:
- previous-month information is used rather than same-month information;
- at least two weak components are required for a flag;
- flagged assets receive the intended penalty;
- no direct positive PDAIS boost is given to unflagged assets;
- weights remain non-negative;
- weights remain within the portfolio cap;
- every rebalance row sums to one.

The complete project test suite also passed.

The validated baseline portfolio remained unchanged.

## Results:

Combined Risk Parity baseline:
- Annualised return: 13.97%
- Annualised volatility: 16.02%
- Sharpe ratio: 0.8964
- Maximum drawdown: -19.84%
- Average monthly turnover: 2.25%
Naive PDAIS overlay:
- Annualised return: 13.34%
- Annualised volatility: 15.70%
- Sharpe ratio: 0.8760
- Maximum drawdown: -19.78%
- Average monthly turnover: 3.17%
PDAIS Investability Guardrail:
- Annualised return: 13.27%
- Annualised volatility: 15.76%
- Sharpe ratio: 0.8695
- Maximum drawdown: -19.85%
- Average monthly turnover: 4.22%
The Guardrail flagged an average of approximately 11.36 assets per rebalance and acted on approximately 14% of baseline portfolio weight.

## Final decision
The experiments provide an honest negative result.
PDAIS consistently reduced portfolio volatility slightly, but the reduction was not sufficient to compensate for lower returns and higher turnover. Neither the continuous overlay nor the defensive Guardrail 
improved the baseline Combined Risk Parity Sharpe ratio.
Therefore, I decided not to deploy PDAIS as an automatic portfolio-weighting rule.
Instead, Vestra will retain PDAIS as an investor-facing asset-quality and investability intelligence layer. This preserves its useful interpretation while avoiding mechanical portfolio adjustments that were not supported by the out-of-sample evidence.
This negative result is retained rather than hidden because it directly informed the final Vestra product design.
