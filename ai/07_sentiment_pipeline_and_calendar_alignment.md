# AI Log 07 — Sentiment Pipeline and Calendar Alignment
## What I wanted
Construct the Project B sector sentiment index from the supplied financial news headlines using VADER.
## What the assistant produced
The AI recommended cleaning and aligning headlines to the equity trading calendar, scoring headline sentiment with VADER, aggregating 
to ticker-day sentiment, and equal-weighting the five constituent tickers within each sector.
## What was wrong or risky
Several implementation risks required correction:
- sentiment code was initially placed in the wrong source file
- merging news and trading dates failed because the datetime columns used different precisions
- Joining several headlines before VADER scoring could make unrelated headlines affect one another's sentiment interpretation
- missing-news ticker-days could cause sectors with fewer headlines to receive unintended weights
## What I changed and why
I kept headline assembly in `features.py` and sentiment modelling in `sentiment.py`.
I explicitly converted both calendar fields to a common `datetime64[ns]` representation before `merge_asof`.
Each headline is scored individually with VADER and the resulting scores are averaged to ticker-day sentiment.
Ticker-days with no news receive neutral sentiment of zero before the five constituents are equaly weighted into each sector index.
## Verification
The final pipeline produced:
- all 50 equity tickers
- all 10 sectors
- sentiment scores bounded within [-1, 1]
- zero missing values in the sector index
- no weekend dates after trading-calendar alignment
The final descriptive sentiment index is saved to:
- results/data/sector_sentiment_index.csv
## Final decision
Retain the VADER-based equal-weight sector sentiment index as Vestra's baseline sentiment signal. The investment use of this signal
will be lagged by one trading day during fusion to prevent look-ahead bias.
