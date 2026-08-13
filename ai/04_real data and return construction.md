# AI Log 04 — Real Data and Return Construction

## What I wanted
Connect the real Project B equity and crypto price data to the portfolio pipeline and verify daily return construction.

## What the assistant produced
The AI recommended implementing `daily_returns()` in `src/features.py` and testing it on the real equity and crypto datasets before aligning the two asset classes.

## What was wrong or risky
The AI initially suggested importing a non-existent `load_prices()` function from `src.data_access.py`. The actual starter API provides separate
functions: `load_equity_prices()` and `load_crypto_prices()`.

## What I changed and why
I checked `src/data_access.py`, rejected the invented API, and used the exact starter functions instead. I kept return calculation in `features.py` because 
it is reusable transformation logic, while `run_part_b.py` should only orchestrate the full pipeline.
