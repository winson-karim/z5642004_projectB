# AI Log 12 — Streamlit App, Guided Allocation and Investor Journey

## What I wanted
Build the investor-facing Streamlit application for Vestra: a six-tab app that
loads only the precomputed results and artifacts and
presents the 12 systematic funds, the sector sentiment index, the fusion
experiments, investability, a risk lab and the raw data without recomputing any
backtest, sentiment or optimisation. I also wanted a guided to use or 
allocation path for retail users on top of the required manual builder.

## Prompts
- "Build the Streamlit app loading only from results/data and results/tables,
  with tabs for Funds, Sentiment, Fusion, Investability, Risk Lab and Data."
- "Add a 'How to Use Vestra' section near the top with four equal columns for
  the main user journeys."
- "Add a 'Vestra Guided Allocation' profile selector (Defensive / Balanced /
  Growth) before the manual builder, and a 'Use This Profile' button that
  pre-fills the manual builder through session state."
- "Reword all investor-facing labels so they are professional and title-cased,
  and remove anything that reads like advice or a guarantee."

## What the assistant produced
A six-tab app that loads `fund_returns.csv`, `fund_weights.csv`,
`sector_sentiment_index.csv`, `performance_metrics.csv` and the fusion and
investability outputs, with the standard Streamlit figures for growth of $1,
drawdown, portfolio weights over time and the sector sentiment index. It also
produced the guided allocation profiles, the session-state wiring for the
"Use This Profile" button, and the manual "Build Your Fund Allocation" builder
with per-fund sliders that sum to 100%.

## What was wrong or risky
- The first builder draft let sliders be edited independently, so the total
  could silently drift away from 100%; some fund names did not match the real
  `FUND_ORDER` members in `results/data/fund_returns.csv`.
- The guided-profile chart initially blended wealth curves incorrectly
  (dividing by the first blended row as if it were a separate baseline), which
  distorted the profile preview.
- Streamlit raised a warning when a widget carried a session-state key that was
  also given a `default=` value: "widget with key created with a default value
  but also had its value set via the Session State API".
- Early labels contained casual wording ("best fund", "recommended for you",
  "Growth of $1" captions, "Sector Sentiment Index Over Time" heading) that
  needed tightening to professional title case, and one caption originally
  described a walk-forward chart as if it were a live look-ahead preview.
- The four profile weight sets had to be checked so each summed to exactly 100%
  and stayed within the long-only caps used by the funds.

## What I changed and why
- I mapped the builder to the real fund list from `results/data/fund_returns.csv`
  and enforced a normalised total with a live total label, so the manual
  builder always sums to 100%.
- I rewrote the profile preview to combine the precomputed daily fund returns
  once and compound them into a growth-of-$1 curve, instead of dividing one
  blended series by itself.
- I moved the builder widgets to explicit session-state initialisation and
  removed the conflicting `default=` so Streamlit logs stay clean and the
  "Use This Profile" pre-fill works reliably on first load.
- I reworded all investor-facing copy to professional title case, removed
  advice and guarantee language, and corrected the walk-forward caption so the
  app never implies look-ahead information.

## Verify the exhibits
- `scripts/check_handin.py` passes all checks (21 passed, 0 failures) with only
  the expected reminders for `report/report.pdf` and `__pycache__` cleanup.
- I ran the app locally and confirmed HTTP 200 on `http://localhost:8501` with
  a clean Streamlit log.
- I exercised three user paths with Streamlit's AppTest: the default guided
  Balanced profile, a warning-state path, and the "Use This Profile" pre-fill
  into the manual builder. The default 4-fund blend shows an ending value of
  $1.46 and a cumulative return of +45.7%; the guided Balanced profile shows
  $1.44 and +44.0%. The app renders dataframes and charts without error and
  the benign `ScriptRunContext` warnings during AppTest runs are expected.
- I confirmed the app never recomputes backtests or sentiment: every tab reads
  precomputed files from `results/` only.

## Final decision
The six-tab app with the guided allocation and the manual builder is the
submitted Vestra application. The investor journey, guided profiles and clean
wording are kept because they add genuine product value while the underlying
analysis stays frozen in `results/`.
