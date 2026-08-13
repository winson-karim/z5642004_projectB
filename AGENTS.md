# AGENTS.md - Agent instructions for Project B

> This file is graded (AI Workflow & Transparency, 20% of the Part). It is YOUR
> work: the instructions below are the ones you actually give your AI coding
> assistant. Do not hand in the starter version of this file unchanged.

The full assignment brief is in PROJECT_BRIEF.md (this folder) - read it first,
along with context/ (the data guide and project context).

If you use Codex (or another assistant that reads AGENTS.md), put your project
conventions, coding rules, and task routing here. Keep your prompt logs in `ai/`.

Suggested things to record:
- what the project is and where the data comes from (see PROJECT_BRIEF.md and context/)
- your coding conventions and folder layout
- rules you want the assistant to follow (for example: no look-ahead in backtests)
- how you check and correct the assistant's output

# Project B AI Instructions
This repository contains my individual FINS3645 Project B submission for Vestra, a retail multi-asset investment platform.
The project covers:
- equity-only, crypto-only and combined systematic funds
- walk-forward out-of-sample portfolio backtesting
- portfolio fact sheets
- equity-sector news sentiment
- sentiment and portfolio fusion
- A deployed Streamlit investor application
- a written report and AI workflow documentation

All work must follow PROJECT_BRIEF.md.
## Working rules
1. Work only inside this Project B folder.
2. Do not access or copy another student's files.
3. My own Project A code may be reused only after it has been reviewed for compatibility.
4. Do not edit files unless I explicitly request an edit.
5. Explain proposed changes before applying them.
6. Never assume code is correct because it runs.
7. Check for:
- look-ahead bias
- incorrect calendar alignment
- incorrect annualisation
- unstable optimisation
- data leakage
- silent solver failure
8. Use only past information when forming portfolio weights.
9. Lag sentiment by at least one trading day before using it in investment decisions.
10. The Streamlit app must load precomputed files from results/ and must not recompute backtests or sentiment.

## Required Project B outputs
The following filenames must be produced exactly:
- results/data/fund_returns.csv
- results/data/fund_weights.csv
- results/data/sector_sentiment_index.csv
- results/tables/performance_metrics.csv

## Validation requirements
For every major implementation:
1. State the assumptions.
2. Explain the method.
3. Identify possible failure points.
4. Test the output.
5. Compare the result with a simple baseline.
6. Record any correction in the AI prompt log.




