"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score the assembled headlines with VADER.

    Returns a long frame with columns date, ticker, sector and sentiment, where
    sentiment is the VADER compound score (-1 to +1) over all headlines for that
    ticker on that trading day. VADER relies on casing, punctuation and negation,
    so the raw titles are joined and scored without stripping.
    """
    required = {"date", "ticker", "sector", "title"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")

    sia = SentimentIntensityAnalyzer()

    joined = (
        panel.sort_values(["ticker", "date", "title"])
        .groupby(["date", "ticker", "sector"], sort=False)["title"]
        .apply(" . ".join)
        .reset_index()
        .rename(columns={"title": "titles"})
    )
    joined["sentiment"] = joined["titles"].map(
        lambda text: sia.polarity_scores(text)["compound"]
    )

    return joined[["date", "ticker", "sector", "sentiment"]]


def sector_sentiment_index(
    scores: pd.DataFrame,
    calendar: pd.Index | None = None,
) -> pd.DataFrame:
    """Build a daily sentiment index per sector, equal-weighting its tickers.

    Ticker-days with no headline are treated as neutral (sentiment 0): no new
    information means the prior view holds. This avoids stale carry-forward
    signals and keeps thin sectors (e.g. Materials, Utilities) from swinging on
    a single headline. Returns a wide date-by-sector panel with no missing values.
    If a trading `calendar` is supplied, the index is reindexed to it (neutral on
    calendar days with no news at all).
    """
    required = {"date", "ticker", "sector", "sentiment"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")

    # Daily per-ticker sentiment, neutral-filled where no news was published.
    ticker_day = scores.pivot_table(
        index="date",
        columns="ticker",
        values="sentiment",
        aggfunc="mean",
    )
    ticker_day = ticker_day.reindex(
        index=ticker_day.index.sort_values(),
        columns=sorted(ticker_day.columns),
    ).fillna(0.0)

    sector_of_ticker = (
        scores[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]
    )

    index = pd.DataFrame(index=ticker_day.index)
    for sector in sorted(sector_of_ticker.unique()):
        members = sector_of_ticker[sector_of_ticker == sector].index
        index[sector] = ticker_day[members].mean(axis=1)

    if calendar is not None:
        index = index.reindex(calendar).fillna(0.0)

    index.index.name = "date"
    return index
