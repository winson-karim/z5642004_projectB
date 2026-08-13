"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
import pandas as pd


def _cap_weights(row: pd.Series, max_weight: float) -> pd.Series:
    """Clip weights at max_weight, redistributing excess to uncapped names."""
    row = row.copy()
    free = row.index[row <= max_weight]
    row[row > max_weight] = max_weight
    excess = (row.sum() - 1.0) if row.sum() > 1.0 else 0.0

    while excess > 1e-12 and len(free) > 0:
        total_free = row[free].sum()
        if total_free <= 0:
            break
        add = excess * row[free] / total_free
        row[free] = row[free] + add
        newly_capped = free[row[free] > max_weight]
        excess = (row[newly_capped] - max_weight).sum()
        row[newly_capped] = max_weight
        free = free[~free.isin(newly_capped)]

    if row.sum() > 0:
        row = row / row.sum()
    return row


def apply_sentiment(
    weights: pd.DataFrame,
    sentiment: pd.DataFrame,
    ticker_sector: pd.Series,
    tilt_strength: float = 0.20,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """Tilt portfolio weights toward sectors with positive sentiment.

    Sentiment is lagged by one trading day before any rebalance date lookup, so
    weights at day t use only sentiment from day t-1 or earlier (no look-ahead).
    Each asset is scaled by (1 + tilt_strength * sector_sentiment); assets with
    no sector mapping or missing sentiment stay at factor 1.0. Rows are
    renormalised to sum to 1 and capped at max_weight.
    """
    weights = weights.copy()

    # Lag the sentiment signal by one trading day to avoid look-ahead.
    lagged = sentiment.shift(1)

    tilted_rows = []
    for date, base_row in weights.iterrows():
        sector_sent = lagged.asof(date) if len(lagged) else pd.Series(dtype=float)

        factors = pd.Series(1.0, index=base_row.index)
        for asset in base_row.index:
            sector = ticker_sector.get(asset)
            if sector is not None and sector in sector_sent.index:
                value = sector_sent[sector]
                if pd.notna(value):
                    factors[asset] = 1.0 + tilt_strength * value

        tilted = (base_row * factors).clip(lower=0.0)
        if tilted.sum() > 0:
            tilted = tilted / tilted.sum()
        tilted = _cap_weights(tilted, max_weight)
        tilted_rows.append(tilted)

    result = pd.DataFrame(
        {date: row for date, row in zip(weights.index, tilted_rows)}
    ).T
    result.columns = weights.columns
    result.index.name = weights.index.name
    return result


def apply_pdais(
    weights: pd.DataFrame,
    scores: pd.DataFrame,
    ticker_map: pd.Series,
    score_col: str = "PDAIS_EW",
    tilt_strength: float = 0.25,
    max_weight: float = 0.40,
) -> pd.DataFrame:
    """Tilt portfolio weights toward assets with higher PDAIS.

    PDAIS is a monthly 0-100 score. A rebalance in calendar month M uses the
    scores from month M-1 (one-month lag), so weights at any date use only past
    information (no look-ahead). The signal is rescaled score/50 - 1 so a score
    of 50 is neutral; each asset is scaled by (1 + tilt_strength * signal).
    Assets without a score mapping stay at factor 1.0. Rows are renormalised to
    sum to 1 and capped at max_weight.
    """
    weights = weights.copy()

    score_lookup = {}
    for (ym, ticker), value in scores.set_index(["year_month", "ticker"])[score_col].items():
        score_lookup[(ym, ticker)] = value

    tilted_rows = []
    for date, base_row in weights.iterrows():
        prev_month = (date - pd.DateOffset(months=1)).strftime("%Y-%m")

        factors = pd.Series(1.0, index=base_row.index)
        for asset in base_row.index:
            key = ticker_map.get(asset)
            if key is None:
                continue
            value = score_lookup.get((prev_month, key))
            if value is not None and pd.notna(value):
                signal = value / 50.0 - 1.0
                factors[asset] = 1.0 + tilt_strength * signal

        tilted = (base_row * factors).clip(lower=0.0)
        if tilted.sum() > 0:
            tilted = tilted / tilted.sum()
        tilted = _cap_weights(tilted, max_weight)
        tilted_rows.append(tilted)

    result = pd.DataFrame(
        {date: row for date, row in zip(weights.index, tilted_rows)}
    ).T
    result.columns = weights.columns
    result.index.name = weights.index.name
    return result


def apply_pdais_guardrail(
    weights: pd.DataFrame,
    scores: pd.DataFrame,
    ticker_map: pd.Series,
    penalty: float = 0.30,
    quartile: float = 0.25,
    max_weight: float = 0.20,
) -> dict:
    """Defensive PDAIS investability guardrail (no alpha boost).

    Treats PDAIS as a risk/investability measure, not a return signal. A
    rebalance in calendar month M uses only the scores from month M-1 (one-month
    lag, no look-ahead). Within each asset class and month, RS_norm, DR_norm and
    LQ_norm are percentile-ranked across the class; an asset is flagged as
    low-investability when at least two of the three components sit in the
    bottom within-class quartile. Flagged holdings are scaled by (1 - penalty);
    every other factor is exactly 1.0, so nothing is ever boosted directly - any
    increase in unflagged weights arises only through row renormalisation. Rows
    are renormalised to sum to 1 and capped at max_weight via _cap_weights.

    Returns a dict with:
      - 'weights': the guarded portfolio weights (same shape as input)
      - 'flags': bool DataFrame (rebalance x asset) of low-investability flags
      - 'penalised_weight': Series of pre-penalty baseline weight held in
        flagged assets at each rebalance
    """
    weights = weights.copy()
    components = ["RS_norm", "DR_norm", "LQ_norm"]
    rank_cols = [f"r_{comp}" for comp in components]

    ranked = scores.copy()
    for comp, rank_col in zip(components, rank_cols):
        ranked[rank_col] = ranked.groupby(
            ["asset_class", "year_month"]
        )[comp].rank(pct=True)

    flag_rows = []
    penalised_rows = []
    guarded_rows = []

    for date, base_row in weights.iterrows():
        prev_month = (date - pd.DateOffset(months=1)).strftime("%Y-%m")
        month_scores = ranked[ranked["year_month"] == prev_month]
        month_lookup = month_scores.set_index("ticker")

        flags = pd.Series(False, index=base_row.index)
        for asset in base_row.index:
            key = ticker_map.get(asset)
            if key is None or key not in month_lookup.index:
                continue
            row = month_lookup.loc[key]
            low_components = sum(
                1
                for rank_col in rank_cols
                if pd.notna(row.get(rank_col)) and row[rank_col] <= quartile
            )
            flags[asset] = low_components >= 2

        factors = pd.Series(1.0, index=base_row.index)
        factors[flags] = 1.0 - penalty

        penalised_weight = float(base_row[flags].sum()) if flags.any() else 0.0
        guarded = (base_row * factors).clip(lower=0.0)
        if guarded.sum() > 0:
            guarded = guarded / guarded.sum()
        guarded = _cap_weights(guarded, max_weight)

        flag_rows.append(flags)
        penalised_rows.append(penalised_weight)
        guarded_rows.append(guarded)

    result = pd.DataFrame(
        guarded_rows,
        index=weights.index,
        columns=weights.columns,
    )
    result.index.name = weights.index.name

    flags_df = pd.DataFrame(
        flag_rows,
        index=weights.index,
        columns=weights.columns,
    )
    flags_df.index.name = weights.index.name

    penalised_series = pd.Series(
        penalised_rows,
        index=weights.index,
        name="penalised_weight",
    )

    return {
        "weights": result,
        "flags": flags_df,
        "penalised_weight": penalised_series,
    }
