"""Station 2 - your features: return features and text assembly.

Build your return features here, and assemble the headlines into a daily text
panel. Scoring the text is the Station 3 sentiment model (see src/sentiment.py).
"""
import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Return a wide date-by-ticker panel of simple daily returns."""
    required = {"date", "ticker", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = prices[["date", "ticker", price_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Compute returns within each ticker before any cross-asset alignment.
    df["return"] = df.groupby("ticker")[price_col].pct_change(fill_method=None)

    returns = (
        df.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .astype(float)
    )
    returns.columns.name = None
    return returns


def assemble_headline_panel(
    news: pd.DataFrame,
    equity_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble headlines into a daily panel per ticker and sector.

    Station 2 is assembly only: drop exact duplicates (ticker + date + title),
    normalise timezone-aware UTC timestamps to naive calendar dates, and align
    every headline to its equity trading day - the same day if it falls on one,
    otherwise the next trading day. Raw headline text is kept so the Station 3
    sentiment model can score it.
    """
    required_news = {"date", "ticker", "sector", "title"}
    missing_news = required_news.difference(news.columns)
    if missing_news:
        raise ValueError(f"Missing news columns: {sorted(missing_news)}")

    if "date" not in equity_prices.columns:
        raise ValueError("equity_prices must contain a 'date' column")

    panel = news[["date", "ticker", "sector", "title"]].copy()

    # Remove exact duplicate headlines for the same ticker and timestamp.
    panel = panel.drop_duplicates(subset=["ticker", "date", "title"]).copy()

    # Convert tz-aware UTC timestamps to tz-naive calendar dates, and cast to a
    # common datetime resolution so merge_asof accepts the join keys.
    panel["date"] = (
        pd.to_datetime(panel["date"], utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    # Equity trading calendar used for sentiment alignment.
    trading_dates = (
        pd.to_datetime(equity_prices["date"])
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .astype("datetime64[ns]")
    )
    trading_calendar = pd.DataFrame({"trading_date": trading_dates})

    # Map each headline to the same trading day, or the next trading day if needed.
    panel = panel.sort_values("date")
    panel = pd.merge_asof(
        panel,
        trading_calendar,
        left_on="date",
        right_on="trading_date",
        direction="forward",
        allow_exact_matches=True,
    )

    # Drop headlines after the final available equity trading day.
    panel = panel.dropna(subset=["trading_date"]).copy()
    panel["date"] = panel["trading_date"]
    panel = panel.drop(columns="trading_date")

    return panel.sort_values(["date", "ticker", "title"]).reset_index(drop=True)


def _rolling_semi_dev(returns: pd.Series, window: int) -> pd.Series:
    """Rolling semi-deviation (downside deviation) over a window.

    Semi-deviation = sqrt( mean( min(r_i, 0)^2 ) ) over the window.
    """
    downside = returns.clip(upper=0)
    return downside.rolling(window=window, min_periods=window).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    )


def _min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 100]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    return (series - lo) / (hi - lo) * 100.0


def _data_reliability_gate(eq_prices: pd.DataFrame, cr_prices: pd.DataFrame) -> dict:
    """Quality gate: per-ticker checks for missing OCHLV, date gaps, duplicates,
    non-positive prices, and invalid OHLC.

    OCHLV columns: open, close, high, low, adjClose, volume.
    Engineered fields like returns are excluded from the missing value check.

    Date gaps: For equities, the master calendar is constructed from the set of
    unique dates across the full equity dataset. For each ticker, gaps are the
    dates in the master calendar not present for that ticker. This records true
    missing trading days with no assumed calendar (e.g., public holidays).

    For crypto, the master calendar is the full daily range between the global
    start and end dates present in the crypto panel.

    Returns a dict with:
      - 'per_ticker': list of dicts with ticker-level gate results
      - 'equity': aggregate summary for equity
      - 'crypto': aggregate summary for crypto
      - 'all_pass': True if every single asset passes
    """
    ochlv = {"open", "high", "low", "close", "adjClose", "volume"}

    def gate_one(df, label):
        ochlv_in_df = [c for c in df.columns if c in ochlv]
        mv_df = df[ochlv_in_df]
        mv = mv_df.isna().sum().sum()
        n_expected_mv = mv_df.shape[0] * mv_df.shape[1]
        missing_pct = mv / n_expected_mv if n_expected_mv > 0 else 0.0

        dup = int(df.duplicated(subset=["ticker", "date"]).sum())

        price_cols = ["open", "high", "low", "close", "adjClose"]
        npp = int(sum((df[c] <= 0).sum() for c in price_cols if c in df.columns))

        has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])
        iohlc = 0
        if has_ohlc:
            iohlc = int(((df["low"] > df["high"]) |
                         (df["open"] < df["low"]) | (df["open"] > df["high"]) |
                         (df["close"] < df["low"]) | (df["close"] > df["high"])).sum())
        tickers = df["ticker"].unique()
        n_tickers = len(tickers)

        if label == "equity":
            master_cal = pd.DatetimeIndex(sorted(df["date"].unique()))
        else:
            start, end = df["date"].min(), df["date"].max()
            master_cal = pd.date_range(start=start, end=end, freq="D")

        per_ticker = []
        total_gaps = 0
        for t in sorted(tickers):
            t_dates = pd.DatetimeIndex(df.loc[df["ticker"] == t, "date"].unique())
            n_gaps = len(master_cal.difference(t_dates))
            total_gaps += n_gaps
            n_gate_fails = int(missing_pct > 0) + int(npp > 0) + int(iohlc > 0) + int(n_gaps > 0) + int(dup > 0)
            all_zero = (missing_pct == 0 and npp == 0 and iohlc == 0 and n_gaps == 0 and dup == 0)
            per_ticker.append(dict(
                ticker=t, missing_pct=missing_pct, n_missing_cells=int(mv),
                duplicates=dup, non_positive_prices=npp, invalid_ohlc=iohlc,
                date_gaps=n_gaps, issues_detected=n_gate_fails, gate_pass=all_zero
            ))
        n_pass = sum(1 for p in per_ticker if p["gate_pass"])
        return dict(
            label=label,
            n_tickers=n_tickers,
            n_pass=n_pass,
            n_fail=n_tickers - n_pass,
            total_missing_cells=int(mv),
            total_duplicates=dup,
            total_non_positive=npp,
            total_invalid_ohlc=iohlc,
            total_date_gaps=total_gaps,
            per_ticker=per_ticker,
            gate_all_pass=n_pass == n_tickers
        )

    equity_gate = gate_one(eq_prices, "equity")
    crypto_gate = gate_one(cr_prices, "crypto")
    gate_all_pass = equity_gate["gate_all_pass"] and crypto_gate["gate_all_pass"]
    return dict(equity=equity_gate, crypto=crypto_gate, all_pass=gate_all_pass)


def compute_investability_scores(
    eq_returns_wide: pd.DataFrame,
    cr_returns_wide: pd.DataFrame,
    eq_prices: pd.DataFrame,
    cr_prices: pd.DataFrame,
    window: int = 21,
) -> dict:
    """Compute the Vestra Dynamic Asset Investability Score.

    Three components (monthly):
      1. Return Stability (RS): 1 / (1 + mean rolling std)
      2. Downside Resilience (DR): 1 / (1 + mean rolling semi-dev)
      3. Liquidity Quality (LQ): log median daily dollar volume

    All normalised 0-100 per asset class per month.
    Data Reliability (DRLC) is a separate quality gate, not a composite component.

    Three weighting scenarios:
      - EW:  RS=1/3, DR=1/3, LQ=1/3
      - RF:  RS=0.50, DR=0.30, LQ=0.20  (risk-focused)
      - LF:  RS=0.20, DR=0.20, LQ=0.60  (liquidity-focused)

    Returns a dict of DataFrames:
      - 'monthly_scores': long format with all raw/normalised scores + composites
      - 'summary': per-asset mean composites + DRLC gate + static LQ rank
      - 'sensitivity': per-month Spearman rho between (EW,RF) and (EW,LF)
                       and mean/max rank changes, per asset class
    """
    results = {}

    drlc = _data_reliability_gate(eq_prices, cr_prices)
    drlc_pass = drlc["all_pass"]
    drlc_per_ticker = {}
    for label in ["equity", "crypto"]:
        for p in drlc[label]["per_ticker"]:
            drlc_per_ticker[p["ticker"]] = p["gate_pass"]
    n_pass = sum(1 for v in drlc_per_ticker.values() if v)
    n_fail = len(drlc_per_ticker) - n_pass
    is_non_discriminating = (n_pass == len(drlc_per_ticker))

    print(f"  Data Reliability Gate: {n_pass} passed, {n_fail} failed. Non-discriminating: {is_non_discriminating}")

    monthly_rows = []
    all_tickers = sorted(eq_returns_wide.columns.tolist() + cr_returns_wide.columns.tolist())

    for label, rw in [("Equity", eq_returns_wide), ("Crypto", cr_returns_wide)]:
        tickers = rw.columns.tolist()
        for ticker in tickers:
            s = rw[ticker].dropna()
            if len(s) < window + 1:
                continue
            rolling_std = s.rolling(window=window, min_periods=window).std()
            rolling_sd = _rolling_semi_dev(s, window)
            df_tmp = pd.DataFrame({
                "raw_rs": 1.0 / (1.0 + rolling_std),
                "raw_dr": 1.0 / (1.0 + rolling_sd),
            }, index=s.index)
            df_tmp.index.name = "date"
            monthly = df_tmp.resample("ME").mean()
            for dt, row in monthly.iterrows():
                monthly_rows.append({
                    "year_month": dt.strftime("%Y-%m"),
                    "ticker": ticker,
                    "asset_class": label,
                    "RS_raw": row["raw_rs"],
                    "DR_raw": row["raw_dr"],
                })
    df_monthly = pd.DataFrame(monthly_rows)

    lq_rows = []
    for label, prices in [("Equity", eq_prices), ("Crypto", cr_prices)]:
        df_p = prices[["ticker", "date", "volume", "adjClose"]].copy()
        df_p["dollar_vol"] = df_p["volume"] * df_p["adjClose"]
        df_p["log_dollar_vol"] = np.log(df_p["dollar_vol"].clip(lower=1))
        df_p["year_month"] = df_p["date"].dt.strftime("%Y-%m")
        lq_monthly = df_p.groupby(["ticker", "year_month"])["log_dollar_vol"].median().reset_index()
        lq_monthly = lq_monthly.rename(columns={"log_dollar_vol": "LQ_raw"})
        lq_monthly["asset_class"] = label
        lq_rows.append(lq_monthly)
    df_lq = pd.concat(lq_rows, ignore_index=True)

    scores = df_monthly.merge(df_lq, on=["ticker", "year_month", "asset_class"], how="left")

    scores["RS_norm"] = np.nan
    scores["DR_norm"] = np.nan
    scores["LQ_norm"] = np.nan
    for ac in scores["asset_class"].unique():
        mask_ac = scores["asset_class"] == ac
        for ym in scores.loc[mask_ac, "year_month"].unique():
            mask_ym = mask_ac & (scores["year_month"] == ym)
            scores.loc[mask_ym, "RS_norm"] = _min_max_norm(scores.loc[mask_ym, "RS_raw"])
            scores.loc[mask_ym, "DR_norm"] = _min_max_norm(scores.loc[mask_ym, "DR_raw"])
            scores.loc[mask_ym, "LQ_norm"] = _min_max_norm(scores.loc[mask_ym, "LQ_raw"])

    scores["PDAIS_EW"] = (scores["RS_norm"] + scores["DR_norm"] + scores["LQ_norm"]) / 3.0
    scores["PDAIS_RF"] = 0.50 * scores["RS_norm"] + 0.30 * scores["DR_norm"] + 0.20 * scores["LQ_norm"]
    scores["PDAIS_LF"] = 0.20 * scores["RS_norm"] + 0.20 * scores["DR_norm"] + 0.60 * scores["LQ_norm"]

    scores = scores.sort_values(["asset_class", "ticker", "year_month"]).reset_index(drop=True)

    eq_sector_map = eq_prices[["ticker", "sector"]].drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
    scores["sector"] = scores["ticker"].map(eq_sector_map).fillna("Crypto")

    results["monthly_scores"] = scores

    summary_rows = []
    for ticker in all_tickers:
        sub = scores[scores["ticker"] == ticker]
        if sub.empty:
            continue
        ac = sub["asset_class"].iloc[0]
        gate_pass = drlc_per_ticker.get(ticker, False)
        summary_rows.append({
            "ticker": ticker,
            "asset_class": ac,
            "DRLC_gate_pass": gate_pass,
            "mean_PDAIS_EW": sub["PDAIS_EW"].mean(),
            "mean_PDAIS_RF": sub["PDAIS_RF"].mean(),
            "mean_PDAIS_LF": sub["PDAIS_LF"].mean(),
            "std_PDAIS_EW": sub["PDAIS_EW"].std(),
            "mean_LQ_raw": sub["LQ_raw"].mean(),
        })
    df_summary = pd.DataFrame(summary_rows)
    results["summary"] = df_summary

    sens_rows = []
    rank_change_rows = []
    for ac in scores["asset_class"].unique():
        mask_ac = scores["asset_class"] == ac
        for ym in scores.loc[mask_ac, "year_month"].unique():
            mask = mask_ac & (scores["year_month"] == ym)
            sub = scores.loc[mask].copy()
            if len(sub) < 3:
                continue
            rho_ew_rf = sub["PDAIS_EW"].corr(sub["PDAIS_RF"], method="spearman")
            rho_ew_lf = sub["PDAIS_EW"].corr(sub["PDAIS_LF"], method="spearman")
            sens_rows.append({
                "year_month": ym, "asset_class": ac,
                "spearman_EW_vs_RF": rho_ew_rf,
                "spearman_EW_vs_LF": rho_ew_lf,
            })
            sub = sub.copy()
            sub["rank_EW"] = sub["PDAIS_EW"].rank(ascending=False)
            sub["rank_RF"] = sub["PDAIS_RF"].rank(ascending=False)
            sub["rank_LF"] = sub["PDAIS_LF"].rank(ascending=False)
            rank_change_rows.append({
                "year_month": ym, "asset_class": ac,
                "mean_abs_rank_change_RF_vs_EW": (sub["rank_RF"] - sub["rank_EW"]).abs().mean(),
                "max_abs_rank_change_RF_vs_EW": (sub["rank_RF"] - sub["rank_EW"]).abs().max(),
                "mean_abs_rank_change_LF_vs_EW": (sub["rank_LF"] - sub["rank_EW"]).abs().mean(),
                "max_abs_rank_change_LF_vs_EW": (sub["rank_LF"] - sub["rank_EW"]).abs().max(),
            })

    results["sensitivity"] = pd.DataFrame(sens_rows)
    results["rank_changes"] = pd.DataFrame(rank_change_rows)

    return results
