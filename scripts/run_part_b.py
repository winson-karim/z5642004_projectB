"""Reproduce Part B portfolio results. Run from the project root:

    python scripts/run_part_b.py
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access  # noqa: E402
from src.features import (
    daily_returns,
    assemble_headline_panel,
    compute_investability_scores,
)  # noqa: E402
from src.fusion import apply_pdais, apply_pdais_guardrail, apply_sentiment  # noqa: E402
from src.portfolios import oos_backtest, performance_metrics  # noqa: E402
from src.sentiment import score_headlines, sector_sentiment_index  # noqa: E402


METHODS = ["equal_weight", "min_variance", "risk_parity", "max_sharpe"]


def _holding_returns(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    live_index: pd.Index,
) -> pd.Series:
    """Apply weights held between consecutive rebalance dates to returns."""
    rebalance_dates = weights.index
    portfolio_returns = pd.Series(index=live_index, dtype=float)

    for i, rd in enumerate(rebalance_dates):
        if i + 1 < len(rebalance_dates):
            next_rd = rebalance_dates[i + 1]
            holding_dates = live_index[(live_index >= rd) & (live_index < next_rd)]
        else:
            holding_dates = live_index[live_index >= rd]

        w = weights.loc[rd]
        assets = w[w > 0].index
        portfolio_returns.loc[holding_dates] = (
            returns.loc[holding_dates, assets].mul(w[assets], axis=1).sum(axis=1)
        )

    return portfolio_returns.dropna()


def _average_one_way_turnover(weights: pd.DataFrame) -> float:
    """Average one-way monthly turnover: mean of 0.5 * sum(|dw|) per rebalance."""
    if len(weights) < 2:
        return 0.0
    return weights.diff().abs().sum(axis=1).iloc[1:].mean() / 2.0


def main():
    eq_prices = data_access.load_equity_prices()
    cr_prices = data_access.load_crypto_prices()

    news = data_access.load_news_headlines()

    headline_panel = assemble_headline_panel(news, eq_prices)
    ticker_sentiment = score_headlines(headline_panel)

    equity_calendar = (
        pd.to_datetime(eq_prices["date"])
        .astype("datetime64[ns]")
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    sector_index = sector_sentiment_index(
        ticker_sentiment,
        calendar=equity_calendar,
    )

    print("Headline panel:", headline_panel.shape)
    print("Ticker-day sentiment:", ticker_sentiment.shape)
    print("Sector sentiment index:", sector_index.shape)

    # Compute returns on each asset class's native calendar first.
    eq_returns = daily_returns(eq_prices).loc[:"2023-12-31"].dropna(how="all")
    cr_returns = daily_returns(cr_prices).loc[:"2023-12-31"].dropna(how="all")

    # The combined universe follows the equity trading calendar.
    cr_on_equity_dates = cr_returns.reindex(eq_returns.index)
    combined_returns = pd.concat(
        [
            eq_returns.add_prefix("EQ_"),
            cr_on_equity_dates.add_prefix("CR_"),
        ],
        axis=1,
    )

    families = {
        "equity": (eq_returns, 252),
        "crypto": (cr_returns, 365),
        "combined": (combined_returns, 252),
    }

    results = {}

    print("Running Project B portfolio backtests...")
    print("Equity:", eq_returns.shape)
    print("Crypto:", cr_returns.shape)
    print("Combined:", combined_returns.shape)

    for family, (returns, periods_per_year) in families.items():
        for method in METHODS:
            fund_name = f"{family}_{method}"
            print(f"Running {fund_name}...")

            results[fund_name] = oos_backtest(
                returns,
                method=method,
                lookback=252,
                periods_per_year=periods_per_year,
            )

    print("\nCompleted funds:", len(results))
    print("\nPerformance summary:")

    summary = pd.DataFrame(
        {name: result["metrics"] for name, result in results.items()}
    ).T
    print(summary.round(4))

    project_root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = project_root / "results" / "data"
    tables_dir = project_root / "results" / "tables"
    figures_dir = project_root / "results" / "figures"

    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save the standalone sector sentiment index (date x sector).
    sector_index.to_csv(data_dir / "sector_sentiment_index.csv")

    # Save daily fund returns: date x fund.
    fund_returns = pd.concat(
        {name: result["daily_returns"] for name, result in results.items()},
        axis=1,
    )
    fund_returns.index.name = "date"
    fund_returns.to_csv(data_dir / "fund_returns.csv")

    # Save weights in long format for Streamlit filtering.
    weight_frames = []
    for fund_name, result in results.items():
        weights_long = (
            result["weights"]
            .reset_index()
            .melt(
                id_vars="rebalance_date",
                var_name="asset",
                value_name="weight",
            )
        )
        weights_long.insert(1, "fund", fund_name)
        weight_frames.append(weights_long)

    fund_weights = pd.concat(weight_frames, ignore_index=True)
    fund_weights.to_csv(data_dir / "fund_weights.csv", index=False)

    # Save 12-fund performance table.
    performance_table = summary.reset_index().rename(columns={"index": "fund"})
    performance_table.to_csv(
        tables_dir / "performance_metrics.csv",
        index=False,
    )

    # Combined fund names used for exhibits.
    combined_funds = [f"combined_{method}" for method in METHODS]

    # Exhibit 1: Growth of $1 across Combined fund methods.
    combined_growth = pd.concat(
        {name: results[name]["growth"] for name in combined_funds},
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for fund_name in combined_funds:
        label = fund_name.replace("combined_", "").replace("_", " ").title()
        ax.plot(combined_growth.index, combined_growth[fund_name], label=label)

    ax.set_title("Combined Funds: Growth of $1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(title="Method")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "combined_growth_of_1.png", dpi=200)
    plt.close(fig)

    # Exhibit 2: Drawdown comparison across Combined fund methods.
    fig, ax = plt.subplots(figsize=(10, 6))

    for fund_name in combined_funds:
        growth = results[fund_name]["growth"]
        drawdown = growth / growth.cummax() - 1.0
        label = fund_name.replace("combined_", "").replace("_", " ").title()
        ax.plot(drawdown.index, drawdown, label=label)

    ax.set_title("Combined Funds: Drawdown Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(title="Method")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "combined_drawdown.png", dpi=200)
    plt.close(fig)

    # Exhibit 3: Combined fund crypto allocation over time by method.
    fig, ax = plt.subplots(figsize=(10, 6))

    for fund_name in combined_funds:
        weights = results[fund_name]["weights"]
        crypto_columns = [c for c in weights.columns if c.startswith("CR_")]
        crypto_weight = weights[crypto_columns].sum(axis=1)
        label = fund_name.replace("combined_", "").replace("_", " ").title()
        ax.plot(crypto_weight.index, crypto_weight, label=label)

    ax.set_title("Combined Funds: Crypto Allocation Over Time")
    ax.set_xlabel("Rebalance Date")
    ax.set_ylabel("Crypto Portfolio Weight")
    ax.set_ylim(0, 1)
    ax.legend(title="Method")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "combined_crypto_weight_over_time.png", dpi=200)
    plt.close(fig)

    # Exhibit 4: Risk-return comparison across all 12 funds.
    fig, ax = plt.subplots(figsize=(10, 7))

    for fund_name, row in performance_table.set_index("fund").iterrows():
        x = row["annualised_volatility"]
        y = row["annualised_return"]
        ax.scatter(x, y, s=70)
        ax.annotate(
            fund_name,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Vestra Funds: Annualised Risk vs Return")
    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "all_funds_risk_return.png", dpi=200)
    plt.close(fig)

    # Exhibit 5: sector sentiment through time.
    sentiment_20d = sector_index.rolling(
        window=20,
        min_periods=5,
    ).mean()

    fig, ax = plt.subplots(figsize=(12, 7))

    for sector in sentiment_20d.columns:
        ax.plot(
            sentiment_20d.index,
            sentiment_20d[sector],
            label=sector,
            linewidth=1.2,
        )

    ax.axhline(0, linewidth=0.8)
    ax.set_title("Vestra Sector Sentiment — 20-Day Rolling Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("VADER Sentiment")
    ax.legend(
        title="Sector",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures_dir / "sector_sentiment_over_time.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Exhibit 6: Combined fund portfolio weights over time, grouped by sector
    # and crypto, across methods.
    sector_map = data_access.load_sector_universe().set_index("ticker")["sector"].to_dict()

    grouped_by_method = {}
    for fund_name in combined_funds:
        weights = results[fund_name]["weights"]
        grouped = pd.DataFrame(index=weights.index)
        for col in weights.columns:
            bucket = (
                "Crypto"
                if col.startswith("CR_")
                else sector_map.get(col.replace("EQ_", ""), "Other")
            )
            grouped[bucket] = grouped.get(bucket, 0.0) + weights[col]
        grouped_by_method[fund_name] = grouped

    all_buckets = sorted({c for g in grouped_by_method.values() for c in g.columns})
    bucket_order = (
        pd.concat([g[all_buckets].fillna(0.0) for g in grouped_by_method.values()])
        .mean()
        .sort_values(ascending=False)
        .index
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    colors = plt.get_cmap("tab20").colors
    for ax, fund_name in zip(axes.ravel(), combined_funds):
        grouped = grouped_by_method[fund_name][bucket_order].fillna(0.0)
        ax.stackplot(
            grouped.index,
            grouped.values.T,
            labels=bucket_order,
            colors=[colors[i % len(colors)] for i in range(len(bucket_order))],
        )
        ax.set_title(fund_name.replace("combined_", "").replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Rebalance Date")
    for ax in axes[:, 0]:
        ax.set_ylabel("Portfolio Weight")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Combined Funds: Portfolio Weights Over Time by Sector and Crypto",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(figures_dir / "portfolio_weights_over_time.png", dpi=200)
    plt.close(fig)

    # ---- Sentiment-fusion comparison (equity equal-weight base) ----
    sector_universe = data_access.load_sector_universe()
    ticker_sector = sector_universe.set_index("ticker")["sector"]

    base_name = "equity_equal_weight"
    base_weights = results[base_name]["weights"]
    base_returns = results[base_name]["daily_returns"]

    fused_weights = apply_sentiment(
        base_weights,
        sector_index,
        ticker_sector,
        tilt_strength=0.20,
        max_weight=0.40,
    )

    fused_returns = _holding_returns(
        eq_returns,
        fused_weights,
        live_index=base_returns.index,
    ).rename(base_name + "_sentiment_fused")

    metric_columns = [
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "max_drawdown",
        "average_monthly_turnover",
    ]

    def _fund_metrics(returns: pd.Series, weights: pd.DataFrame) -> dict:
        metrics = performance_metrics(returns, periods_per_year=252)
        metrics["average_monthly_turnover"] = _average_one_way_turnover(weights)
        return metrics

    base_metrics = _fund_metrics(base_returns, base_weights)
    fused_metrics = _fund_metrics(fused_returns, fused_weights)

    comparison = pd.DataFrame(
        {
            "portfolio": [base_name, base_name + "_sentiment_fused"],
            **{
                col: [base_metrics[col], fused_metrics[col]]
                for col in metric_columns
            },
        }
    )

    print("\nVestra sentiment-fusion comparison:")
    print(comparison.round(4).to_string(index=False))

    comparison.to_csv(tables_dir / "sentiment_fusion_comparison.csv", index=False)
    fused_returns.to_csv(data_dir / "sentiment_fusion_returns.csv")
    fused_weights.to_csv(data_dir / "sentiment_fusion_weights.csv")

    # Exhibit 7: Sentiment-fusion before-vs-after (base vs fused), growth and
    # drawdown on a shared date axis.
    def _wealth_and_dd(returns: pd.Series) -> tuple:
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        return wealth, drawdown

    base_growth, base_dd = _wealth_and_dd(base_returns)
    fused_growth, fused_dd = _wealth_and_dd(fused_returns)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(base_growth.index, base_growth, label="Base: Equal-Weight Equity", linewidth=1.6)
    ax1.plot(fused_growth.index, fused_growth, label="Sentiment-Augmented", linewidth=1.6)
    ax1.set_ylabel("Value of $1")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left")

    ax2.plot(base_dd.index, base_dd * 100, label="Base", linewidth=1.6)
    ax2.plot(fused_dd.index, fused_dd * 100, label="Sentiment-Augmented", linewidth=1.6)
    ax2.set_ylabel("Drawdown from Peak (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower left")
    dd_min = min(base_dd.min(), fused_dd.min()) * 100
    ax2.set_ylim(dd_min * 1.05, 0)

    fig.suptitle(
        "Sentiment Fusion: Base vs Sentiment-Augmented Equal-Weight Equity Fund\n"
        f"{base_growth.index.min().date()} to {base_growth.index.max().date()}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        figures_dir / "sentiment_fusion_before_after.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ---- PDAIS investability scores (Project A methodology, capped sample) ----
    eq_prices_capped = eq_prices[eq_prices["date"] <= "2023-12-31"]
    cr_prices_capped = cr_prices[cr_prices["date"] <= "2023-12-31"]

    invest = compute_investability_scores(
        eq_returns,
        cr_returns,
        eq_prices_capped,
        cr_prices_capped,
    )

    invest["monthly_scores"].to_csv(
        data_dir / "investability_monthly_scores.csv",
        index=False,
    )
    invest["summary"].to_csv(
        tables_dir / "investability_summary.csv",
        index=False,
    )
    invest["sensitivity"].to_csv(
        tables_dir / "investability_weight_sensitivity.csv",
        index=False,
    )
    invest["rank_changes"].to_csv(
        tables_dir / "investability_rank_changes.csv",
        index=False,
    )

    # ---- PDAIS 2.0 overlay experiment (combined_risk_parity baseline) ----
    pdais_base_name = "combined_risk_parity"
    pdais_base_weights = results[pdais_base_name]["weights"]
    pdais_base_returns = results[pdais_base_name]["daily_returns"]

    # Map EQ_<ticker> / CR_<ticker> back to the plain tickers used by PDAIS.
    pdais_ticker_map = pd.Series(
        {
            col: col[len("EQ_"):] if col.startswith("EQ_") else col[len("CR_"):]
            for col in pdais_base_weights.columns
        },
        index=pdais_base_weights.columns,
        name="ticker",
    )

    pdais_overlay_weights = apply_pdais(
        pdais_base_weights,
        invest["monthly_scores"],
        pdais_ticker_map,
        score_col="PDAIS_EW",
        tilt_strength=0.25,
        max_weight=0.20,
    )

    pdais_overlay_returns = _holding_returns(
        combined_returns,
        pdais_overlay_weights,
        live_index=pdais_base_returns.index,
    ).rename(pdais_base_name + "_pdais_overlay")

    def _overlay_metrics(returns: pd.Series, weights: pd.DataFrame) -> dict:
        metrics = performance_metrics(returns, periods_per_year=252)
        metrics["average_monthly_turnover"] = _average_one_way_turnover(weights)
        metrics["average_maximum_asset_weight"] = weights.max(axis=1).mean()
        crypto_columns = [c for c in weights.columns if c.startswith("CR_")]
        metrics["average_crypto_exposure"] = weights[crypto_columns].sum(axis=1).mean()
        return metrics

    pdais_comparison_columns = [
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "max_drawdown",
        "average_monthly_turnover",
        "average_maximum_asset_weight",
        "average_crypto_exposure",
    ]

    pdais_overlay_comparison = pd.DataFrame(
        {
            "portfolio": [pdais_base_name, pdais_base_name + "_pdais_overlay"],
            **{
                col: [
                    _overlay_metrics(pdais_base_returns, pdais_base_weights)[col],
                    _overlay_metrics(pdais_overlay_returns, pdais_overlay_weights)[col],
                ]
                for col in pdais_comparison_columns
            },
        }
    )

    print("\nVestra PDAIS 2.0 comparison:")
    print(pdais_overlay_comparison.round(4).to_string(index=False))

    pdais_overlay_comparison.to_csv(
        tables_dir / "pdais_overlay_comparison.csv",
        index=False,
    )
    pdais_overlay_returns.to_csv(data_dir / "pdais_overlay_returns.csv")
    pdais_overlay_weights.to_csv(data_dir / "pdais_overlay_weights.csv")

    # ---- PDAIS investability guardrail (defensive, no alpha boost) ----
    guard = apply_pdais_guardrail(
        pdais_base_weights,
        invest["monthly_scores"],
        pdais_ticker_map,
        penalty=0.30,
        quartile=0.25,
        max_weight=0.20,
    )

    pdais_guard_weights = guard["weights"]
    pdais_guard_returns = _holding_returns(
        combined_returns,
        pdais_guard_weights,
        live_index=pdais_base_returns.index,
    ).rename(pdais_base_name + "_pdais_guardrail")

    base_metrics = _overlay_metrics(pdais_base_returns, pdais_base_weights)
    overlay_metrics = _overlay_metrics(pdais_overlay_returns, pdais_overlay_weights)
    guard_metrics = _overlay_metrics(pdais_guard_returns, pdais_guard_weights)

    guard_flags_per_rebalance = guard["flags"].sum(axis=1)
    guard_avg_flags = float(guard_flags_per_rebalance.mean())
    guard_pct_with_flags = float((guard_flags_per_rebalance > 0).mean())
    guard_avg_penalised_weight = float(guard["penalised_weight"].mean())
    guard_max_weight = float(pdais_guard_weights.max().max())
    guard_row_sums_ok = bool(
        (pdais_guard_weights.sum(axis=1) - 1.0).abs().max() < 1e-8
    )

    pdais_guard_comparison = pd.DataFrame(
        {
            "portfolio": [
                pdais_base_name,
                pdais_base_name + "_pdais_overlay",
                pdais_base_name + "_pdais_guardrail",
            ],
            "annualised_return": [
                base_metrics["annualised_return"],
                overlay_metrics["annualised_return"],
                guard_metrics["annualised_return"],
            ],
            "annualised_volatility": [
                base_metrics["annualised_volatility"],
                overlay_metrics["annualised_volatility"],
                guard_metrics["annualised_volatility"],
            ],
            "sharpe": [
                base_metrics["sharpe"],
                overlay_metrics["sharpe"],
                guard_metrics["sharpe"],
            ],
            "max_drawdown": [
                base_metrics["max_drawdown"],
                overlay_metrics["max_drawdown"],
                guard_metrics["max_drawdown"],
            ],
            "average_monthly_turnover": [
                base_metrics["average_monthly_turnover"],
                overlay_metrics["average_monthly_turnover"],
                guard_metrics["average_monthly_turnover"],
            ],
            "incremental_turnover": [
                0.0,
                overlay_metrics["average_monthly_turnover"] - base_metrics["average_monthly_turnover"],
                guard_metrics["average_monthly_turnover"] - base_metrics["average_monthly_turnover"],
            ],
            "average_maximum_asset_weight": [
                base_metrics["average_maximum_asset_weight"],
                overlay_metrics["average_maximum_asset_weight"],
                guard_metrics["average_maximum_asset_weight"],
            ],
            "average_crypto_exposure": [
                base_metrics["average_crypto_exposure"],
                overlay_metrics["average_crypto_exposure"],
                guard_metrics["average_crypto_exposure"],
            ],
            "average_flags_per_rebalance": [0.0, 0.0, guard_avg_flags],
            "percent_rebalances_with_flags": [0.0, 0.0, guard_pct_with_flags],
            "average_penalised_weight": [0.0, 0.0, guard_avg_penalised_weight],
        }
    )

    print("\nVestra PDAIS investability guardrail comparison:")
    print(pdais_guard_comparison.round(4).to_string(index=False))
    print("\nGuardrail diagnostics:")
    print(f"  Average flags per rebalance: {guard_avg_flags:.3f}")
    print(f"  % rebalances with >=1 flag: {guard_pct_with_flags:.1%}")
    print(f"  Average penalised baseline weight: {guard_avg_penalised_weight:.4f}")
    print(f"  Maximum resulting asset weight: {guard_max_weight:.4f}")
    print(f"  Weight-row sums valid: {guard_row_sums_ok}")

    pdais_guard_comparison.to_csv(
        tables_dir / "pdais_guardrail_comparison.csv",
        index=False,
    )
    pdais_guard_returns.to_csv(data_dir / "pdais_guardrail_returns.csv")
    pdais_guard_weights.to_csv(data_dir / "pdais_guardrail_weights.csv")

    print("\nSaved portfolio outputs:")
    print(data_dir / "fund_returns.csv")
    print(data_dir / "fund_weights.csv")
    print(data_dir / "sector_sentiment_index.csv")
    print(tables_dir / "performance_metrics.csv")
    print(figures_dir / "combined_growth_of_1.png")
    print(figures_dir / "combined_drawdown.png")
    print(figures_dir / "combined_crypto_weight_over_time.png")
    print(figures_dir / "all_funds_risk_return.png")
    print(figures_dir / "sector_sentiment_over_time.png")
    print(figures_dir / "portfolio_weights_over_time.png")
    print(figures_dir / "sentiment_fusion_before_after.png")
    print(data_dir / "sentiment_fusion_returns.csv")
    print(data_dir / "sentiment_fusion_weights.csv")
    print(tables_dir / "sentiment_fusion_comparison.csv")
    print(data_dir / "investability_monthly_scores.csv")
    print(tables_dir / "investability_summary.csv")
    print(tables_dir / "investability_weight_sensitivity.csv")
    print(tables_dir / "investability_rank_changes.csv")
    print(data_dir / "pdais_overlay_returns.csv")
    print(data_dir / "pdais_overlay_weights.csv")
    print(tables_dir / "pdais_overlay_comparison.csv")
    print(data_dir / "pdais_guardrail_returns.csv")
    print(data_dir / "pdais_guardrail_weights.csv")
    print(tables_dir / "pdais_guardrail_comparison.csv")


if __name__ == "__main__":
    main()
