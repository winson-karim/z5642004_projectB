"""Vestra Systematic Funds - Project B investor app.

This dashboard loads precomputed artifacts from results/ and never recomputes
backtests or sentiment. Run locally with: streamlit run streamlit_app.py
"""
import pathlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "results" / "data"
TABLES_DIR = BASE_DIR / "results" / "tables"

FUND_ORDER = [
    "equity_equal_weight",
    "equity_min_variance",
    "equity_risk_parity",
    "equity_max_sharpe",
    "crypto_equal_weight",
    "crypto_min_variance",
    "crypto_risk_parity",
    "crypto_max_sharpe",
    "combined_equal_weight",
    "combined_min_variance",
    "combined_risk_parity",
    "combined_max_sharpe",
]


@st.cache_data(show_spinner="Loading fund returns...")
def load_fund_returns():
    df = pd.read_csv(DATA_DIR / "fund_returns.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


@st.cache_data(show_spinner="Loading fund weights...")
def load_fund_weights():
    df = pd.read_csv(DATA_DIR / "fund_weights.csv", parse_dates=["rebalance_date"])
    return df


@st.cache_data(show_spinner="Loading performance metrics...")
def load_performance_metrics():
    df = pd.read_csv(TABLES_DIR / "performance_metrics.csv")
    return df


@st.cache_data(show_spinner="Loading sector sentiment...")
def load_sector_sentiment():
    df = pd.read_csv(DATA_DIR / "sector_sentiment_index.csv", parse_dates=["date"])
    return df


@st.cache_data(show_spinner="Loading fusion outputs...")
def load_fusion():
    returns = pd.read_csv(
        DATA_DIR / "sentiment_fusion_returns.csv",
        parse_dates=["date"],
    )
    comparison = pd.read_csv(
        TABLES_DIR / "sentiment_fusion_comparison.csv"
    )
    return returns.set_index("date").sort_index(), comparison


@st.cache_data(show_spinner="Loading investability scores...")
def load_investability_scores():
    return pd.read_csv(
        DATA_DIR / "investability_monthly_scores.csv"
    )


@st.cache_data(show_spinner="Loading investability summary...")
def load_investability_summary():
    return pd.read_csv(
        TABLES_DIR / "investability_summary.csv"
    )


@st.cache_data(show_spinner="Loading investability weight sensitivity...")
def load_investability_weight_sensitivity():
    return pd.read_csv(
        TABLES_DIR / "investability_weight_sensitivity.csv"
    )


@st.cache_data(show_spinner="Loading investability rank changes...")
def load_investability_rank_changes():
    return pd.read_csv(
        TABLES_DIR / "investability_rank_changes.csv"
    )


@st.cache_data(show_spinner="Loading PDAIS overlay comparison...")
def load_pdais_overlay_comparison():
    return pd.read_csv(
        TABLES_DIR / "pdais_overlay_comparison.csv"
    )


@st.cache_data(show_spinner="Loading PDAIS guardrail comparison...")
def load_pdais_guardrail_comparison():
    return pd.read_csv(
        TABLES_DIR / "pdais_guardrail_comparison.csv"
    )


@st.cache_data(show_spinner="Loading PDAIS guardrail returns...")
def load_pdais_guardrail_returns():
    df = pd.read_csv(
        DATA_DIR / "pdais_guardrail_returns.csv",
        parse_dates=["date"],
    )
    return df.set_index("date").sort_index()


def growth_of_one(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()

    if clean.empty:
        return clean

    return (1.0 + clean).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    clean = returns.dropna()

    if clean.empty:
        return clean

    wealth = (1.0 + clean).cumprod()

    return wealth / wealth.cummax() - 1.0


def line(fig, x, y, name, color=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="lines",
            line=dict(color=color),
        )
    )


st.set_page_config(
    page_title="Vestra Systematic Funds",
    layout="wide",
)

st.title("Vestra Systematic Multi-Asset Funds")

st.caption(
    "Project B dashboard. All figures are precomputed from walk-forward "
    "out-of-sample backtests; nothing is recomputed in the app."
)

st.markdown("### How to Use Vestra")

col_journey1, col_journey2, col_journey3, col_journey4 = st.columns(4)

with col_journey1:

    st.markdown("**1. Compare Funds**")

    st.caption(
        "Review historical return, risk, drawdown and portfolio composition "
        "across Vestra funds."
    )

with col_journey2:

    st.markdown("**2. Build Your Allocation**")

    st.caption(
        "Combine selected funds and explore how different fund weights "
        "affect historical portfolio performance."
    )

with col_journey3:

    st.markdown("**3. Review Sentiment**")

    st.caption(
        "Examine sector-level news sentiment and how lagged sentiment "
        "signals affected the equity strategy."
    )

with col_journey4:

    st.markdown("**4. Assess Investability**")

    st.caption(
        "Use Vestra's Investability Score as an additional decision-support "
        "tool for asset quality and portfolio risk."
    )

st.caption(
    "Vestra combines systematic fund analysis, market sentiment and "
    "investability analytics to support an informed investment process."
)


fund_returns = load_fund_returns()
fund_weights = load_fund_weights()
perf = load_performance_metrics()
sentiment = load_sector_sentiment()

fusion_returns, fusion_comparison = load_fusion()

investability_scores = load_investability_scores()
investability_summary = load_investability_summary()
investability_sensitivity = load_investability_weight_sensitivity()
investability_rank_changes = load_investability_rank_changes()

overlay_comparison = load_pdais_overlay_comparison()
guardrail_comparison = load_pdais_guardrail_comparison()
guardrail_returns = load_pdais_guardrail_returns()


tab_funds, tab_sentiment, tab_fusion, tab_investability, tab_risk, tab_data = st.tabs(
    [
        "Funds",
        "Sentiment",
        "Fusion",
        "Investability",
        "Risk Lab",
        "Data",
    ]
)


with tab_funds:

    col_picker, col_metrics = st.columns([1, 2])

    with col_picker:

        selected = st.selectbox(
            "Select a fund",
            FUND_ORDER,
            format_func=lambda f: f.replace("_", " ").title(),
        )

    with col_metrics:

        st.subheader("Performance summary")

        row = perf[
            perf["fund"] == selected
        ].iloc[0]

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Metric": "Annualised return",
                        "Value": f"{row['annualised_return']:.2%}",
                    },
                    {
                        "Metric": "Annualised volatility",
                        "Value": f"{row['annualised_volatility']:.2%}",
                    },
                    {
                        "Metric": "Sharpe ratio",
                        "Value": f"{row['sharpe']:.2f}",
                    },
                    {
                        "Metric": "Max drawdown",
                        "Value": f"{row['max_drawdown']:.2%}",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    returns = fund_returns[selected]

    growth = growth_of_one(returns)

    dd = drawdown(returns)

    period = (
        f"{growth.index.min().date()} "
        f"to {growth.index.max().date()}"
    )

    st.subheader("Growth of $1")

    fig = go.Figure()

    line(
        fig,
        growth.index,
        growth.values,
        "Growth of $1",
        "#1f77b4",
    )

    fig.update_layout(
        title=(
            f"{selected.replace('_', ' ').title()} - Growth of $1 ({period})"
        ),
        xaxis_title="Date",
        yaxis_title="Value of $1",
        height=420,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.subheader("Drawdown")

    fig2 = go.Figure()

    line(
        fig2,
        dd.index,
        dd.values,
        "Drawdown",
        "#d62728",
    )

    fig2.update_layout(
        title=(
            f"{selected.replace('_', ' ').title()} - Drawdown ({period})"
        ),
        xaxis_title="Date",
        yaxis_title="Drawdown from peak",
        height=360,
    )

    st.plotly_chart(
        fig2,
        width="stretch",
    )

    st.subheader("Holdings at the latest rebalance")

    latest_date = (
        fund_weights[
            fund_weights["fund"] == selected
        ]["rebalance_date"]
        .max()
    )

    holdings = fund_weights[
        (fund_weights["fund"] == selected)
        & (
            fund_weights["rebalance_date"]
            == latest_date
        )
    ].sort_values(
        "weight",
        ascending=False,
    )

    top = holdings.head(10)

    other = (
        float(
            holdings["weight"]
            .iloc[10:]
            .sum()
        )
        if len(holdings) > 10
        else 0.0
    )

    fig3 = go.Figure()

    fig3.add_trace(
        go.Bar(
            x=top["weight"],
            y=top["asset"],
            orientation="h",
            name="Holdings",
        )
    )

    if other > 0:

        fig3.add_trace(
            go.Bar(
                x=[other],
                y=["Other"],
                orientation="h",
                name="Other",
            )
        )

    fig3.update_layout(
        title=(
            f"{selected.replace('_', ' ').title()} - Latest Portfolio "
            f"Weights ({latest_date.date()})"
        ),
        xaxis_title="Weight",
        yaxis_title="Asset",
        height=380,
        barmode="stack",
    )

    st.plotly_chart(
        fig3,
        width="stretch",
    )

    st.subheader("Portfolio weights over time")

    st.markdown(
        "See how the selected fund's target allocation changed at each rebalance "
        "during the walk-forward backtest."
    )

    weights_history = fund_weights[
        fund_weights["fund"] == selected
    ].copy()

    weights_pivot = (
        weights_history.pivot_table(
            index="rebalance_date",
            columns="asset",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    fig_weights_time = go.Figure()

    for asset_name in weights_pivot.columns:
        fig_weights_time.add_trace(
            go.Scatter(
                x=weights_pivot.index,
                y=weights_pivot[asset_name],
                mode="lines",
                name=asset_name,
                stackgroup="one",
                hovertemplate=(
                    f"<b>{asset_name}</b><br>"
                    "Rebalance date: %{x|%Y-%m-%d}<br>"
                    "Weight: %{y:.2%}<extra></extra>"
                ),
            )
        )

    fig_weights_time.update_layout(
        title=(
            f"{selected.replace('_', ' ').title()} - Portfolio Weights "
            "Over Time"
        ),
        xaxis_title="Rebalance Date",
        yaxis_title="Portfolio Weight",
        height=500,
        hovermode="x unified",
        legend_title="Asset",
    )

    fig_weights_time.update_yaxes(
        tickformat=".0%",
        range=[0, 1],
    )

    st.plotly_chart(
        fig_weights_time,
        width="stretch",
    )

    st.caption(
        "Each rebalance uses only information available at that point in time, "
        "avoiding look-ahead bias."
    )

    st.divider()

    st.subheader("Compare Funds")

    st.markdown(
        "Compare all available Vestra funds across return, risk, Sharpe ratio, "
        "drawdown, and cumulative performance."
    )

    comparison_table = perf.copy()

    comparison_table["Fund"] = comparison_table["fund"].str.replace(
        "_", " ", regex=False
    ).str.title()

    comparison_table["Annualised Return"] = comparison_table[
        "annualised_return"
    ].map(lambda x: f"{x:.2%}")

    comparison_table["Annualised Volatility"] = comparison_table[
        "annualised_volatility"
    ].map(lambda x: f"{x:.2%}")

    comparison_table["Sharpe Ratio"] = comparison_table["sharpe"].map(
        lambda x: f"{x:.3f}"
    )

    comparison_table["Max Drawdown"] = comparison_table[
        "max_drawdown"
    ].map(lambda x: f"{x:.2%}")

    st.dataframe(
        comparison_table[
            [
                "Fund",
                "Annualised Return",
                "Annualised Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Growth of $1 across methods")

    family_label = st.radio(
        "Asset family",
        ["Combined", "Equity", "Crypto"],
        horizontal=True,
    )

    family_prefix = family_label.lower()

    family_funds = [
        f for f in FUND_ORDER if f.startswith(f"{family_prefix}_")
    ]

    fig_compare_growth = go.Figure()

    for fund_name in family_funds:
        series = growth_of_one(fund_returns[fund_name])

        line(
            fig_compare_growth,
            series.index,
            series.values,
            fund_name.replace("_", " ").title(),
        )

    fig_compare_growth.update_layout(
        title=f"{family_label} funds - growth of $1",
        xaxis_title="Date",
        yaxis_title="Value of $1",
        height=440,
    )

    st.plotly_chart(
        fig_compare_growth,
        width="stretch",
    )

    st.subheader("Return vs risk across funds")

    risk_return = perf.copy()

    risk_return["Fund"] = risk_return["fund"].str.replace(
        "_", " ", regex=False
    ).str.title()

    fig_risk_return = go.Figure()

    fig_risk_return.add_trace(
        go.Scatter(
            x=risk_return["annualised_volatility"],
            y=risk_return["annualised_return"],
            mode="markers",
            text=risk_return["Fund"],
            customdata=risk_return[["sharpe", "max_drawdown"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Annualised volatility: %{x:.2%}<br>"
                "Annualised return: %{y:.2%}<br>"
                "Sharpe ratio: %{customdata[0]:.3f}<br>"
                "Max drawdown: %{customdata[1]:.2%}<extra></extra>"
            ),
        )
    )

    fig_risk_return.update_layout(
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        height=440,
    )

    fig_risk_return.update_xaxes(tickformat=".0%")
    fig_risk_return.update_yaxes(tickformat=".0%")

    st.plotly_chart(
        fig_risk_return,
        width="stretch",
    )

    st.divider()

    st.subheader(
        "Vestra Guided Allocation"
    )

    st.markdown(
        "Choose an illustrative starting profile based on different historical "
        "risk preferences. You can then customise the fund weights below."
    )

    guided_profiles = {
        "Defensive": {
            "combined_min_variance": 50.0,
            "combined_risk_parity": 35.0,
            "combined_equal_weight": 15.0,
            "combined_max_sharpe": 0.0,
        },
        "Balanced": {
            "combined_min_variance": 25.0,
            "combined_risk_parity": 35.0,
            "combined_equal_weight": 25.0,
            "combined_max_sharpe": 15.0,
        },
        "Growth": {
            "combined_min_variance": 10.0,
            "combined_risk_parity": 20.0,
            "combined_equal_weight": 25.0,
            "combined_max_sharpe": 45.0,
        },
    }

    guided_descriptions = {
        "Defensive": (
            "Prioritises historically lower-risk portfolio methods, with "
            "greater emphasis on Minimum Variance and Risk Parity."
        ),
        "Balanced": (
            "Combines defensive and growth-oriented methods to provide a more "
            "even balance between historical risk and return."
        ),
        "Growth": (
            "Places greater emphasis on historically higher-return strategies "
            "while accepting greater portfolio risk and variability."
        ),
    }

    guided_profile = st.radio(
        "Allocation profile",
        list(guided_profiles),
        index=list(guided_profiles).index("Balanced"),
        horizontal=True,
    )

    st.markdown(
        guided_descriptions[guided_profile]
    )

    guided_weights = guided_profiles[guided_profile]

    guided_table = pd.DataFrame(
        {
            "Fund": [
                "Combined Equal Weight",
                "Combined Minimum Variance",
                "Combined Risk Parity",
                "Combined Max Sharpe",
            ],
            "Illustrative Allocation": [
                f"{guided_weights[fund]:.0f}%"
                for fund in [
                    "combined_equal_weight",
                    "combined_min_variance",
                    "combined_risk_parity",
                    "combined_max_sharpe",
                ]
            ],
        }
    )

    st.dataframe(
        guided_table,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Illustrative allocations are based on historical fund characteristics "
        "and are provided for exploration only. They are not personalised "
        "investment recommendations, and past performance does not guarantee "
        "future results."
    )

    if st.button("Use This Profile"):

        st.session_state["allocation_builder_funds"] = list(
            guided_weights.keys()
        )

        for fund_name, weight_pct in guided_weights.items():

            st.session_state[f"allocation_{fund_name}"] = weight_pct

        st.success(
            "The selected profile has been loaded into the manual allocation "
            "builder below. You can still adjust the weights."
        )

    st.subheader(
        "Selected Profile - Historical Growth of $1"
    )

    guided_return_series = (
        fund_returns[list(guided_weights.keys())]
        .dropna()
        .mul(
            {
                fund_name: weight_pct / 100.0
                for fund_name, weight_pct in guided_weights.items()
            },
            axis=1,
        )
        .sum(axis=1)
    )

    guided_growth = growth_of_one(guided_return_series)

    fig_guided = go.Figure()

    line(
        fig_guided,
        guided_growth.index,
        guided_growth.values,
        "Guided profile",
        "#1f77b4",
    )

    fig_guided.update_layout(
        xaxis_title="Date",
        yaxis_title="Value of $1",
        height=360,
    )

    st.plotly_chart(
        fig_guided,
        width="stretch",
    )

    st.metric(
        "Ending value of $1",
        f"${guided_growth.iloc[-1]:.2f}",
    )

    st.metric(
        "Historical cumulative return",
        f"{(guided_growth.iloc[-1] - 1.0):+.1%}",
    )

    st.caption(
        "This preview combines existing precomputed fund returns using the "
        "selected illustrative weights. It does not rerun or optimise the "
        "underlying fund strategies."
    )

    st.subheader(
        "Build Your Fund Allocation"
    )

    st.markdown(
        "Choose up to four of the platform funds and set a target weight for "
        "each. The blended path below is computed from the same precomputed "
        "fund return series used across the app."
    )

    if "allocation_builder_funds" not in st.session_state:
        st.session_state["allocation_builder_funds"] = [
            "combined_risk_parity",
            "combined_max_sharpe",
            "combined_min_variance",
            "combined_equal_weight",
        ]

    builder_funds = st.multiselect(
        "Funds to include",
        FUND_ORDER,
        key="allocation_builder_funds",
    )

    if builder_funds:

        st.markdown("Set Your Fund Weights (Total = 100%)")

        weights = {}

        for fund_name in builder_funds:

            if f"allocation_{fund_name}" not in st.session_state:
                st.session_state[f"allocation_{fund_name}"] = 25.0

            weights[fund_name] = st.number_input(
                fund_name.replace("_", " ").title(),
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                key=f"allocation_{fund_name}",
            )

        total = sum(weights.values())

        st.metric(
            "Total Portfolio Allocation",
            f"{total:.0f}%",
        )

        allocation_df = pd.DataFrame(
            {
                "Fund": [
                    fund_name.replace("_", " ").title()
                    for fund_name in weights.keys()
                ],
                "Weight": [f"{w:.0f}%" for w in weights.values()],
            }
        )

        st.dataframe(
            allocation_df,
            width="stretch",
            hide_index=True,
        )

        if abs(total - 100.0) > 0.01:

            st.warning(
                f"Your weights sum to {total:.0f}% - adjust them to 100% to "
                "see the blended path."
            )

        else:

            norm = {k: v / 100.0 for k, v in weights.items()}

            blended = (
                fund_returns[builder_funds]
                .dropna()
                .mul(norm, axis=1)
                .sum(axis=1)
            )

            blended = growth_of_one(blended)

            fig_builder = go.Figure()

            builder_line = go.Scatter(
                x=blended.index,
                y=blended.values,
                mode="lines",
                name="Blended allocation",
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Value of $1: %{y:.2f}<extra></extra>"
                ),
                line=dict(color="#2ca02c"),
            )

            fig_builder.add_trace(builder_line)

            fig_builder.update_layout(
                title="Custom allocation - historical growth of $1",
                xaxis_title="Date",
                yaxis_title="Value of $1",
                height=420,
            )

            st.plotly_chart(
                fig_builder,
                width="stretch",
            )

            st.metric("Starting value", "$1.00")
            st.metric("Ending value", f"${blended.iloc[-1]:.2f}")
            st.metric(
                "Cumulative return",
                f"{(blended.iloc[-1] - 1.0):+.1%}",
            )

            st.caption(
                "The blended path is computed from the same precomputed fund "
                "return series used elsewhere in this app; no backtest is "
                "re-run when you change the allocation."
            )

    else:

        st.info(
            "Pick at least one fund to build your own allocation."
        )


with tab_sentiment:

    st.subheader(
        "Sector Sentiment Index Over Time"
    )

    sectors = sentiment.columns.drop(
        "date"
    ).tolist()

    chosen = st.multiselect(
        "Sectors",
        sectors,
        default=sectors[:5],
    )

    fig4 = go.Figure()

    for s in chosen:

        line(
            fig4,
            sentiment["date"],
            sentiment[s],
            s,
        )

    fig4.update_layout(
        title=(
            f"Sector Sentiment Index "
            f"({sentiment['date'].min().date()} "
            f"to {sentiment['date'].max().date()})"
        ),
        xaxis_title="Date",
        yaxis_title=(
            "Sentiment "
            "(VADER, equal-weight sector)"
        ),
        height=480,
    )

    st.plotly_chart(
        fig4,
        width="stretch",
    )

    st.subheader(
        "Monthly average sentiment by sector"
    )

    monthly = (
        sentiment
        .set_index("date")
        .resample("ME")
        .mean()
    )

    fig5 = go.Figure(
        go.Heatmap(
            z=monthly[sectors].T.values,
            x=monthly.index.strftime(
                "%Y-%m"
            ),
            y=sectors,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(
                title="Sentiment"
            ),
        )
    )

    fig5.update_layout(
        height=480,
        xaxis_nticks=48,
    )

    st.plotly_chart(
        fig5,
        width="stretch",
    )


with tab_fusion:

    st.subheader(
        "Sentiment-fusion comparison"
    )

    st.markdown(
        "The sentiment-fused variant applies a one-trading-day lagged sector "
        "sentiment tilt (strength 0.20, 0.40 cap) to the equal-weight equity "
        "fund and holds the tilted weights between monthly rebalances."
    )

    st.dataframe(
        fusion_comparison.round(4),
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Growth of $1 - base vs sentiment-fused"
    )

    base = growth_of_one(
        fund_returns[
            "equity_equal_weight"
        ]
    )

    fused = growth_of_one(
        fusion_returns[
            "equity_equal_weight_sentiment_fused"
        ]
    )

    fig6 = make_subplots(
        specs=[
            [
                {
                    "secondary_y": False
                }
            ]
        ]
    )

    line(
        fig6,
        base.index,
        base.values,
        "Equal-weight equity (base)",
        "#1f77b4",
    )

    line(
        fig6,
        fused.index,
        fused.values,
        "Sentiment-fused",
        "#ff7f0e",
    )

    fig6.update_layout(
        title=(
            "Growth of $1 - "
            "base vs sentiment-fused"
        ),
        xaxis_title="Date",
        yaxis_title="Value of $1",
        height=420,
    )

    st.plotly_chart(
        fig6,
        width="stretch",
    )

    st.subheader(
        "Average one-way turnover added by the tilt"
    )

    st.metric(
        "Monthly one-way turnover (fused)",
        f"{fusion_comparison.iloc[1]['average_monthly_turnover']:.2%}",
    )


with tab_investability:

    st.subheader(
        "Vestra Investability Score"
    )

    st.markdown(
        "Vestra's Investability Score rates each asset from 0 to 100 using three "
        "ideas: how stable its returns are, how well it holds up during "
        "negative-return periods, and how easy it is to trade. Stocks and crypto "
        "are scored separately, so rankings should only be compared within their "
        "own asset group."
    )

    lens_map = {
        "Balanced score": "PDAIS_EW",
        "Risk-focused score": "PDAIS_RF",
        "Liquidity-focused score": "PDAIS_LF",
    }

    lens_label = st.selectbox(
        "Investability scoring style",
        list(lens_map),
    )

    lens = lens_map[
        lens_label
    ]

    latest_month = (
        investability_scores[
            "year_month"
        ].max()
    )

    latest = investability_scores[
        investability_scores[
            "year_month"
        ]
        == latest_month
    ]

    col_kpi1, col_kpi2, col_kpi3 = st.columns(
        3
    )

    col_kpi1.metric(
        "Assets scored",
        int(
            latest[
                "ticker"
            ].nunique()
        ),
    )

    col_kpi2.metric(
        "Latest scoring month",
        latest_month,
    )

    col_kpi3.metric(
        "Median investability score",
        f"{latest[lens].dropna().median():.1f} / 100",
    )

    st.caption(
        "Scores are relative within each asset class and should not be interpreted as expected returns."
    )

    st.subheader(
        f"Top-ranked assets - {latest_month}"
    )

    equity_rank = (
        latest[
            latest[
                "asset_class"
            ]
            == "Equity"
        ]
        .dropna(
            subset=[lens]
        )
        .sort_values(
            lens,
            ascending=False,
        )
        .head(10)[
            [
                "ticker",
                lens,
            ]
        ]
        .rename(
            columns={
                lens: (
                    "Investability score"
                )
            }
        )
        .round(1)
        .reset_index(
            drop=True
        )
    )

    crypto_rank = (
        latest[
            latest[
                "asset_class"
            ]
            == "Crypto"
        ]
        .dropna(
            subset=[lens]
        )
        .sort_values(
            lens,
            ascending=False,
        )
        .head(10)[
            [
                "ticker",
                lens,
            ]
        ]
        .rename(
            columns={
                lens: (
                    "Investability score"
                )
            }
        )
        .round(1)
        .reset_index(
            drop=True
        )
    )

    col_eq, col_cr = st.columns(
        2
    )

    with col_eq:

        st.markdown(
            "**Top 10 stocks**"
        )

        st.dataframe(
            equity_rank,
            hide_index=True,
            width="stretch",
        )

    with col_cr:

        st.markdown(
            "**Top 10 crypto**"
        )

        st.dataframe(
            crypto_rank,
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "Stocks and crypto are shown separately because Vestra normalises each "
        "asset class independently."
    )

    st.subheader(
        "Investability component history"
    )

    asset_type = st.radio(
        "Asset type",
        [
            "Stocks",
            "Crypto",
        ],
    )

    class_filter = {
        "Stocks": "Equity",
        "Crypto": "Crypto",
    }[
        asset_type
    ]

    tickers = sorted(
        investability_scores[
            investability_scores[
                "asset_class"
            ]
            == class_filter
        ][
            "ticker"
        ].unique()
    )

    asset = st.selectbox(
        "Asset",
        tickers,
    )

    hist = (
        investability_scores[
            investability_scores[
                "ticker"
            ]
            == asset
        ]
        .sort_values(
            "year_month"
        )
    )

    hist_x = pd.to_datetime(
        hist[
            "year_month"
        ],
        format="%Y-%m",
    )

    fig7 = go.Figure()

    line(
        fig7,
        hist_x,
        hist["RS_norm"],
        "Return stability",
        "#1f77b4",
    )

    line(
        fig7,
        hist_x,
        hist["DR_norm"],
        "Downside resilience",
        "#2ca02c",
    )

    line(
        fig7,
        hist_x,
        hist["LQ_norm"],
        "Liquidity quality",
        "#d62728",
    )

    fig7.update_layout(
        title=(
            f"{asset} - "
            "what drives its investability score"
        ),
        xaxis_title="Month",
        yaxis_title=(
            "Normalised score (0-100)"
        ),
        height=420,
    )

    st.plotly_chart(
        fig7,
        width="stretch",
    )

    col_sens, col_rank = st.columns(
        2
    )

    with col_sens:

        st.markdown(
            "**How stable are the rankings across scoring styles?**"
        )

        st.caption(
            "Higher values mean changing the scoring emphasis does not change "
            "the rankings much."
        )

        st.dataframe(
            investability_sensitivity,
            hide_index=True,
            width="stretch",
        )

    with col_rank:

        st.markdown(
            "**How much do asset rankings move?**"
        )

        st.caption(
            "Smaller changes mean the result is more robust to the chosen "
            "scoring style."
        )

        st.dataframe(
            investability_rank_changes,
            hide_index=True,
            width="stretch",
        )


with tab_risk:

    st.subheader(
        "Test 1 - Direct investability weighting"
    )

    st.markdown(
        "Vestra first tested a simple idea: give slightly more weight to "
        "higher-scoring assets and slightly less weight to lower-scoring "
        "assets. Scores were delayed by one month to avoid using future "
        "information."
    )

    overlay_display = overlay_comparison.rename(
        columns={
            "portfolio": "Portfolio",
            "annualised_return": "Annualised Return",
            "annualised_volatility": "Annualised Volatility",
            "sharpe": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "average_monthly_turnover": "Turnover",
        }
    ).copy()

    overlay_display["Portfolio"] = overlay_display["Portfolio"].replace(
        {
            "combined_risk_parity": "Combined Risk Parity",
            "combined_risk_parity_pdais_overlay": "Risk Parity + Investability Weighting",
        }
    )

    for c in [
        "Annualised Return",
        "Annualised Volatility",
        "Max Drawdown",
        "Turnover",
    ]:

        if c in overlay_display.columns:

            overlay_display[c] = overlay_display[c].map(
                lambda x: f"{x:.2%}"
            )

    if "Sharpe Ratio" in overlay_display.columns:

        overlay_display[
            "Sharpe Ratio"
        ] = overlay_display[
            "Sharpe Ratio"
        ].map(
            lambda x: f"{x:.3f}"
        )

    st.dataframe(
        overlay_display,
        hide_index=True,
        width="stretch",
    )

    st.info(
        "Result: direct investability weighting slightly reduced volatility, but it also "
        "reduced annualised return and the Sharpe ratio. This suggests higher "
        "investability scores should not automatically receive larger portfolio weights."
    )

    st.subheader(
        "Test 2 - Defensive investability filter"
    )

    st.markdown(
        "Vestra then tested a more defensive approach. When at least two of "
        "the three investability components were among the weakest in their "
        "asset group, that holding was reduced by 30%. No asset received an "
        "automatic positive boost."
    )

    guardrail_display = guardrail_comparison.rename(
        columns={
            "portfolio": "Portfolio",
            "annualised_return": "Annualised Return",
            "annualised_volatility": "Annualised Volatility",
            "sharpe": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "average_monthly_turnover": "Turnover",
        }
    ).copy()

    guardrail_display["Portfolio"] = guardrail_display["Portfolio"].replace(
        {
            "combined_risk_parity": "Combined Risk Parity",
            "combined_risk_parity_pdais_overlay": "Risk Parity + Investability Weighting",
            "combined_risk_parity_pdais_guardrail": "Risk Parity + Defensive Filter",
        }
    )

    for c in [
        "Annualised Return",
        "Annualised Volatility",
        "Max Drawdown",
        "Turnover",
    ]:

        if c in guardrail_display.columns:

            guardrail_display[c] = guardrail_display[c].map(
                lambda x: f"{x:.2%}"
            )

    if "Sharpe Ratio" in guardrail_display.columns:

        guardrail_display[
            "Sharpe Ratio"
        ] = guardrail_display[
            "Sharpe Ratio"
        ].map(
            lambda x: f"{x:.3f}"
        )

    st.dataframe(
        guardrail_display,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Did the defensive filter improve the portfolio?"
    )

    base_growth = growth_of_one(
        fund_returns[
            "combined_risk_parity"
        ]
    )

    guardrail_growth = growth_of_one(
        guardrail_returns[
            "combined_risk_parity_pdais_guardrail"
        ]
    )

    fig8 = go.Figure()

    line(
        fig8,
        base_growth.index,
        base_growth.values,
        "Combined risk parity (base)",
        "#1f77b4",
    )

    line(
        fig8,
        guardrail_growth.index,
        guardrail_growth.values,
        "Defensive investability filter",
        "#ff7f0e",
    )

    fig8.update_layout(
        title=(
            "Growth of $1 - Base vs Defensive Filter"
        ),
        xaxis_title="Date",
        yaxis_title="Value of $1",
        height=420,
    )

    st.plotly_chart(
        fig8,
        width="stretch",
    )

    base_row = guardrail_comparison[
        guardrail_comparison[
            "portfolio"
        ]
        == "combined_risk_parity"
    ].iloc[0]

    guardrail_row = guardrail_comparison[
        guardrail_comparison[
            "portfolio"
        ]
        == "combined_risk_parity_pdais_guardrail"
    ].iloc[0]

    st.info(
        f"Result: the defensive filter reduced annualised volatility slightly "
        f"({base_row['annualised_volatility']:.2%} to "
        f"{guardrail_row['annualised_volatility']:.2%}), but the Sharpe ratio "
        f"fell ({base_row['sharpe']:.3f} to "
        f"{guardrail_row['sharpe']:.3f}) and "
        f"monthly trading increased "
        f"({base_row['average_monthly_turnover']:.2%} to "
        f"{guardrail_row['average_monthly_turnover']:.2%}). "
        f"Vestra therefore keeps the Investability Score as decision support "
        f"rather than forcing it into the core portfolio."
    )
with tab_data:

    st.subheader("Precomputed results")

    st.write(
        "All files below are produced by scripts/run_part_b.py and stored in "
        "results/; the app only reads them."
    )

    for label, df in [
        ("fund_returns.csv (1208 days x 12 funds)", fund_returns),
        ("fund_weights.csv (rebalances x funds x assets)", fund_weights),
        ("performance_metrics.csv", perf),
        ("sector_sentiment_index.csv", sentiment),
        ("sentiment_fusion_returns.csv", fusion_returns),
        ("investability_monthly_scores.csv", investability_scores),
        ("investability_summary.csv", investability_summary),
        ("investability_weight_sensitivity.csv", investability_sensitivity),
        ("investability_rank_changes.csv", investability_rank_changes),
        ("pdais_overlay_comparison.csv", overlay_comparison),
        ("pdais_guardrail_comparison.csv", guardrail_comparison),
        ("pdais_guardrail_returns.csv", guardrail_returns),
    ]:

        with st.expander(label):

            st.dataframe(
                df.head(50),
                width="stretch",
            )
