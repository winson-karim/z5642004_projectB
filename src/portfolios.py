"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
def _minimum_variance_weights(history: pd.DataFrame, max_weight: float = 0.20) -> pd.Series:
    """Compute long-only minimum-variance weights with a per-asset cap."""
    cov = history.cov().values * 1_000_000.0
    n_assets = history.shape[1]
    x0 = np.repeat(1.0 / n_assets, n_assets)

    def objective(w):
        return float(w @ cov @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n_assets

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = np.clip(result.x, 0.0, max_weight)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return pd.Series(x0, index=history.columns, dtype=float)

    weights = weights / weights.sum()

    if (
        not np.isclose(weights.sum(), 1.0, atol=1e-8)
        or (weights < -1e-10).any()
        or (weights > max_weight + 1e-8).any()
    ):
        return pd.Series(x0, index=history.columns, dtype=float)

    return pd.Series(weights, index=history.columns, dtype=float)


# --- Risk Parity Helper ---
# --- Risk Parity Helper ---
def _risk_parity_weights(history: pd.DataFrame, max_weight: float = 0.20) -> pd.Series:
    """Compute long-only risk-parity weights with a per-asset cap."""
    cov = history.cov().values
    n_assets = history.shape[1]
    x0 = np.repeat(1.0 / n_assets, n_assets)

    def objective(w):
        marginal = cov @ w
        contributions = w * marginal
        total_variance = float(w @ cov @ w)

        if total_variance <= 0 or not np.isfinite(total_variance):
            return 1e6

        risk_shares = contributions / total_variance
        target = np.repeat(1.0 / n_assets, n_assets)
        return float(np.sum((risk_shares - target) ** 2))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n_assets

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = np.clip(result.x, 0.0, max_weight)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return pd.Series(x0, index=history.columns, dtype=float)

    weights = weights / weights.sum()

    if (
        not np.isclose(weights.sum(), 1.0, atol=1e-8)
        or (weights < -1e-10).any()
        or (weights > max_weight + 1e-8).any()
    ):
        return pd.Series(x0, index=history.columns, dtype=float)

    return pd.Series(weights, index=history.columns, dtype=float)

# --- Max Sharpe Helper ---
def _max_sharpe_weights(history: pd.DataFrame, max_weight: float = 0.20) -> pd.Series:
    """Compute long-only maximum-Sharpe weights with a per-asset cap and rf=0."""
    mean_returns = history.mean().values
    cov = history.cov().values
    n_assets = history.shape[1]
    x0 = np.repeat(1.0 / n_assets, n_assets)

    def objective(w):
        portfolio_return = float(w @ mean_returns)
        portfolio_variance = float(w @ cov @ w)

        if portfolio_variance <= 0 or not np.isfinite(portfolio_variance):
            return 1e6

        portfolio_volatility = np.sqrt(portfolio_variance)
        return -portfolio_return / portfolio_volatility

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n_assets

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    weights = np.clip(result.x, 0.0, max_weight)

    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return pd.Series(x0, index=history.columns, dtype=float)

    weights = weights / weights.sum()

    if (
        not np.isclose(weights.sum(), 1.0, atol=1e-8)
        or (weights < -1e-10).any()
        or (weights > max_weight + 1e-8).any()
    ):
        return pd.Series(x0, index=history.columns, dtype=float)

    return pd.Series(weights, index=history.columns, dtype=float)


def oos_backtest(
    returns: pd.DataFrame,
    method: str = "equal_weight",
    lookback: int = 252,
    periods_per_year: int = 252,
) -> dict:
    """Run a monthly walk-forward out-of-sample portfolio backtest.

    The first `lookback` observations are used only for estimation. At each
    month's first available date, portfolio weights are formed using data
    strictly before that date and then held until the next rebalance.
    """
    r = returns.copy().sort_index().astype(float)
    r = r.dropna(how="all")

    valid_methods = {"equal_weight", "min_variance", "risk_parity", "max_sharpe"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}")
    if len(r) <= lookback:
        raise ValueError("Not enough observations for the requested lookback window.")
    if r.shape[1] == 0:
        raise ValueError("Returns must contain at least one asset column.")

    # The first eligible OOS observation comes only after the estimation window.
    live = r.iloc[lookback:].copy()

    # Rebalance on the first available observation of each calendar month.
    month_key = live.index.to_period("M")
    rebalance_dates = live.groupby(month_key).apply(lambda x: x.index[0]).tolist()

    portfolio_returns = pd.Series(index=live.index, dtype=float, name="portfolio_return")
    weight_rows = []

    for i, rebalance_date in enumerate(rebalance_dates):
        history = r.loc[r.index < rebalance_date].tail(lookback)
        available_assets = history.columns[history.notna().all()].tolist()

        if not available_assets:
            continue

        weights = pd.Series(0.0, index=r.columns, dtype=float)

        if method == "equal_weight":
            method_weights = pd.Series(
                1.0 / len(available_assets),
                index=available_assets,
                dtype=float,
            )
        elif method == "min_variance":
            method_weights = _minimum_variance_weights(history[available_assets])
        elif method == "risk_parity":
            method_weights = _risk_parity_weights(history[available_assets])
        else:
            method_weights = _max_sharpe_weights(history[available_assets])

        weights.loc[available_assets] = method_weights
        weights.name = rebalance_date
        weight_rows.append(weights)

        if i + 1 < len(rebalance_dates):
            next_rebalance = rebalance_dates[i + 1]
            holding_dates = live.index[
                (live.index >= rebalance_date) & (live.index < next_rebalance)
            ]
        else:
            holding_dates = live.index[live.index >= rebalance_date]

        holding_returns = r.loc[holding_dates, available_assets]
        portfolio_returns.loc[holding_dates] = holding_returns.mul(
            weights.loc[available_assets], axis=1
        ).sum(axis=1, min_count=len(available_assets))

    portfolio_returns = portfolio_returns.dropna()
    weights_over_time = pd.DataFrame(weight_rows)
    weights_over_time.index.name = "rebalance_date"
    growth = (1.0 + portfolio_returns).cumprod().rename("growth_of_1")
    metrics = performance_metrics(portfolio_returns, periods_per_year=periods_per_year)

    return {
        "daily_returns": portfolio_returns,
        "weights": weights_over_time,
        "growth": growth,
        "metrics": metrics,
    }


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Calculate annualised return, volatility, Sharpe ratio, and max drawdown."""
    r = pd.Series(daily_returns, dtype=float).dropna()

    if r.empty:
        return {
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    growth = (1.0 + r).cumprod()
    annualised_return = growth.iloc[-1] ** (periods_per_year / len(r)) - 1.0
    daily_volatility = r.std(ddof=1)
    annualised_volatility = daily_volatility * np.sqrt(periods_per_year)
    sharpe = (
        r.mean() / daily_volatility * np.sqrt(periods_per_year)
        if daily_volatility > 0
        else np.nan
    )
    drawdown = growth / growth.cummax() - 1.0
    max_drawdown = drawdown.min()

    return {
        "annualised_return": annualised_return,
        "annualised_volatility": annualised_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
