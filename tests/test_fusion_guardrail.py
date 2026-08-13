"""Unit tests for the PDAIS investability guardrail (apply_pdais_guardrail).

Run with the repo interpreter:  python -m pytest tests/test_fusion_guardrail.py
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.fusion import apply_pdais_guardrail  # noqa: E402


def _make_scores() -> pd.DataFrame:
    """Four assets, one asset class, three months.

    2021-01 (used by the 2021-02-01 rebalance):
      A low on RS and DR (bottom within-class quartile), mid on LQ  -> flagged
      B low only on LQ                                                 -> not
      C, D high everywhere                                             -> not
    2021-02 (used by the 2021-03-01 rebalance): A low on all three -> flagged.
    2021-03 (must be ignored by every rebalance present).
    """
    def rows(ym, vals):
        out = []
        for ticker, (r, d, l) in vals.items():
            out.append({
                "year_month": ym,
                "ticker": ticker,
                "asset_class": "Equity",
                "RS_norm": r,
                "DR_norm": d,
                "LQ_norm": l,
                "PDAIS_EW": (r + d + l) / 3.0,
            })
        return out

    return pd.DataFrame(
        rows("2020-12", {
            "A": (0.5, 0.5, 0.5), "B": (0.5, 0.5, 0.5),
            "C": (0.5, 0.5, 0.5), "D": (0.5, 0.5, 0.5),
        })
        + rows("2021-01", {
            "A": (0.10, 0.10, 0.50), "B": (0.40, 0.40, 0.10),
            "C": (0.70, 0.70, 0.70), "D": (0.90, 0.90, 0.90),
        })
        + rows("2021-02", {
            "A": (0.00, 0.00, 0.00), "B": (0.50, 0.50, 0.50),
            "C": (0.70, 0.70, 0.70), "D": (0.90, 0.90, 0.90),
        })
        + rows("2021-03", {
            "A": (0.90, 0.90, 0.90), "B": (0.10, 0.10, 0.10),
            "C": (0.50, 0.50, 0.50), "D": (0.50, 0.50, 0.50),
        })
    )


def _make_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0.25, 0.25],
            "B": [0.25, 0.25],
            "C": [0.25, 0.25],
            "D": [0.25, 0.25],
        },
        index=pd.to_datetime(["2021-02-01", "2021-03-01"]),
    )


def _ticker_map() -> pd.Series:
    return pd.Series({"A": "A", "B": "B", "C": "C", "D": "D"})


def test_flag_rule_requires_two_of_three():
    out = apply_pdais_guardrail(
        _make_weights(), _make_scores(), _ticker_map(), max_weight=0.40,
    )
    flags = out["flags"]
    # 2021-02-01 reads 2021-01: A flagged (RS+DR low), B not (only LQ low).
    assert bool(flags.loc["2021-02-01", "A"]) is True
    assert bool(flags.loc["2021-02-01", "B"]) is False
    assert bool(flags.loc["2021-02-01", "C"]) is False
    assert bool(flags.loc["2021-02-01", "D"]) is False
    # 2021-03-01 reads 2021-02: A low on all three -> flagged.
    assert bool(flags.loc["2021-03-01", "A"]) is True


def test_penalty_applied_only_to_flagged():
    weights = _make_weights()
    out = apply_pdais_guardrail(
        weights, _make_scores(), _ticker_map(), penalty=0.30, max_weight=0.40,
    )
    row = out["weights"].loc["2021-02-01"]
    # A penalised by 30% pre-normalisation; B/C/D untouched; then row sums to 1.
    pre = pd.Series(
        {"A": 0.25 * 0.70, "B": 0.25, "C": 0.25, "D": 0.25},
        dtype=float,
    )
    expected = pre / pre.sum()
    assert abs(row["A"] - expected["A"]) < 1e-12
    assert abs(row["B"] - expected["B"]) < 1e-12
    assert abs(row["C"] - expected["C"]) < 1e-12
    assert abs(row["D"] - expected["D"]) < 1e-12


def test_no_direct_boost_for_unflagged():
    weights = _make_weights()
    out = apply_pdais_guardrail(
        weights, _make_scores(), _ticker_map(), penalty=0.30, max_weight=0.40,
    )
    row = out["weights"].loc["2021-02-01"]
    pen = out["penalised_weight"].loc["2021-02-01"]
    # raw = guarded * raw_sum, raw_sum = 1 - penalty * penalised_weight.
    raw_sum = 1.0 - 0.30 * pen
    base_row = weights.loc["2021-02-01"]
    factors = row * raw_sum / base_row
    assert abs(factors["A"] - 0.70) < 1e-12
    assert abs(factors["B"] - 1.0) < 1e-12
    assert abs(factors["C"] - 1.0) < 1e-12
    assert abs(factors["D"] - 1.0) < 1e-12
    assert (factors <= 1.0 + 1e-12).all()


def test_penalised_weight_diagnostic():
    out = apply_pdais_guardrail(
        _make_weights(), _make_scores(), _ticker_map(), max_weight=0.40,
    )
    # 2021-02-01: only A flagged, pre-penalty baseline weight 0.25.
    assert abs(out["penalised_weight"].loc["2021-02-01"] - 0.25) < 1e-12
    # 2021-03-01: only A flagged (2021-02 scores) -> 0.25.
    assert abs(out["penalised_weight"].loc["2021-03-01"] - 0.25) < 1e-12


def test_row_sums_and_cap():
    # Real risk-parity baseline weights are far below the 0.20 cap, so the cap
    # is exercised only as an upper bound; use a non-binding cap here.
    out = apply_pdais_guardrail(
        _make_weights(), _make_scores(), _ticker_map(), max_weight=0.40,
    )
    w = out["weights"]
    assert (w.sum(axis=1) - 1.0).abs().max() < 1e-10
    assert (w.values >= -1e-10).all()
    assert w.values.max() <= 0.40 + 1e-10


def test_lookahead_uses_previous_month_only():
    # 2021-03 scores make A the best name, but no rebalance may see them; the
    # 2021-03-01 rebalance uses 2021-02 scores, so A stays flagged.
    out = apply_pdais_guardrail(
        _make_weights(), _make_scores(), _ticker_map(), max_weight=0.40,
    )
    # 2021-02-01 rebalance uses 2021-01 scores -> A flagged.
    assert bool(out["flags"].loc["2021-02-01", "A"]) is True
    # 2021-03-01 rebalance uses 2021-02 scores -> A flagged.
    assert bool(out["flags"].loc["2021-03-01", "A"]) is True
    # 2021-03 scores (A best) never enter any weight decision here.


def test_crypto_flags_are_within_asset_class():
    # Ranks must be within class: a low crypto name is flagged against crypto
    # peers only, never against equity names.
    rows = []
    for ym in ["2021-01"]:
        for ticker, ac, vals in [
            ("E1", "Equity", (0.5, 0.5, 0.5)),
            ("E2", "Equity", (0.6, 0.6, 0.6)),
            ("C1", "Crypto", (0.10, 0.10, 0.10)),
            ("C2", "Crypto", (0.60, 0.60, 0.60)),
            ("C3", "Crypto", (0.80, 0.80, 0.80)),
            ("C4", "Crypto", (0.90, 0.90, 0.90)),
        ]:
            r, d, l = vals
            rows.append({
                "year_month": ym, "ticker": ticker, "asset_class": ac,
                "RS_norm": r, "DR_norm": d, "LQ_norm": l,
                "PDAIS_EW": (r + d + l) / 3.0,
            })
    scores = pd.DataFrame(rows)
    weights = pd.DataFrame(
        {
            "E1": [0.20], "E2": [0.20],
            "C1": [0.20], "C2": [0.20], "C3": [0.10], "C4": [0.10],
        },
        index=pd.to_datetime(["2021-02-01"]),
    )
    ticker_map = pd.Series(
        {"E1": "E1", "E2": "E2", "C1": "C1", "C2": "C2", "C3": "C3", "C4": "C4"}
    )
    out = apply_pdais_guardrail(
        weights, scores, ticker_map, max_weight=0.40,
    )
    flags = out["flags"].loc["2021-02-01"]
    assert bool(flags["C1"]) is True
    assert bool(flags["C2"]) is False
    assert bool(flags["C3"]) is False
    assert bool(flags["C4"]) is False
    # Equity names are never flagged: E1/E2 are not bottom-quartile in RS/DR/LQ
    # relative to each other at the same time.
    assert bool(flags["E1"]) is False
    assert bool(flags["E2"]) is False


if __name__ == "__main__":
    tests = [
        test_flag_rule_requires_two_of_three,
        test_penalty_applied_only_to_flagged,
        test_no_direct_boost_for_unflagged,
        test_penalised_weight_diagnostic,
        test_row_sums_and_cap,
        test_lookahead_uses_previous_month_only,
        test_crypto_flags_are_within_asset_class,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
