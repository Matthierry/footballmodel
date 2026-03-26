from __future__ import annotations

import polars as pl
import pytest

from footballmodel.markets.value import attach_value_flags, credibility_score


def test_credibility_score_returns_zero_for_missing_edge():
    assert credibility_score(0.4, None) == 0.0


def test_attach_value_flags_marks_missing_edge_rows_non_value():
    market_df = pl.DataFrame(
        {
            "fixture_id": ["f1", "f1"],
            "market": ["1X2", "BTTS"],
            "outcome": ["home", "btts_yes"],
            "model_probability": [0.5, 0.55],
            "model_fair_odds": [2.0, 1.82],
            "current_price": [2.2, None],
        }
    )

    out = attach_value_flags(market_df, edge_threshold=0.025, credibility_threshold=0.4)

    home = out.filter(pl.col("outcome") == "home").to_dicts()[0]
    btts_yes = out.filter(pl.col("outcome") == "btts_yes").to_dicts()[0]

    assert home["benchmark_available"] is True
    assert home["edge"] == pytest.approx(0.2)
    assert home["value_status"] == "assessed"

    assert btts_yes["benchmark_available"] is False
    assert btts_yes["edge"] is None
    assert btts_yes["credibility_score"] == 0.0
    assert btts_yes["value_status"] == "missing_benchmark"
    assert btts_yes["value_flag"] is False
    assert btts_yes["growth_score"] == 0.0


def test_attach_value_flags_valid_edge_still_behaves_normally():
    market_df = pl.DataFrame(
        {
            "fixture_id": ["f2"],
            "market": ["1X2"],
            "outcome": ["home"],
            "model_probability": [0.5],
            "model_fair_odds": [2.0],
            "current_price": [2.3],
        }
    )

    out = attach_value_flags(market_df, edge_threshold=0.025, credibility_threshold=0.4)
    row = out.to_dicts()[0]

    assert row["benchmark_available"] is True
    assert row["value_status"] == "assessed"
    assert row["value_flag"] is True
    assert row["credibility_score"] > 0.4
    assert row["growth_score"] > 0.0
