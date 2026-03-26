from __future__ import annotations

import polars as pl
import pytest

from footballmodel.config.settings import load_app_config
from footballmodel.markets.value import attach_value_flags, credibility_score
from footballmodel.orchestration.pipeline import run_fixture_prediction


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


def test_run_fixture_prediction_maps_benchmark_prices_for_supported_markets(sample_matches):
    cfg = load_app_config("config/runtime.yaml")
    history = sample_matches.filter(pl.col("home_goals").is_not_null())
    fixture = sample_matches.filter(pl.col("home_goals").is_null()).to_dicts()[0]
    elo_history = pl.DataFrame(
        {
            "elo_date": [fixture["match_date"]] * 3,
            "team": ["A", "B", "C"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1700.0, 1680.0, 1650.0],
        }
    )

    prediction = run_fixture_prediction(history, fixture, elo_history, cfg)
    markets = pl.DataFrame(prediction["markets"])

    assert markets.filter(pl.col("market") == "1X2").select(pl.col("benchmark_source").is_not_null().all()).item()
    assert markets.filter(pl.col("market") == "OU25").select(pl.col("benchmark_source").is_not_null().all()).item()
    assert markets.filter(pl.col("market") == "BTTS").select(pl.col("benchmark_source").is_not_null().all()).item()
    assert markets.filter(pl.col("market") == "AH").select(pl.col("benchmark_source").is_not_null().all()).item()


def test_exchange_fallback_to_market_average_for_ou25_btts_and_ah(sample_matches):
    cfg = load_app_config("config/runtime.yaml")
    history = sample_matches.filter(pl.col("home_goals").is_not_null())
    fixture = sample_matches.filter(pl.col("home_goals").is_null()).to_dicts()[0]

    fixture["bf_over_2_5_odds"] = None
    fixture["bf_under_2_5_odds"] = None
    fixture["bf_btts_yes_odds"] = None
    fixture["bf_btts_no_odds"] = None
    fixture["bf_ah_home_odds"] = None
    fixture["bf_ah_away_odds"] = None

    elo_history = pl.DataFrame(
        {
            "elo_date": [fixture["match_date"]] * 3,
            "team": ["A", "B", "C"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1700.0, 1680.0, 1650.0],
        }
    )

    markets = pl.DataFrame(run_fixture_prediction(history, fixture, elo_history, cfg)["markets"])

    assert markets.filter(pl.col("market") == "OU25").select((pl.col("benchmark_source") == "market_average").all()).item()
    assert markets.filter(pl.col("market") == "BTTS").select((pl.col("benchmark_source") == "market_average").all()).item()
    assert markets.filter(pl.col("market") == "AH").select((pl.col("benchmark_source") == "market_average").all()).item()


def test_non_1x2_rows_without_true_benchmark_stay_safe_missing_benchmark(sample_matches):
    cfg = load_app_config("config/runtime.yaml")
    history = sample_matches.filter(pl.col("home_goals").is_not_null())
    fixture = sample_matches.filter(pl.col("home_goals").is_null()).to_dicts()[0]

    for key in (
        "avg_over_2_5_odds",
        "avg_under_2_5_odds",
        "bf_over_2_5_odds",
        "bf_under_2_5_odds",
        "avg_btts_yes_odds",
        "avg_btts_no_odds",
        "bf_btts_yes_odds",
        "bf_btts_no_odds",
        "avg_ah_home_odds",
        "avg_ah_away_odds",
        "bf_ah_home_odds",
        "bf_ah_away_odds",
    ):
        fixture[key] = None

    elo_history = pl.DataFrame(
        {
            "elo_date": [fixture["match_date"]] * 3,
            "team": ["A", "B", "C"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1700.0, 1680.0, 1650.0],
        }
    )

    markets = pl.DataFrame(run_fixture_prediction(history, fixture, elo_history, cfg)["markets"])
    non_1x2 = markets.filter(pl.col("market") != "1X2")

    assert non_1x2.select((pl.col("value_status") == "missing_benchmark").all()).item()
    assert non_1x2.select((pl.col("value_flag") == False).all()).item()


def test_non_1x2_markets_can_produce_valid_assessed_edges(sample_matches):
    cfg = load_app_config("config/runtime.yaml")
    history = sample_matches.filter(pl.col("home_goals").is_not_null())
    fixture = sample_matches.filter(pl.col("home_goals").is_null()).to_dicts()[0]

    fixture["bf_over_2_5_odds"] = 5.0
    fixture["bf_under_2_5_odds"] = 5.0
    fixture["bf_btts_yes_odds"] = 5.0
    fixture["bf_btts_no_odds"] = 5.0
    fixture["bf_ah_home_odds"] = 5.0
    fixture["bf_ah_away_odds"] = 5.0

    elo_history = pl.DataFrame(
        {
            "elo_date": [fixture["match_date"]] * 3,
            "team": ["A", "B", "C"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1700.0, 1680.0, 1650.0],
        }
    )

    markets = pl.DataFrame(run_fixture_prediction(history, fixture, elo_history, cfg)["markets"])
    assessed = markets.filter((pl.col("market") != "1X2") & (pl.col("value_status") == "assessed"))

    assert assessed.height >= 6
    assert assessed.select(pl.col("edge").is_not_null().all()).item()
    assert assessed.select(pl.col("benchmark_source").is_not_null().all()).item()
