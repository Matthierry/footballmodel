from __future__ import annotations

from datetime import date

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("numpy")
pytest.importorskip("scipy")

from footballmodel.backtest.walkforward import BacktestRequest, run_walkforward
from footballmodel.config.settings import load_app_config
from footballmodel.ingestion.clubelo import elo_as_of
from footballmodel.orchestration.pipeline import run_fixture_prediction


def test_walkforward_uses_only_prior_rows(sample_matches: pl.DataFrame):
    seen_hist_ids: list[list[str]] = []

    def predict_fn(hist: pl.DataFrame, fixture: dict):
        seen_hist_ids.append(hist["fixture_id"].to_list())
        return {"pred": 1.0}

    req = BacktestRequest(start_date=date(2024, 8, 1), end_date=date(2024, 8, 30), leagues=["ENG1"])
    out = run_walkforward(sample_matches, predict_fn, req)

    assert out.height == 4
    assert seen_hist_ids == [[], ["f1"], ["f1", "f2"], ["f1", "f2", "f3"]]


def test_leakage_tripwire_future_result_would_change_answer_if_leaked():
    matches = pl.DataFrame(
        {
            "fixture_id": ["p1", "p2"],
            "league": ["ENG1", "ENG1"],
            "match_date": [date(2024, 1, 1), date(2024, 1, 8)],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_goals": [0, 9],
            "away_goals": [0, 0],
        }
    )

    def predict_fn(hist: pl.DataFrame, fixture: dict):
        return {"mean_home": float(hist.filter(pl.col("home_team") == fixture["home_team"]).select(pl.mean("home_goals")).item() or 0.0)}

    req = BacktestRequest(start_date=date(2024, 1, 1), end_date=date(2024, 1, 8), leagues=["ENG1"])
    out = run_walkforward(matches, predict_fn, req).sort("fixture_id")

    assert out.filter(pl.col("fixture_id") == "p1")["mean_home"][0] == 0.0
    assert out.filter(pl.col("fixture_id") == "p2")["mean_home"][0] == 0.0


def test_historical_elo_lookup_is_date_correct_and_not_current():
    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2024, 1, 1), date(2024, 6, 1), date(2025, 1, 1)],
            "team": ["A", "A", "A"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1500.0, 1600.0, 1900.0],
        }
    )

    assert elo_as_of(elo_history, "A", "2024-03-01") == 1500.0
    assert elo_as_of(elo_history, "A", "2024-12-30") == 1600.0
    assert elo_as_of(elo_history, "A", "2025-01-02") == 1900.0


def test_run_fixture_prediction_uses_date_appropriate_elo_for_promoted_team(sample_matches: pl.DataFrame):
    history = sample_matches.filter(pl.col("home_goals").is_not_null())
    fixture = sample_matches.filter(pl.col("fixture_id") == "f4").to_dicts()[0]
    fixture["home_team"] = "Promoted FC"

    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2023, 8, 1), date(2024, 8, 1), date(2025, 8, 1)],
            "team": ["Promoted FC", "A", "Promoted FC"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1450.0, 1650.0, 1800.0],
        }
    )

    cfg = load_app_config("config/runtime.yaml")
    result = run_fixture_prediction(history, fixture, elo_history, cfg)

    assert result["expected_home_goals"] < 1.7
    assert result["expected_away_goals"] > 0.5
