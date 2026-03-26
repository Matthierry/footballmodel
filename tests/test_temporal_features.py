from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.features.feature_builder import build_match_features
from footballmodel.ingestion.clubelo import elo_as_of


def test_rolling_window_respects_lookback_cutoff():
    matches = pl.DataFrame(
        {
            "match_date": [date(2024, 1, 1), date(2024, 6, 1), date(2024, 8, 15)],
            "home_team": ["A", "A", "A"],
            "away_team": ["B", "C", "D"],
            "home_goals": [10.0, 0.0, 2.0],
            "away_goals": [0.0, 1.0, 1.0],
        }
    )

    features = build_match_features(matches, lookback_days=60, half_life_days=30)
    atk = features.filter(pl.col("team") == "A")["atk_strength"][0]

    assert atk < 2.01
    assert atk > 0.01


def test_recency_weighting_prioritizes_newer_matches():
    matches = pl.DataFrame(
        {
            "match_date": [date(2024, 7, 1), date(2024, 8, 15)],
            "home_team": ["A", "A"],
            "away_team": ["B", "C"],
            "home_goals": [0.0, 4.0],
            "away_goals": [0.0, 0.0],
        }
    )

    weighted = build_match_features(matches, lookback_days=365, half_life_days=10)
    long_half_life = build_match_features(matches, lookback_days=365, half_life_days=10_000)

    weighted_atk = weighted.filter(pl.col("team") == "A")["atk_strength"][0]
    near_unweighted_atk = long_half_life.filter(pl.col("team") == "A")["atk_strength"][0]

    assert weighted_atk > near_unweighted_atk
    assert weighted_atk > 2.0


def test_elo_as_of_selects_latest_historical_value_at_date_cutoff():
    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2024, 1, 1), date(2024, 8, 1), date(2025, 1, 1)],
            "team": ["Promoted FC", "Promoted FC", "Promoted FC"],
            "country": ["ENG", "ENG", "ENG"],
            "elo": [1450.0, 1510.0, 1650.0],
        }
    )

    assert elo_as_of(elo_history, "Promoted FC", "2024-07-31") == 1450.0
    assert elo_as_of(elo_history, "Promoted FC", "2024-08-20") == 1510.0
    assert elo_as_of(elo_history, "Promoted FC", "2025-01-15") == 1650.0
