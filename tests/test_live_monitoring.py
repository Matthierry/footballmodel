from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.config.settings import load_app_config
from footballmodel.live.monitoring import build_live_review_rows


def test_named_default_live_config_is_resolvable():
    cfg = load_app_config("config/runtime.yaml")
    config_name, live_cfg = cfg.resolve_live_config()

    assert config_name == "champion_v1"
    assert live_cfg.version == "2026.03.1"
    assert live_cfg.runtime.value_edge_threshold > 0
    assert len(live_cfg.leagues) >= 1


def test_live_review_rows_include_clv_and_settlement_status():
    predictions = pl.DataFrame(
        {
            "live_run_id": ["live_1", "live_1"],
            "run_timestamp_utc": ["2026-01-01T10:00:00", "2026-01-01T10:00:00"],
            "config_name": ["champion_v1", "champion_v1"],
            "config_version": ["2026.03.1", "2026.03.1"],
            "fixture_id": ["f1", "f2"],
            "prediction_timestamp_utc": ["2026-01-01T10:00:00", "2026-01-01T10:00:00"],
            "market": ["1X2", "1X2"],
            "outcome": ["home", "home"],
            "line": [None, None],
            "raw_probability": [0.5, 0.4],
            "calibrated_probability": [0.5, 0.4],
            "current_price": [2.2, 2.5],
            "benchmark_source": ["exchange", "exchange"],
            "value_flag": [True, False],
            "value_status": ["assessed", "assessed"],
        }
    )
    snapshots = pl.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "market": ["1X2", "1X2"],
            "outcome": ["home", "home"],
            "line": [None, None],
            "benchmark_price": [2.0, None],
            "benchmark_source": ["exchange", "exchange"],
            "snapshot_type": ["closing", "closing_surrogate"],
            "snapshot_timestamp_utc": ["2026-01-01T19:00:00", "2026-01-01T19:00:00"],
        }
    )
    matches = pl.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "match_date": [date(2026, 1, 1), date(2026, 1, 2)],
            "league": ["ENG1", "ENG1"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_goals": [2, None],
            "away_goals": [1, None],
        }
    )

    review = build_live_review_rows(predictions, snapshots, matches)
    settled = review.filter(pl.col("fixture_id") == "f1").row(0, named=True)
    pending = review.filter(pl.col("fixture_id") == "f2").row(0, named=True)

    assert settled["settlement_status"] == "settled"
    assert settled["later_snapshot_type"] == "closing"
    assert settled["clv"] is not None
    assert pending["settlement_status"] == "pending"
    assert pending["later_snapshot_type"] is None
    assert pending["clv"] is None
