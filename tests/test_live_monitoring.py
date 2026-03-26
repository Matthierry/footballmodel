from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.config.settings import load_app_config
from footballmodel.live.monitoring import (
    build_email_alert_events,
    build_live_review_rows,
    build_live_run_summary,
    detect_drift_alerts,
)


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


def test_live_run_summary_rolls_up_audit_metrics():
    run_predictions = pl.DataFrame(
        {
            "live_run_id": ["live_1"],
            "run_timestamp_utc": ["2026-01-01T10:00:00"],
            "config_name": ["champion_v1"],
            "config_version": ["2026.03.1"],
            "fixture_id": ["f1"],
        }
    )
    run_market_predictions = pl.DataFrame({"fixture_id": ["f1", "f1"], "market": ["1X2", "BTTS"]})
    run_review = pl.DataFrame(
        {
            "settlement_status": ["settled", "pending", "settled"],
            "value_flag": [True, True, False],
            "result_status": ["won", "pending", "lost"],
            "clv": [0.02, None, -0.01],
            "later_benchmark_price": [1.95, None, 3.1],
        }
    )

    summary = build_live_run_summary(run_predictions, run_market_predictions, run_review)
    row = summary.row(0, named=True)

    assert row["live_run_id"] == "live_1"
    assert row["fixtures_scored"] == 1
    assert row["market_predictions"] == 2
    assert row["review_rows"] == 3
    assert row["settled_rows"] == 2
    assert row["pending_rows"] == 1
    assert row["settled_value_rows"] == 1
    assert row["value_hit_rate"] == 1.0
    assert round(row["benchmark_coverage_rate"], 4) == round(2 / 3, 4)


def test_detect_drift_alerts_flags_core_deterioration():
    rows = []
    for idx in range(40):
        rows.append(
            {
                "run_timestamp_utc": f"2026-02-{idx % 20 + 1:02d}T00:00:00",
                "prediction_snapshot_timestamp_utc": f"2026-02-{idx % 20 + 1:02d}T00:00:00",
                "config_name": "champion_v1",
                "config_version": "2026.03.1",
                "settlement_status": "settled",
                "clv": 0.02,
                "value_flag": True,
                "result_status": "won",
                "later_benchmark_price": 2.1,
                "target": 1,
                "calibrated_probability": 0.8,
                "market": "1X2",
                "league": "ENG1",
            }
        )
    for idx in range(40):
        rows.append(
            {
                "run_timestamp_utc": f"2026-03-{20 + (idx % 7):02d}T00:00:00",
                "prediction_snapshot_timestamp_utc": f"2026-03-{20 + (idx % 7):02d}T00:00:00",
                "config_name": "champion_v1",
                "config_version": "2026.03.1",
                "settlement_status": "settled",
                "clv": -0.02,
                "value_flag": True,
                "result_status": "lost",
                "later_benchmark_price": None if idx < 20 else 2.0,
                "target": 0,
                "calibrated_probability": 0.9,
                "market": "1X2",
                "league": "ENG1",
            }
        )
    review = pl.DataFrame(rows)
    run_summaries = pl.DataFrame(
        {
            "run_timestamp_utc": ["2026-03-20T00:00:00"],
            "fixtures_scored": [10],
            "market_predictions": [30],
            "review_rows": [30],
            "pending_rows": [1],
        }
    )
    snapshots = pl.DataFrame(
        {
            "snapshot_timestamp_utc": ["2026-03-24T00:00:00"],
        }
    )
    thresholds = {
        "windows_days": [7],
        "baseline_days": 60,
        "min_settled_rows": 10,
        "min_value_rows": 10,
        "min_calibration_rows": 10,
        "min_concentration_rows": 10,
        "clv_drop_abs": 0.01,
        "value_hit_rate_drop_abs": 0.1,
        "benchmark_coverage_drop_abs": 0.1,
        "calibration_error_increase_abs": 0.05,
        "value_volume_ratio_low": 0.5,
        "value_volume_ratio_high": 1.8,
        "concentration_negative_clv_share": 0.6,
        "stale_snapshot_hours": 12,
        "stale_run_hours": 12,
        "pending_rows_ratio_high": 0.9,
    }
    alerts = detect_drift_alerts(
        review_history=review,
        run_summaries_history=run_summaries,
        benchmark_snapshots=snapshots,
        config_name="champion_v1",
        config_version="2026.03.1",
        run_timestamp_utc="2026-03-26T00:00:00+00:00",
        thresholds=thresholds,
    )
    alert_types = set(alerts["alert_type"].to_list())
    assert "clv_deterioration" in alert_types
    assert "value_hit_rate_deterioration" in alert_types
    assert "benchmark_coverage_deterioration" in alert_types
    assert "calibration_deterioration" in alert_types
    assert "source_freshness_stale" in alert_types


def test_build_email_alert_events_only_emits_severe_rows():
    alerts = pl.DataFrame(
        {
            "alert_id": ["a1", "a2"],
            "alert_timestamp_utc": ["2026-03-26T00:00:00+00:00", "2026-03-26T00:00:00+00:00"],
            "alert_type": ["x", "y"],
            "severity": ["warning", "critical"],
            "config_name": ["champion_v1", "champion_v1"],
            "config_version": ["2026.03.1", "2026.03.1"],
        }
    )
    events = build_email_alert_events(alerts, enabled=True)
    assert events.height == 1
    assert events.row(0, named=True)["alert_id"] == "a2"
