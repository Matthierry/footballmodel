from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.backtest.walkforward import BacktestRequest, run_backtest
from footballmodel.config.settings import load_app_config
from footballmodel.markets.benchmark_snapshots import (
    SNAPSHOT_TYPE_CLOSING,
    SNAPSHOT_TYPE_PREDICTION_TIME,
    benchmark_snapshot_rows_from_fixture,
    choose_later_snapshot,
)
from footballmodel.storage.repository import DuckRepository


def test_benchmark_snapshot_upsert_and_missing_source(tmp_path):
    repo = DuckRepository(str(tmp_path / "snap.duckdb"))
    fixture = {"fixture_id": "f1", "match_date": date(2026, 1, 1), "avg_home_odds": 2.2}
    prediction = benchmark_snapshot_rows_from_fixture(fixture, SNAPSHOT_TYPE_PREDICTION_TIME, "2026-01-01T00:00:00")
    assert prediction.filter(pl.col("benchmark_source") == "unavailable").height > 0

    repo.upsert_benchmark_snapshots(prediction)
    updated = prediction.with_columns(pl.lit("2026-01-01T01:00:00").alias("snapshot_timestamp_utc"))
    repo.upsert_benchmark_snapshots(updated)
    stored = repo.read_df("select * from benchmark_snapshots where fixture_id = 'f1'")
    repo.close()
    assert stored.height == prediction.height
    assert stored["snapshot_timestamp_utc"].n_unique() == 1


def test_choose_later_snapshot_requires_real_later_snapshot():
    rows = pl.DataFrame(
        {
            "snapshot_type": [SNAPSHOT_TYPE_PREDICTION_TIME, SNAPSHOT_TYPE_CLOSING],
            "benchmark_price": [2.1, None],
            "snapshot_timestamp_utc": ["2026-01-01T00:00:00", "2026-01-01T00:10:00"],
            "benchmark_source": ["exchange", "unavailable"],
        }
    )
    assert choose_later_snapshot(rows) is None


def test_backtest_clv_uses_snapshot_structure_without_false_signal(sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    elo_history = pl.DataFrame({"elo_date": [date(2024, 1, 1)], "team": ["A"], "country": ["ENG"], "elo": [1600.0]})
    req = BacktestRequest(start_date=date(2024, 8, 1), end_date=date(2024, 8, 20), leagues=["ENG1"], calibrate_probabilities=False)
    _run_id, predictions, _metrics = run_backtest(sample_matches, elo_history, cfg, req)

    assert "prediction_snapshot_type" in predictions.columns
    assert "later_snapshot_type" in predictions.columns
    assert predictions.filter(pl.col("close_price").is_null()).filter(pl.col("clv").is_not_null()).is_empty()
