from __future__ import annotations

import polars as pl
import pytest

from footballmodel.storage.repository import DuckRepository


def test_append_df_rejects_zero_column_dataframe(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo.duckdb"))
    try:
        with pytest.raises(ValueError, match="zero columns"):
            repo.append_df("model_runs", pl.DataFrame([]))
    finally:
        repo.close()


def test_append_df_allows_empty_dataframe_with_schema(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo_empty_schema.duckdb"))
    frame = pl.DataFrame(schema={"live_run_id": pl.Utf8, "run_timestamp_utc": pl.Utf8})
    try:
        repo.append_df("model_runs", frame)
        persisted_columns = [row[0] for row in repo.con.execute("describe model_runs").fetchall()]
        persisted_count = repo.con.execute("select count(*) from model_runs").fetchone()[0]
    finally:
        repo.close()

    assert persisted_count == 0
    assert persisted_columns == [
        "fixture_id",
        "timestamp_utc",
        "home_team",
        "away_team",
        "expected_home_goals",
        "expected_away_goals",
        "live_run_id",
        "run_timestamp_utc",
        "config_name",
        "config_version",
    ]


def test_read_table_or_empty_returns_schema_when_optional_table_absent(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo_optional.duckdb"))
    try:
        snapshots = repo.read_table_or_empty("benchmark_snapshots")
        model_runs = repo.read_table_or_empty("model_runs", order_by="timestamp_utc desc", limit=5)
    finally:
        repo.close()

    assert snapshots.is_empty()
    assert snapshots.columns == [
        "fixture_id",
        "market",
        "outcome",
        "line",
        "benchmark_price",
        "benchmark_source",
        "snapshot_type",
        "snapshot_timestamp_utc",
    ]
    assert model_runs.is_empty()
    assert "fixture_id" in model_runs.columns


def test_read_table_or_empty_returns_schema_when_optional_table_exists_but_empty(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo_optional_existing_empty.duckdb"))
    try:
        repo.ensure_optional_tables(["benchmark_snapshots", "model_runs", "model_market_predictions"])
        snapshots = repo.read_table_or_empty("benchmark_snapshots")
        model_runs = repo.read_table_or_empty("model_runs")
        market_predictions = repo.read_table_or_empty("model_market_predictions")
    finally:
        repo.close()

    assert snapshots.is_empty()
    assert snapshots.columns == [
        "fixture_id",
        "market",
        "outcome",
        "line",
        "benchmark_price",
        "benchmark_source",
        "snapshot_type",
        "snapshot_timestamp_utc",
    ]
    assert model_runs.is_empty()
    assert model_runs.columns == [
        "fixture_id",
        "timestamp_utc",
        "home_team",
        "away_team",
        "expected_home_goals",
        "expected_away_goals",
        "live_run_id",
        "run_timestamp_utc",
        "config_name",
        "config_version",
    ]
    assert market_predictions.is_empty()
    assert market_predictions.columns == [
        "live_run_id",
        "run_timestamp_utc",
        "config_name",
        "config_version",
        "fixture_id",
        "prediction_timestamp_utc",
        "market",
        "outcome",
        "line",
        "raw_probability",
        "calibrated_probability",
        "calibration_method",
        "model_fair_odds",
        "current_price",
        "benchmark_source",
        "benchmark_snapshot_type",
        "benchmark_snapshot_timestamp_utc",
        "value_flag",
        "value_status",
        "edge",
    ]


def test_ensure_optional_tables_bootstraps_expected_tables(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo_bootstrap.duckdb"))
    try:
        repo.ensure_optional_tables(["benchmark_snapshots", "model_runs", "model_market_predictions"])
        created = {
            row[0]
            for row in repo.con.execute(
                """
                select table_name from information_schema.tables
                where table_schema='main' and table_name in ('benchmark_snapshots','model_runs','model_market_predictions')
                """
            ).fetchall()
        }
    finally:
        repo.close()

    assert created == {"benchmark_snapshots", "model_runs", "model_market_predictions"}


def test_append_df_aligns_columns_for_legacy_model_runs_table(tmp_path):
    repo = DuckRepository(str(tmp_path / "repo_legacy_model_runs.duckdb"))
    try:
        repo.con.execute(
            """
            create table model_runs (
                fixture_id varchar,
                timestamp_utc varchar,
                home_team varchar,
                away_team varchar,
                expected_home_goals double,
                expected_away_goals double,
                live_run_id varchar,
                run_timestamp_utc varchar,
                config_name varchar,
                config_version varchar
            )
            """
        )
        repo.append_df(
            "model_runs",
            pl.DataFrame(
                {
                    "fixture_id": ["f_1"],
                    "timestamp_utc": ["2026-04-07T00:00:00+00:00"],
                    "home_team": ["Alpha"],
                    "away_team": ["Beta"],
                    "expected_home_goals": [1.23],
                    "expected_away_goals": [0.87],
                    "live_run_id": ["live_1"],
                    "run_timestamp_utc": ["2026-04-07T00:00:00+00:00"],
                    "config_name": ["default"],
                    "config_version": ["v1"],
                    "score_matrix": [[[0.1, 0.2], [0.3, 0.4]]],
                    "markets": [[{"market": "1X2", "outcome": "H"}]],
                    "correct_score_top5": [[{"score": "1-0", "prob": 0.12}]],
                    "asian_handicap": [[{"line": -0.5, "home_prob": 0.55}]],
                }
            ),
        )
        persisted = repo.read_df("select * from model_runs")
    finally:
        repo.close()

    assert persisted.height == 1
    assert persisted.columns == [
        "fixture_id",
        "timestamp_utc",
        "home_team",
        "away_team",
        "expected_home_goals",
        "expected_away_goals",
        "live_run_id",
        "run_timestamp_utc",
        "config_name",
        "config_version",
    ]
    assert persisted.row(0, named=True)["fixture_id"] == "f_1"
