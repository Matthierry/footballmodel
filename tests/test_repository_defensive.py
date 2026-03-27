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
    assert persisted_columns == ["live_run_id", "run_timestamp_utc"]


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
