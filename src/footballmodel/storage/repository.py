from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from footballmodel.config.runtime_env import resolve_duckdb_path


OPTIONAL_TABLE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "benchmark_snapshots": {
        "fixture_id": pl.Utf8,
        "market": pl.Utf8,
        "outcome": pl.Utf8,
        "line": pl.Float64,
        "benchmark_price": pl.Float64,
        "benchmark_source": pl.Utf8,
        "snapshot_type": pl.Utf8,
        "snapshot_timestamp_utc": pl.Utf8,
    },
    "model_runs": {
        "fixture_id": pl.Utf8,
        "timestamp_utc": pl.Utf8,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "expected_home_goals": pl.Float64,
        "expected_away_goals": pl.Float64,
        "live_run_id": pl.Utf8,
        "run_timestamp_utc": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
    },
    "model_market_predictions": {
        "live_run_id": pl.Utf8,
        "run_timestamp_utc": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "fixture_id": pl.Utf8,
        "prediction_timestamp_utc": pl.Utf8,
        "market": pl.Utf8,
        "outcome": pl.Utf8,
        "line": pl.Float64,
        "raw_probability": pl.Float64,
        "calibrated_probability": pl.Float64,
        "calibration_method": pl.Utf8,
        "model_fair_odds": pl.Float64,
        "current_price": pl.Float64,
        "benchmark_source": pl.Utf8,
        "benchmark_snapshot_type": pl.Utf8,
        "benchmark_snapshot_timestamp_utc": pl.Utf8,
        "value_flag": pl.Boolean,
        "value_status": pl.Utf8,
        "edge": pl.Float64,
    },
    "live_prediction_history": {
        "live_run_id": pl.Utf8,
        "run_timestamp_utc": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "fixture_id": pl.Utf8,
        "prediction_timestamp_utc": pl.Utf8,
        "market": pl.Utf8,
        "outcome": pl.Utf8,
        "line": pl.Float64,
        "raw_probability": pl.Float64,
        "calibrated_probability": pl.Float64,
        "calibration_method": pl.Utf8,
        "model_fair_odds": pl.Float64,
        "current_price": pl.Float64,
        "benchmark_source": pl.Utf8,
        "benchmark_snapshot_type": pl.Utf8,
        "benchmark_snapshot_timestamp_utc": pl.Utf8,
        "value_flag": pl.Boolean,
        "value_status": pl.Utf8,
        "edge": pl.Float64,
    },
    "live_review_history": {
        "live_run_id": pl.Utf8,
        "run_timestamp_utc": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "fixture_id": pl.Utf8,
        "match_date": pl.Date,
        "league": pl.Utf8,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "market": pl.Utf8,
        "outcome": pl.Utf8,
        "line": pl.Float64,
        "raw_probability": pl.Float64,
        "calibrated_probability": pl.Float64,
        "value_flag": pl.Boolean,
        "value_status": pl.Utf8,
        "prediction_benchmark_price": pl.Float64,
        "prediction_benchmark_source": pl.Utf8,
        "prediction_snapshot_type": pl.Utf8,
        "prediction_snapshot_timestamp_utc": pl.Utf8,
        "later_benchmark_price": pl.Float64,
        "later_snapshot_type": pl.Utf8,
        "later_snapshot_source": pl.Utf8,
        "later_snapshot_timestamp_utc": pl.Utf8,
        "clv": pl.Float64,
        "settlement_status": pl.Utf8,
        "result_status": pl.Utf8,
        "target": pl.Int64,
        "reviewed_at_utc": pl.Utf8,
    },
    "live_run_summaries_history": {
        "live_run_id": pl.Utf8,
        "run_timestamp_utc": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "fixtures_scored": pl.Int64,
        "market_predictions": pl.Int64,
        "review_rows": pl.Int64,
        "pending_rows": pl.Int64,
        "settled_rows": pl.Int64,
        "value_rows": pl.Int64,
        "settled_value_rows": pl.Int64,
        "value_hit_rate": pl.Float64,
        "avg_clv": pl.Float64,
        "benchmark_coverage_rate": pl.Float64,
        "summary_created_at_utc": pl.Utf8,
    },
    "live_alert_history": {
        "alert_id": pl.Utf8,
        "alert_timestamp_utc": pl.Utf8,
        "alert_type": pl.Utf8,
        "severity": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "market": pl.Utf8,
        "league": pl.Utf8,
        "metric": pl.Utf8,
        "window_days": pl.Int64,
        "observed_value": pl.Float64,
        "baseline_value": pl.Float64,
        "delta_value": pl.Float64,
        "status": pl.Utf8,
        "details": pl.Utf8,
    },
    "live_open_alerts": {
        "alert_id": pl.Utf8,
        "alert_timestamp_utc": pl.Utf8,
        "alert_type": pl.Utf8,
        "severity": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "market": pl.Utf8,
        "league": pl.Utf8,
        "metric": pl.Utf8,
        "window_days": pl.Int64,
        "observed_value": pl.Float64,
        "baseline_value": pl.Float64,
        "delta_value": pl.Float64,
        "status": pl.Utf8,
        "details": pl.Utf8,
    },
    "live_alert_notifications_history": {
        "alert_id": pl.Utf8,
        "alert_timestamp_utc": pl.Utf8,
        "alert_type": pl.Utf8,
        "severity": pl.Utf8,
        "config_name": pl.Utf8,
        "config_version": pl.Utf8,
        "notification_status": pl.Utf8,
        "channel": pl.Utf8,
    },
    "experiment_sweep_metadata": {
        "sweep_id": pl.Utf8,
        "created_at": pl.Utf8,
        "run_count": pl.Int64,
    },
}


class DuckRepository:
    def __init__(self, db_path: str | Path | None = None):
        resolved = Path(db_path) if db_path is not None else resolve_duckdb_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(resolved))

    def close(self) -> None:
        self.con.close()

    def has_table(self, table: str) -> bool:
        exists = self.con.execute(
            "select 1 from information_schema.tables where table_schema='main' and table_name = ? limit 1",
            [table],
        ).fetchone()
        return exists is not None

    def empty_optional_table(self, table: str) -> pl.DataFrame:
        schema = OPTIONAL_TABLE_SCHEMAS.get(table)
        return pl.DataFrame(schema=schema) if schema else pl.DataFrame([])

    def ensure_optional_tables(self, tables: list[str] | None = None) -> None:
        table_names = tables if tables is not None else list(OPTIONAL_TABLE_SCHEMAS)
        for table in table_names:
            schema = OPTIONAL_TABLE_SCHEMAS.get(table)
            if schema is None or self.has_table(table):
                continue
            self.con.register("tmp_df", pl.DataFrame(schema=schema).to_arrow())
            self.con.execute(f"create table if not exists {table} as select * from tmp_df where 1=0")

    def write_df(self, table: str, df: pl.DataFrame) -> None:
        self._validate_dataframe(table, df, operation="write")
        self.con.register("tmp_df", df.to_arrow())
        self.con.execute(f"create or replace table {table} as select * from tmp_df")

    def append_df(self, table: str, df: pl.DataFrame) -> None:
        self._validate_dataframe(table, df, operation="append")
        self.con.register("tmp_df", df.to_arrow())
        self.con.execute(f"create table if not exists {table} as select * from tmp_df where 1=0")
        self.con.execute(f"insert into {table} select * from tmp_df")

    def read_df(self, query: str) -> pl.DataFrame:
        return pl.from_arrow(self.con.execute(query).arrow())

    def read_table_or_empty(self, table: str, *, order_by: str | None = None, limit: int | None = None) -> pl.DataFrame:
        if not self.has_table(table):
            return self.empty_optional_table(table)
        query = f"select * from {table}"
        if order_by:
            query = f"{query} order by {order_by}"
        if limit is not None:
            query = f"{query} limit {limit}"
        return self.read_df(query)

    def upsert_benchmark_snapshots(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
        self._validate_dataframe("benchmark_snapshots", df, operation="upsert")
        self.con.register("tmp_df", df.to_arrow())
        self.con.execute("create table if not exists benchmark_snapshots as select * from tmp_df where 1=0")
        self.con.execute(
            """
            delete from benchmark_snapshots as tgt
            using tmp_df as src
            where tgt.fixture_id = src.fixture_id
              and tgt.market = src.market
              and tgt.outcome = src.outcome
              and tgt.snapshot_type = src.snapshot_type
              and tgt.line is not distinct from src.line
            """
        )
        self.con.execute("insert into benchmark_snapshots select * from tmp_df")

    @staticmethod
    def _validate_dataframe(table: str, df: pl.DataFrame, operation: str) -> None:
        if df.width == 0:
            raise ValueError(
                f"Cannot {operation} dataframe with zero columns into '{table}'. "
                "Provide an explicit schema or skip persistence for this table."
            )
