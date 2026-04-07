from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from footballmodel.config.runtime_env import resolve_duckdb_path

logger = logging.getLogger(__name__)


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
        self._ensure_table_for_dataframe_writes(table)
        self._migrate_table_to_expected_schema(table)
        self._insert_aligned_from_tmp_df(table=table, source_columns=df.columns)

    def read_df(self, query: str) -> pl.DataFrame:
        # NOTE:
        # DuckDB's `.arrow()` can yield a RecordBatchReader with zero batches for
        # empty results. `polars.from_arrow(...)` then raises:
        # "Must pass schema, or at least one RecordBatch".
        # Using DuckDB's native `.pl()` conversion preserves schema on empty reads.
        return self.con.execute(query).pl()

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
        self._ensure_table_for_dataframe_writes("benchmark_snapshots")
        self._migrate_table_to_expected_schema("benchmark_snapshots")
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
        self._insert_aligned_from_tmp_df(table="benchmark_snapshots", source_columns=df.columns)

    @staticmethod
    def _validate_dataframe(table: str, df: pl.DataFrame, operation: str) -> None:
        if df.width == 0:
            raise ValueError(
                f"Cannot {operation} dataframe with zero columns into '{table}'. "
                "Provide an explicit schema or skip persistence for this table."
            )

    def _table_columns(self, table: str) -> list[str]:
        return [
            row[1]
            for row in self.con.execute(f"pragma table_info('{table}')").fetchall()
        ]

    def _ensure_table_for_dataframe_writes(self, table: str) -> None:
        if self.has_table(table):
            return
        schema = OPTIONAL_TABLE_SCHEMAS.get(table)
        if schema is None:
            self.con.execute(f"create table if not exists {self._quote_identifier(table)} as select * from tmp_df where 1=0")
            return
        self.con.register("tmp_empty_df", pl.DataFrame(schema=schema).to_arrow())
        self.con.execute(f"create table if not exists {self._quote_identifier(table)} as select * from tmp_empty_df where 1=0")

    def _migrate_table_to_expected_schema(self, table: str) -> None:
        expected_schema = OPTIONAL_TABLE_SCHEMAS.get(table)
        if expected_schema is None or not self.has_table(table):
            return
        existing_columns = set(self._table_columns(table))
        for column, dtype in expected_schema.items():
            if column in existing_columns:
                continue
            duckdb_type = self._duckdb_type_from_polars(dtype)
            self.con.execute(
                f"alter table {self._quote_identifier(table)} add column {self._quote_identifier(column)} {duckdb_type}"
            )
            logger.info(
                "Migrated DuckDB table by adding missing column.",
                extra={"table": table, "column": column, "duckdb_type": duckdb_type},
            )

    def _insert_aligned_from_tmp_df(self, *, table: str, source_columns: list[str]) -> None:
        target_columns = self._table_columns(table)
        source_column_set = set(source_columns)
        missing_source_columns = [column for column in target_columns if column not in source_column_set]
        extra_source_columns = sorted(source_column_set - set(target_columns))
        if missing_source_columns or extra_source_columns:
            logger.info(
                "Aligning dataframe and table schemas for DuckDB write.",
                extra={
                    "table": table,
                    "missing_source_columns": missing_source_columns,
                    "extra_source_columns": extra_source_columns,
                },
            )

        target_column_sql = ", ".join(self._quote_identifier(column) for column in target_columns)
        select_sql = ", ".join(
            [
                self._quote_identifier(column)
                if column in source_column_set
                else f"null as {self._quote_identifier(column)}"
                for column in target_columns
            ]
        )
        self.con.execute(
            f"insert into {self._quote_identifier(table)} ({target_column_sql}) select {select_sql} from tmp_df"
        )

    @staticmethod
    def _duckdb_type_from_polars(dtype: pl.DataType) -> str:
        if dtype == pl.Utf8:
            return "VARCHAR"
        if dtype == pl.Float64:
            return "DOUBLE"
        if dtype == pl.Int64:
            return "BIGINT"
        if dtype == pl.Boolean:
            return "BOOLEAN"
        if dtype == pl.Date:
            return "DATE"
        raise ValueError(f"Unsupported optional table dtype for DuckDB migration: {dtype}")

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return f'"{identifier.replace(chr(34), chr(34) * 2)}"'
