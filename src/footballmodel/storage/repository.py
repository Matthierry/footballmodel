from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


class DuckRepository:
    def __init__(self, db_path: str = "data/footballmodel.duckdb"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(db_path)

    def close(self) -> None:
        self.con.close()

    def write_df(self, table: str, df: pl.DataFrame) -> None:
        self.con.register("tmp_df", df.to_arrow())
        self.con.execute(f"create or replace table {table} as select * from tmp_df")

    def append_df(self, table: str, df: pl.DataFrame) -> None:
        self.con.register("tmp_df", df.to_arrow())
        self.con.execute(f"create table if not exists {table} as select * from tmp_df where 1=0")
        self.con.execute(f"insert into {table} select * from tmp_df")

    def read_df(self, query: str) -> pl.DataFrame:
        return pl.from_arrow(self.con.execute(query).arrow())

    def upsert_benchmark_snapshots(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return
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
