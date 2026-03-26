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
