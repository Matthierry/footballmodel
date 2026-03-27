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
