from __future__ import annotations

from pathlib import Path

import polars as pl


def load_clubelo_csv(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    expected = {
        "Date": "elo_date",
        "Club": "team",
        "Country": "country",
        "Elo": "elo",
    }
    keep = [k for k in expected if k in df.columns]
    return df.select(keep).rename({k: expected[k] for k in keep}).with_columns(
        pl.col("elo_date").str.strptime(pl.Date, strict=False)
    )


def elo_as_of(elo_history: pl.DataFrame, team: str, dt: str) -> float:
    rows = (
        elo_history.filter((pl.col("team") == team) & (pl.col("elo_date") <= pl.lit(dt).str.strptime(pl.Date)))
        .sort("elo_date")
        .tail(1)
    )
    if rows.is_empty():
        return 1500.0
    return float(rows["elo"][0])
