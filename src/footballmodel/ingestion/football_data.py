from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

FOOTBALL_DATA_MAPPING = {
    "Date": "match_date",
    "Div": "league",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "B365H": "avg_home_odds",
    "B365D": "avg_draw_odds",
    "B365A": "avg_away_odds",
    "BFH": "bf_home_odds",
    "BFD": "bf_draw_odds",
    "BFA": "bf_away_odds",
}


def load_football_data_csv(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path, ignore_errors=True)
    keep_cols = [c for c in FOOTBALL_DATA_MAPPING if c in df.columns]
    df = df.select(keep_cols).rename({k: FOOTBALL_DATA_MAPPING[k] for k in keep_cols})
    if "match_date" in df.columns:
        df = df.with_columns(
            pl.col("match_date")
            .map_elements(lambda x: datetime.strptime(str(x), "%d/%m/%Y").date() if x else None, return_dtype=pl.Date)
        )
    df = df.with_row_index("fixture_num").with_columns(
        (pl.col("league") + "_" + pl.col("fixture_num").cast(pl.Utf8)).alias("fixture_id")
    )
    return df
