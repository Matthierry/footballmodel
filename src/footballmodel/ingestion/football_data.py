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
    "AHh": "ah_line",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "avg_over_2_5_odds": ("B365>2.5", "B365O2.5"),
    "avg_under_2_5_odds": ("B365<2.5", "B365U2.5"),
    "bf_over_2_5_odds": ("P>2.5", "PO2.5"),
    "bf_under_2_5_odds": ("P<2.5", "PU2.5"),
    "avg_btts_yes_odds": ("B365BTTS", "B365BTS"),
    "avg_btts_no_odds": ("B365NBTS", "B365BTTS_No"),
    "bf_btts_yes_odds": ("PBBTS", "PBTTS"),
    "bf_btts_no_odds": ("PBNBTS", "PBTTS_No"),
    "avg_ah_home_odds": ("B365AHH",),
    "avg_ah_away_odds": ("B365AHA",),
    "bf_ah_home_odds": ("PAHH",),
    "bf_ah_away_odds": ("PAHA",),
}


def load_football_data_csv(path: str | Path) -> pl.DataFrame:
    raw_df = pl.read_csv(path, ignore_errors=True)
    keep_cols = [c for c in FOOTBALL_DATA_MAPPING if c in raw_df.columns]
    df = raw_df.select(keep_cols).rename({k: FOOTBALL_DATA_MAPPING[k] for k in keep_cols})

    for target, aliases in COLUMN_ALIASES.items():
        present_aliases = [alias for alias in aliases if alias in raw_df.columns]
        if present_aliases:
            alias_series = raw_df.select(
                pl.coalesce([pl.col(alias).cast(pl.Float64, strict=False) for alias in present_aliases]).alias(target)
            ).get_column(target)
            df = df.with_columns(alias_series)

    if "match_date" in df.columns:
        df = df.with_columns(
            pl.col("match_date")
            .map_elements(lambda x: datetime.strptime(str(x), "%d/%m/%Y").date() if x else None, return_dtype=pl.Date)
        )
    df = df.with_row_index("fixture_num").with_columns(
        (pl.col("league") + "_" + pl.col("fixture_num").cast(pl.Utf8)).alias("fixture_id")
    )
    return df
