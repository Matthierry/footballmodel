from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(slots=True)
class BacktestRequest:
    start_date: date
    end_date: date
    leagues: list[str]
    stake: float = 1.0


def run_walkforward(matches: pl.DataFrame, predict_fn, req: BacktestRequest) -> pl.DataFrame:
    scoped = matches.filter(
        (pl.col("match_date") >= pl.lit(req.start_date))
        & (pl.col("match_date") <= pl.lit(req.end_date))
        & (pl.col("league").is_in(req.leagues))
    ).sort("match_date")

    rows: list[dict[str, object]] = []
    for fixture in scoped.iter_rows(named=True):
        hist = matches.filter(pl.col("match_date") < pl.lit(fixture["match_date"]))
        pred = predict_fn(hist, fixture)
        pred["fixture_id"] = fixture["fixture_id"]
        pred["match_date"] = fixture["match_date"]
        rows.append(pred)
    return pl.DataFrame(rows)
