from __future__ import annotations

from datetime import timedelta

import polars as pl


def decay_weight(days_old: pl.Expr, half_life_days: int) -> pl.Expr:
    return (2.0 ** (-days_old.cast(pl.Float64) / half_life_days)).alias("decay_w")


def build_match_features(matches: pl.DataFrame, lookback_days: int = 180, half_life_days: int = 60) -> pl.DataFrame:
    if matches.is_empty():
        return matches

    max_date = matches.select(pl.max("match_date")).item()
    min_date = max_date - timedelta(days=lookback_days)
    scoped = matches.filter(pl.col("match_date") >= pl.lit(min_date))

    scoped = scoped.with_columns(
        (pl.lit(max_date) - pl.col("match_date")).dt.total_days().alias("days_old")
    ).with_columns(decay_weight(pl.col("days_old"), half_life_days))

    team_attack = (
        scoped.group_by("home_team")
        .agg((pl.col("home_goals") * pl.col("decay_w")).sum() / pl.col("decay_w").sum())
        .rename({"home_team": "team", "home_goals": "atk_strength"})
    )

    team_def = (
        scoped.group_by("home_team")
        .agg((pl.col("away_goals") * pl.col("decay_w")).sum() / pl.col("decay_w").sum())
        .rename({"home_team": "team", "away_goals": "def_concede"})
    )

    return team_attack.join(team_def, on="team", how="outer_coalesce")
