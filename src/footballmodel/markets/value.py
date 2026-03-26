from __future__ import annotations

import polars as pl


def credibility_score(model_probability: float, edge: float) -> float:
    prob_anchor = min(1.0, model_probability / 0.5)
    edge_penalty = max(0.0, 1 - abs(edge) * 3)
    return float(max(0.0, min(1.0, 0.7 * prob_anchor + 0.3 * edge_penalty)))


def attach_value_flags(
    market_df: pl.DataFrame,
    edge_threshold: float,
    credibility_threshold: float,
) -> pl.DataFrame:
    df = market_df.with_columns(
        (pl.col("current_price") - pl.col("model_fair_odds")).alias("edge"),
    ).with_columns(
        pl.struct(["model_probability", "edge"]).map_elements(
            lambda row: credibility_score(row["model_probability"], row["edge"]), return_dtype=pl.Float64
        ).alias("credibility_score")
    ).with_columns(
        ((pl.col("edge") >= edge_threshold) & (pl.col("credibility_score") >= credibility_threshold)).alias("value_flag"),
        (pl.col("edge") * pl.col("credibility_score") * pl.col("model_probability").sqrt()).alias("growth_score"),
    )
    return df
