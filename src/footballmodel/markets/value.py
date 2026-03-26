from __future__ import annotations

import polars as pl


def credibility_score(model_probability: float | None, edge: float | None) -> float:
    """Score confidence in a value assessment.

    Missing probability or edge implies we cannot credibly assess value,
    so we return 0.0 rather than raising.
    """
    if model_probability is None or edge is None:
        return 0.0

    prob_anchor = min(1.0, model_probability / 0.5)
    edge_penalty = max(0.0, 1 - abs(edge) * 3)
    return float(max(0.0, min(1.0, 0.7 * prob_anchor + 0.3 * edge_penalty)))


def attach_value_flags(
    market_df: pl.DataFrame,
    edge_threshold: float,
    credibility_threshold: float,
) -> pl.DataFrame:
    benchmark_available = pl.col("current_price").is_not_null() & pl.col("model_fair_odds").is_not_null()

    df = market_df.with_columns(
        benchmark_available.alias("benchmark_available"),
        pl.when(benchmark_available)
        .then(pl.col("current_price") - pl.col("model_fair_odds"))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("edge"),
    ).with_columns(
        pl.struct(["model_probability", "edge"]).map_elements(
            lambda row: credibility_score(row["model_probability"], row["edge"]), return_dtype=pl.Float64
        ).alias("credibility_score")
    ).with_columns(
        pl.when(~pl.col("benchmark_available"))
        .then(pl.lit("missing_benchmark"))
        .when(pl.col("edge").is_null())
        .then(pl.lit("missing_edge"))
        .otherwise(pl.lit("assessed"))
        .alias("value_status"),
        (
            pl.col("benchmark_available")
            & (pl.col("edge") >= edge_threshold)
            & (pl.col("credibility_score") >= credibility_threshold)
        ).alias("value_flag"),
        pl.when(pl.col("benchmark_available") & pl.col("edge").is_not_null())
        .then(pl.col("edge") * pl.col("credibility_score") * pl.col("model_probability").sqrt())
        .otherwise(pl.lit(0.0))
        .alias("growth_score"),
    )
    return df
