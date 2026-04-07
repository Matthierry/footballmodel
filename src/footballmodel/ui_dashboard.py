from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.config.runtime_env import get_app_password
from footballmodel.storage.repository import DuckRepository

MARKET_ORDER = ["1X2", "Over/Under 2.5", "BTTS", "Asian Handicap", "Correct Score"]
MARKET_GROUPS = {
    "1X2": "1X2",
    "Over/Under 2.5": "Over/Under 2.5",
    "BTTS": "BTTS",
    "Asian Handicap": "Asian Handicap",
}


def require_password_gate() -> None:
    """Enforce a shared password gate across all Streamlit pages."""
    if st.session_state.get("authenticated") is True:
        return
    password = st.sidebar.text_input("Password", type="password")
    if password == get_app_password(streamlit_module=st):
        st.session_state["authenticated"] = True
        return
    st.warning("Enter valid password to access dashboard")
    st.stop()


def safe_table(repo: DuckRepository, table: str, *, order_by: str | None = None, limit: int | None = None) -> pl.DataFrame:
    try:
        return repo.read_table_or_empty(table, order_by=order_by, limit=limit)
    except Exception:
        return pl.DataFrame([])


def load_core_data(repo: DuckRepository) -> dict[str, pl.DataFrame]:
    review = safe_table(repo, "live_review_history", order_by="run_timestamp_utc desc")
    runs = safe_table(repo, "live_run_summaries_history", order_by="run_timestamp_utc desc")
    matches = safe_table(repo, "curated_matches")
    snapshots = safe_table(repo, "benchmark_snapshots", order_by="snapshot_timestamp_utc desc")
    for col, dtype in (("match_date", pl.Date), ("run_timestamp_utc", pl.Datetime), ("kickoff_utc", pl.Datetime)):
        if col in review.columns:
            review = review.with_columns(pl.col(col).cast(dtype, strict=False))

    if "prediction_benchmark_price" not in review.columns:
        review = review.with_columns(pl.lit(None).cast(pl.Float64).alias("prediction_benchmark_price"))
    if "model_fair_price" not in review.columns:
        review = review.with_columns(pl.lit(None).cast(pl.Float64).alias("model_fair_price"))

    if "edge" not in review.columns:
        review = review.with_columns((pl.col("prediction_benchmark_price") - pl.col("model_fair_price")).alias("edge"))

    if "value_flag" not in review.columns:
        review = review.with_columns(pl.lit(False).alias("value_flag"))

    review = review.with_columns(
        pl.col("edge").cast(pl.Float64, strict=False).alias("edge"),
        pl.col("prediction_benchmark_price").cast(pl.Float64, strict=False).alias("prediction_benchmark_price"),
        pl.col("model_fair_price").cast(pl.Float64, strict=False).alias("model_fair_price"),
        pl.col("value_flag").fill_null(False).cast(pl.Boolean).alias("value_flag"),
    )

    edge_valid = pl.col("edge").is_not_null() & pl.col("edge").is_finite()
    has_benchmark = pl.col("prediction_benchmark_price").is_not_null() & pl.col("prediction_benchmark_price").is_finite()
    has_fair = pl.col("model_fair_price").is_not_null() & pl.col("model_fair_price").is_finite()
    effective_value = pl.when(pl.col("value_flag") & edge_valid & has_benchmark & has_fair).then(True).otherwise(False)

    review = review.with_columns(
        effective_value.alias("value_flag"),
        pl.when(~has_benchmark)
        .then(pl.lit("Missing benchmark"))
        .when(~edge_valid)
        .then(pl.lit("Unavailable"))
        .when(pl.col("value_flag"))
        .then(pl.lit("Value"))
        .otherwise(pl.lit("Assessed"))
        .alias("status"),
    )
    return {"review": review, "runs": runs, "matches": matches, "snapshots": snapshots}


def fixture_label(_df: pl.DataFrame) -> pl.Expr:
    if {"home_team", "away_team"}.issubset(set(_df.columns)):
        return (pl.col("home_team").fill_null("?") + pl.lit(" vs ") + pl.col("away_team").fill_null("?")).alias("fixture")
    return pl.lit("Unknown fixture").alias("fixture")


def format_percent_expr(col: str, decimals: int = 1) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
        .then(pl.lit("N/A"))
        .otherwise((pl.col(col) * 100).round(decimals).cast(pl.Utf8) + pl.lit("%"))
    )


def apply_prediction_filters(review: pl.DataFrame, *, market: str | None = None, league: str | None = None, fixture_search: str = "", min_edge: float = 0.0, value_only: bool = False) -> pl.DataFrame:
    scoped = review
    if market and market != "All":
        scoped = scoped.filter(pl.col("market") == market)
    if league and league != "All":
        scoped = scoped.filter(pl.col("league") == league)
    if fixture_search and {"home_team", "away_team"}.issubset(set(scoped.columns)):
        q = fixture_search.lower()
        scoped = scoped.filter(
            pl.col("home_team").fill_null("").str.to_lowercase().str.contains(q)
            | pl.col("away_team").fill_null("").str.to_lowercase().str.contains(q)
            | pl.col("fixture_id").fill_null("").str.to_lowercase().str.contains(q)
        )
    if "edge" in scoped.columns and min_edge > 0:
        scoped = scoped.filter(pl.col("edge").is_not_null() & pl.col("edge").is_finite() & (pl.col("edge") >= min_edge))
    if value_only:
        scoped = scoped.filter(pl.col("value_flag") == True)
    return scoped


def prediction_display_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    confidence_expr = pl.lit("N/A")
    if "credibility" in df.columns:
        confidence_expr = pl.when(pl.col("credibility").is_not_null()).then((pl.col("credibility") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%"))
    if "confidence" in df.columns:
        confidence_expr = pl.when(pl.col("confidence").is_not_null()).then((pl.col("confidence") * 100).round(0).cast(pl.Int64).cast(pl.Utf8) + pl.lit("%")).otherwise(confidence_expr)

    augmented = df.with_columns(
        fixture_label(df),
        format_percent_expr("edge").alias("edge"),
        pl.when(pl.col("status") == "Value")
        .then(pl.lit("🟢 Value"))
        .when(pl.col("status") == "Missing benchmark")
        .then(pl.lit("🟠 Missing benchmark"))
        .when(pl.col("status") == "Unavailable")
        .then(pl.lit("⚪ Unavailable"))
        .otherwise(pl.lit("🔵 Assessed"))
        .alias("status"),
        confidence_expr.alias("confidence"),
    )
    cols = [c for c in ["fixture", "market", "outcome", "edge", "status", "confidence"] if c in augmented.columns]
    return augmented.select(cols)


def prediction_detail_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    kickoff_expr = pl.lit("-").alias("kickoff_utc")
    if "kickoff_utc" in df.columns:
        kickoff_expr = pl.when(pl.col("kickoff_utc").is_null()).then(pl.lit("-")).otherwise(pl.col("kickoff_utc").cast(pl.Utf8)).alias("kickoff_utc")
    status_expr = pl.lit("Assessed").alias("status")
    if "status" in df.columns:
        status_expr = pl.col("status").alias("status")

    augmented = df.with_columns(
        fixture_label(df),
        format_percent_expr("edge").alias("edge"),
        pl.when(pl.col("model_fair_price").is_null()).then(pl.lit("N/A")).otherwise(pl.col("model_fair_price").round(3).cast(pl.Utf8)).alias("model_fair_price"),
        pl.when(pl.col("prediction_benchmark_price").is_null())
        .then(pl.lit("Missing benchmark"))
        .otherwise(pl.col("prediction_benchmark_price").round(3).cast(pl.Utf8))
        .alias("prediction_benchmark_price"),
        kickoff_expr,
        status_expr,
    )
    cols = [
        c
        for c in [
            "fixture",
            "market",
            "outcome",
            "edge",
            "status",
            "league",
            "kickoff_utc",
            "model_fair_price",
            "prediction_benchmark_price",
        ]
        if c in augmented.columns
    ]
    return augmented.select(cols)


def market_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "market" not in df.columns:
        return pl.DataFrame([])
    return (
        df.group_by("market")
        .agg(
            pl.len().alias("assessed"),
            (pl.col("status") == "Value").sum().alias("value"),
            (pl.col("status") == "Missing benchmark").sum().alias("missing_benchmark"),
        )
        .sort("value", descending=True)
    )


def today_scope(review: pl.DataFrame) -> pl.DataFrame:
    if "match_date" not in review.columns:
        return review.head(0)
    return review.filter(pl.col("match_date") == pl.lit(date.today()))
