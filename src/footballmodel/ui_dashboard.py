from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.config.runtime_env import get_app_password
from footballmodel.storage.repository import DuckRepository

MARKET_ORDER = ["1X2", "Over/Under 2.5", "BTTS", "Asian Handicap", "Correct Score"]


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
    if "value_flag" not in review.columns:
        review = review.with_columns(pl.lit(None).alias("value_flag"))
    if "edge" not in review.columns:
        review = review.with_columns((pl.col("prediction_benchmark_price") - pl.col("model_fair_price")).alias("edge") if {"prediction_benchmark_price", "model_fair_price"}.issubset(set(review.columns)) else pl.lit(None).alias("edge"))
    return {"review": review, "runs": runs, "matches": matches, "snapshots": snapshots}


def fixture_label(df: pl.DataFrame) -> pl.Series:
    if {"home_team", "away_team"}.issubset(set(df.columns)):
        return (pl.col("home_team").fill_null("?") + pl.lit(" vs ") + pl.col("away_team").fill_null("?")).alias("fixture")
    return pl.lit("Unknown fixture").alias("fixture")


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
    if "edge" in scoped.columns:
        scoped = scoped.filter(pl.col("edge").fill_null(0.0) >= min_edge)
    if value_only:
        scoped = scoped.filter(pl.col("value_flag") == True)
    return scoped


def prediction_display_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    augmented = df.with_columns(fixture_label(df))
    cols = [c for c in ["fixture", "market", "outcome", "edge", "value_flag", "credibility", "confidence"] if c in augmented.columns]
    if "credibility" not in augmented.columns and "confidence" in augmented.columns and "confidence" not in cols:
        cols.append("confidence")
    return augmented.select(cols)


def prediction_detail_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    augmented = df.with_columns(fixture_label(df))
    cols = [c for c in ["fixture", "league", "kickoff_utc", "model_fair_price", "prediction_benchmark_price", "market", "outcome", "edge", "value_flag"] if c in augmented.columns]
    return augmented.select(cols)


def market_breakdown(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "market" not in df.columns:
        return pl.DataFrame([])
    return (
        df.group_by("market")
        .agg(
            pl.len().alias("assessed"),
            (pl.col("value_flag") == True).sum().alias("value"),
        )
        .sort("value", descending=True)
    )


def today_scope(review: pl.DataFrame) -> pl.DataFrame:
    if "match_date" not in review.columns:
        return review.head(0)
    return review.filter(pl.col("match_date") == pl.lit(date.today()))
