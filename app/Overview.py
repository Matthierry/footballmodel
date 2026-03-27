from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.config.runtime_env import get_app_password
from footballmodel.storage.repository import DuckRepository

st.set_page_config(page_title="FootballModel", layout="wide")

password = st.sidebar.text_input("Password", type="password")
if password != get_app_password(streamlit_module=st):
    st.warning("Enter valid password to access dashboard")
    st.stop()

st.title("FootballModel Overview")
repo: DuckRepository | None = None
try:
    repo = DuckRepository()

    def _safe_read_table(
        table: str,
        *,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> pl.DataFrame:
        if repo is None:
            st.info("Repository unavailable; showing empty fallback state.")
            return pl.DataFrame([])
        try:
            return repo.read_table_or_empty(table, order_by=order_by, limit=limit)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Could not read table '{table}': {exc}")
            return pl.DataFrame([])

    latest_summary = _safe_read_table("live_run_summaries_history", order_by="run_timestamp_utc desc", limit=1)
    latest_predictions = _safe_read_table("model_runs", order_by="run_timestamp_utc desc, timestamp_utc desc", limit=500)
    matches = _safe_read_table("curated_matches")
    snapshots = _safe_read_table("benchmark_snapshots")

    latest_summary_row = latest_summary.row(0, named=True) if latest_summary.height else None

    if latest_summary_row:
        st.metric("Latest pipeline run", f"Completed @ {latest_summary_row['run_timestamp_utc']}")
        st.caption(
            f"Config: {latest_summary_row.get('config_name') or 'unknown'} "
            f"(version {latest_summary_row.get('config_version') or 'unknown'})"
        )
        st.metric(
            "Latest live run summary",
            (
                f"fixtures={latest_summary_row.get('fixtures_scored', 0) or 0}, "
                f"markets={latest_summary_row.get('market_predictions', 0) or 0}, "
                f"review={latest_summary_row.get('review_rows', 0) or 0}"
            ),
        )
    else:
        st.metric("Latest pipeline run", "No run summary available yet")
        st.metric("Latest live run summary", "No run summary available yet")
        st.info("Pipeline summary table is empty; run the pipeline to populate latest run status.")

    st.metric("Canonical match count", matches.height)
    if matches.height == 0:
        st.info("curated_matches is empty; run ingestion/build to populate the canonical dataset.")

    latest_refresh = (
        matches.select("fetched_at_utc").drop_nulls().sort("fetched_at_utc", descending=True).head(1)
        if "fetched_at_utc" in matches.columns
        else matches.head(0)
    )
    if latest_refresh.height:
        st.metric("Latest raw refresh/build", str(latest_refresh["fetched_at_utc"][0]))
    else:
        st.metric("Latest raw refresh/build", "No refresh timestamp available")
        st.info("Raw ingestion metadata is empty; verify Football-Data/ClubElo refresh completed.")

    upcoming_count = 0
    if matches.height:
        future_expr = (
            pl.col("is_future_fixture")
            if "is_future_fixture" in matches.columns
            else (pl.col("home_goals").is_null() & pl.col("away_goals").is_null())
        )
        if "match_date" in matches.columns:
            upcoming_count = matches.filter(future_expr & (pl.col("match_date") >= pl.lit(date.today()))).height
        else:
            upcoming_count = matches.filter(future_expr).height
    st.metric("Upcoming fixtures available", upcoming_count)
    if upcoming_count == 0:
        st.info("No upcoming fixtures are currently available in curated_matches.")

    latest_predictions_count = 0
    if latest_summary_row:
        latest_predictions_count = int(latest_summary_row.get("market_predictions") or 0)
    st.metric("Predictions in latest run", latest_predictions_count)
    if latest_predictions_count == 0:
        st.info("Latest run produced zero market predictions.")

    st.metric("Benchmark snapshots stored", snapshots.height)
    if snapshots.height == 0:
        st.info("No benchmark snapshots stored yet.")

    latest_config_name = None
    latest_config_version = None
    if latest_summary_row:
        latest_config_name = latest_summary_row.get("config_name")
        latest_config_version = latest_summary_row.get("config_version")
    elif latest_predictions.height:
        latest_config_name = latest_predictions["config_name"][0] if "config_name" in latest_predictions.columns else None
        latest_config_version = (
            latest_predictions["config_version"][0] if "config_version" in latest_predictions.columns else None
        )
    if latest_config_name or latest_config_version:
        st.metric("Latest live config", f"{latest_config_name or 'unknown'} / {latest_config_version or 'unknown'}")
    else:
        st.metric("Latest live config", "Not available")
        st.info("No live config metadata found yet.")

    st.subheader("Recent run predictions")
    if latest_predictions.height:
        show_cols = [
            c
            for c in ["fixture_id", "home_team", "away_team", "expected_home_goals", "expected_away_goals", "run_timestamp_utc"]
            if c in latest_predictions.columns
        ]
        st.dataframe(latest_predictions.select(show_cols).head(50))
    else:
        st.info("No model runs yet.")
except Exception as exc:  # noqa: BLE001
    st.metric("Latest pipeline run", "Unavailable")
    st.metric("Canonical match count", 0)
    st.metric("Upcoming fixtures available", 0)
    st.metric("Predictions in latest run", 0)
    st.info(f"Overview fallback activated due to dashboard error: {exc}")
finally:
    if repo is not None:
        repo.close()

st.caption("Use pages for 1X2, OU2.5, BTTS, Correct Score, Asian Handicap, Drilldown, Backtest Lab, Experiments, Run Control, and History.")
