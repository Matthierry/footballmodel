from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import load_core_data, require_password_gate

require_password_gate()
st.title("History")
st.caption("Review prediction runs and benchmark snapshots captured at prediction-time for later settlement analysis.")

repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
runs = data["runs"]
snapshots = data["snapshots"]

st.markdown("**What this page means**")
st.caption("Prediction-time snapshot = benchmark price captured when selection was produced. Later snapshot = benchmark refresh used for CLV and review.")

if not runs.is_empty():
    st.subheader("Recent prediction runs")
    cols = [c for c in ["run_timestamp_utc", "live_run_id", "config_name", "config_version", "fixtures_scored", "market_predictions"] if c in runs.columns]
    st.dataframe(runs.select(cols).head(30), use_container_width=True)

if review.is_empty():
    st.info("No review rows yet. Settlements and post-run analysis will appear after pipeline snapshots are stored.")
else:
    if "fixture_id" not in review.columns:
        st.info("History rows exist but fixture identifiers are missing in this dataset.")
    else:
        needed = [c for c in ["fixture_id", "home_team", "away_team"] if c in review.columns]
        fixtures = review.select(needed).unique(subset=["fixture_id"]).sort("fixture_id")
        if "home_team" not in fixtures.columns:
            fixtures = fixtures.with_columns(pl.lit("?").alias("home_team"))
        if "away_team" not in fixtures.columns:
            fixtures = fixtures.with_columns(pl.lit("?").alias("away_team"))
        labels = [f"{r['fixture_id']} · {r.get('home_team', '?')} vs {r.get('away_team', '?')}" for r in fixtures.to_dicts()]
        selected_idx = st.selectbox("Fixture", options=list(range(len(labels))), format_func=lambda x: labels[x])
        fixture_id = fixtures.row(selected_idx, named=True)["fixture_id"]

        scoped = review.filter(pl.col("fixture_id") == fixture_id)
        st.subheader("Fixture review table")
        cols = [c for c in ["match_date", "league", "market", "outcome", "prediction_benchmark_price", "later_benchmark_price", "edge", "value_flag", "settlement_status", "result_status", "clv"] if c in scoped.columns]
        st.dataframe(scoped.select(cols), use_container_width=True)

        st.subheader("Latest benchmark by market/outcome")
        if snapshots.is_empty() or "fixture_id" not in snapshots.columns:
            st.info("No benchmark snapshots are currently stored.")
        else:
            snap = snapshots.filter(pl.col("fixture_id") == fixture_id)
            if snap.is_empty():
                st.info("No snapshot history available for this fixture yet.")
            else:
                latest = (
                    snap.sort("snapshot_timestamp_utc")
                    .group_by([c for c in ["market", "outcome", "line"] if c in snap.columns])
                    .agg(
                        pl.col("benchmark_price").last().alias("latest_benchmark_price"),
                        pl.col("benchmark_source").last().alias("latest_benchmark_source"),
                        pl.col("snapshot_type").last().alias("latest_snapshot_type"),
                        pl.col("snapshot_timestamp_utc").last().alias("latest_snapshot_timestamp_utc"),
                    )
                )
                st.dataframe(latest, use_container_width=True)
                st.subheader("Snapshot timeline")
                st.dataframe(snap.sort("snapshot_timestamp_utc", descending=True), use_container_width=True)

repo.close()
