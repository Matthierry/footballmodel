from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import load_core_data, prediction_detail_table, require_password_gate

require_password_gate()
st.title("Fixture Detail")
st.caption("Deep dive into one fixture across model outputs, value flags, and benchmark snapshots.")

repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
snapshots = data["snapshots"]
runs = data["runs"]

if review.is_empty() or "fixture_id" not in review.columns:
    st.info("No fixture-level prediction rows are available yet.")
else:
    options = (
        review.with_columns((pl.col("home_team").fill_null("?") + pl.lit(" vs ") + pl.col("away_team").fill_null("?")).alias("fixture"))
        .select(["fixture_id", "fixture", "match_date", "league"])
        .unique(subset=["fixture_id"])
        .sort("match_date", descending=True)
    )
    labels = [f"{r['fixture']} ({r.get('league', 'NA')} · {r.get('match_date', 'NA')})" for r in options.to_dicts()]
    idx = st.selectbox("Fixture", options=list(range(len(labels))), format_func=lambda x: labels[x])
    row = options.row(idx, named=True)
    fixture_id = row["fixture_id"]

    scoped = review.filter(pl.col("fixture_id") == fixture_id)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Fixture", row.get("fixture", "N/A"))
    s2.metric("Kickoff date", str(row.get("match_date", "N/A")))
    s3.metric("League", str(row.get("league", "N/A")))
    s4.metric("Value rows", scoped.filter(pl.col("value_flag") == True).height if "value_flag" in scoped.columns else 0)

    latest_run = runs.row(0, named=True) if runs.height else {}
    st.caption(f"Latest run/config: {latest_run.get('run_timestamp_utc', 'N/A')} | {latest_run.get('config_name', 'N/A')} {latest_run.get('config_version', '')}")

    st.subheader("Match summary")
    xg_cols = [c for c in ["expected_home_goals", "expected_away_goals"] if c in scoped.columns]
    if xg_cols:
        st.dataframe(scoped.select([c for c in ["market", "outcome", *xg_cols, "edge", "value_flag"] if c in scoped.columns]).head(12), use_container_width=True)
    else:
        st.info("Expected goals are not available for this fixture in the current review table.")

    st.subheader("Value opportunities")
    value_rows = scoped.filter(pl.col("value_flag") == True) if "value_flag" in scoped.columns else scoped.head(0)
    if value_rows.is_empty():
        st.info("No value opportunities are currently flagged for this fixture.")
    else:
        st.dataframe(prediction_detail_table(value_rows), use_container_width=True)

    st.subheader("Full market breakdown")
    st.dataframe(prediction_detail_table(scoped), use_container_width=True)

    st.subheader("Benchmarks captured")
    if snapshots.is_empty() or "fixture_id" not in snapshots.columns:
        st.info("No benchmark snapshots were captured yet for this fixture.")
    else:
        snap = snapshots.filter(pl.col("fixture_id") == fixture_id)
        if snap.is_empty():
            st.info("No benchmark snapshots found for selected fixture.")
        else:
            st.dataframe(snap.sort("snapshot_timestamp_utc", descending=True), use_container_width=True)

    st.subheader("Technical details")
    st.dataframe(scoped, use_container_width=True)

repo.close()
