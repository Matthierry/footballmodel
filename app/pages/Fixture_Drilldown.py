from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import MARKET_GROUPS, load_core_data, prediction_detail_table, require_password_gate

require_password_gate()
st.title("Fixture Detail")
st.caption("Deep dive one fixture with value context and benchmark coverage.")

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

    s1, s2, s3 = st.columns(3)
    s1.metric("Fixture", row.get("fixture", "N/A"))
    s2.metric("Kickoff date", str(row.get("match_date", "N/A")))
    s3.metric("League", str(row.get("league", "N/A")))

    best = scoped.filter(pl.col("status") == "Value") if "status" in scoped.columns else scoped.head(0)
    if best.is_empty() and "edge" in scoped.columns:
        best = scoped.filter(pl.col("edge").is_not_null()).sort("edge", descending=True).head(1)
    else:
        best = best.sort("edge", descending=True).head(1)

    st.subheader("Best opportunity on this fixture")
    if best.is_empty():
        st.info("No opportunity can be highlighted yet (missing benchmark or no assessed edges).")
    else:
        b = best.row(0, named=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("Market / Outcome", f"{b.get('market', 'N/A')} · {b.get('outcome', 'N/A')}")
        b2.metric("Edge", f"{round(float((b.get('edge') or 0.0) * 100), 1)}%" if b.get("edge") is not None else "N/A")
        b3.metric("Credibility", f"{round(float((b.get('credibility') or b.get('confidence') or 0.0) * 100), 0)}%")
        st.caption(f"Market price: {b.get('prediction_benchmark_price', 'Missing benchmark')} | Model fair odds: {b.get('model_fair_price', 'N/A')}")

    latest_run = runs.row(0, named=True) if runs.height else {}
    st.caption(f"Latest run: {latest_run.get('run_timestamp_utc', 'N/A')} | {latest_run.get('config_name', 'N/A')} {latest_run.get('config_version', '')}")

    st.subheader("Expected goals context")
    xg_cols = [c for c in ["expected_home_goals", "expected_away_goals"] if c in scoped.columns]
    if xg_cols:
        st.dataframe(scoped.select([c for c in ["market", "outcome", *xg_cols, "edge", "status"] if c in scoped.columns]), use_container_width=True)
    else:
        st.info("Expected goals are unavailable for this fixture. This can happen when upstream feature rows were missing at run time.")

    st.subheader("Market breakdown")
    for group in ["1X2", "Over/Under 2.5", "BTTS", "Asian Handicap"]:
        block = scoped.filter(pl.col("market") == group) if "market" in scoped.columns else scoped.head(0)
        st.markdown(f"**{group}**")
        if block.is_empty():
            st.caption("No assessed rows for this market.")
        else:
            st.dataframe(prediction_detail_table(block), use_container_width=True)

    other = scoped.filter(~pl.col("market").is_in(list(MARKET_GROUPS.keys()))) if "market" in scoped.columns else scoped.head(0)
    if not other.is_empty():
        st.markdown("**Other markets**")
        st.dataframe(prediction_detail_table(other), use_container_width=True)

    st.subheader("Benchmarks captured")
    if snapshots.is_empty() or "fixture_id" not in snapshots.columns:
        st.info("No benchmark snapshots were captured yet for this fixture.")
    else:
        snap = snapshots.filter(pl.col("fixture_id") == fixture_id)
        if snap.is_empty():
            st.info("No benchmark snapshots found for selected fixture.")
        else:
            st.dataframe(snap.sort("snapshot_timestamp_utc", descending=True), use_container_width=True)

repo.close()
