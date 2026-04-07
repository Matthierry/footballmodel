from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import (
    MARKET_ORDER,
    load_core_data,
    market_breakdown,
    prediction_display_table,
    require_password_gate,
    today_scope,
)

st.set_page_config(page_title="Stattack Dashboard", layout="wide")
require_password_gate()

st.title("Dashboard")
st.caption("Daily football model workflow: review today's opportunities, inspect fixtures, run pipeline, and audit outcomes.")

repo: DuckRepository | None = None
try:
    repo = DuckRepository()
    data = load_core_data(repo)
    review = data["review"]
    runs = data["runs"]
    matches = data["matches"]

    latest_run = runs.row(0, named=True) if runs.height else {}
    today = today_scope(review)
    today_fixtures = today.select("fixture_id").n_unique() if "fixture_id" in today.columns and today.height else 0

    upcoming_fixtures = 0
    if not matches.is_empty() and "match_date" in matches.columns:
        upcoming_fixtures = matches.filter(pl.col("match_date") >= pl.lit(date.today())).height

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Latest run status", "OK" if runs.height else "No runs yet")
    k2.metric("Last successful run", str(latest_run.get("run_timestamp_utc", "N/A")))
    k3.metric("Fixtures in scope today", today_fixtures)
    k4.metric("Assessed selections today", today.height)
    value_count = today.filter(today["value_flag"] == True).height if today.height and "value_flag" in today.columns else 0
    k5.metric("Value selections today", value_count)

    st.metric("Upcoming fixtures available", upcoming_fixtures)
    predictions_count = int(latest_run.get("market_predictions", 0) or 0)
    st.metric("Predictions in latest run", predictions_count)
    if predictions_count == 0:
        st.info("Latest run produced zero market predictions.")

    st.subheader("Value opportunities by market")
    breakdown = market_breakdown(today)
    if breakdown.is_empty():
        st.info("No eligible fixtures are in scope today. This usually means the latest run found no upcoming fixtures or benchmarks.")
    else:
        show = {row["market"]: row["value"] for row in breakdown.to_dicts()}
        cols = st.columns(len(MARKET_ORDER))
        for idx, market in enumerate(MARKET_ORDER):
            cols[idx].metric(market, int(show.get(market, 0)))

    st.subheader("Top value opportunities today")
    top = today.filter(today["value_flag"] == True) if today.height and "value_flag" in today.columns else today.head(0)
    if top.is_empty():
        st.info("No value selections are currently flagged for today's fixture set.")
    else:
        if "edge" in top.columns:
            top = top.sort("edge", descending=True)
        st.dataframe(prediction_display_table(top).head(20), use_container_width=True)

    st.subheader("Quick actions")
    st.markdown("- Open **Today's Value Bets** page\n- Open **Fixture Detail** page\n- Open **Run Control** page\n- Open **History** page")

    if matches.is_empty():
        st.info("No curated fixture dataset found yet. Run pipeline ingestion before relying on dashboard metrics.")
except Exception as exc:  # noqa: BLE001
    st.info(f"Dashboard fallback activated due to error: {exc}")
finally:
    if repo is not None:
        repo.close()
