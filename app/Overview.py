from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import MARKET_ORDER, load_core_data, market_breakdown, prediction_display_table, require_password_gate, today_scope

st.set_page_config(page_title="Dashboard", layout="wide")
require_password_gate()

st.title("Dashboard")
st.caption("Daily workflow: inspect one fixture, review today's value bets, run pipeline checks, and audit history.")

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
    upcoming_fixtures = matches.filter(pl.col("match_date") >= pl.lit(date.today())).height if (not matches.is_empty() and "match_date" in matches.columns) else 0
    value_count = today.filter(pl.col("status") == "Value").height if today.height else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Run status", "Ready" if runs.height else "No runs yet")
    k2.metric("Latest run", str(latest_run.get("run_timestamp_utc", "N/A")))
    k3.metric("In-scope fixtures today", today_fixtures)
    k4.metric("Value selections today", value_count)

    m1, m2, m3 = st.columns(3)
    m1.metric("Assessed selections today", today.height)
    m2.metric("Upcoming fixtures available", upcoming_fixtures)
    predictions_count = int(latest_run.get("market_predictions", 0) or 0)
    m3.metric("Predictions in latest run", predictions_count)

    if predictions_count == 0:
        st.info("Latest run produced zero market predictions.")

    st.subheader("Value opportunities by market")
    breakdown = market_breakdown(today)
    if breakdown.is_empty():
        st.info("No eligible fixtures are in scope today.")
    else:
        show = {row["market"]: row for row in breakdown.to_dicts()}
        cols = st.columns(len(MARKET_ORDER))
        for idx, market in enumerate(MARKET_ORDER):
            row = show.get(market, {})
            cols[idx].metric(market, f"{int(row.get('value', 0))} value / {int(row.get('assessed', 0))} assessed")

    st.subheader("Top value opportunities today")
    top = today.filter(pl.col("status") == "Value") if today.height else today.head(0)
    if top.is_empty():
        st.info("No value selections are currently flagged for today's fixture set.")
    else:
        st.dataframe(prediction_display_table(top.sort("edge", descending=True)).head(20), use_container_width=True)

    st.subheader("Quick actions")
    st.markdown(
        "- **Today's Value Bets**: action shortlist\n"
        "- **Fixture Detail**: deep-dive one match\n"
        "- **Run Pipeline**: run status + manual trigger state\n"
        "- **History**: review and settlement checks"
    )

    if matches.is_empty():
        st.info("No curated fixture dataset found yet. Run pipeline ingestion before relying on dashboard metrics.")
except Exception as exc:  # noqa: BLE001
    st.info(f"Dashboard fallback activated due to error: {exc}")
finally:
    if repo is not None:
        repo.close()
