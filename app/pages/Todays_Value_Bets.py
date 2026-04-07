from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import (
    apply_prediction_filters,
    apply_premium_dark_theme,
    dark_dataframe,
    load_core_data,
    prediction_detail_table,
    prediction_display_table,
    render_empty_state,
    require_password_gate,
    today_scope,
)

apply_premium_dark_theme("Todays Value Bets")
require_password_gate()
st.title("Today's Value Bets")
st.caption("Decision shortlist for today's fixtures.")

repo = DuckRepository()
data = load_core_data(repo)
review = today_scope(data["review"])

if review.is_empty():
    render_empty_state("No assessed selections for today yet. Run the pipeline or widen fixture date coverage.")
else:
    markets = ["All"] + sorted(review["market"].drop_nulls().unique().to_list()) if "market" in review.columns else ["All"]
    leagues = ["All"] + sorted(review["league"].drop_nulls().unique().to_list()) if "league" in review.columns else ["All"]

    f1, f2, f3, f4, f5 = st.columns(5)
    market = f1.selectbox("Market", options=markets, index=0)
    league = f2.selectbox("League", options=leagues, index=0)
    value_mode = f3.selectbox("Rows", options=["Value only", "All assessed", "Missing benchmark"], index=0)
    min_edge = f4.number_input("Min edge", min_value=0.0, value=0.0, step=0.005)
    fixture_q = f5.text_input("Fixture search")

    scoped = apply_prediction_filters(
        review,
        market=market,
        league=league,
        fixture_search=fixture_q,
        min_edge=float(min_edge),
        value_only=(value_mode == "Value only"),
    )
    if value_mode == "Missing benchmark":
        scoped = scoped.filter(pl.col("status") == "Missing benchmark")

    c1, c2, c3 = st.columns(3)
    c1.metric("Value", scoped.filter(pl.col("status") == "Value").height)
    c2.metric("Assessed", scoped.filter(pl.col("status") == "Assessed").height)
    c3.metric("Missing benchmark", scoped.filter(pl.col("status") == "Missing benchmark").height)

    st.subheader("Scan view")
    st.caption("Edge = model advantage vs benchmark price.")
    st.caption("Rows can be assessed but not value when thresholds are not met or benchmark is missing.")

    if scoped.is_empty():
        if review.is_empty():
            render_empty_state("No assessed selections are available for today's fixtures yet.")
        else:
            render_empty_state("No rows match current filters. Try widening market, league, or edge filters.")
            fallback = apply_prediction_filters(review, market=market, league=league, fixture_search=fixture_q, min_edge=0.0, value_only=False)
            if not fallback.is_empty():
                st.caption("Top assessed opportunities by edge (fallback view).")
                dark_dataframe(prediction_display_table(fallback.sort("edge", descending=True, nulls_last=True).head(12)))
    else:
        dark_dataframe(prediction_display_table(scoped.sort("edge", descending=True, nulls_last=True)))
        st.subheader("Details (secondary)")
        dark_dataframe(prediction_detail_table(scoped))

repo.close()
