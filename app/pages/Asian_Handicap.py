from __future__ import annotations

import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import apply_prediction_filters, load_core_data, prediction_display_table, require_password_gate, today_scope

TITLE = "Asian Handicap"
MARKET = "Asian Handicap"

require_password_gate()
st.title(TITLE)
st.caption("Focused market view powered by the same dataset as Today's Value Bets.")

repo = DuckRepository()
data = load_core_data(repo)
review = today_scope(data["review"])

if review.is_empty():
    st.info("No assessed selections for today.")
else:
    leagues = ["All"] + sorted(review["league"].drop_nulls().unique().to_list()) if "league" in review.columns else ["All"]
    c1, c2, c3 = st.columns(3)
    league = c1.selectbox("League", options=leagues, index=0)
    value_only = c2.selectbox("Rows", options=["Value only", "All assessed"], index=0) == "Value only"
    min_edge = c3.number_input("Min edge", min_value=0.0, value=0.0, step=0.005)

    scoped = apply_prediction_filters(review, market=MARKET, league=league, min_edge=float(min_edge), value_only=value_only)
    if scoped.is_empty():
        st.info("No rows match filters for this market.")
    else:
        st.dataframe(prediction_display_table(scoped), use_container_width=True)

repo.close()
