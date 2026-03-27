from __future__ import annotations

import streamlit as st
import polars as pl

from footballmodel.config.runtime_env import get_app_password
from footballmodel.storage.repository import DuckRepository

st.set_page_config(page_title="FootballModel", layout="wide")

password = st.sidebar.text_input("Password", type="password")
if password != get_app_password(streamlit_module=st):
    st.warning("Enter valid password to access dashboard")
    st.stop()

st.title("FootballModel Overview")
repo = DuckRepository()
try:
    runs = repo.read_df("select * from model_runs order by timestamp_utc desc limit 200")
    st.metric("Predictions", runs.height)
    if runs.height:
        st.dataframe(runs.select(["fixture_id", "home_team", "away_team", "expected_home_goals", "expected_away_goals"]))
finally:
    repo.close()

st.caption("Use pages for 1X2, OU2.5, BTTS, Correct Score, Asian Handicap, Drilldown, Backtest Lab, Experiments, Run Control, and History.")
