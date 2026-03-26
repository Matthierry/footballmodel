from __future__ import annotations

from datetime import date

import streamlit as st

from footballmodel.backtest.walkforward import BacktestRequest, persist_backtest, run_backtest
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository

st.title("Backtest Lab")
st.caption("Run walk-forward paper-trading experiments for 1X2, OU2.5, BTTS, and Asian Handicap.")

repo = DuckRepository()
try:
    matches = repo.read_df("select * from curated_matches")
    elo_history = repo.read_df("select * from elo_history")
except Exception:
    st.warning("Missing curated_matches / elo_history tables. Run the data pipeline first.")
    repo.close()
    st.stop()

cfg = load_app_config("config/runtime.yaml")

min_day = matches["match_date"].min() or date(2020, 1, 1)
max_day = matches["match_date"].max() or date.today()
league_options = sorted(matches["league"].drop_nulls().unique().to_list())
seasons = sorted(
    {
        f"{d.year if d.month >= 7 else d.year - 1}/{(d.year if d.month >= 7 else d.year - 1) + 1}"
        for d in matches["match_date"].drop_nulls().to_list()
    }
)

col_a, col_b = st.columns(2)
with col_a:
    start_date = st.date_input("Start date", value=min_day, min_value=min_day, max_value=max_day)
    end_date = st.date_input("End date", value=max_day, min_value=min_day, max_value=max_day)
    leagues = st.multiselect("Leagues", options=league_options, default=league_options)
    selected_seasons = st.multiselect("Seasons (optional)", options=seasons, default=seasons)
with col_b:
    stake = st.number_input("Flat stake", min_value=0.1, value=1.0, step=0.1)
    dixon = st.number_input("Dixon-Coles weight", min_value=0.0, max_value=1.0, value=float(cfg.weights.dixon_coles), step=0.05)
    elo = st.number_input("ELO weight", min_value=0.0, max_value=1.0, value=float(cfg.weights.elo_prior), step=0.05)
    shot = st.number_input("Shot weight", min_value=0.0, max_value=1.0, value=float(cfg.weights.shot_adjustment), step=0.05)
    edge = st.number_input("Value edge threshold", min_value=0.0, max_value=1.0, value=float(cfg.runtime.value_edge_threshold), step=0.005)
    cred = st.number_input("Credibility threshold", min_value=0.0, max_value=1.0, value=float(cfg.runtime.credibility_threshold), step=0.01)

if st.button("Run walk-forward backtest"):
    request = BacktestRequest(
        start_date=start_date,
        end_date=end_date,
        leagues=leagues,
        seasons=selected_seasons,
        stake=float(stake),
        dixon_coles_weight=float(dixon),
        elo_prior_weight=float(elo),
        shot_adjustment_weight=float(shot),
        value_edge_threshold=float(edge),
        credibility_threshold=float(cred),
    )
    run_id, predictions, metrics = run_backtest(matches, elo_history, cfg, request)
    persist_backtest(repo, request, run_id, predictions, metrics)

    st.success(f"Backtest run complete: {run_id}")
    st.metric("Selections", predictions.height)
    st.metric("Metrics rows", metrics.height)
    st.dataframe(metrics.sort(["breakdown", "group_key"]))

repo.close()
