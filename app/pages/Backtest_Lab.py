from __future__ import annotations

from datetime import date

import streamlit as st

from footballmodel.backtest.walkforward import BacktestRequest, persist_backtest, run_backtest
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import require_password_gate

require_password_gate()
st.title("Backtest Lab")
st.caption("Run walk-forward simulation experiments and compare model settings before promoting changes.")

repo = DuckRepository()
try:
    matches = repo.read_df("select * from curated_matches")
    elo_history = repo.read_df("select * from elo_history")
except Exception:
    st.warning("Missing curated_matches / elo_history tables. Run the data pipeline first.")
    repo.close()
    st.stop()

cfg = load_app_config("config/runtime.yaml")

st.subheader("Date range")
min_day = matches["match_date"].min() or date(2020, 1, 1)
max_day = matches["match_date"].max() or date.today()
c1, c2 = st.columns(2)
start_date = c1.date_input("Start date", value=min_day, min_value=min_day, max_value=max_day)
end_date = c2.date_input("End date", value=max_day, min_value=min_day, max_value=max_day)

st.subheader("League universe")
league_options = sorted(matches["league"].drop_nulls().unique().to_list())
leagues = st.multiselect("Leagues", options=league_options, default=league_options)

st.subheader("Model weights")
w1, w2, w3 = st.columns(3)
dixon = w1.number_input("Dixon-Coles", min_value=0.0, max_value=1.0, value=float(cfg.weights.dixon_coles), step=0.05)
elo = w2.number_input("ELO prior", min_value=0.0, max_value=1.0, value=float(cfg.weights.elo_prior), step=0.05)
shot = w3.number_input("Shot adjustment", min_value=0.0, max_value=1.0, value=float(cfg.weights.shot_adjustment), step=0.05)

st.subheader("Runtime thresholds")
t1, t2, t3, t4 = st.columns(4)
edge = t1.number_input("Value edge", min_value=0.0, max_value=1.0, value=float(cfg.runtime.value_edge_threshold), step=0.005)
cred = t2.number_input("Credibility", min_value=0.0, max_value=1.0, value=float(cfg.runtime.credibility_threshold), step=0.01)
lookback_days = t3.number_input("Lookback days", min_value=30, max_value=730, value=int(cfg.lookback.team_form_days), step=15)
half_life_days = t4.number_input("Half-life days", min_value=5, max_value=365, value=int(cfg.half_life.team_form_days), step=5)

stake = st.number_input("Flat stake", min_value=0.1, value=1.0, step=0.1)
calibration_min_samples = st.number_input("Calibration min samples", min_value=10, max_value=500, value=50, step=10)

if st.button("Run backtest"):
    request = BacktestRequest(
        start_date=start_date,
        end_date=end_date,
        leagues=leagues,
        seasons=[],
        stake=float(stake),
        dixon_coles_weight=float(dixon),
        elo_prior_weight=float(elo),
        shot_adjustment_weight=float(shot),
        value_edge_threshold=float(edge),
        credibility_threshold=float(cred),
        lookback_days=int(lookback_days),
        half_life_days=int(half_life_days),
        calibration_min_samples=int(calibration_min_samples),
    )
    run_id, predictions, metrics = run_backtest(matches, elo_history, cfg, request)
    persist_backtest(repo, request, run_id, predictions, metrics)
    st.success(f"Backtest complete: {run_id}")
    st.metric("Selections", predictions.height)
    st.metric("Metric rows", metrics.height)
    st.dataframe(metrics.sort(["breakdown", "group_key"]), use_container_width=True)
else:
    st.info("Choose a date range + parameters, then run backtest to populate results.")

repo.close()
