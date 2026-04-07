from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.backtest.walkforward import (
    SweepRequest,
    build_champion_view,
    persist_sweep_results,
    rank_experiment_runs,
    run_experiment_sweep,
)
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import apply_premium_dark_theme, render_empty_state, require_password_gate

apply_premium_dark_theme("Experiments")
require_password_gate()
st.title("Experiments")
st.caption("Optimisation sweeps test combinations of weights and thresholds to find robust model settings.")

repo = DuckRepository()
cfg = load_app_config("config/runtime.yaml")

try:
    matches = repo.read_df("select * from curated_matches")
    elo_history = repo.read_df("select * from elo_history")
except Exception:
    matches = pl.DataFrame([])
    elo_history = pl.DataFrame([])

if matches.is_empty() or elo_history.is_empty():
    st.info("Missing source tables for experiment sweep. Run ingestion first.")
else:
    st.subheader("Sweep setup")
    st.caption("Define window and leagues before generating candidate configurations.")
    min_day = matches["match_date"].min() or date(2020, 1, 1)
    max_day = matches["match_date"].max() or date.today()
    leagues = sorted(matches["league"].drop_nulls().unique().to_list())

    c1, c2 = st.columns(2)
    sweep_start = c1.date_input("Start", value=min_day, min_value=min_day, max_value=max_day)
    sweep_end = c2.date_input("End", value=max_day, min_value=min_day, max_value=max_day)
    sweep_leagues = st.multiselect("Leagues", options=leagues, default=leagues)

    st.subheader("Search space")
    st.caption("Smaller search spaces run faster and are easier to interpret.")
    dixon_opts = st.multiselect("Dixon weights", [0.4, 0.5, 0.6, 0.7], default=[cfg.weights.dixon_coles])
    elo_opts = st.multiselect("ELO weights", [0.1, 0.2, 0.25, 0.3, 0.4], default=[cfg.weights.elo_prior])
    shot_opts = st.multiselect("Shot weights", [0.1, 0.15, 0.2, 0.25], default=[cfg.weights.shot_adjustment])
    edge_opts = st.multiselect("Edge thresholds", [0.01, 0.02, 0.025, 0.03, 0.04], default=[cfg.runtime.value_edge_threshold])
    cred_opts = st.multiselect("Credibility thresholds", [0.45, 0.5, 0.55, 0.6, 0.65], default=[cfg.runtime.credibility_threshold])

    if st.button("Run optimisation sweep"):
        st.caption("Running optimisation sweep; results will populate below.")
        req = SweepRequest(
            start_date=sweep_start,
            end_date=sweep_end,
            leagues=sweep_leagues,
            stake=1.0,
            dixon_coles_weights=dixon_opts,
            elo_prior_weights=elo_opts,
            shot_adjustment_weights=shot_opts,
            value_edge_thresholds=edge_opts,
            credibility_thresholds=cred_opts,
            lookback_days_options=[cfg.lookback.team_form_days],
            half_life_days_options=[cfg.half_life.team_form_days],
            calibrate_probabilities=True,
        )
        sweep_id, summary, ranking = run_experiment_sweep(matches, elo_history, cfg, req)
        persist_sweep_results(repo, sweep_id, summary, ranking)
        st.success(f"Sweep complete: {sweep_id}")
        st.dataframe(ranking, use_container_width=True)
        st.dataframe(build_champion_view(ranking), use_container_width=True)

try:
    runs = repo.read_df("select * from backtest_runs order by created_at desc")
except Exception:
    runs = pl.DataFrame([])

if runs.is_empty():
    st.info("No saved experiment/backtest runs yet.")
else:
    selected = st.multiselect("Compare run IDs", options=runs["run_id"].to_list(), default=runs["run_id"].to_list()[:2])
    if selected:
        id_list = ",".join(f"'{run_id}'" for run_id in selected)
        metrics = repo.read_df(f"select * from backtest_metrics where run_id in ({id_list})")
        st.subheader("Run comparison")
        st.dataframe(metrics, use_container_width=True)
        summary = metrics.filter((metrics["breakdown"] == "overall") & (metrics["group_key"] == "all"))
        if not summary.is_empty():
            st.subheader("Ranking")
            st.dataframe(rank_experiment_runs(summary), use_container_width=True)

repo.close()
