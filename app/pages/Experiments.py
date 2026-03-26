from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.backtest.walkforward import SweepRequest, persist_sweep_results, rank_experiment_runs, run_experiment_sweep
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository

st.title("Experiments")
st.caption("Run optimisation sweeps and compare raw vs calibrated predictive quality.")

repo = DuckRepository()
cfg = load_app_config("config/runtime.yaml")

try:
    runs = repo.read_df("select * from backtest_runs order by created_at desc")
except Exception:
    runs = pl.DataFrame([])

try:
    matches = repo.read_df("select * from curated_matches")
    elo_history = repo.read_df("select * from elo_history")
except Exception:
    matches = pl.DataFrame([])
    elo_history = pl.DataFrame([])

if not matches.is_empty() and not elo_history.is_empty():
    st.subheader("Calibration + optimisation sweep")
    min_day = matches["match_date"].min() or date(2020, 1, 1)
    max_day = matches["match_date"].max() or date.today()
    leagues = sorted(matches["league"].drop_nulls().unique().to_list())

    col_a, col_b = st.columns(2)
    with col_a:
        sweep_start = st.date_input("Sweep start", value=min_day, min_value=min_day, max_value=max_day)
        sweep_end = st.date_input("Sweep end", value=max_day, min_value=min_day, max_value=max_day)
        sweep_leagues = st.multiselect("Sweep leagues", options=leagues, default=leagues)
        dixon_opts = st.multiselect("Dixon weight options", [0.4, 0.5, 0.6, 0.7], default=[cfg.weights.dixon_coles])
        elo_opts = st.multiselect("ELO weight options", [0.1, 0.2, 0.25, 0.3, 0.4], default=[cfg.weights.elo_prior])
        shot_opts = st.multiselect("Shot weight options", [0.1, 0.15, 0.2, 0.25], default=[cfg.weights.shot_adjustment])
    with col_b:
        edge_opts = st.multiselect("Edge threshold options", [0.01, 0.02, 0.025, 0.03, 0.04], default=[cfg.runtime.value_edge_threshold])
        cred_opts = st.multiselect("Cred threshold options", [0.45, 0.5, 0.55, 0.6, 0.65], default=[cfg.runtime.credibility_threshold])
        lookback_opts = st.multiselect("Lookback options", [90, 120, 180, 240, 365], default=[cfg.lookback.team_form_days])
        half_life_opts = st.multiselect("Half-life options", [15, 30, 45, 60, 90], default=[cfg.half_life.team_form_days])
        stake = st.number_input("Flat stake", min_value=0.1, value=1.0, step=0.1)

    if st.button("Run optimisation sweep"):
        sweep_request = SweepRequest(
            start_date=sweep_start,
            end_date=sweep_end,
            leagues=sweep_leagues,
            stake=float(stake),
            dixon_coles_weights=dixon_opts,
            elo_prior_weights=elo_opts,
            shot_adjustment_weights=shot_opts,
            value_edge_thresholds=edge_opts,
            credibility_thresholds=cred_opts,
            lookback_days_options=lookback_opts,
            half_life_days_options=half_life_opts,
            calibrate_probabilities=True,
        )
        sweep_id, summary, ranking = run_experiment_sweep(matches, elo_history, cfg, sweep_request)
        persist_sweep_results(repo, sweep_id, summary, ranking)

        st.success(f"Sweep complete: {sweep_id} ({summary.height} runs)")
        if not ranking.is_empty():
            st.dataframe(ranking)

if runs.is_empty():
    st.info("No experiment runs yet. Use Backtest Lab to create one.")
    repo.close()
    st.stop()

run_ids = runs["run_id"].to_list()
selected = st.multiselect("Select runs to compare", options=run_ids, default=run_ids[:2] if len(run_ids) > 1 else run_ids)

if selected:
    id_list = ",".join(f"'{run_id}'" for run_id in selected)
    metrics = repo.read_df(f"select * from backtest_metrics where run_id in ({id_list}) order by run_id, breakdown, group_key")
    st.dataframe(metrics)

    summary = metrics.filter((metrics["breakdown"] == "overall") & (metrics["group_key"] == "all"))
    if not summary.is_empty():
        ranked = rank_experiment_runs(summary)
        st.subheader("Overall comparison (predictive-first ranking)")
        display_cols = [
            "run_id",
            "ranking_score",
            "raw_log_loss",
            "calibrated_log_loss",
            "raw_brier_score",
            "calibrated_brier_score",
            "raw_calibration_error",
            "calibrated_calibration_error",
            "avg_clv",
            "median_clv",
            "share_beating_close",
            "flat_stake_roi",
            "strike_rate",
            "max_drawdown",
            "bets",
        ]
        st.dataframe(ranked.select([c for c in display_cols if c in ranked.columns]))
        st.subheader("Best and worst runs")
        st.dataframe(pl.concat([ranked.head(3), ranked.tail(3)]))

    st.subheader("Calibration by market")
    st.dataframe(metrics.filter(pl.col("breakdown") == "market"))

    st.subheader("Calibration by league")
    st.dataframe(metrics.filter(pl.col("breakdown") == "league"))

repo.close()
