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
            st.subheader("Champion vs challengers")
            st.dataframe(build_champion_view(ranking))

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
    ranked = pl.DataFrame([])
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
            "robustness_score",
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

    st.subheader("Champion vs challengers")
    st.dataframe(build_champion_view(ranked) if not summary.is_empty() else pl.DataFrame([]))

    st.subheader("Raw vs calibrated by market")
    st.dataframe(metrics.filter(pl.col("breakdown") == "market"))

    st.subheader("Raw vs calibrated by league")
    st.dataframe(metrics.filter(pl.col("breakdown") == "league"))

    st.subheader("Strongest and weakest segments")
    segment_view = metrics.filter(pl.col("breakdown").is_in(["league", "market", "edge_bucket", "benchmark_source"]))
    if not segment_view.is_empty():
        sorted_segments = segment_view.sort("calibrated_log_loss")
        st.markdown("**Strongest**")
        st.dataframe(sorted_segments.head(10))
        st.markdown("**Weakest**")
        st.dataframe(sorted_segments.tail(10))

try:
    sweep_meta = repo.read_df("select * from experiment_sweep_metadata order by created_at desc")
except Exception:
    try:
        sweep_meta = repo.read_df(
            "select sweep_id, max(created_at) as created_at, count(*) as run_count from experiment_rankings group by sweep_id order by created_at desc"
        )
    except Exception:
        sweep_meta = pl.DataFrame([])

if not sweep_meta.is_empty():
    st.subheader("Historical sweep review")
    sweep_options = sweep_meta["sweep_id"].to_list()
    default_idx = 0 if sweep_options else None
    selected_sweep_id = st.selectbox("Select sweep ID", options=sweep_options, index=default_idx)

    st.markdown("**Sweep metadata**")
    st.dataframe(sweep_meta.filter(pl.col("sweep_id") == selected_sweep_id))

    compare_ids = st.multiselect(
        "Compare sweeps (summary level)",
        options=sweep_options,
        default=sweep_options[:2] if len(sweep_options) > 1 else sweep_options[:1],
    )
    if compare_ids:
        compare_clause = ",".join(f"'{x}'" for x in compare_ids)
        compare = repo.read_df(
            f"""
            select sweep_id, run_id, ranking_score, calibrated_log_loss, calibrated_brier_score,
                   calibrated_calibration_error, avg_clv, robustness_score, flat_stake_roi
            from experiment_rankings
            where sweep_id in ({compare_clause})
            order by sweep_id, ranking_score
            """
        )
        st.markdown("**Sweep comparison (ranked outcomes)**")
        st.dataframe(compare)

    champion = repo.read_df(
        f"select * from experiment_champion_view where sweep_id = '{selected_sweep_id}' order by selection_role, ranking_score"
    )
    st.markdown("**Champion + challengers**")
    st.dataframe(champion)

    calib_buckets = repo.read_df(
        f"""
        select * from experiment_calibration_buckets
        where sweep_id = '{selected_sweep_id}'
        order by run_id, market, league, probability_bucket
        """
    )
    st.markdown("**Calibration buckets (selected sweep)**")
    st.dataframe(calib_buckets)

    clv_seg = repo.read_df(
        f"select * from experiment_clv_segments where sweep_id = '{selected_sweep_id}' order by run_id, market, league"
    )
    st.markdown("**CLV by market/league (selected sweep)**")
    st.dataframe(clv_seg)

    value_hit = repo.read_df(
        f"select * from experiment_value_flag_hit_rate where sweep_id = '{selected_sweep_id}' order by run_id, bets desc"
    )
    st.markdown("**Value-flag hit rate (selected sweep)**")
    st.dataframe(value_hit)

    false_pos = repo.read_df(
        f"select * from experiment_false_positive_zones where sweep_id = '{selected_sweep_id}' order by false_positives desc limit 100"
    )
    st.markdown("**False-positive concentration zones (selected sweep)**")
    st.dataframe(false_pos)

repo.close()
