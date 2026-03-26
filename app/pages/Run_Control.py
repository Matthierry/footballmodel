from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import streamlit as st

from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository

st.title("Live Monitoring / Model Review")

cfg = load_app_config("config/runtime.yaml")
active_name, active_cfg = cfg.resolve_live_config()
st.caption(f"Default scheduled config: **{active_name}** (version **{active_cfg.version}**)" )

repo = DuckRepository()
try:
    review = repo.read_df("select * from live_review_history order by run_timestamp_utc desc")
    runs = repo.read_df("select * from live_run_summaries_history order by run_timestamp_utc desc")
except Exception:
    review = pl.DataFrame([])
    runs = pl.DataFrame([])

if review.is_empty():
    st.info("No live monitoring rows yet. Run the pipeline to populate live_model_review.")
else:
    for col in ("match_date", "run_timestamp_utc"):
        if col in review.columns:
            cast_type = pl.Date if col == "match_date" else pl.Datetime
            review = review.with_columns(pl.col(col).cast(cast_type, strict=False))

    min_day = review["match_date"].min() or date.today()
    max_day = review["match_date"].max() or date.today()

    c1, c2, c3 = st.columns(3)
    with c1:
        from_day = st.date_input("From", value=min_day, min_value=min_day, max_value=max_day)
    with c2:
        to_day = st.date_input("To", value=max_day, min_value=min_day, max_value=max_day)
    with c3:
        benchmark_sources = sorted([x for x in review["prediction_benchmark_source"].drop_nulls().unique().to_list()])
        selected_sources = st.multiselect("Benchmark source", options=benchmark_sources, default=benchmark_sources)

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        leagues = sorted(review["league"].drop_nulls().unique().to_list())
        selected_leagues = st.multiselect("League", options=leagues, default=leagues)
    with f2:
        markets = sorted(review["market"].drop_nulls().unique().to_list())
        selected_markets = st.multiselect("Market", options=markets, default=markets)
    with f3:
        config_versions = sorted(review["config_version"].drop_nulls().unique().to_list())
        selected_versions = st.multiselect("Config version", options=config_versions, default=config_versions)
    with f4:
        settlement = st.selectbox("Settlement", options=["all", "pending", "settled"], index=0)
    with f5:
        value_mode = st.selectbox("Value filter", options=["all", "value", "non_value"], index=0)

    filtered = review.filter((pl.col("match_date") >= pl.lit(from_day)) & (pl.col("match_date") <= pl.lit(to_day)))
    if selected_sources:
        filtered = filtered.filter(pl.col("prediction_benchmark_source").is_in(selected_sources))
    if selected_leagues:
        filtered = filtered.filter(pl.col("league").is_in(selected_leagues))
    if selected_markets:
        filtered = filtered.filter(pl.col("market").is_in(selected_markets))
    if selected_versions:
        filtered = filtered.filter(pl.col("config_version").is_in(selected_versions))
    if settlement != "all":
        filtered = filtered.filter(pl.col("settlement_status") == settlement)
    if value_mode == "value":
        filtered = filtered.filter(pl.col("value_flag") == True)
    elif value_mode == "non_value":
        filtered = filtered.filter((pl.col("value_flag") == False) | pl.col("value_flag").is_null())

    last_success = runs["run_timestamp_utc"][0] if not runs.is_empty() and "run_timestamp_utc" in runs.columns else "N/A"
    active_cfg_name = runs["config_name"][0] if not runs.is_empty() and "config_name" in runs.columns else active_name
    active_cfg_version = runs["config_version"][0] if not runs.is_empty() and "config_version" in runs.columns else active_cfg.version

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active config", f"{active_cfg_name} ({active_cfg_version})")
    m2.metric("Last successful run", str(last_success))
    m3.metric("Pending selections", filtered.filter(pl.col("settlement_status") == "pending").height)
    m4.metric("Settled selections", filtered.filter(pl.col("settlement_status") == "settled").height)

    today_rows = filtered.filter(pl.col("match_date") == pl.lit(date.today()))
    st.subheader("Today's predicted fixtures")
    st.dataframe(today_rows.select(["match_date", "league", "fixture_id", "home_team", "away_team", "market", "outcome", "value_flag"]))

    settled = filtered.filter(pl.col("settlement_status") == "settled")
    recent_clv = settled.filter(pl.col("clv").is_not_null())
    value_rows = settled.filter(pl.col("value_flag") == True)
    value_hit_rate = (
        value_rows.filter(pl.col("result_status") == "won").height / value_rows.height
        if value_rows.height
        else 0.0
    )

    s1, s2 = st.columns(2)
    s1.metric("Recent avg CLV", round(float(recent_clv["clv"].mean()), 4) if recent_clv.height else 0.0)
    s2.metric("Recent value-flag hit rate", round(value_hit_rate, 4))

    st.subheader("7-day / 14-day / 30-day summary")
    if "run_timestamp_utc" in filtered.columns:
        as_of = filtered["run_timestamp_utc"].max()
        window_rows = []
        for days in (7, 14, 30):
            since = as_of - timedelta(days=days)
            scoped = filtered.filter(pl.col("run_timestamp_utc") >= since)
            settled_scoped = scoped.filter(pl.col("settlement_status") == "settled")
            value_scoped = settled_scoped.filter(pl.col("value_flag") == True)
            window_rows.append(
                {
                    "window_days": days,
                    "rows": scoped.height,
                    "settled_rows": settled_scoped.height,
                    "avg_clv": float(settled_scoped["clv"].mean()) if settled_scoped.height else None,
                    "value_hit_rate": (
                        value_scoped.filter(pl.col("result_status") == "won").height / value_scoped.height
                        if value_scoped.height
                        else None
                    ),
                    "benchmark_coverage": (
                        scoped.filter(pl.col("later_benchmark_price").is_not_null()).height / scoped.height
                        if scoped.height
                        else None
                    ),
                }
            )
        st.dataframe(pl.DataFrame(window_rows))

    st.subheader("CLV trend over time")
    clv_trend = (
        filtered.filter(pl.col("settlement_status") == "settled")
        .filter(pl.col("clv").is_not_null())
        .with_columns(pl.col("run_timestamp_utc").dt.date().alias("run_day"))
        .group_by("run_day")
        .agg(pl.len().alias("rows"), pl.col("clv").mean().alias("avg_clv"))
        .sort("run_day")
    )
    st.dataframe(clv_trend)

    st.subheader("Value-flag hit-rate trend")
    value_trend = (
        filtered.filter(pl.col("settlement_status") == "settled")
        .filter(pl.col("value_flag") == True)
        .with_columns(pl.col("run_timestamp_utc").dt.date().alias("run_day"))
        .group_by("run_day")
        .agg(
            pl.len().alias("value_rows"),
            (pl.col("result_status") == "won").sum().alias("wins"),
        )
        .with_columns((pl.col("wins") / pl.col("value_rows")).alias("value_hit_rate"))
        .sort("run_day")
    )
    st.dataframe(value_trend)

    st.subheader("Strongest / weakest leagues and markets over time")
    perf = (
        settled.filter(pl.col("clv").is_not_null())
        .with_columns(pl.col("run_timestamp_utc").dt.date().alias("run_day"))
        .group_by(["run_day", "league", "market"])
        .agg(pl.col("clv").mean().alias("avg_clv"), pl.len().alias("samples"))
        .sort("avg_clv", descending=True)
    )
    st.dataframe(perf)

    st.subheader("Benchmark coverage trend")
    coverage = filtered.with_columns(pl.col("run_timestamp_utc").dt.date().alias("run_day")).group_by(
        ["run_day", "prediction_benchmark_source"]
    ).agg(
        pl.len().alias("rows"),
        pl.col("later_benchmark_price").is_not_null().sum().alias("has_later_snapshot"),
        pl.col("later_benchmark_price").is_null().sum().alias("missing_later_snapshot"),
    )
    st.dataframe(coverage)

    st.subheader("Source / data health summary")
    health = filtered.group_by(["prediction_benchmark_source", "settlement_status"]).agg(
        pl.len().alias("rows"),
        pl.col("prediction_benchmark_price").is_null().sum().alias("missing_prediction_price"),
        pl.col("later_benchmark_price").is_null().sum().alias("missing_later_price"),
    )
    st.dataframe(health)

    st.subheader("Config-version comparison over time")
    config_compare = (
        filtered.with_columns(pl.col("run_timestamp_utc").dt.date().alias("run_day"))
        .group_by(["run_day", "config_name", "config_version"])
        .agg(
            pl.len().alias("rows"),
            pl.col("clv").mean().alias("avg_clv"),
            (pl.col("value_flag") == True).sum().alias("value_rows"),
            pl.col("later_benchmark_price").is_not_null().sum().alias("rows_with_later_snapshot"),
        )
        .sort(["run_day", "config_name", "config_version"])
    )
    st.dataframe(config_compare)

    st.subheader("Filtered live review rows")
    st.dataframe(filtered)

repo.close()
