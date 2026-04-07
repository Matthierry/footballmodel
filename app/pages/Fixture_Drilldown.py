from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import (
    MARKET_GROUPS,
    apply_premium_dark_theme,
    load_core_data,
    prediction_detail_table,
    render_empty_state,
    render_metric_card,
    require_password_gate,
    status_badge,
)

apply_premium_dark_theme("Fixture Detail", "🎯")
require_password_gate()
st.title("Fixture Detail")
st.caption("Premium deep-dive on one fixture: probabilities, edge signals, and benchmark context.")

repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
snapshots = data["snapshots"]
runs = data["runs"]

if review.is_empty() or "fixture_id" not in review.columns:
    render_empty_state("No fixture-level prediction rows are available yet.")
else:
    options = (
        review.with_columns((pl.col("home_team").fill_null("?") + pl.lit(" vs ") + pl.col("away_team").fill_null("?")).alias("fixture"))
        .select(["fixture_id", "fixture", "match_date", "league"])
        .unique(subset=["fixture_id"])
        .sort("match_date", descending=True)
    )
    labels = [f"{r['fixture']} ({r.get('league', 'NA')} · {r.get('match_date', 'NA')})" for r in options.to_dicts()]
    idx = st.selectbox("Fixture", options=list(range(len(labels))), format_func=lambda x: labels[x])
    row = options.row(idx, named=True)
    fixture_id = row["fixture_id"]

    scoped = review.filter(pl.col("fixture_id") == fixture_id)
    latest_run = runs.row(0, named=True) if runs.height else {}

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        render_metric_card("Fixture", row.get("fixture", "N/A"), tone="info")
    with h2:
        render_metric_card("Kickoff date", str(row.get("match_date", "N/A")), tone="info")
    with h3:
        render_metric_card("League", str(row.get("league", "N/A")), tone="info")
    with h4:
        value_rows = scoped.filter(pl.col("status") == "Value").height if "status" in scoped.columns else 0
        render_metric_card("Value signals", value_rows, tone="value" if value_rows else "warn", delta=str(latest_run.get("run_timestamp_utc", "N/A")))

    st.subheader("Probability distribution")
    probability_col = "prediction_probability" if "prediction_probability" in scoped.columns else "probability"

    def market_chart(market: str, title: str) -> None:
        block = scoped.filter(pl.col("market") == market) if "market" in scoped.columns else scoped.head(0)
        if block.is_empty() or probability_col not in block.columns:
            render_empty_state(f"{title}: unavailable for this fixture.")
            return
        chart_df = block.select(
            [
                "outcome",
                pl.col(probability_col).alias("Model probability"),
                (pl.when(pl.col("prediction_benchmark_price").is_not_null() & (pl.col("prediction_benchmark_price") > 0)).then(1 / pl.col("prediction_benchmark_price")).otherwise(None)).alias("Market implied"),
            ]
        ).to_pandas().set_index("outcome")
        st.markdown(f"**{title}**")
        st.bar_chart(chart_df)

    c1, c2 = st.columns(2)
    with c1:
        market_chart("1X2", "1X2")
        market_chart("Over/Under 2.5", "Over/Under 2.5")
    with c2:
        market_chart("BTTS", "BTTS")
        ah = scoped.filter(pl.col("market") == "Asian Handicap") if "market" in scoped.columns else scoped.head(0)
        st.markdown("**Asian Handicap**")
        if ah.is_empty():
            render_empty_state("Asian Handicap not available for this fixture.")
        else:
            ah_table = ah.select([c for c in ["outcome", "line", probability_col, "prediction_benchmark_price", "edge", "status"] if c in ah.columns])
            st.dataframe(ah_table.sort("edge", descending=True, nulls_last=True), use_container_width=True, hide_index=True)

    best = scoped.filter(pl.col("status") == "Value") if "status" in scoped.columns else scoped.head(0)
    if best.is_empty() and "edge" in scoped.columns:
        best = scoped.filter(pl.col("edge").is_not_null()).sort("edge", descending=True, nulls_last=True).head(1)
    else:
        best = best.sort("edge", descending=True, nulls_last=True).head(1)

    st.subheader("Best opportunity on this fixture")
    if best.is_empty():
        render_empty_state("No clear opportunity yet (missing benchmark, unavailable edge, or no assessed value).")
    else:
        b = best.row(0, named=True)
        o1, o2, o3 = st.columns(3)
        with o1:
            render_metric_card("Market / Outcome", f"{b.get('market', 'N/A')} · {b.get('outcome', 'N/A')}", tone="info")
        with o2:
            edge_txt = f"{round(float((b.get('edge') or 0.0) * 100), 2)}%" if b.get("edge") is not None else "N/A"
            render_metric_card("Edge", edge_txt, tone="value" if (b.get("edge") or 0) > 0 else "neg")
        with o3:
            cred = b.get("credibility") or b.get("confidence")
            render_metric_card("Confidence", f"{round(float(cred * 100), 0)}%" if cred is not None else "N/A", tone="info")

        st.markdown(
            f"{status_badge(b.get('status'))} &nbsp;"
            f"Market odds: **{b.get('prediction_benchmark_price') if b.get('prediction_benchmark_price') is not None else 'Missing benchmark'}** &nbsp;|&nbsp; "
            f"Model fair odds: **{b.get('model_fair_price') if b.get('model_fair_price') is not None else 'N/A'}**",
            unsafe_allow_html=True,
        )
        st.caption("Why it matters: a positive edge indicates the model estimates a better-than-market chance at the quoted odds.")

    st.subheader("Market context and odds vs model")
    compare_cols = [
        c
        for c in [
            "market",
            "outcome",
            probability_col,
            "prediction_benchmark_price",
            "model_fair_price",
            "edge",
            "status",
        ]
        if c in scoped.columns
    ]
    st.dataframe(scoped.select(compare_cols).sort(["market", "edge"], descending=[False, True], nulls_last=True), use_container_width=True, hide_index=True)

    st.subheader("Correct Score grid")
    cs = scoped.filter(pl.col("market") == "Correct Score") if "market" in scoped.columns else scoped.head(0)
    if cs.is_empty():
        render_empty_state("Correct score rows are unavailable for this fixture.")
    else:
        cs_cols = [c for c in ["outcome", probability_col, "prediction_benchmark_price", "model_fair_price", "edge", "status"] if c in cs.columns]
        st.dataframe(cs.select(cs_cols).sort("edge", descending=True, nulls_last=True), use_container_width=True, hide_index=True)

    st.subheader("Deeper fixture context")
    xg_cols = [c for c in ["expected_home_goals", "expected_away_goals", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target", "credibility", "confidence"] if c in scoped.columns]
    if xg_cols:
        st.dataframe(scoped.select([c for c in ["market", "outcome", *xg_cols, "status"] if c in scoped.columns]), use_container_width=True, hide_index=True)
    else:
        render_empty_state("Expected goals / form / shot context unavailable for this fixture.")

    st.markdown("### Technical market tables")
    for group in ["1X2", "Over/Under 2.5", "BTTS", "Asian Handicap"]:
        block = scoped.filter(pl.col("market") == group) if "market" in scoped.columns else scoped.head(0)
        st.markdown(f"**{group}**")
        if block.is_empty():
            st.caption("No assessed rows for this market.")
        else:
            st.dataframe(prediction_detail_table(block), use_container_width=True, hide_index=True)

    other = scoped.filter(~pl.col("market").is_in(list(MARKET_GROUPS.keys()))) if "market" in scoped.columns else scoped.head(0)
    if not other.is_empty():
        st.markdown("**Other markets**")
        st.dataframe(prediction_detail_table(other), use_container_width=True, hide_index=True)

    st.subheader("Benchmarks captured")
    if snapshots.is_empty() or "fixture_id" not in snapshots.columns:
        render_empty_state("No benchmark snapshots captured yet for this fixture.")
    else:
        snap = snapshots.filter(pl.col("fixture_id") == fixture_id)
        if snap.is_empty():
            render_empty_state("No benchmark snapshots found for selected fixture.")
        else:
            st.dataframe(snap.sort("snapshot_timestamp_utc", descending=True), use_container_width=True, hide_index=True)

repo.close()
