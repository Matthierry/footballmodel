from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import (
    dark_dataframe,
    MARKET_ORDER,
    apply_premium_dark_theme,
    load_core_data,
    market_breakdown,
    prediction_display_table,
    render_empty_state,
    render_dark_bar_comparison,
    render_metric_card,
    require_password_gate,
    today_scope,
)

apply_premium_dark_theme("Dashboard", "🏁")
require_password_gate()

st.title("Dashboard")
st.caption("Command center for fixture deep-dives, value shortlist triage, pipeline checks, and performance review.")

repo: DuckRepository | None = None
try:
    repo = DuckRepository()
    data = load_core_data(repo)
    review = data["review"]
    runs = data["runs"]
    matches = data["matches"]

    latest_run = runs.row(0, named=True) if runs.height else {}
    today = today_scope(review)
    today_fixtures = today.select("fixture_id").n_unique() if "fixture_id" in today.columns and today.height else 0
    assessed_count = today.height
    value_count = today.filter(pl.col("status") == "Value").height if today.height else 0

    predictions_count = int(latest_run.get("market_predictions", 0) or 0)
    st.metric("Predictions in latest run", predictions_count)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Run status", "READY" if runs.height else "NO RUNS", tone="info", delta=str(latest_run.get("run_timestamp_utc", "N/A")))
    with k2:
        render_metric_card("Fixtures today", today_fixtures, tone="info")
    with k3:
        render_metric_card("Markets analysed", assessed_count, tone="info")
    with k4:
        render_metric_card("Value opportunities", value_count, tone="value" if value_count else "warn")

    left, right = st.columns(2)
    with left:
        st.subheader("Value opportunities by market")
        breakdown = market_breakdown(today)
        if breakdown.is_empty():
            render_empty_state("No eligible fixtures are in scope today yet.")
        else:
            ordered = breakdown.with_columns(
                pl.col("market").replace_strict({m: i for i, m in enumerate(MARKET_ORDER)}, default=999).alias("order")
            ).sort("order")
            chart_data = ordered.select(
                pl.col("market").alias("outcome"),
                (pl.col("value") / pl.col("assessed").clip(lower_bound=1)).cast(pl.Float64).alias("model_probability"),
                (pl.col("assessed") / pl.col("assessed").max().clip(lower_bound=1)).cast(pl.Float64).alias("market_probability"),
            )
            render_dark_bar_comparison(chart_data, title="Value rate vs assessment depth", category="outcome")
            dark_dataframe(ordered.select(["market", "value", "assessed", "missing_benchmark"]))

    with right:
        st.subheader("Top opportunities today")
        top = today.filter(pl.col("status") == "Value") if today.height else today.head(0)
        if top.is_empty():
            render_empty_state("No value selections flagged currently. Monitor run health and benchmark coverage.")
        else:
            dark_dataframe(
                prediction_display_table(top.sort("edge", descending=True, nulls_last=True)).head(8),
            )
        if top.is_empty() and today.height:
            st.caption("No value opportunities currently. Showing top assessed edges instead.")
            alt_top = today.sort("edge", descending=True, nulls_last=True).head(8)
            dark_dataframe(prediction_display_table(alt_top))

    b1, b2 = st.columns(2)
    with b1:
        st.subheader("Recent run trend")
        if runs.is_empty() or "run_timestamp_utc" not in runs.columns:
            render_empty_state("No run history available yet.")
        else:
            run_trend = runs.select([c for c in ["run_timestamp_utc", "fixtures_scored", "market_predictions", "review_rows"] if c in runs.columns]).head(12)
            if run_trend.is_empty() or run_trend.width < 2:
                render_empty_state("Insufficient run data for trend chart.")
            else:
                trend_pdf = run_trend.reverse().to_pandas().set_index("run_timestamp_utc")
                if hasattr(st, "line_chart"):
                    st.line_chart(trend_pdf)
                else:
                    dark_dataframe(run_trend)

    with b2:
        st.subheader("Quick actions")
        if hasattr(st, "page_link"):
            q1, q2 = st.columns(2)
            q1.page_link("pages/Fixture_Drilldown.py", label="🔎 Inspect Fixture", icon="🎯")
            q2.page_link("pages/Todays_Value_Bets.py", label="💡 Today Value List", icon="📈")
            q3, q4 = st.columns(2)
            q3.page_link("pages/Run_Control.py", label="⚙️ Run Pipeline", icon="🛠️")
            q4.page_link("pages/History.py", label="🧾 Review History", icon="📚")
        else:
            st.markdown("- Fixture Detail\n- Today's Value Bets\n- Run Pipeline\n- History")
        upcoming_fixtures = matches.filter(pl.col("match_date") >= pl.lit(date.today())).height if (not matches.is_empty() and "match_date" in matches.columns) else 0
        st.metric("Upcoming fixtures available", upcoming_fixtures)

    if predictions_count == 0:
        st.info("Latest run produced zero market predictions.")

    if matches.is_empty():
        st.info("No curated fixture dataset found yet. Run pipeline ingestion before relying on dashboard metrics.")
except Exception as exc:  # noqa: BLE001
    st.info(f"Dashboard fallback activated due to error: {exc}")
finally:
    if repo is not None:
        repo.close()
