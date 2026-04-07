from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import (
    apply_premium_dark_theme,
    dark_dataframe,
    load_core_data,
    render_dark_bar_comparison,
    render_empty_state,
    render_metric_card,
    require_password_gate,
    status_badge,
)

apply_premium_dark_theme("Fixture Detail", "🎯")
require_password_gate()
st.title("Fixture Detail")
st.caption("Premium deep-dive on one fixture: real model maths vs market benchmarks.")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _fmt_odds(value: float | None, missing: str = "N/A") -> str:
    if value is None:
        return missing
    return f"{value:.3f}"


def _safe_implied(price_col: str) -> pl.Expr:
    return pl.when(pl.col(price_col).is_not_null() & (pl.col(price_col) > 0)).then(1 / pl.col(price_col)).otherwise(None)


repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
snapshots = data["snapshots"]
runs = data["runs"]
live_predictions = data["live_predictions"]
model_predictions = data["model_predictions"]
model_runs = data["model_runs"]

base = live_predictions if not live_predictions.is_empty() else model_predictions

if base.is_empty() or "fixture_id" not in base.columns:
    render_empty_state("No fixture-level prediction rows are available yet.")
else:
    fixtures = (
        review.select([c for c in ["fixture_id", "home_team", "away_team", "match_date", "league"] if c in review.columns])
        .unique(subset=["fixture_id"]) if not review.is_empty() else pl.DataFrame([])
    )
    if fixtures.is_empty():
        fixtures = base.select([c for c in ["fixture_id"] if c in base.columns]).unique(subset=["fixture_id"]).with_columns(
            pl.lit("?").alias("home_team"),
            pl.lit("?").alias("away_team"),
            pl.lit(None).cast(pl.Date).alias("match_date"),
            pl.lit("N/A").alias("league"),
        )
    fixtures = fixtures.with_columns((pl.col("home_team").fill_null("?") + pl.lit(" vs ") + pl.col("away_team").fill_null("?")).alias("fixture")).sort("match_date", descending=True, nulls_last=True)

    labels = [f"{r.get('fixture', 'N/A')} ({r.get('league', 'N/A')} · {r.get('match_date', 'N/A')})" for r in fixtures.to_dicts()]
    idx = st.selectbox("Fixture", options=list(range(len(labels))), format_func=lambda x: labels[x])
    fixture_row = fixtures.row(idx, named=True)
    fixture_id = fixture_row["fixture_id"]

    scoped = base.filter(pl.col("fixture_id") == fixture_id)
    if "run_timestamp_utc" in scoped.columns:
        latest_ts = scoped.select(pl.col("run_timestamp_utc").max().alias("run_timestamp_utc")).row(0, named=True).get("run_timestamp_utc")
        scoped = scoped.filter(pl.col("run_timestamp_utc") == latest_ts)

    optional_schema: list[tuple[str, pl.DataType, object]] = [
        ("market", pl.Utf8, "Unavailable"),
        ("outcome", pl.Utf8, "Unavailable"),
        ("raw_probability", pl.Float64, None),
        ("calibrated_probability", pl.Float64, None),
        ("model_fair_odds", pl.Float64, None),
        ("current_price", pl.Float64, None),
        ("edge", pl.Float64, None),
        ("value_flag", pl.Boolean, False),
        ("line", pl.Float64, None),
    ]
    for col, dtype, default in optional_schema:
        if col not in scoped.columns:
            scoped = scoped.with_columns(pl.lit(default).cast(dtype, strict=False).alias(col))

    scoped = scoped.with_columns(
        pl.col("raw_probability").cast(pl.Float64, strict=False),
        pl.col("calibrated_probability").cast(pl.Float64, strict=False),
        pl.col("model_fair_odds").cast(pl.Float64, strict=False),
        pl.col("current_price").cast(pl.Float64, strict=False),
        pl.col("edge").cast(pl.Float64, strict=False),
        pl.col("value_flag").fill_null(False).cast(pl.Boolean),
        _safe_implied("current_price").alias("market_implied_probability"),
    )

    probability_col = "calibrated_probability" if "calibrated_probability" in scoped.columns else "raw_probability"
    scoped = scoped.with_columns(pl.col(probability_col).cast(pl.Float64, strict=False).alias("model_probability"))
    has_market_level_detail = (
        ("market" in base.columns)
        and ("outcome" in base.columns)
        and ("raw_probability" in base.columns or "calibrated_probability" in base.columns)
    )

    latest_run = runs.row(0, named=True) if runs.height else {}

    st.subheader("Fixture hero")
    h1, h2, h3, h4, h5 = st.columns(5)
    with h1:
        render_metric_card("Fixture", fixture_row.get("fixture", "N/A"), tone="info")
    with h2:
        render_metric_card("Kickoff", str(fixture_row.get("match_date", "N/A")), tone="info")
    with h3:
        render_metric_card("League", str(fixture_row.get("league", "N/A")), tone="info")
    with h4:
        run_context = f"{latest_run.get('config_name', 'N/A')} · {latest_run.get('config_version', 'N/A')}"
        render_metric_card("Run / config", run_context, tone="info", delta=str(scoped.select(pl.col("run_timestamp_utc").max()).item() if "run_timestamp_utc" in scoped.columns and scoped.height else "N/A"))
    with h5:
        value_rows = scoped.filter(pl.col("value_flag") & pl.col("edge").is_not_null()).height
        render_metric_card("Value signals", value_rows, tone="value" if value_rows else "warn")

    best = scoped.filter(pl.col("edge").is_not_null()).sort("edge", descending=True, nulls_last=True).head(1)
    st.subheader("Best opportunity")
    if best.is_empty():
        render_empty_state("No clear edge is available for this fixture yet. Markets may be pending benchmark capture.")
    else:
        row = best.row(0, named=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Market / Outcome", f"{row.get('market', 'N/A')} · {row.get('outcome', 'N/A')}")
        b2.metric("Edge", _fmt_pct(row.get("edge")))
        b3.metric("Model probability", _fmt_pct(row.get("model_probability")))
        b4.metric("Market implied", _fmt_pct(row.get("market_implied_probability")))
        st.markdown(
            f"{status_badge('Value' if row.get('value_flag') and row.get('edge') is not None else 'Assessed')} &nbsp; "
            f"Model fair odds: **{_fmt_odds(row.get('model_fair_odds'))}** &nbsp;|&nbsp; "
            f"Market odds: **{_fmt_odds(row.get('current_price'), 'Missing benchmark')}**",
            unsafe_allow_html=True,
        )
        st.caption("Standout rationale: this market has the strongest model-vs-market gap on currently available benchmark prices.")

    st.subheader("Probability distributions")
    if not has_market_level_detail:
        render_empty_state(
            "Detailed market probability data is unavailable in this dataset view. "
            "Fixture-level context is shown, but model-vs-market distributions require market-level prediction rows."
        )

    if has_market_level_detail:
        def _market_block(market: str, title: str) -> None:
            block = scoped.filter(pl.col("market") == market)
            if block.is_empty():
                render_empty_state(f"{title} unavailable for this fixture.")
                return
            chart_df = block.select([
                pl.col("outcome").cast(pl.Utf8),
                pl.col("model_probability"),
                pl.col("market_implied_probability").alias("market_probability"),
            ])
            render_dark_bar_comparison(chart_df, title=title)
            table = block.select([
                pl.col("outcome"),
                pl.col("model_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%").alias("model_probability"),
                pl.when(pl.col("market_implied_probability").is_null()).then(pl.lit("Missing benchmark")).otherwise(pl.col("market_implied_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("market_implied_probability"),
                pl.when(pl.col("model_fair_odds").is_null()).then(pl.lit("N/A")).otherwise(pl.col("model_fair_odds").round(3).cast(pl.Utf8)).alias("model_fair_odds"),
                pl.when(pl.col("current_price").is_null()).then(pl.lit("Missing benchmark")).otherwise(pl.col("current_price").round(3).cast(pl.Utf8)).alias("market_odds"),
                pl.when(pl.col("edge").is_null()).then(pl.lit("Unavailable")).otherwise(pl.col("edge").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("edge"),
                pl.when(pl.col("value_flag") & pl.col("edge").is_not_null()).then(pl.lit("Value")).otherwise(pl.lit("Assessed")).alias("value_status"),
            ])
            dark_dataframe(table)

        c1, c2 = st.columns(2)
        with c1:
            _market_block("1X2", "1X2 model vs market")
            _market_block("Over/Under 2.5", "Over/Under 2.5 model vs market")
        with c2:
            _market_block("BTTS", "BTTS model vs market")
            st.markdown("**Asian Handicap**")
            ah = scoped.filter(pl.col("market") == "Asian Handicap")
            if ah.is_empty():
                render_empty_state("Asian Handicap unavailable for this fixture.")
            else:
                ah_table = ah.select([
                    pl.col("line"),
                    pl.col("outcome"),
                    pl.col("model_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%").alias("model_probability"),
                    pl.when(pl.col("market_implied_probability").is_null()).then(pl.lit("Missing benchmark")).otherwise(pl.col("market_implied_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("market_implied_probability"),
                    pl.when(pl.col("edge").is_null()).then(pl.lit("Unavailable")).otherwise(pl.col("edge").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("edge"),
                    pl.when(pl.col("value_flag") & pl.col("edge").is_not_null()).then(pl.lit("Value")).otherwise(pl.lit("Assessed")).alias("value_status"),
                ]).sort(["line", "edge"], descending=[False, True], nulls_last=True)
                dark_dataframe(ah_table)

        st.subheader("Model vs market detail")
        detail = scoped.select([
            pl.col("market"),
            pl.col("line"),
            pl.col("outcome"),
            pl.col("model_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%").alias("model_probability"),
            pl.when(pl.col("market_implied_probability").is_null()).then(pl.lit("Missing benchmark")).otherwise(pl.col("market_implied_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("market_implied_probability"),
            pl.when(pl.col("model_fair_odds").is_null()).then(pl.lit("N/A")).otherwise(pl.col("model_fair_odds").round(3).cast(pl.Utf8)).alias("model_fair_odds"),
            pl.when(pl.col("current_price").is_null()).then(pl.lit("Missing benchmark")).otherwise(pl.col("current_price").round(3).cast(pl.Utf8)).alias("market_odds"),
            pl.when(pl.col("edge").is_null()).then(pl.lit("Unavailable")).otherwise(pl.col("edge").mul(100).round(2).cast(pl.Utf8) + pl.lit("%")).alias("edge"),
            pl.when(pl.col("current_price").is_null()).then(pl.lit("Missing benchmark")).when(pl.col("edge").is_null()).then(pl.lit("Unavailable")).when(pl.col("value_flag")).then(pl.lit("Value")).otherwise(pl.lit("Assessed")).alias("status"),
        ]).sort(["market", "edge"], descending=[False, True], nulls_last=True)
        dark_dataframe(detail)
    else:
        st.subheader("Model vs market detail")
        render_empty_state("Model-vs-market table unavailable without market-level prediction rows.")

    st.subheader("Correct score probability grid")
    cs = scoped.filter(pl.col("market") == "Correct Score") if has_market_level_detail else scoped.head(0)
    if cs.is_empty():
        render_empty_state("Correct Score probabilities are unavailable for this fixture.")
    else:
        cs_grid = cs.select([
            pl.col("outcome"),
            pl.col("model_probability").mul(100).round(2).cast(pl.Utf8) + pl.lit("%").alias("probability"),
        ]).sort("probability", descending=True)
        dark_dataframe(cs_grid)

    st.subheader("Technical & context details")
    team_context_cols = [
        c
        for c in [
            "expected_home_goals",
            "expected_away_goals",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "credibility",
            "confidence",
        ]
        if c in review.columns
    ]
    context = review.filter(pl.col("fixture_id") == fixture_id) if not review.is_empty() else review
    if context.is_empty() or not team_context_cols:
        if not model_runs.is_empty() and "fixture_id" in model_runs.columns:
            run_ctx = model_runs.filter(pl.col("fixture_id") == fixture_id)
            if not run_ctx.is_empty():
                dark_dataframe(run_ctx.select([c for c in ["run_timestamp_utc", "expected_home_goals", "expected_away_goals", "config_name", "config_version"] if c in run_ctx.columns]).head(1))
            else:
                render_empty_state("Expected goals, form, and confidence context are unavailable for this fixture.")
        else:
            render_empty_state("Expected goals, form, and confidence context are unavailable for this fixture.")
    else:
        dark_dataframe(context.select([c for c in ["market", "outcome", *team_context_cols] if c in context.columns]).head(20))

    st.subheader("Benchmarks captured")
    if snapshots.is_empty() or "fixture_id" not in snapshots.columns:
        render_empty_state("No benchmark snapshots captured yet for this fixture.")
    else:
        snap = snapshots.filter(pl.col("fixture_id") == fixture_id)
        if snap.is_empty():
            render_empty_state("No benchmark snapshots found for selected fixture.")
        else:
            dark_dataframe(snap.sort("snapshot_timestamp_utc", descending=True))

repo.close()
