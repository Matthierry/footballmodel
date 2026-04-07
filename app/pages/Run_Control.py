from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import load_core_data, require_password_gate, today_scope

require_password_gate()
st.title("Run Pipeline")
st.caption("Predict → snapshot benchmarks → review value and settlement readiness.")

cfg = load_app_config("config/runtime.yaml")
active_name, active_cfg = cfg.resolve_live_config()

repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
runs = data["runs"]

latest = runs.row(0, named=True) if runs.height else {}
today = today_scope(review)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Run status", "Ready" if runs.height else "No runs")
m2.metric("Latest run", str(latest.get("run_timestamp_utc", "N/A")))
m3.metric("In-scope fixtures", today.select("fixture_id").n_unique() if today.height and "fixture_id" in today.columns else 0)
m4.metric("Value selections", today.filter(pl.col("status") == "Value").height if today.height and "status" in today.columns else 0)

st.caption(f"Active config: {latest.get('config_name', active_name)} {latest.get('config_version', active_cfg.version)}")

st.markdown("**What a run does**")
st.caption("1) score fixtures, 2) capture benchmark snapshots, 3) write review rows for decision and later settlement.")
st.button("Manual run (disabled – use scripts/run_pipeline.py)", disabled=True)
st.caption("Manual trigger is intentionally disabled in UI until safe execution controls are wired.")

if not runs.is_empty():
    st.subheader("Recent run log")
    cols = [c for c in ["run_timestamp_utc", "live_run_id", "config_name", "config_version", "fixtures_scored", "market_predictions", "review_rows"] if c in runs.columns]
    st.dataframe(runs.select(cols).head(25), use_container_width=True)

if today.is_empty():
    st.info("No in-scope fixtures today. Check ingestion coverage and benchmark snapshot availability.")
else:
    value_today = today.filter(pl.col("status") == "Value").height
    if value_today == 0:
        st.info("No value selections today. Rows may be assessed but below threshold or missing benchmark.")

if int(latest.get("market_predictions", 0) or 0) == 0:
    st.warning("Latest run produced zero predictions.")

if not review.is_empty() and "run_timestamp_utc" in review.columns and "status" in review.columns:
    stale = review.filter((pl.col("match_date") == date.today()) & (pl.col("status") == "Unavailable")) if "match_date" in review.columns else review.head(0)
    if stale.height:
        st.warning("Some rows are unavailable due to missing edge inputs. Verify benchmark capture and model fair prices.")

repo.close()
