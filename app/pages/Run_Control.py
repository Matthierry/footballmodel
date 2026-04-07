from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository
from footballmodel.ui_dashboard import load_core_data, require_password_gate, today_scope

require_password_gate()
st.title("Run Pipeline")
st.caption("Operational run control for the snapshot workflow (predict -> snapshot -> review).")

cfg = load_app_config("config/runtime.yaml")
active_name, active_cfg = cfg.resolve_live_config()

repo = DuckRepository()
data = load_core_data(repo)
review = data["review"]
runs = data["runs"]

latest = runs.row(0, named=True) if runs.height else {}
today = today_scope(review)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest run status", "OK" if runs.height else "No runs")
m2.metric("Last successful run", str(latest.get("run_timestamp_utc", "N/A")))
m3.metric("Today's fixtures", today.select("fixture_id").n_unique() if today.height and "fixture_id" in today.columns else 0)
m4.metric("Today's value selections", today.filter(today["value_flag"] == True).height if today.height and "value_flag" in today.columns else 0)

st.caption(f"Latest run id/config: {latest.get('live_run_id', 'N/A')} | {latest.get('config_name', active_name)} {latest.get('config_version', active_cfg.version)}")

if st.button("Manual run control (coming soon)"):
    st.info("Manual trigger is not wired in this UI yet. Use scripts/run_pipeline.py until button execution is enabled.")

if not runs.is_empty():
    st.subheader("Recent run log")
    cols = [c for c in ["run_timestamp_utc", "live_run_id", "config_name", "config_version", "fixtures_scored", "market_predictions", "review_rows"] if c in runs.columns]
    st.dataframe(runs.select(cols).head(25), use_container_width=True)

if today.is_empty():
    st.info("No fixtures were eligible today. Check upstream ingestion and benchmark snapshot availability.")

if not review.is_empty() and "run_timestamp_utc" in review.columns and "value_flag" in review.columns:
    stale = review.filter((pl.col("match_date") == date.today()) & pl.col("value_flag").is_null()) if "match_date" in review.columns else review.head(0)
    if stale.height:
        st.warning("Some rows are assessed without value classification. Verify benchmark capture and threshold settings.")

repo.close()
