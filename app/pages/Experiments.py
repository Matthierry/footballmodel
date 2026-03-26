from __future__ import annotations

import streamlit as st

from footballmodel.storage.repository import DuckRepository

st.title("Experiments")
st.caption("Compare historical walk-forward runs and inspect summary metrics by breakdown.")

repo = DuckRepository()
try:
    runs = repo.read_df("select * from backtest_runs order by created_at desc")
except Exception:
    runs = None

if runs is None or runs.is_empty():
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
        st.subheader("Overall comparison")
        st.dataframe(
            summary.select(
                [
                    "run_id",
                    "log_loss",
                    "brier_score",
                    "calibration_error",
                    "avg_clv",
                    "share_beating_close",
                    "flat_stake_roi",
                    "strike_rate",
                    "max_drawdown",
                    "bets",
                ]
            )
        )

repo.close()
