from __future__ import annotations

import polars as pl
import streamlit as st

from footballmodel.storage.repository import DuckRepository

st.title("History")
st.caption("Audit benchmark snapshots and prediction-time vs later benchmark movements.")

repo = DuckRepository()
snapshots = repo.read_table_or_empty("benchmark_snapshots", order_by="snapshot_timestamp_utc desc")

if snapshots.is_empty():
    st.info("No benchmark snapshots persisted yet.")
else:
    fixture_ids = snapshots["fixture_id"].unique().to_list()
    selected_fixture = st.selectbox("Fixture ID", fixture_ids, index=0 if fixture_ids else None)
    scoped = snapshots.filter(pl.col("fixture_id") == selected_fixture)
    st.subheader("Snapshots")
    st.dataframe(scoped.sort(["market", "outcome", "snapshot_timestamp_utc"]))

    st.subheader("Latest benchmark by market/outcome")
    latest = (
        scoped.sort("snapshot_timestamp_utc")
        .group_by(["market", "outcome", "line"])
        .agg(
            pl.col("benchmark_price").last().alias("latest_benchmark_price"),
            pl.col("benchmark_source").last().alias("latest_benchmark_source"),
            pl.col("snapshot_type").last().alias("latest_snapshot_type"),
            pl.col("snapshot_timestamp_utc").last().alias("latest_snapshot_timestamp_utc"),
        )
        .sort(["market", "outcome"])
    )
    st.dataframe(latest)

repo.close()
