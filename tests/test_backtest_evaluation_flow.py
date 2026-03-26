from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.backtest.walkforward import BacktestRequest, persist_backtest, run_backtest
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository


def test_backtest_generates_predictions_and_metrics(sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2024, 1, 1)],
            "team": ["A"],
            "country": ["ENG"],
            "elo": [1600.0],
        }
    )

    req = BacktestRequest(
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 20),
        leagues=["ENG1"],
        seasons=["2024/2025"],
        stake=1.0,
    )

    run_id, predictions, metrics = run_backtest(sample_matches, elo_history, cfg, req)

    assert run_id.startswith("bt_")
    assert predictions.height > 0
    assert {"1X2", "OU25", "BTTS", "AH"}.issubset(set(predictions["market"].unique().to_list()))
    assert {"overall", "market", "league", "edge_bucket", "benchmark_source", "value_flag", "value_status"}.issubset(
        set(metrics["breakdown"].unique().to_list())
    )


def test_backtest_persistence_writes_auditable_outputs(tmp_path, sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    req = BacktestRequest(start_date=date(2024, 8, 1), end_date=date(2024, 8, 20), leagues=["ENG1"])
    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2024, 1, 1)],
            "team": ["A"],
            "country": ["ENG"],
            "elo": [1600.0],
        }
    )

    run_id, predictions, metrics = run_backtest(sample_matches, elo_history, cfg, req)

    repo = DuckRepository(str(tmp_path / "backtest.duckdb"))
    persist_backtest(repo, req, run_id, predictions, metrics)

    stored_runs = repo.read_df("select * from backtest_runs")
    stored_predictions = repo.read_df("select * from backtest_predictions")
    stored_metrics = repo.read_df("select * from backtest_metrics")
    repo.close()

    assert stored_runs.height == 1
    assert stored_predictions.filter(pl.col("run_id") == run_id).height == predictions.height
    assert stored_metrics.filter(pl.col("run_id") == run_id).height == metrics.height
