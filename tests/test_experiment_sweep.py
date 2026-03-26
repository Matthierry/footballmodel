from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.backtest.walkforward import SweepRequest, rank_experiment_runs, run_experiment_sweep
from footballmodel.config.settings import load_app_config


def test_experiment_sweep_runs_and_ranks(sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    elo_history = pl.DataFrame(
        {
            "elo_date": [date(2024, 1, 1)],
            "team": ["A"],
            "country": ["ENG"],
            "elo": [1600.0],
        }
    )

    request = SweepRequest(
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 20),
        leagues=["ENG1"],
        seasons=["2024/2025"],
        dixon_coles_weights=[0.5, 0.6],
        elo_prior_weights=[0.2],
        shot_adjustment_weights=[0.2],
        value_edge_thresholds=[0.02],
        credibility_thresholds=[0.5],
        lookback_days_options=[90],
        half_life_days_options=[30],
    )

    sweep_id, summary, ranking = run_experiment_sweep(sample_matches, elo_history, cfg, request)

    assert sweep_id.startswith("sw_")
    assert summary.height == 2
    assert ranking.height == summary.height
    assert "ranking_score" in ranking.columns
    assert ranking["ranking_score"].to_list() == sorted(ranking["ranking_score"].to_list())


def test_rank_experiment_runs_prioritises_predictive_quality():
    summary = pl.DataFrame(
        {
            "run_id": ["bt_a", "bt_b"],
            "calibrated_log_loss": [0.4, 0.6],
            "calibrated_brier_score": [0.2, 0.1],
            "calibrated_calibration_error": [0.05, 0.04],
            "avg_clv": [0.01, 0.05],
        }
    )

    ranked = rank_experiment_runs(summary)
    assert ranked["run_id"][0] == "bt_a"
