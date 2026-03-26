from __future__ import annotations

from datetime import date

import polars as pl

from footballmodel.backtest.walkforward import (
    SweepRequest,
    build_champion_view,
    persist_sweep_results,
    rank_experiment_runs,
    run_experiment_sweep,
)
from footballmodel.config.settings import load_app_config
from footballmodel.storage.repository import DuckRepository


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
            "robustness_score": [0.9, 0.1],
            "flat_stake_roi": [0.01, 0.8],
        }
    )

    ranked = rank_experiment_runs(summary)
    assert ranked["run_id"][0] == "bt_a"


def test_build_champion_view_returns_champion_and_challengers():
    summary = pl.DataFrame(
        {
            "run_id": ["bt_a", "bt_b", "bt_c"],
            "calibrated_log_loss": [0.4, 0.45, 0.5],
            "calibrated_brier_score": [0.2, 0.21, 0.25],
            "calibrated_calibration_error": [0.05, 0.06, 0.08],
            "avg_clv": [0.01, 0.02, -0.01],
            "robustness_score": [0.8, 0.7, 0.2],
            "flat_stake_roi": [0.03, 0.01, 0.04],
        }
    )
    view = build_champion_view(summary)
    assert view.height == 3
    assert view.filter(pl.col("selection_role") == "champion").height == 1


def test_sweep_persists_champion_and_diagnostics(tmp_path, sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    elo_history = pl.DataFrame({"elo_date": [date(2024, 1, 1)], "team": ["A"], "country": ["ENG"], "elo": [1600.0]})
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
    repo = DuckRepository(str(tmp_path / "diag.duckdb"))
    persist_sweep_results(repo, sweep_id, summary, ranking)
    champion = repo.read_df("select * from experiment_champion_view")
    calibration_buckets = repo.read_df("select * from experiment_calibration_buckets")
    false_positive = repo.read_df("select * from experiment_false_positive_zones")
    metadata = repo.read_df("select * from experiment_sweep_metadata")
    repo.close()
    assert champion.height >= 1
    assert calibration_buckets.height >= 1
    assert false_positive.height >= 0
    assert metadata.height == 1


def test_historical_sweep_rows_are_queryable_by_selected_sweep(tmp_path, sample_matches: pl.DataFrame):
    cfg = load_app_config("config/runtime.yaml")
    elo_history = pl.DataFrame({"elo_date": [date(2024, 1, 1)], "team": ["A"], "country": ["ENG"], "elo": [1600.0]})
    req = SweepRequest(
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 20),
        leagues=["ENG1"],
        dixon_coles_weights=[0.5],
        elo_prior_weights=[0.2],
        shot_adjustment_weights=[0.2],
    )
    repo = DuckRepository(str(tmp_path / "history.duckdb"))
    sweep_a, summary_a, ranking_a = run_experiment_sweep(sample_matches, elo_history, cfg, req)
    persist_sweep_results(repo, sweep_a, summary_a, ranking_a)
    sweep_b, summary_b, ranking_b = run_experiment_sweep(sample_matches, elo_history, cfg, req)
    persist_sweep_results(repo, sweep_b, summary_b, ranking_b)

    selected = repo.read_df(f"select * from experiment_champion_view where sweep_id = '{sweep_a}'")
    latest = repo.read_df("select sweep_id from experiment_sweep_metadata order by created_at desc limit 1")
    repo.close()

    assert selected.height >= 1
    assert latest.height == 1
