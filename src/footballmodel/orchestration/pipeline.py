from __future__ import annotations

from datetime import datetime

import polars as pl

from footballmodel.config.settings import AppConfig
from footballmodel.ingestion.clubelo import elo_as_of
from footballmodel.markets.benchmark import resolve_benchmark_price
from footballmodel.markets.derivation import derive_ah, derive_correct_score_top5, matrix_to_market_table
from footballmodel.markets.value import attach_value_flags
from footballmodel.model.blending import SubModelSignal, blend_signals, elo_to_goal_prior
from footballmodel.model.score_engine import GoalModelInputs, UnifiedScoreEngine


def default_dc_signal(history: pl.DataFrame, fixture: dict) -> SubModelSignal:
    home_avg = history.filter(pl.col("home_team") == fixture["home_team"]).select(pl.mean("home_goals")).item() or 1.3
    away_avg = history.filter(pl.col("away_team") == fixture["away_team"]).select(pl.mean("away_goals")).item() or 1.1
    return SubModelSignal(float(home_avg), float(away_avg))


def shot_signal(history: pl.DataFrame, fixture: dict) -> SubModelSignal:
    h = history.filter(pl.col("home_team") == fixture["home_team"]).select(pl.mean("home_sot")).item() or 4.5
    a = history.filter(pl.col("away_team") == fixture["away_team"]).select(pl.mean("away_sot")).item() or 4.0
    return SubModelSignal(0.2 + h * 0.22, 0.2 + a * 0.22)


def run_fixture_prediction(history: pl.DataFrame, fixture: dict, elo_history: pl.DataFrame, cfg: AppConfig) -> dict:
    dc = default_dc_signal(history, fixture)
    h_elo = elo_as_of(elo_history, fixture["home_team"], str(fixture["match_date"]))
    a_elo = elo_as_of(elo_history, fixture["away_team"], str(fixture["match_date"]))
    elo = elo_to_goal_prior(h_elo, a_elo)
    shot = shot_signal(history, fixture)

    blended = blend_signals(dc, elo, shot, cfg.weights.dixon_coles, cfg.weights.elo_prior, cfg.weights.shot_adjustment)
    engine = UnifiedScoreEngine(max_goals=cfg.runtime.max_goals)
    matrix = engine.score_matrix(GoalModelInputs(home_xg=blended.home_xg, away_xg=blended.away_xg))

    market_df = matrix_to_market_table(fixture["fixture_id"], matrix)

    market_rows = []
    for row in market_df.to_dicts():
        benchmark = resolve_benchmark_price(fixture, market=str(row["market"]), outcome=str(row["outcome"]))
        row["current_price"] = benchmark.current_price
        row["benchmark_source"] = benchmark.benchmark_source
        market_rows.append(row)

    market_df = attach_value_flags(
        pl.DataFrame(market_rows),
        cfg.runtime.value_edge_threshold,
        cfg.runtime.credibility_threshold,
    )

    return {
        "fixture_id": fixture["fixture_id"],
        "timestamp_utc": datetime.utcnow().isoformat(),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "expected_home_goals": blended.home_xg,
        "expected_away_goals": blended.away_xg,
        "score_matrix": matrix.tolist(),
        "markets": market_df.to_dicts(),
        "correct_score_top5": derive_correct_score_top5(matrix),
        "asian_handicap": derive_ah(matrix),
    }
