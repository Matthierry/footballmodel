from __future__ import annotations

from datetime import date, datetime, timedelta
from math import exp, log

import polars as pl

from footballmodel.config.settings import AppConfig
from footballmodel.ingestion.clubelo import elo_as_of
from footballmodel.markets.benchmark import resolve_benchmark_price
from footballmodel.markets.benchmark_snapshots import (
    SNAPSHOT_TYPE_PREDICTION_TIME,
    SNAPSHOT_TYPE_PRE_KICKOFF_LATEST,
    benchmark_snapshot_rows_from_fixture,
)
from footballmodel.markets.derivation import derive_ah, derive_correct_score_top5, matrix_to_market_table
from footballmodel.markets.value import attach_value_flags
from footballmodel.model.blending import SubModelSignal, blend_signals, elo_to_goal_prior
from footballmodel.model.score_engine import GoalModelInputs, UnifiedScoreEngine


def _weighted_team_signal(
    history: pl.DataFrame,
    fixture_date: date,
    team_col: str,
    team: str,
    value_col: str,
    lookback_days: int,
    half_life_days: int,
    default: float,
) -> float:
    if value_col not in history.columns:
        return default

    scoped = history.filter(pl.col(team_col) == team).filter(pl.col("match_date") < pl.lit(fixture_date))
    if lookback_days > 0:
        cutoff = fixture_date - timedelta(days=lookback_days)
        scoped = scoped.filter(pl.col("match_date") >= pl.lit(cutoff))
    if scoped.is_empty():
        return default

    weighted_total = 0.0
    total_weight = 0.0
    for row in scoped.select(["match_date", value_col]).iter_rows(named=True):
        if row[value_col] is None:
            continue
        age_days = max(0, (fixture_date - row["match_date"]).days)
        decay = exp(-log(2) * age_days / max(1, half_life_days))
        weighted_total += float(row[value_col]) * decay
        total_weight += decay
    if total_weight == 0:
        return default
    return weighted_total / total_weight


def default_dc_signal(history: pl.DataFrame, fixture: dict, cfg: AppConfig) -> SubModelSignal:
    fixture_date = fixture["match_date"]
    home_avg = _weighted_team_signal(
        history,
        fixture_date,
        "home_team",
        fixture["home_team"],
        "home_goals",
        cfg.lookback.team_form_days,
        cfg.half_life.team_form_days,
        1.3,
    )
    away_avg = _weighted_team_signal(
        history,
        fixture_date,
        "away_team",
        fixture["away_team"],
        "away_goals",
        cfg.lookback.team_form_days,
        cfg.half_life.team_form_days,
        1.1,
    )
    return SubModelSignal(float(home_avg), float(away_avg))


def shot_signal(history: pl.DataFrame, fixture: dict, cfg: AppConfig) -> SubModelSignal:
    fixture_date = fixture["match_date"]
    h = _weighted_team_signal(
        history,
        fixture_date,
        "home_team",
        fixture["home_team"],
        "home_sot",
        cfg.lookback.team_form_days,
        cfg.half_life.team_form_days,
        4.5,
    )
    a = _weighted_team_signal(
        history,
        fixture_date,
        "away_team",
        fixture["away_team"],
        "away_sot",
        cfg.lookback.team_form_days,
        cfg.half_life.team_form_days,
        4.0,
    )
    return SubModelSignal(0.2 + h * 0.22, 0.2 + a * 0.22)


def run_fixture_prediction(history: pl.DataFrame, fixture: dict, elo_history: pl.DataFrame, cfg: AppConfig) -> dict:
    dc = default_dc_signal(history, fixture, cfg)
    h_elo = elo_as_of(elo_history, fixture["home_team"], str(fixture["match_date"]))
    a_elo = elo_as_of(elo_history, fixture["away_team"], str(fixture["match_date"]))
    elo = elo_to_goal_prior(h_elo, a_elo)
    shot = shot_signal(history, fixture, cfg)

    blended = blend_signals(dc, elo, shot, cfg.weights.dixon_coles, cfg.weights.elo_prior, cfg.weights.shot_adjustment)
    engine = UnifiedScoreEngine(max_goals=cfg.runtime.max_goals)
    matrix = engine.score_matrix(GoalModelInputs(home_xg=blended.home_xg, away_xg=blended.away_xg))

    market_df = matrix_to_market_table(fixture["fixture_id"], matrix)

    prediction_timestamp = datetime.utcnow().isoformat()
    market_rows = []
    for row in market_df.to_dicts():
        benchmark = resolve_benchmark_price(fixture, market=str(row["market"]), outcome=str(row["outcome"]))
        row["current_price"] = benchmark.current_price
        row["benchmark_source"] = benchmark.benchmark_source
        row["benchmark_snapshot_type"] = SNAPSHOT_TYPE_PREDICTION_TIME
        row["benchmark_snapshot_timestamp_utc"] = prediction_timestamp
        market_rows.append(row)

    market_df = attach_value_flags(
        pl.DataFrame(market_rows),
        cfg.runtime.value_edge_threshold,
        cfg.runtime.credibility_threshold,
    )

    return {
        "fixture_id": fixture["fixture_id"],
        "timestamp_utc": prediction_timestamp,
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "expected_home_goals": blended.home_xg,
        "expected_away_goals": blended.away_xg,
        "score_matrix": matrix.tolist(),
        "markets": market_df.to_dicts(),
        "correct_score_top5": derive_correct_score_top5(matrix),
        "asian_handicap": derive_ah(matrix),
    }


def build_prediction_time_benchmark_snapshots(fixture: dict, prediction_timestamp_utc: str | None = None) -> pl.DataFrame:
    return benchmark_snapshot_rows_from_fixture(
        fixture=fixture,
        snapshot_type=SNAPSHOT_TYPE_PREDICTION_TIME,
        snapshot_timestamp_utc=prediction_timestamp_utc,
    )


def build_pre_kickoff_benchmark_snapshots(fixtures: pl.DataFrame, snapshot_timestamp_utc: str | None = None) -> pl.DataFrame:
    rows = [
        benchmark_snapshot_rows_from_fixture(
            fixture=fixture,
            snapshot_type=SNAPSHOT_TYPE_PRE_KICKOFF_LATEST,
            snapshot_timestamp_utc=snapshot_timestamp_utc,
        )
        for fixture in fixtures.iter_rows(named=True)
    ]
    if not rows:
        return pl.DataFrame([])
    return pl.concat(rows)
