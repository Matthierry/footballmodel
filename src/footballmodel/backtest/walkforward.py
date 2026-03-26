from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from math import log
from uuid import uuid4

import polars as pl

from footballmodel.config.settings import AppConfig
from footballmodel.markets.benchmark_snapshots import (
    SNAPSHOT_TYPE_CLOSING_SURROGATE,
    SNAPSHOT_TYPE_PREDICTION_TIME,
    benchmark_snapshot_rows_from_fixture,
    choose_later_snapshot,
)
from footballmodel.orchestration.pipeline import run_fixture_prediction
from footballmodel.storage.repository import DuckRepository


@dataclass(slots=True)
class BacktestRequest:
    start_date: date
    end_date: date
    leagues: list[str]
    seasons: list[str] | None = None
    stake: float = 1.0
    dixon_coles_weight: float | None = None
    elo_prior_weight: float | None = None
    shot_adjustment_weight: float | None = None
    value_edge_threshold: float | None = None
    credibility_threshold: float | None = None
    lookback_days: int | None = None
    half_life_days: int | None = None
    calibrate_probabilities: bool = True
    calibration_min_samples: int = 50


@dataclass(slots=True)
class SweepRequest:
    start_date: date
    end_date: date
    leagues: list[str]
    seasons: list[str] | None = None
    stake: float = 1.0
    dixon_coles_weights: list[float] | None = None
    elo_prior_weights: list[float] | None = None
    shot_adjustment_weights: list[float] | None = None
    value_edge_thresholds: list[float] | None = None
    credibility_thresholds: list[float] | None = None
    lookback_days_options: list[int] | None = None
    half_life_days_options: list[int] | None = None
    calibrate_probabilities: bool = True


def _season_label(match_date: date) -> str:
    start = match_date.year if match_date.month >= 7 else match_date.year - 1
    return f"{start}/{start + 1}"


def _close_price(fixture: dict, market: str, outcome: str) -> float | None:
    def _val(column: str) -> float | None:
        value = fixture.get(column)
        return float(value) if value is not None else None

    if market == "1X2":
        return _val({"home": "avg_home_odds", "draw": "avg_draw_odds", "away": "avg_away_odds"}[outcome])
    if market == "OU25":
        return _val({"over_2_5": "avg_over_2_5_odds", "under_2_5": "avg_under_2_5_odds"}[outcome])
    if market == "BTTS":
        return _val({"btts_yes": "avg_btts_yes_odds", "btts_no": "avg_btts_no_odds"}[outcome])
    if market == "AH":
        return _val("avg_ah_home_odds" if outcome.startswith("home_") else "avg_ah_away_odds")
    return None


def _line_for_market_outcome(market: str, outcome: str) -> float | None:
    if market == "OU25":
        return 2.5
    if market == "AH":
        try:
            return float(outcome.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            return None
    return None


def _build_row_snapshot_index(
    fixture: dict,
    market_row: dict[str, object],
    run_timestamp: str,
) -> pl.DataFrame:
    market = str(market_row["market"])
    outcome = str(market_row["outcome"])
    line = _line_for_market_outcome(market, outcome)
    pred_snapshot = pl.DataFrame(
        [
            {
                "fixture_id": fixture["fixture_id"],
                "market": market,
                "outcome": outcome,
                "line": line,
                "benchmark_price": market_row.get("current_price"),
                "benchmark_source": market_row.get("benchmark_source"),
                "snapshot_type": SNAPSHOT_TYPE_PREDICTION_TIME,
                "snapshot_timestamp_utc": run_timestamp,
            }
        ]
    ).with_columns(pl.col("line").cast(pl.Float64))
    surrogate_rows = benchmark_snapshot_rows_from_fixture(
        fixture=fixture,
        snapshot_type=SNAPSHOT_TYPE_CLOSING_SURROGATE,
        snapshot_timestamp_utc=run_timestamp,
    )
    surrogate = surrogate_rows.select(pred_snapshot.columns).with_columns(pl.col("line").cast(pl.Float64)).filter(
        (pl.col("market") == market)
        & (pl.col("outcome") == outcome)
        & (pl.col("line").is_null() if line is None else pl.col("line") == line)
    )
    return pl.concat([pred_snapshot, surrogate]) if not surrogate.is_empty() else pred_snapshot


def _actual_target(fixture: dict, market: str, outcome: str) -> tuple[int | None, bool]:
    home_goals = fixture.get("home_goals")
    away_goals = fixture.get("away_goals")
    if home_goals is None or away_goals is None:
        return None, False

    if market == "1X2":
        actual = "home" if home_goals > away_goals else "away" if home_goals < away_goals else "draw"
        return (1 if outcome == actual else 0), False

    if market == "OU25":
        actual = "over_2_5" if (home_goals + away_goals) > 2 else "under_2_5"
        return (1 if outcome == actual else 0), False

    if market == "BTTS":
        actual = "btts_yes" if (home_goals > 0 and away_goals > 0) else "btts_no"
        return (1 if outcome == actual else 0), False

    if market == "AH":
        line = float(outcome.split("_", maxsplit=1)[1])
        adjusted = (home_goals - away_goals) - line
        if adjusted == 0:
            return None, True
        actual = "home" if adjusted > 0 else "away"
        return (1 if outcome.startswith(actual) else 0), False

    return None, False


def _edge_bucket(edge: float | None) -> str:
    if edge is None:
        return "unknown"
    if edge < 0:
        return "negative"
    if edge < 0.02:
        return "0-2%"
    if edge < 0.05:
        return "2-5%"
    return ">=5%"


def _max_drawdown(pnl: list[float]) -> float:
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return max_dd


def _binary_log_loss(probability: float, target: int) -> float:
    p = min(1 - 1e-12, max(1e-12, float(probability)))
    return -(target * log(p) + (1 - target) * log(1 - p))


def _credibility_score(probability: float | None, edge: float | None) -> float:
    if probability is None or edge is None:
        return 0.0
    prob_anchor = min(1.0, probability / 0.5)
    edge_penalty = max(0.0, 1 - abs(edge) * 3)
    return float(max(0.0, min(1.0, 0.7 * prob_anchor + 0.3 * edge_penalty)))


def _revalue_with_calibration(
    probability: float,
    current_price: float | None,
    edge_threshold: float,
    credibility_threshold: float,
) -> tuple[float | None, float | None, bool, str]:
    if current_price is None or probability <= 0:
        return None, None, False, "missing_benchmark"
    fair_odds = 1.0 / max(probability, 1e-12)
    edge = current_price - fair_odds
    cred = _credibility_score(probability, edge)
    is_value = edge >= edge_threshold and cred >= credibility_threshold
    return fair_odds, edge, is_value, "assessed"


def _calibrated_probability(
    history_rows: list[dict[str, object]],
    market: str,
    outcome: str,
    raw_probability: float,
    min_samples: int,
) -> tuple[float, str]:
    eligible = [
        row
        for row in history_rows
        if row.get("market") == market and row.get("outcome") == outcome and row.get("target") is not None and not row.get("is_push", False)
    ]
    if len(eligible) < min_samples:
        return raw_probability, "identity"

    bins: dict[float, list[int]] = {}
    for row in eligible:
        key = float(min(0.9, max(0.0, (float(row["raw_probability"]) * 10) // 1 / 10)))
        bins.setdefault(key, []).append(int(row["target"]))

    bucket = float(min(0.9, max(0.0, (raw_probability * 10) // 1 / 10)))
    if bucket not in bins:
        return raw_probability, "identity"

    positives = sum(bins[bucket])
    count = len(bins[bucket])
    calibrated = (positives + 1) / (count + 2)
    return float(min(1 - 1e-6, max(1e-6, calibrated))), "bin_laplace"


def run_walkforward(matches: pl.DataFrame, predict_fn, req: BacktestRequest) -> pl.DataFrame:
    scoped = matches.filter(
        (pl.col("match_date") >= pl.lit(req.start_date))
        & (pl.col("match_date") <= pl.lit(req.end_date))
        & (pl.col("league").is_in(req.leagues))
    ).sort("match_date")

    rows: list[dict[str, object]] = []
    for fixture in scoped.iter_rows(named=True):
        hist = matches.filter(pl.col("match_date") < pl.lit(fixture["match_date"]))
        pred = predict_fn(hist, fixture)
        pred["fixture_id"] = fixture["fixture_id"]
        pred["match_date"] = fixture["match_date"]
        rows.append(pred)
    return pl.DataFrame(rows)


def run_backtest(matches: pl.DataFrame, elo_history: pl.DataFrame, cfg: AppConfig, req: BacktestRequest) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    run_id = f"bt_{uuid4().hex[:12]}"
    run_cfg = cfg.model_copy(deep=True)
    if req.dixon_coles_weight is not None:
        run_cfg.weights.dixon_coles = req.dixon_coles_weight
    if req.elo_prior_weight is not None:
        run_cfg.weights.elo_prior = req.elo_prior_weight
    if req.shot_adjustment_weight is not None:
        run_cfg.weights.shot_adjustment = req.shot_adjustment_weight
    if req.value_edge_threshold is not None:
        run_cfg.runtime.value_edge_threshold = req.value_edge_threshold
    if req.credibility_threshold is not None:
        run_cfg.runtime.credibility_threshold = req.credibility_threshold
    if req.lookback_days is not None:
        run_cfg.lookback.team_form_days = req.lookback_days
    if req.half_life_days is not None:
        run_cfg.half_life.team_form_days = req.half_life_days

    completed = matches.filter(pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null()).with_columns(
        pl.col("match_date").map_elements(_season_label, return_dtype=pl.Utf8).alias("season")
    )
    scoped = completed.filter(
        (pl.col("match_date") >= pl.lit(req.start_date))
        & (pl.col("match_date") <= pl.lit(req.end_date))
        & pl.col("league").is_in(req.leagues)
        & (pl.col("season").is_in(req.seasons) if req.seasons else pl.lit(True))
    ).sort("match_date")

    rows: list[dict[str, object]] = []
    for fixture in scoped.iter_rows(named=True):
        history = completed.filter(pl.col("match_date") < pl.lit(fixture["match_date"]))
        pred = run_fixture_prediction(history, fixture, elo_history, run_cfg)
        for market_row in pred["markets"]:
            target, is_push = _actual_target(fixture, str(market_row["market"]), str(market_row["outcome"]))
            snapshot_index = _build_row_snapshot_index(fixture, market_row, datetime.utcnow().isoformat())
            prediction_snapshot = snapshot_index.filter(pl.col("snapshot_type") == SNAPSHOT_TYPE_PREDICTION_TIME)
            prediction_snapshot_row = prediction_snapshot.row(0, named=True) if not prediction_snapshot.is_empty() else None
            later_snapshot_row = choose_later_snapshot(snapshot_index)

            current = prediction_snapshot_row["benchmark_price"] if prediction_snapshot_row else None
            close_price = later_snapshot_row["benchmark_price"] if later_snapshot_row else None
            clv = (1 / current - 1 / close_price) if (current is not None and close_price is not None) else None
            beat_close = clv is not None and clv > 0

            raw_probability = float(market_row["model_probability"])
            calibrated_probability = raw_probability
            calibration_method = "disabled"
            if req.calibrate_probabilities:
                calibrated_probability, calibration_method = _calibrated_probability(
                    rows,
                    str(market_row["market"]),
                    str(market_row["outcome"]),
                    raw_probability,
                    req.calibration_min_samples,
                )
            cal_fair_odds, cal_edge, cal_value_flag, cal_status = _revalue_with_calibration(
                calibrated_probability,
                current,
                run_cfg.runtime.value_edge_threshold,
                run_cfg.runtime.credibility_threshold,
            )

            implied_stake = req.stake if cal_value_flag else 0.0
            pnl = 0.0
            if implied_stake > 0 and current:
                if is_push:
                    pnl = 0.0
                elif target == 1:
                    pnl = implied_stake * (float(current) - 1.0)
                else:
                    pnl = -implied_stake

            rows.append(
                {
                    "run_id": run_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "fixture_id": fixture["fixture_id"],
                    "match_date": fixture["match_date"],
                    "season": fixture["season"],
                    "league": fixture["league"],
                    "home_team": fixture["home_team"],
                    "away_team": fixture["away_team"],
                    "market": market_row["market"],
                    "outcome": market_row["outcome"],
                    "raw_probability": raw_probability,
                    "model_probability": raw_probability,
                    "model_fair_odds": market_row["model_fair_odds"],
                    "calibrated_probability": calibrated_probability,
                    "calibrated_fair_odds": cal_fair_odds,
                    "calibration_method": calibration_method,
                    "current_price": current,
                    "close_price": close_price,
                    "benchmark_source": prediction_snapshot_row.get("benchmark_source") if prediction_snapshot_row else "unavailable",
                    "prediction_snapshot_type": SNAPSHOT_TYPE_PREDICTION_TIME,
                    "prediction_snapshot_timestamp_utc": prediction_snapshot_row.get("snapshot_timestamp_utc")
                    if prediction_snapshot_row
                    else None,
                    "later_snapshot_type": later_snapshot_row.get("snapshot_type") if later_snapshot_row else None,
                    "later_snapshot_source": later_snapshot_row.get("benchmark_source") if later_snapshot_row else "unavailable",
                    "later_snapshot_timestamp_utc": later_snapshot_row.get("snapshot_timestamp_utc") if later_snapshot_row else None,
                    "benchmark_available": bool(market_row.get("benchmark_available")),
                    "edge": market_row.get("edge"),
                    "calibrated_edge": cal_edge,
                    "edge_bucket": _edge_bucket(cal_edge),
                    "value_flag": bool(market_row.get("value_flag")),
                    "calibrated_value_flag": cal_value_flag,
                    "value_status": market_row.get("value_status"),
                    "calibrated_value_status": cal_status,
                    "credibility_score": market_row.get("credibility_score"),
                    "calibrated_credibility_score": _credibility_score(calibrated_probability, cal_edge),
                    "target": target,
                    "is_push": is_push,
                    "raw_log_loss_component": _binary_log_loss(raw_probability, target) if target is not None else None,
                    "calibrated_log_loss_component": _binary_log_loss(calibrated_probability, target) if target is not None else None,
                    "raw_brier_component": (raw_probability - float(target)) ** 2 if target is not None else None,
                    "calibrated_brier_component": (calibrated_probability - float(target)) ** 2 if target is not None else None,
                    "clv": clv,
                    "beat_close": beat_close,
                    "stake": implied_stake,
                    "pnl": pnl,
                }
            )

    predictions = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"run_id": pl.Utf8})
    metrics = compute_backtest_metrics(predictions)
    return run_id, predictions, metrics


def compute_backtest_metrics(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.is_empty():
        return pl.DataFrame([])

    dims: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("market", ["market"]),
        ("league", ["league"]),
        ("edge_bucket", ["edge_bucket"]),
        ("benchmark_source", ["benchmark_source"]),
        ("value_flag", ["calibrated_value_flag"]),
        ("value_status", ["calibrated_value_status"]),
    ]

    out_rows: list[dict[str, object]] = []
    grouped_fx = predictions.group_by(["run_id", "fixture_id", "market"]).agg(
        pl.col("league").first().alias("league"),
        pl.col("edge_bucket").first().alias("edge_bucket"),
        pl.col("benchmark_source").first().alias("benchmark_source"),
        pl.col("calibrated_value_flag").max().alias("calibrated_value_flag"),
        pl.col("calibrated_value_status").first().alias("calibrated_value_status"),
        pl.col("raw_log_loss_component").sum().alias("raw_log_loss"),
        pl.col("calibrated_log_loss_component").sum().alias("calibrated_log_loss"),
        pl.col("raw_brier_component").sum().alias("raw_brier_score"),
        pl.col("calibrated_brier_component").sum().alias("calibrated_brier_score"),
        pl.col("match_date").first().alias("match_date"),
    )

    for breakdown, keys in dims:
        frame = grouped_fx if keys else grouped_fx.with_columns(pl.lit("all").alias("group_key"))
        gb_keys = ["group_key"] if not keys else keys

        for key_data, sub in frame.group_by(gb_keys):
            key_tuple = key_data if isinstance(key_data, tuple) else (key_data,)
            key_val = "all" if not keys else "|".join(str(v) for v in key_tuple)

            row_subset = predictions.filter(
                pl.lit(True)
                if not keys
                else pl.all_horizontal([pl.col(k) == pl.lit(sub[k][0]) for k in keys])
            )
            raw_calib = _calibration_error(row_subset, "raw_probability")
            cal_calib = _calibration_error(row_subset, "calibrated_probability")
            market = row_subset.filter(pl.col("clv").is_not_null())
            avg_clv = market["clv"].mean() if market.height else None
            med_clv = market["clv"].median() if market.height else None
            share_beat = market["beat_close"].mean() if market.height else None
            share_exchange = market.filter(pl.col("benchmark_source") == "exchange").height / market.height if market.height else None
            share_market_avg = (
                market.filter(pl.col("benchmark_source") == "market_average").height / market.height if market.height else None
            )

            bets = row_subset.filter(pl.col("stake") > 0)
            settled = bets.filter(~pl.col("is_push"))
            total_stake = float(bets["stake"].sum()) if bets.height else 0.0
            total_pnl = float(bets["pnl"].sum()) if bets.height else 0.0
            roi = (total_pnl / total_stake) if total_stake > 0 else None
            strike = settled.filter(pl.col("pnl") > 0).height / settled.height if settled.height else None
            max_dd = _max_drawdown(bets.sort("match_date")["pnl"].to_list()) if bets.height else None

            out_rows.append(
                {
                    "run_id": row_subset["run_id"][0],
                    "breakdown": breakdown,
                    "group_key": key_val,
                    "fixtures": int(sub.height),
                    "raw_log_loss": sub["raw_log_loss"].mean(),
                    "calibrated_log_loss": sub["calibrated_log_loss"].mean(),
                    "raw_brier_score": sub["raw_brier_score"].mean(),
                    "calibrated_brier_score": sub["calibrated_brier_score"].mean(),
                    "calibration_error": cal_calib,
                    "raw_calibration_error": raw_calib,
                    "calibrated_calibration_error": cal_calib,
                    "avg_clv": avg_clv,
                    "median_clv": med_clv,
                    "share_beating_close": share_beat,
                    "benchmark_exchange_share": share_exchange,
                    "benchmark_market_average_share": share_market_avg,
                    "flat_stake_roi": roi,
                    "yield": roi,
                    "strike_rate": strike,
                    "max_drawdown": max_dd,
                    "bets": int(bets.height),
                }
            )

    return pl.DataFrame(out_rows)


def _calibration_error(rows: pl.DataFrame, probability_col: str) -> float | None:
    usable = rows.filter(pl.col("target").is_not_null())
    if usable.is_empty():
        return None

    bucketed = usable.with_columns(((pl.col(probability_col) * 10).floor() / 10).alias("prob_bucket"))
    total = bucketed.height
    ece = 0.0
    for _, bucket in bucketed.group_by("prob_bucket"):
        ece += abs(float(bucket["target"].mean()) - float(bucket[probability_col].mean())) * (bucket.height / total)
    return float(ece)


def _run_robustness_score(metrics: pl.DataFrame) -> float:
    segments = metrics.filter(pl.col("breakdown").is_in(["market", "league"]))
    if segments.is_empty():
        return 0.0
    stable = segments.filter(
        (pl.col("calibrated_log_loss").is_not_null())
        & (pl.col("calibrated_calibration_error").is_not_null())
        & (pl.col("calibrated_log_loss") <= segments["calibrated_log_loss"].median() * 1.25)
        & (pl.col("calibrated_calibration_error") <= segments["calibrated_calibration_error"].median() * 1.25)
    )
    return float(stable.height / segments.height)


def build_run_diagnostics(run_id: str, predictions: pl.DataFrame, metrics: pl.DataFrame) -> dict[str, pl.DataFrame]:
    if predictions.is_empty():
        empty = pl.DataFrame([])
        return {
            "segment_strength": empty,
            "raw_vs_calibrated": empty,
            "calibration_buckets": empty,
            "clv_segments": empty,
            "value_flag_hit_rate": empty,
            "false_positive_zones": empty,
        }

    segment_strength = (
        metrics.filter(pl.col("breakdown").is_in(["market", "league", "edge_bucket", "benchmark_source"]))
        .with_columns(
            pl.lit(run_id).alias("run_id"),
            ((pl.col("calibrated_log_loss") - pl.col("raw_log_loss")) * -1).alias("lift_log_loss"),
            ((pl.col("calibrated_brier_score") - pl.col("raw_brier_score")) * -1).alias("lift_brier"),
        )
        .select(
            "run_id",
            "breakdown",
            "group_key",
            "fixtures",
            "raw_log_loss",
            "calibrated_log_loss",
            "raw_brier_score",
            "calibrated_brier_score",
            "calibrated_calibration_error",
            "avg_clv",
            "flat_stake_roi",
            "lift_log_loss",
            "lift_brier",
        )
    )

    raw_vs_calibrated = (
        pl.concat(
            [
                metrics.filter(pl.col("breakdown") == "market").with_columns(pl.lit("market").alias("dimension")),
                metrics.filter(pl.col("breakdown") == "league").with_columns(pl.lit("league").alias("dimension")),
            ]
        )
        .with_columns(pl.lit(run_id).alias("run_id"))
        .select(
            "run_id",
            "dimension",
            pl.col("group_key").alias("segment"),
            "raw_log_loss",
            "calibrated_log_loss",
            "raw_brier_score",
            "calibrated_brier_score",
            "raw_calibration_error",
            "calibrated_calibration_error",
        )
    )

    bucketed = predictions.filter(pl.col("target").is_not_null()).with_columns(
        ((pl.col("raw_probability") * 10).floor() / 10).alias("raw_prob_bucket"),
        ((pl.col("calibrated_probability") * 10).floor() / 10).alias("cal_prob_bucket"),
    )
    calibration_buckets = (
        bucketed.group_by(["market", "league", "cal_prob_bucket"])
        .agg(
            pl.len().alias("samples"),
            pl.col("target").mean().alias("actual_rate"),
            pl.col("raw_probability").mean().alias("avg_raw_probability"),
            pl.col("calibrated_probability").mean().alias("avg_calibrated_probability"),
        )
        .with_columns(
            pl.lit(run_id).alias("run_id"),
            (pl.col("actual_rate") - pl.col("avg_calibrated_probability")).abs().alias("bucket_calibration_error"),
        )
        .rename({"cal_prob_bucket": "probability_bucket"})
    )

    clv_segments = (
        predictions.filter(pl.col("clv").is_not_null())
        .group_by(["market", "league"])
        .agg(
            pl.len().alias("samples"),
            pl.col("clv").mean().alias("avg_clv"),
            pl.col("clv").median().alias("median_clv"),
            pl.col("beat_close").mean().alias("share_beating_close"),
        )
        .with_columns(pl.lit(run_id).alias("run_id"))
    )

    value_flag_hit_rate = (
        predictions.filter(pl.col("calibrated_value_flag"))
        .filter(~pl.col("is_push"))
        .group_by(["market", "league"])
        .agg(
            pl.len().alias("bets"),
            pl.col("target").mean().alias("hit_rate"),
            pl.col("pnl").sum().alias("total_pnl"),
        )
        .with_columns(pl.lit(run_id).alias("run_id"))
    )

    false_positive_zones = (
        predictions.filter(pl.col("calibrated_value_flag") & (pl.col("target") == 0))
        .with_columns(((pl.col("calibrated_probability") * 10).floor() / 10).alias("probability_bucket"))
        .group_by(["market", "league", "edge_bucket", "benchmark_source", "probability_bucket"])
        .agg(pl.len().alias("false_positives"))
        .with_columns(pl.lit(run_id).alias("run_id"))
        .sort("false_positives", descending=True)
    )

    return {
        "segment_strength": segment_strength,
        "raw_vs_calibrated": raw_vs_calibrated,
        "calibration_buckets": calibration_buckets,
        "clv_segments": clv_segments,
        "value_flag_hit_rate": value_flag_hit_rate,
        "false_positive_zones": false_positive_zones,
    }


def build_champion_view(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return pl.DataFrame([])
    ranked = rank_experiment_runs(summary)
    champion = ranked.head(1).with_columns(pl.lit("champion").alias("selection_role"))
    challengers = ranked.slice(1, min(4, max(0, ranked.height - 1))).with_columns(pl.lit("challenger").alias("selection_role"))
    return pl.concat([champion, challengers]) if challengers.height else champion


def rank_experiment_runs(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty():
        return summary

    ranked = summary.with_columns(
        pl.col("calibrated_log_loss").rank(method="ordinal").alias("rank_log_loss"),
        pl.col("calibrated_brier_score").rank(method="ordinal").alias("rank_brier"),
        pl.col("calibrated_calibration_error").rank(method="ordinal").alias("rank_calibration"),
        pl.col("avg_clv").rank(method="ordinal", descending=True).alias("rank_clv"),
        pl.col("robustness_score").rank(method="ordinal", descending=True).alias("rank_robustness"),
        pl.col("flat_stake_roi").rank(method="ordinal", descending=True).alias("rank_roi_support"),
    ).with_columns(
        (
            pl.col("rank_log_loss") * 10000
            + pl.col("rank_brier") * 1000
            + pl.col("rank_calibration") * 100
            + pl.col("rank_clv") * 10
            + pl.col("rank_robustness") * 5
            + pl.col("rank_roi_support")
        ).alias(
            "ranking_score"
        )
    )
    return ranked.sort(["ranking_score", "calibrated_log_loss", "calibrated_brier_score", "calibrated_calibration_error", "avg_clv"])


def run_experiment_sweep(
    matches: pl.DataFrame,
    elo_history: pl.DataFrame,
    cfg: AppConfig,
    request: SweepRequest,
) -> tuple[str, pl.DataFrame, pl.DataFrame]:
    sweep_id = f"sw_{uuid4().hex[:12]}"
    dc = request.dixon_coles_weights or [cfg.weights.dixon_coles]
    elo = request.elo_prior_weights or [cfg.weights.elo_prior]
    shot = request.shot_adjustment_weights or [cfg.weights.shot_adjustment]
    edge = request.value_edge_thresholds or [cfg.runtime.value_edge_threshold]
    cred = request.credibility_thresholds or [cfg.runtime.credibility_threshold]
    lookback = request.lookback_days_options or [cfg.lookback.team_form_days]
    half_life = request.half_life_days_options or [cfg.half_life.team_form_days]

    rows: list[dict[str, object]] = []
    diagnostics: dict[str, list[pl.DataFrame]] = {
        "segment_strength": [],
        "raw_vs_calibrated": [],
        "calibration_buckets": [],
        "clv_segments": [],
        "value_flag_hit_rate": [],
        "false_positive_zones": [],
    }
    for dixon_w, elo_w, shot_w, edge_t, cred_t, lookback_d, half_life_d in product(dc, elo, shot, edge, cred, lookback, half_life):
        bt_request = BacktestRequest(
            start_date=request.start_date,
            end_date=request.end_date,
            leagues=request.leagues,
            seasons=request.seasons,
            stake=request.stake,
            dixon_coles_weight=float(dixon_w),
            elo_prior_weight=float(elo_w),
            shot_adjustment_weight=float(shot_w),
            value_edge_threshold=float(edge_t),
            credibility_threshold=float(cred_t),
            lookback_days=int(lookback_d),
            half_life_days=int(half_life_d),
            calibrate_probabilities=request.calibrate_probabilities,
        )
        run_id, predictions, metrics = run_backtest(matches, elo_history, cfg, bt_request)
        overall = metrics.filter((pl.col("breakdown") == "overall") & (pl.col("group_key") == "all"))
        if overall.is_empty():
            continue

        row = overall.row(0, named=True)
        row["robustness_score"] = _run_robustness_score(metrics)
        row.update(
            {
                "sweep_id": sweep_id,
                "run_id": run_id,
                "dixon_coles_weight": float(dixon_w),
                "elo_prior_weight": float(elo_w),
                "shot_adjustment_weight": float(shot_w),
                "value_edge_threshold": float(edge_t),
                "credibility_threshold": float(cred_t),
                "lookback_days": int(lookback_d),
                "half_life_days": int(half_life_d),
                "calibrate_probabilities": request.calibrate_probabilities,
                "created_at": datetime.utcnow().isoformat(),
            }
        )
        rows.append(row)
        run_diagnostics = build_run_diagnostics(run_id, predictions, metrics)
        for key, value in run_diagnostics.items():
            if not value.is_empty():
                diagnostics[key].append(value)

    summary = pl.DataFrame(rows) if rows else pl.DataFrame([])
    ranking = rank_experiment_runs(summary)
    run_experiment_sweep.last_diagnostics = {
        key: pl.concat(frames) if frames else pl.DataFrame([])
        for key, frames in diagnostics.items()
    }
    run_experiment_sweep.last_champion_view = build_champion_view(ranking)
    run_experiment_sweep.last_sweep_metadata = build_sweep_metadata(sweep_id, request, summary.height)
    return sweep_id, summary, ranking


def persist_backtest(repo: DuckRepository, request: BacktestRequest, run_id: str, predictions: pl.DataFrame, metrics: pl.DataFrame) -> None:
    run_df = pl.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at": datetime.utcnow().isoformat(),
                "start_date": request.start_date,
                "end_date": request.end_date,
                "leagues": ",".join(request.leagues),
                "seasons": ",".join(request.seasons or []),
                "stake": request.stake,
                "dixon_coles_weight": request.dixon_coles_weight,
                "elo_prior_weight": request.elo_prior_weight,
                "shot_adjustment_weight": request.shot_adjustment_weight,
                "value_edge_threshold": request.value_edge_threshold,
                "credibility_threshold": request.credibility_threshold,
                "lookback_days": request.lookback_days,
                "half_life_days": request.half_life_days,
                "calibrate_probabilities": request.calibrate_probabilities,
            }
        ]
    )
    repo.append_df("backtest_runs", run_df)
    if not predictions.is_empty():
        repo.append_df("backtest_predictions", predictions)
    if not metrics.is_empty():
        repo.append_df("backtest_metrics", metrics)


def persist_sweep_results(repo: DuckRepository, sweep_id: str, summary: pl.DataFrame, ranking: pl.DataFrame) -> None:
    if not summary.is_empty():
        repo.append_df("experiment_sweeps", summary.with_columns(pl.lit(sweep_id).alias("sweep_id")))
    if not ranking.is_empty():
        repo.append_df("experiment_rankings", ranking.with_columns(pl.lit(sweep_id).alias("sweep_id")))
    champion_view = getattr(run_experiment_sweep, "last_champion_view", pl.DataFrame([]))
    if isinstance(champion_view, pl.DataFrame) and not champion_view.is_empty():
        repo.append_df("experiment_champion_view", champion_view.with_columns(pl.lit(sweep_id).alias("sweep_id")))
    diagnostics = getattr(run_experiment_sweep, "last_diagnostics", {})
    if isinstance(diagnostics, dict):
        table_map = {
            "segment_strength": "experiment_segment_strength",
            "raw_vs_calibrated": "experiment_raw_vs_calibrated",
            "calibration_buckets": "experiment_calibration_buckets",
            "clv_segments": "experiment_clv_segments",
            "value_flag_hit_rate": "experiment_value_flag_hit_rate",
            "false_positive_zones": "experiment_false_positive_zones",
        }
        for key, table in table_map.items():
            frame = diagnostics.get(key, pl.DataFrame([]))
            if isinstance(frame, pl.DataFrame) and not frame.is_empty():
                repo.append_df(table, frame.with_columns(pl.lit(sweep_id).alias("sweep_id")))
    metadata = getattr(run_experiment_sweep, "last_sweep_metadata", pl.DataFrame([]))
    if isinstance(metadata, pl.DataFrame) and not metadata.is_empty():
        repo.append_df("experiment_sweep_metadata", metadata.with_columns(pl.lit(sweep_id).alias("sweep_id")))


def build_sweep_metadata(sweep_id: str, request: SweepRequest, run_count: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "sweep_id": sweep_id,
                "created_at": datetime.utcnow().isoformat(),
                "start_date": request.start_date,
                "end_date": request.end_date,
                "leagues": ",".join(request.leagues),
                "seasons": ",".join(request.seasons or []),
                "run_count": run_count,
                "ranking_method_version": "predictive_first_v1",
                "dixon_coles_weights": ",".join(str(v) for v in (request.dixon_coles_weights or [])),
                "elo_prior_weights": ",".join(str(v) for v in (request.elo_prior_weights or [])),
                "shot_adjustment_weights": ",".join(str(v) for v in (request.shot_adjustment_weights or [])),
                "value_edge_thresholds": ",".join(str(v) for v in (request.value_edge_thresholds or [])),
                "credibility_thresholds": ",".join(str(v) for v in (request.credibility_thresholds or [])),
                "lookback_days_options": ",".join(str(v) for v in (request.lookback_days_options or [])),
                "half_life_days_options": ",".join(str(v) for v in (request.half_life_days_options or [])),
                "summary_note": "Auto-generated sweep metadata",
            }
        ]
    )
