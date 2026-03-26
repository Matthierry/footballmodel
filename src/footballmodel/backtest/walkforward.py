from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import log
from uuid import uuid4

import polars as pl

from footballmodel.config.settings import AppConfig
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
            close_price = _close_price(fixture, str(market_row["market"]), str(market_row["outcome"]))
            current = market_row.get("current_price")
            clv = (1 / current - 1 / close_price) if (current and close_price) else None
            beat_close = (clv is not None and clv > 0)
            implied_stake = req.stake if bool(market_row.get("value_flag")) else 0.0
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
                    "model_probability": market_row["model_probability"],
                    "model_fair_odds": market_row["model_fair_odds"],
                    "current_price": current,
                    "close_price": close_price,
                    "benchmark_source": market_row.get("benchmark_source"),
                    "benchmark_available": bool(market_row.get("benchmark_available")),
                    "edge": market_row.get("edge"),
                    "edge_bucket": _edge_bucket(market_row.get("edge")),
                    "value_flag": bool(market_row.get("value_flag")),
                    "value_status": market_row.get("value_status"),
                    "credibility_score": market_row.get("credibility_score"),
                    "target": target,
                    "is_push": is_push,
                    "log_loss_component": -log(max(float(market_row["model_probability"]), 1e-12)) if target == 1 else None,
                    "brier_component": (float(market_row["model_probability"]) - float(target)) ** 2 if target is not None else None,
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
        ("value_flag", ["value_flag"]),
        ("value_status", ["value_status"]),
    ]

    out_rows: list[dict[str, object]] = []
    grouped_fx = predictions.group_by(["run_id", "fixture_id", "market"]).agg(
        pl.col("league").first().alias("league"),
        pl.col("edge_bucket").first().alias("edge_bucket"),
        pl.col("benchmark_source").first().alias("benchmark_source"),
        pl.col("value_flag").max().alias("value_flag"),
        pl.col("value_status").first().alias("value_status"),
        pl.col("log_loss_component").max().alias("log_loss"),
        pl.col("brier_component").sum().alias("brier_score"),
        pl.col("match_date").first().alias("match_date"),
    )

    for breakdown, keys in dims:
        frame = grouped_fx if keys else grouped_fx.with_columns(pl.lit("all").alias("group_key"))
        gb_keys = ["group_key"] if not keys else keys

        for key_data, sub in frame.group_by(gb_keys):
            key_tuple = key_data if isinstance(key_data, tuple) else (key_data,)
            key_val = "all" if not keys else "|".join(str(v) for v in key_tuple)
            log_loss = sub.filter(pl.col("log_loss").is_not_null())["log_loss"].mean()
            brier = sub["brier_score"].mean()

            row_subset = predictions.filter(
                pl.lit(True)
                if not keys
                else pl.all_horizontal([pl.col(k) == pl.lit(sub[k][0]) for k in keys])
            )
            calib = _calibration_error(row_subset)
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
                    "log_loss": log_loss,
                    "brier_score": brier,
                    "calibration_error": calib,
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


def _calibration_error(rows: pl.DataFrame) -> float | None:
    usable = rows.filter(pl.col("target").is_not_null())
    if usable.is_empty():
        return None

    bucketed = usable.with_columns(((pl.col("model_probability") * 10).floor() / 10).alias("prob_bucket"))
    total = bucketed.height
    ece = 0.0
    for _, bucket in bucketed.group_by("prob_bucket"):
        ece += abs(float(bucket["target"].mean()) - float(bucket["model_probability"].mean())) * (bucket.height / total)
    return float(ece)


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
            }
        ]
    )
    repo.append_df("backtest_runs", run_df)
    if not predictions.is_empty():
        repo.append_df("backtest_predictions", predictions)
    if not metrics.is_empty():
        repo.append_df("backtest_metrics", metrics)
