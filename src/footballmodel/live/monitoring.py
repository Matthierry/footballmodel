from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from footballmodel.markets.benchmark_snapshots import SNAPSHOT_TYPE_PREDICTION_TIME, choose_later_snapshot


def _line_for_market_outcome(market: str, outcome: str) -> float | None:
    if market == "OU25":
        return 2.5
    if market == "AH":
        try:
            return float(outcome.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            return None
    return None


def _actual_target(fixture: dict[str, object], market: str, outcome: str) -> tuple[int | None, bool]:
    home_goals = fixture.get("home_goals")
    away_goals = fixture.get("away_goals")
    if home_goals is None or away_goals is None:
        return None, False

    hg, ag = int(home_goals), int(away_goals)
    if market == "1X2":
        actual = "home" if hg > ag else "away" if hg < ag else "draw"
        return (1 if outcome == actual else 0), False
    if market == "OU25":
        actual = "over_2_5" if (hg + ag) > 2 else "under_2_5"
        return (1 if outcome == actual else 0), False
    if market == "BTTS":
        actual = "btts_yes" if (hg > 0 and ag > 0) else "btts_no"
        return (1 if outcome == actual else 0), False
    if market == "AH":
        line = float(outcome.split("_", maxsplit=1)[1])
        adjusted = (hg - ag) - line
        if adjusted == 0:
            return None, True
        actual = "home" if adjusted > 0 else "away"
        return (1 if outcome.startswith(actual) else 0), False
    return None, False


def build_live_review_rows(
    market_predictions: pl.DataFrame,
    benchmark_snapshots: pl.DataFrame,
    matches: pl.DataFrame,
) -> pl.DataFrame:
    if market_predictions.is_empty():
        return pl.DataFrame([])

    fixture_lookup = {str(row["fixture_id"]): row for row in matches.iter_rows(named=True)}
    snapshot_keys = ["fixture_id", "market", "outcome", "line", "benchmark_price", "benchmark_source", "snapshot_type", "snapshot_timestamp_utc"]
    rows: list[dict[str, object]] = []

    for prediction in market_predictions.iter_rows(named=True):
        fixture_id = str(prediction["fixture_id"])
        market = str(prediction["market"])
        outcome = str(prediction["outcome"])
        line = prediction.get("line")
        if line is None:
            line = _line_for_market_outcome(market, outcome)

        pred_snapshot = pl.DataFrame(
            [
                {
                    "fixture_id": fixture_id,
                    "market": market,
                    "outcome": outcome,
                    "line": line,
                    "benchmark_price": prediction.get("current_price"),
                    "benchmark_source": prediction.get("benchmark_source"),
                    "snapshot_type": SNAPSHOT_TYPE_PREDICTION_TIME,
                    "snapshot_timestamp_utc": prediction.get("prediction_timestamp_utc"),
                }
            ]
        ).with_columns(pl.col("line").cast(pl.Float64))

        matching_snapshots = benchmark_snapshots.with_columns(pl.col("line").cast(pl.Float64)).filter(
            (pl.col("fixture_id") == fixture_id)
            & (pl.col("market") == market)
            & (pl.col("outcome") == outcome)
            & (pl.col("line").is_null() if line is None else pl.col("line") == float(line))
        )

        combined = pl.concat([pred_snapshot.select(snapshot_keys), matching_snapshots.select(snapshot_keys)], how="vertical")
        later = choose_later_snapshot(combined)

        current_price = prediction.get("current_price")
        close_price = later.get("benchmark_price") if later else None
        clv = (1 / float(current_price) - 1 / float(close_price)) if (current_price is not None and close_price is not None) else None
        fixture = fixture_lookup.get(fixture_id, {})
        target, is_push = _actual_target(fixture, market, outcome)
        settled = target is not None or is_push

        rows.append(
            {
                "live_run_id": prediction.get("live_run_id"),
                "run_timestamp_utc": prediction.get("run_timestamp_utc"),
                "config_name": prediction.get("config_name"),
                "config_version": prediction.get("config_version"),
                "fixture_id": fixture_id,
                "match_date": fixture.get("match_date"),
                "league": fixture.get("league"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "market": market,
                "outcome": outcome,
                "line": line,
                "raw_probability": prediction.get("raw_probability"),
                "calibrated_probability": prediction.get("calibrated_probability"),
                "value_flag": prediction.get("value_flag"),
                "value_status": prediction.get("value_status"),
                "prediction_benchmark_price": current_price,
                "prediction_benchmark_source": prediction.get("benchmark_source"),
                "prediction_snapshot_type": SNAPSHOT_TYPE_PREDICTION_TIME,
                "prediction_snapshot_timestamp_utc": prediction.get("prediction_timestamp_utc"),
                "later_benchmark_price": close_price,
                "later_snapshot_type": later.get("snapshot_type") if later else None,
                "later_snapshot_source": later.get("benchmark_source") if later else None,
                "later_snapshot_timestamp_utc": later.get("snapshot_timestamp_utc") if later else None,
                "clv": clv,
                "settlement_status": "settled" if settled else "pending",
                "result_status": "push" if is_push else ("won" if target == 1 else "lost" if target == 0 else "pending"),
                "target": target,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    return pl.DataFrame(rows)
