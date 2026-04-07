from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def _settlement_snapshot_for_prediction(
    benchmark_snapshots: pl.DataFrame,
    fixture_id: str,
    market: str,
    outcome: str,
    line: float | None,
) -> dict[str, object] | None:
    if benchmark_snapshots.is_empty():
        return None
    matching = benchmark_snapshots.with_columns(pl.col("line").cast(pl.Float64)).filter(
        (pl.col("fixture_id") == fixture_id)
        & (pl.col("market") == market)
        & (pl.col("outcome") == outcome)
        & (pl.col("line").is_null() if line is None else pl.col("line") == float(line))
    )
    return choose_later_snapshot(matching)


def build_snapshot_review_rows(
    market_predictions: pl.DataFrame,
    benchmark_snapshots: pl.DataFrame,
    matches: pl.DataFrame,
) -> pl.DataFrame:
    if market_predictions.is_empty():
        return pl.DataFrame([])

    fixture_lookup = {str(row["fixture_id"]): row for row in matches.iter_rows(named=True)}
    rows: list[dict[str, object]] = []

    for prediction in market_predictions.iter_rows(named=True):
        fixture_id = str(prediction["fixture_id"])
        market = str(prediction["market"])
        outcome = str(prediction["outcome"])
        line = prediction.get("line")
        if line is None:
            line = _line_for_market_outcome(market, outcome)

        settlement_snapshot = _settlement_snapshot_for_prediction(
            benchmark_snapshots=benchmark_snapshots,
            fixture_id=fixture_id,
            market=market,
            outcome=outcome,
            line=line,
        )

        current_price = prediction.get("current_price")
        close_price = settlement_snapshot.get("benchmark_price") if settlement_snapshot else None
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
                "later_snapshot_type": settlement_snapshot.get("snapshot_type") if settlement_snapshot else None,
                "later_snapshot_source": settlement_snapshot.get("benchmark_source") if settlement_snapshot else None,
                "later_snapshot_timestamp_utc": settlement_snapshot.get("snapshot_timestamp_utc") if settlement_snapshot else None,
                "clv": clv,
                "settlement_status": "settled" if settled else "pending",
                "result_status": "push" if is_push else ("won" if target == 1 else "lost" if target == 0 else "pending"),
                "target": target,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    return pl.DataFrame(rows)


def build_live_review_rows(
    market_predictions: pl.DataFrame,
    benchmark_snapshots: pl.DataFrame,
    matches: pl.DataFrame,
) -> pl.DataFrame:
    return build_snapshot_review_rows(market_predictions, benchmark_snapshots, matches)


def build_live_run_summary(
    run_predictions: pl.DataFrame,
    run_market_predictions: pl.DataFrame,
    run_review_rows: pl.DataFrame,
) -> pl.DataFrame:
    if run_predictions.is_empty():
        return pl.DataFrame([])

    run_meta = run_predictions.row(0, named=True)
    settled = run_review_rows.filter(pl.col("settlement_status") == "settled") if not run_review_rows.is_empty() else pl.DataFrame([])
    value_rows = settled.filter(pl.col("value_flag") == True) if not settled.is_empty() else pl.DataFrame([])
    with_clv = settled.filter(pl.col("clv").is_not_null()) if not settled.is_empty() else pl.DataFrame([])

    value_hit_rate = (
        value_rows.filter(pl.col("result_status") == "won").height / value_rows.height
        if value_rows.height
        else None
    )

    return pl.DataFrame(
        [
            {
                "live_run_id": run_meta.get("live_run_id"),
                "run_timestamp_utc": run_meta.get("run_timestamp_utc"),
                "config_name": run_meta.get("config_name"),
                "config_version": run_meta.get("config_version"),
                "fixtures_scored": run_predictions.height,
                "market_predictions": run_market_predictions.height,
                "review_rows": run_review_rows.height,
                "pending_rows": run_review_rows.filter(pl.col("settlement_status") == "pending").height if not run_review_rows.is_empty() else 0,
                "settled_rows": settled.height,
                "value_rows": run_review_rows.filter(pl.col("value_flag") == True).height if not run_review_rows.is_empty() else 0,
                "settled_value_rows": value_rows.height,
                "value_hit_rate": value_hit_rate,
                "avg_clv": float(with_clv["clv"].mean()) if with_clv.height else None,
                "benchmark_coverage_rate": (
                    run_review_rows.filter(pl.col("later_benchmark_price").is_not_null()).height / run_review_rows.height
                    if run_review_rows.height
                    else None
                ),
                "summary_created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )


def _window_stats(frame: pl.DataFrame, min_value_rows: int, min_calibration_rows: int) -> dict[str, float | int | None]:
    settled = frame.filter(pl.col("settlement_status") == "settled")
    with_clv = settled.filter(pl.col("clv").is_not_null())
    value_rows = settled.filter(pl.col("value_flag") == True)
    calibration_rows = settled.filter(pl.col("target").is_not_null() & pl.col("calibrated_probability").is_not_null())

    calibration_error = None
    if calibration_rows.height >= min_calibration_rows:
        calibration_error = float(
            (
                (pl.Series(calibration_rows["calibrated_probability"]) - pl.Series(calibration_rows["target"])).abs().mean()
            )
        )
    return {
        "rows": frame.height,
        "settled_rows": settled.height,
        "avg_clv": float(with_clv["clv"].mean()) if with_clv.height else None,
        "value_hit_rate": (
            value_rows.filter(pl.col("result_status") == "won").height / value_rows.height
            if value_rows.height >= min_value_rows
            else None
        ),
        "coverage_rate": (
            frame.filter(pl.col("later_benchmark_price").is_not_null()).height / frame.height
            if frame.height
            else None
        ),
        "missingness_rate": (
            frame.filter(pl.col("later_benchmark_price").is_null()).height / frame.height
            if frame.height
            else None
        ),
        "value_volume": value_rows.height,
        "value_volume_rate": (value_rows.height / settled.height) if settled.height else None,
        "calibration_error": calibration_error,
    }


def _severity(delta: float, medium_threshold: float = 1.5) -> str:
    return "critical" if delta >= medium_threshold else "warning"


def detect_drift_alerts(
    review_history: pl.DataFrame,
    run_summaries_history: pl.DataFrame,
    benchmark_snapshots: pl.DataFrame,
    config_name: str,
    config_version: str,
    run_timestamp_utc: str,
    thresholds: dict[str, object],
) -> pl.DataFrame:
    if review_history.is_empty():
        return pl.DataFrame([])

    windows_days = [int(x) for x in thresholds.get("windows_days", [7, 14, 30])]
    baseline_days = int(thresholds.get("baseline_days", 120))
    min_settled_rows = int(thresholds.get("min_settled_rows", 30))
    min_value_rows = int(thresholds.get("min_value_rows", 20))
    min_calibration_rows = int(thresholds.get("min_calibration_rows", 30))
    min_concentration_rows = int(thresholds.get("min_concentration_rows", 20))

    as_of = datetime.fromisoformat(run_timestamp_utc.replace("Z", "+00:00"))
    as_of_naive = as_of.replace(tzinfo=None)
    df = review_history.with_columns(
        pl.col("run_timestamp_utc").cast(pl.Datetime, strict=False),
        pl.col("prediction_snapshot_timestamp_utc").cast(pl.Datetime, strict=False),
    ).filter(pl.col("config_name") == config_name)

    alerts: list[dict[str, object]] = []
    alert_counter = 1

    for days in windows_days:
        recent_start = as_of_naive - timedelta(days=days)
        baseline_start = as_of_naive - timedelta(days=baseline_days)
        recent = df.filter(pl.col("run_timestamp_utc") >= pl.lit(recent_start))
        baseline = df.filter((pl.col("run_timestamp_utc") >= pl.lit(baseline_start)) & (pl.col("run_timestamp_utc") < pl.lit(recent_start)))
        version_baseline = baseline.filter(pl.col("config_version") == config_version)
        baseline_scope = version_baseline if not version_baseline.is_empty() else baseline

        recent_stats = _window_stats(recent, min_value_rows=min_value_rows, min_calibration_rows=min_calibration_rows)
        baseline_stats = _window_stats(baseline_scope, min_value_rows=min_value_rows, min_calibration_rows=min_calibration_rows)

        if recent_stats["settled_rows"] >= min_settled_rows and baseline_stats["settled_rows"] >= min_settled_rows:
            if recent_stats["avg_clv"] is not None and baseline_stats["avg_clv"] is not None:
                diff = float(baseline_stats["avg_clv"]) - float(recent_stats["avg_clv"])
                if diff >= float(thresholds.get("clv_drop_abs", 0.015)):
                    alerts.append(
                        {
                            "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                            "alert_timestamp_utc": run_timestamp_utc,
                            "alert_type": "clv_deterioration",
                            "severity": _severity(diff / float(thresholds.get("clv_drop_abs", 0.015))),
                            "config_name": config_name,
                            "config_version": config_version,
                            "market": None,
                            "league": None,
                            "metric": f"avg_clv_{days}d",
                            "window_days": days,
                            "observed_value": recent_stats["avg_clv"],
                            "baseline_value": baseline_stats["avg_clv"],
                            "delta_value": -diff,
                            "status": "open",
                            "details": f"recent_settled={recent_stats['settled_rows']} baseline_settled={baseline_stats['settled_rows']}",
                        }
                    )
                    alert_counter += 1

            if recent_stats["value_hit_rate"] is not None and baseline_stats["value_hit_rate"] is not None:
                drop = float(baseline_stats["value_hit_rate"]) - float(recent_stats["value_hit_rate"])
                if drop >= float(thresholds.get("value_hit_rate_drop_abs", 0.08)):
                    alerts.append(
                        {
                            "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                            "alert_timestamp_utc": run_timestamp_utc,
                            "alert_type": "value_hit_rate_deterioration",
                            "severity": _severity(drop / float(thresholds.get("value_hit_rate_drop_abs", 0.08))),
                            "config_name": config_name,
                            "config_version": config_version,
                            "market": None,
                            "league": None,
                            "metric": f"value_hit_rate_{days}d",
                            "window_days": days,
                            "observed_value": recent_stats["value_hit_rate"],
                            "baseline_value": baseline_stats["value_hit_rate"],
                            "delta_value": -drop,
                            "status": "open",
                            "details": f"recent_value_rows={recent_stats['value_volume']} baseline_value_rows={baseline_stats['value_volume']}",
                        }
                    )
                    alert_counter += 1

            if recent_stats["coverage_rate"] is not None and baseline_stats["coverage_rate"] is not None:
                coverage_drop = float(baseline_stats["coverage_rate"]) - float(recent_stats["coverage_rate"])
                if coverage_drop >= float(thresholds.get("benchmark_coverage_drop_abs", 0.08)):
                    alerts.append(
                        {
                            "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                            "alert_timestamp_utc": run_timestamp_utc,
                            "alert_type": "benchmark_coverage_deterioration",
                            "severity": _severity(coverage_drop / float(thresholds.get("benchmark_coverage_drop_abs", 0.08))),
                            "config_name": config_name,
                            "config_version": config_version,
                            "market": None,
                            "league": None,
                            "metric": f"coverage_rate_{days}d",
                            "window_days": days,
                            "observed_value": recent_stats["coverage_rate"],
                            "baseline_value": baseline_stats["coverage_rate"],
                            "delta_value": -coverage_drop,
                            "status": "open",
                            "details": f"recent_rows={recent_stats['rows']} baseline_rows={baseline_stats['rows']}",
                        }
                    )
                    alert_counter += 1

            if recent_stats["calibration_error"] is not None and baseline_stats["calibration_error"] is not None:
                calibration_increase = float(recent_stats["calibration_error"]) - float(baseline_stats["calibration_error"])
                if calibration_increase >= float(thresholds.get("calibration_error_increase_abs", 0.03)):
                    alerts.append(
                        {
                            "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                            "alert_timestamp_utc": run_timestamp_utc,
                            "alert_type": "calibration_deterioration",
                            "severity": _severity(calibration_increase / float(thresholds.get("calibration_error_increase_abs", 0.03))),
                            "config_name": config_name,
                            "config_version": config_version,
                            "market": None,
                            "league": None,
                            "metric": f"calibration_error_{days}d",
                            "window_days": days,
                            "observed_value": recent_stats["calibration_error"],
                            "baseline_value": baseline_stats["calibration_error"],
                            "delta_value": calibration_increase,
                            "status": "open",
                            "details": f"recent_rows={recent_stats['settled_rows']} baseline_rows={baseline_stats['settled_rows']}",
                        }
                    )
                    alert_counter += 1

        if recent_stats["value_volume_rate"] is not None and baseline_stats["value_volume_rate"] is not None:
            base = float(baseline_stats["value_volume_rate"]) if float(baseline_stats["value_volume_rate"]) > 0 else None
            if base:
                ratio = float(recent_stats["value_volume_rate"]) / base
                if ratio <= float(thresholds.get("value_volume_ratio_low", 0.5)) or ratio >= float(thresholds.get("value_volume_ratio_high", 1.8)):
                    alerts.append(
                        {
                            "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                            "alert_timestamp_utc": run_timestamp_utc,
                            "alert_type": "value_volume_anomaly",
                            "severity": "warning" if 0.4 <= ratio <= 2.2 else "critical",
                            "config_name": config_name,
                            "config_version": config_version,
                            "market": None,
                            "league": None,
                            "metric": f"value_volume_rate_{days}d",
                            "window_days": days,
                            "observed_value": recent_stats["value_volume_rate"],
                            "baseline_value": baseline_stats["value_volume_rate"],
                            "delta_value": ratio,
                            "status": "open",
                            "details": f"ratio={ratio:.4f} recent_value={recent_stats['value_volume']}",
                        }
                    )
                    alert_counter += 1

    recent_7d = df.filter(pl.col("run_timestamp_utc") >= pl.lit(as_of_naive - timedelta(days=7)))
    settled_recent = recent_7d.filter(pl.col("settlement_status") == "settled").filter(pl.col("clv").is_not_null())
    if settled_recent.height >= min_concentration_rows:
        scoped = (
            settled_recent.group_by(["league", "market"])
            .agg(pl.len().alias("rows"), pl.col("clv").mean().alias("avg_clv"))
            .filter(pl.col("avg_clv") < 0)
            .with_columns((pl.col("rows") / pl.col("rows").sum()).alias("share"))
            .sort("share", descending=True)
        )
        if not scoped.is_empty():
            top = scoped.row(0, named=True)
            if float(top["share"]) >= float(thresholds.get("concentration_negative_clv_share", 0.7)):
                alerts.append(
                    {
                        "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                        "alert_timestamp_utc": run_timestamp_utc,
                        "alert_type": "performance_concentration",
                        "severity": "warning",
                        "config_name": config_name,
                        "config_version": config_version,
                        "market": top.get("market"),
                        "league": top.get("league"),
                        "metric": "negative_clv_concentration_7d",
                        "window_days": 7,
                        "observed_value": top.get("share"),
                        "baseline_value": thresholds.get("concentration_negative_clv_share", 0.7),
                        "delta_value": float(top["share"]) - float(thresholds.get("concentration_negative_clv_share", 0.7)),
                        "status": "open",
                        "details": f"rows={top.get('rows')} avg_clv={top.get('avg_clv')}",
                    }
                )
                alert_counter += 1

    if not benchmark_snapshots.is_empty():
        snapshots = benchmark_snapshots.with_columns(pl.col("snapshot_timestamp_utc").cast(pl.Datetime, strict=False))
        max_snap = snapshots["snapshot_timestamp_utc"].max()
        if max_snap is not None:
            age_hours = (as_of_naive - max_snap).total_seconds() / 3600
            if age_hours >= float(thresholds.get("stale_snapshot_hours", 30)):
                alerts.append(
                    {
                        "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                        "alert_timestamp_utc": run_timestamp_utc,
                        "alert_type": "source_freshness_stale",
                        "severity": "critical",
                        "config_name": config_name,
                        "config_version": config_version,
                        "market": None,
                        "league": None,
                        "metric": "latest_snapshot_age_hours",
                        "window_days": None,
                        "observed_value": age_hours,
                        "baseline_value": thresholds.get("stale_snapshot_hours", 30),
                        "delta_value": age_hours - float(thresholds.get("stale_snapshot_hours", 30)),
                        "status": "open",
                        "details": f"latest_snapshot={max_snap.isoformat()}",
                    }
                )
                alert_counter += 1

    if not run_summaries_history.is_empty():
        summaries = run_summaries_history.with_columns(pl.col("run_timestamp_utc").cast(pl.Datetime, strict=False)).sort("run_timestamp_utc", descending=True)
        latest = summaries.row(0, named=True)
        if latest.get("run_timestamp_utc") is not None:
            run_age_hours = (as_of_naive - latest["run_timestamp_utc"]).total_seconds() / 3600
            if run_age_hours > float(thresholds.get("stale_run_hours", 18)):
                alerts.append(
                    {
                        "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                        "alert_timestamp_utc": run_timestamp_utc,
                        "alert_type": "run_staleness",
                        "severity": "critical",
                        "config_name": config_name,
                        "config_version": config_version,
                        "market": None,
                        "league": None,
                        "metric": "hours_since_last_run",
                        "window_days": None,
                        "observed_value": run_age_hours,
                        "baseline_value": thresholds.get("stale_run_hours", 18),
                        "delta_value": run_age_hours - float(thresholds.get("stale_run_hours", 18)),
                        "status": "open",
                        "details": f"last_run={latest.get('run_timestamp_utc')}",
                    }
                )
                alert_counter += 1

        pending_ratio = (latest.get("pending_rows", 0) / latest.get("review_rows", 1)) if latest.get("review_rows", 0) else 0.0
        if latest.get("fixtures_scored", 0) == 0 or latest.get("market_predictions", 0) == 0:
            alerts.append(
                {
                    "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                    "alert_timestamp_utc": run_timestamp_utc,
                    "alert_type": "failed_or_partial_run",
                    "severity": "critical",
                    "config_name": config_name,
                    "config_version": config_version,
                    "market": None,
                    "league": None,
                    "metric": "market_predictions",
                    "window_days": None,
                    "observed_value": latest.get("market_predictions"),
                    "baseline_value": 1,
                    "delta_value": None,
                    "status": "open",
                    "details": f"fixtures_scored={latest.get('fixtures_scored')}",
                }
            )
            alert_counter += 1
        elif pending_ratio >= float(thresholds.get("pending_rows_ratio_high", 0.8)):
            alerts.append(
                {
                    "alert_id": f"ALRT-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{alert_counter:04d}",
                    "alert_timestamp_utc": run_timestamp_utc,
                    "alert_type": "failed_or_partial_run",
                    "severity": "warning",
                    "config_name": config_name,
                    "config_version": config_version,
                    "market": None,
                    "league": None,
                    "metric": "pending_ratio",
                    "window_days": None,
                    "observed_value": pending_ratio,
                    "baseline_value": thresholds.get("pending_rows_ratio_high", 0.8),
                    "delta_value": pending_ratio - float(thresholds.get("pending_rows_ratio_high", 0.8)),
                    "status": "open",
                    "details": f"pending_rows={latest.get('pending_rows')} review_rows={latest.get('review_rows')}",
                }
            )
            alert_counter += 1

    if not alerts:
        return pl.DataFrame([])
    return pl.DataFrame(alerts).with_columns(pl.col("status").cast(pl.Utf8))


def build_open_alerts(alert_history: pl.DataFrame) -> pl.DataFrame:
    if alert_history.is_empty():
        return pl.DataFrame([])
    return (
        alert_history.with_columns(pl.col("alert_timestamp_utc").cast(pl.Datetime, strict=False))
        .sort("alert_timestamp_utc", descending=True)
        .filter(pl.col("status") == "open")
    )


def build_email_alert_events(alerts: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    if alerts.is_empty() or not enabled:
        return pl.DataFrame([])
    severe = alerts.filter(pl.col("severity").is_in(["high", "critical"]))
    if severe.is_empty():
        return pl.DataFrame([])
    return severe.select(
        [
            pl.col("alert_id"),
            pl.col("alert_timestamp_utc"),
            pl.col("alert_type"),
            pl.col("severity"),
            pl.col("config_name"),
            pl.col("config_version"),
            pl.lit("queued").alias("notification_status"),
            pl.lit("snapshot_review").alias("channel"),
        ]
    )
