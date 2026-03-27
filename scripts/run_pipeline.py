from __future__ import annotations

import argparse
from datetime import datetime, timezone

import polars as pl

from footballmodel.config.runtime_env import resolve_raw_data_paths
from footballmodel.config.settings import load_app_config
from footballmodel.ingestion.clubelo import build_clubelo_raw_file, load_clubelo_csv
from footballmodel.ingestion.football_data import build_football_data_raw_file, load_football_data_csv
from footballmodel.live.monitoring import (
    build_email_alert_events,
    build_live_review_rows,
    build_live_run_summary,
    build_open_alerts,
    detect_drift_alerts,
)
from footballmodel.orchestration.pipeline import (
    build_pre_kickoff_benchmark_snapshots,
    build_prediction_time_benchmark_snapshots,
    run_fixture_prediction,
)
from footballmodel.storage.repository import DuckRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run football model live pipeline")
    parser.add_argument("--config-name", default=None, help="Named live config to run (defaults to config.default_live_config)")
    parser.add_argument(
        "--football-data-config",
        default="config/football_data_sources.yaml",
        help="Football-Data source configuration file",
    )
    parser.add_argument(
        "--skip-football-data-fetch",
        action="store_true",
        help="Skip automatic Football-Data ingestion and reuse existing raw file",
    )
    parser.add_argument(
        "--clubelo-config",
        default="config/clubelo_sources.yaml",
        help="ClubElo source configuration file",
    )
    parser.add_argument(
        "--refresh-clubelo",
        action="store_true",
        help="Force ClubElo rebuild even when raw file already exists",
    )
    parser.add_argument(
        "--skip-clubelo-fetch",
        action="store_true",
        help="Skip automatic ClubElo ingestion and reuse existing raw file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_app_config("config/runtime.yaml")
    selected_config_name, live_cfg = cfg.resolve_live_config(args.config_name)

    repo = DuckRepository()

    matches_path, elo_path = resolve_raw_data_paths()
    if not args.skip_football_data_fetch:
        try:
            ingestion_result = build_football_data_raw_file(
                config_path=args.football_data_config,
                output_path=matches_path,
                snapshots_dir=matches_path.parent / "football_data_sources",
            )
            print(
                f"Built {matches_path} from {len(ingestion_result.fetched_sources)} sources "
                f"({ingestion_result.rows_before_dedup} -> {ingestion_result.rows_after_dedup} rows after dedupe)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Football-Data ingestion failed: {exc}")
            return

    needs_clubelo_refresh = args.refresh_clubelo or not elo_path.exists()
    if needs_clubelo_refresh and args.skip_clubelo_fetch:
        print("ClubElo raw file missing/stale and --skip-clubelo-fetch is set; cannot continue safely")
        return
    if needs_clubelo_refresh:
        if not matches_path.exists():
            print(f"ClubElo build requires matches at {matches_path}, but file is missing")
            return
        try:
            clubelo_result = build_clubelo_raw_file(
                config_path=args.clubelo_config,
                output_path=elo_path,
                football_data_path=matches_path,
                snapshots_dir=elo_path.parent / "clubelo_sources",
            )
            print(
                f"Built {elo_path} from {len(clubelo_result.fetched_dates)} date snapshots "
                f"({clubelo_result.rows_written} rows)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ClubElo ingestion failed: {exc}")
            return

    if not matches_path.exists() or not elo_path.exists():
        print(f"Raw files missing; expected {matches_path} and {elo_path}")
        return

    matches = load_football_data_csv(matches_path)
    elos = load_clubelo_csv(elo_path)

    upcoming = matches.filter(pl.col("home_goals").is_null() & pl.col("league").is_in(live_cfg.leagues))
    history = matches.filter(pl.col("home_goals").is_not_null())

    run_timestamp = datetime.now(timezone.utc).isoformat()
    live_run_id = f"live_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    predictions = []
    prediction_markets: list[dict[str, object]] = []
    snapshots: list[pl.DataFrame] = []
    for fixture in upcoming.iter_rows(named=True):
        pred = run_fixture_prediction(history, fixture, elos, live_cfg)
        pred["live_run_id"] = live_run_id
        pred["run_timestamp_utc"] = run_timestamp
        pred["config_name"] = selected_config_name
        pred["config_version"] = live_cfg.version
        predictions.append(pred)
        snapshots.append(build_prediction_time_benchmark_snapshots(fixture, pred["timestamp_utc"]))
        for row in pred["markets"]:
            raw_probability = row["model_probability"]
            calibrated_probability = raw_probability if live_cfg.calibration.enabled else raw_probability
            prediction_markets.append(
                {
                    "live_run_id": live_run_id,
                    "run_timestamp_utc": run_timestamp,
                    "config_name": selected_config_name,
                    "config_version": live_cfg.version,
                    "fixture_id": pred["fixture_id"],
                    "prediction_timestamp_utc": pred["timestamp_utc"],
                    "market": row["market"],
                    "outcome": row["outcome"],
                    "line": float(str(row["outcome"]).split("_", maxsplit=1)[1]) if str(row["market"]) == "AH" else (2.5 if str(row["market"]) == "OU25" else None),
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "calibration_method": "identity" if live_cfg.calibration.enabled else "disabled",
                    "model_fair_odds": row["model_fair_odds"],
                    "current_price": row.get("current_price"),
                    "benchmark_source": row.get("benchmark_source"),
                    "benchmark_snapshot_type": row.get("benchmark_snapshot_type"),
                    "benchmark_snapshot_timestamp_utc": row.get("benchmark_snapshot_timestamp_utc"),
                    "value_flag": row.get("value_flag"),
                    "value_status": row.get("value_status"),
                    "edge": row.get("edge"),
                }
            )

    run_predictions_df = pl.DataFrame(predictions)
    prediction_history_df = pl.DataFrame(prediction_markets)

    repo.write_df("curated_matches", matches)
    repo.write_df("elo_history", elos)
    repo.append_df("model_runs", run_predictions_df)
    repo.append_df("model_market_predictions", prediction_history_df)
    if snapshots:
        repo.upsert_benchmark_snapshots(pl.concat(snapshots))
    if upcoming.height:
        repo.upsert_benchmark_snapshots(build_pre_kickoff_benchmark_snapshots(upcoming))

    benchmark_snapshots = repo.read_df("select * from benchmark_snapshots")
    live_review = build_live_review_rows(prediction_history_df, benchmark_snapshots, matches)
    live_summary = build_live_run_summary(run_predictions_df, prediction_history_df, live_review)

    try:
        review_history = repo.read_df("select * from live_review_history")
    except Exception:
        review_history = pl.DataFrame([])
    review_for_alerts = pl.concat([review_history, live_review], how="vertical") if not review_history.is_empty() else live_review

    try:
        run_summary_history = repo.read_df("select * from live_run_summaries_history")
    except Exception:
        run_summary_history = pl.DataFrame([])
    runs_for_alerts = pl.concat([run_summary_history, live_summary], how="vertical") if not run_summary_history.is_empty() else live_summary

    alerts = detect_drift_alerts(
        review_history=review_for_alerts,
        run_summaries_history=runs_for_alerts,
        benchmark_snapshots=benchmark_snapshots,
        config_name=selected_config_name,
        config_version=live_cfg.version,
        run_timestamp_utc=run_timestamp,
        thresholds=cfg.drift_alerts.model_dump(),
    )
    try:
        alert_history = repo.read_df("select * from live_alert_history")
    except Exception:
        alert_history = pl.DataFrame([])
    alert_history_updated = pl.concat([alert_history, alerts], how="vertical") if not alert_history.is_empty() else alerts
    open_alerts = build_open_alerts(alert_history_updated)
    email_events = build_email_alert_events(alerts, enabled=cfg.drift_alerts.severe_email_enabled)

    repo.append_df("live_run_summaries_history", live_summary)
    repo.append_df("live_prediction_history", prediction_history_df)
    repo.append_df("live_review_history", live_review)
    repo.append_df("live_alert_history", alerts)
    repo.write_df("live_open_alerts", open_alerts)
    repo.append_df("live_alert_notifications_history", email_events)
    repo.write_df("live_model_review", live_review)
    repo.close()

    print(
        f"Wrote {len(predictions)} predictions using config={selected_config_name}"
        f" version={live_cfg.version} run_id={live_run_id}"
    )


if __name__ == "__main__":
    main()
