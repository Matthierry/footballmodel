from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import polars as pl

from footballmodel.config.runtime_env import resolve_raw_data_paths
from footballmodel.config.settings import load_app_config
from footballmodel.ingestion.clubelo import build_clubelo_raw_file, load_clubelo_csv
from footballmodel.ingestion.football_data import (
    build_football_data_raw_file,
    load_football_data_config,
    load_football_data_csv,
)
from footballmodel.live.monitoring import (
    build_email_alert_events,
    build_live_review_rows,
    build_live_run_summary,
    build_open_alerts,
    detect_drift_alerts,
)
from footballmodel.orchestration.pipeline import (
    build_prediction_time_benchmark_snapshots,
    run_fixture_prediction,
)
from footballmodel.storage.repository import DuckRepository


LIVE_REVIEW_SCHEMA: dict[str, pl.DataType] = {
    "live_run_id": pl.Utf8,
    "run_timestamp_utc": pl.Utf8,
    "config_name": pl.Utf8,
    "config_version": pl.Utf8,
    "fixture_id": pl.Utf8,
    "match_date": pl.Date,
    "league": pl.Utf8,
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "market": pl.Utf8,
    "outcome": pl.Utf8,
    "line": pl.Float64,
    "raw_probability": pl.Float64,
    "calibrated_probability": pl.Float64,
    "value_flag": pl.Boolean,
    "value_status": pl.Utf8,
    "prediction_benchmark_price": pl.Float64,
    "prediction_benchmark_source": pl.Utf8,
    "prediction_snapshot_type": pl.Utf8,
    "prediction_snapshot_timestamp_utc": pl.Utf8,
    "later_benchmark_price": pl.Float64,
    "later_snapshot_type": pl.Utf8,
    "later_snapshot_source": pl.Utf8,
    "later_snapshot_timestamp_utc": pl.Utf8,
    "clv": pl.Float64,
    "settlement_status": pl.Utf8,
    "result_status": pl.Utf8,
    "target": pl.Int64,
    "reviewed_at_utc": pl.Utf8,
}

LIVE_RUN_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "live_run_id": pl.Utf8,
    "run_timestamp_utc": pl.Utf8,
    "config_name": pl.Utf8,
    "config_version": pl.Utf8,
    "fixtures_scored": pl.Int64,
    "market_predictions": pl.Int64,
    "review_rows": pl.Int64,
    "pending_rows": pl.Int64,
    "settled_rows": pl.Int64,
    "value_rows": pl.Int64,
    "settled_value_rows": pl.Int64,
    "value_hit_rate": pl.Float64,
    "avg_clv": pl.Float64,
    "benchmark_coverage_rate": pl.Float64,
    "summary_created_at_utc": pl.Utf8,
}

ALERT_SCHEMA: dict[str, pl.DataType] = {
    "alert_id": pl.Utf8,
    "alert_timestamp_utc": pl.Utf8,
    "alert_type": pl.Utf8,
    "severity": pl.Utf8,
    "config_name": pl.Utf8,
    "config_version": pl.Utf8,
    "market": pl.Utf8,
    "league": pl.Utf8,
    "metric": pl.Utf8,
    "window_days": pl.Int64,
    "observed_value": pl.Float64,
    "baseline_value": pl.Float64,
    "delta_value": pl.Float64,
    "status": pl.Utf8,
    "details": pl.Utf8,
}

EMAIL_ALERT_SCHEMA: dict[str, pl.DataType] = {
    "alert_id": pl.Utf8,
    "alert_timestamp_utc": pl.Utf8,
    "alert_type": pl.Utf8,
    "severity": pl.Utf8,
    "config_name": pl.Utf8,
    "config_version": pl.Utf8,
    "notification_status": pl.Utf8,
    "channel": pl.Utf8,
}

MODEL_RUN_COLUMNS = [
    "fixture_id",
    "timestamp_utc",
    "home_team",
    "away_team",
    "expected_home_goals",
    "expected_away_goals",
    "live_run_id",
    "run_timestamp_utc",
    "config_name",
    "config_version",
]


def _frame_with_schema(rows: list[dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def _ensure_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if df.width == 0:
        return pl.DataFrame(schema=schema)
    return df


def _append_if_valid(repo: DuckRepository, table: str, df: pl.DataFrame, *, empty_message: str) -> None:
    if df.width == 0:
        print(f"Skipping append to {table}: {empty_message}")
        return
    repo.append_df(table, df)


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
    football_data_cfg = load_football_data_config(args.football_data_config)
    csv_to_league = {source.csv_code: source.league_code for source in football_data_cfg.sources}
    selected_config_name, live_cfg = cfg.resolve_live_config(args.config_name)

    repo = DuckRepository()
    repo.ensure_optional_tables(["benchmark_snapshots", "model_runs", "model_market_predictions"])

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
            print(
                "Upcoming fixture retention:"
                f" rows_fetched={ingestion_result.future_fixtures_rows_fetched}"
                f" fetched_future={ingestion_result.future_fixtures_fetched}"
                f" normalized_future={ingestion_result.future_fixtures_after_normalization}"
                f" source_div_column_found={ingestion_result.source_div_column_found}"
                f" league_code_created_from_source_div={ingestion_result.league_code_created_from_source_div}"
                f" normalized_future_with_league_code={ingestion_result.future_fixtures_with_league_code_after_normalization}"
                f" deduped_future={ingestion_result.future_fixtures_after_dedup}"
                f" deduped_future_with_league_code={ingestion_result.future_fixtures_with_league_code_after_dedup}"
                f" with_published_odds={ingestion_result.future_fixtures_with_published_odds}"
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

    matches = load_football_data_csv(matches_path, csv_to_league=csv_to_league)
    elos = load_clubelo_csv(elo_path)

    odds_cols = [c for c in ["avg_home_odds", "avg_draw_odds", "avg_away_odds", "bf_home_odds", "bf_draw_odds", "bf_away_odds"] if c in matches.columns]
    has_price_expr = pl.any_horizontal([pl.col(c).is_not_null() for c in odds_cols]) if odds_cols else pl.lit(False)
    today = date.today()
    future_flag_expr = (
        pl.col("is_future_fixture")
        if "is_future_fixture" in matches.columns
        else (pl.col("home_goals").is_null() & pl.col("away_goals").is_null())
    )
    future_date_expr = pl.col("match_date") >= pl.lit(today)
    eligible_future_expr = future_flag_expr & future_date_expr
    league_scope_column = "league_code" if "league_code" in matches.columns else "league"
    upcoming = matches.filter(
        eligible_future_expr
        & pl.col(league_scope_column).cast(pl.Utf8, strict=False).is_in(live_cfg.leagues)
        & has_price_expr
    )
    history = matches.filter(pl.col("home_goals").is_not_null())
    future_matches_count = matches.filter(future_date_expr).height
    eligible_future_count = matches.filter(eligible_future_expr).height
    eligible_future_with_odds_count = matches.filter(eligible_future_expr & has_price_expr).height
    print(
        "Future fixture diagnostics:"
        f" curated_future_rows={future_matches_count}"
        f" eligible_future_rows={eligible_future_count}"
        f" eligible_future_with_published_odds={eligible_future_with_odds_count}"
        f" future_rows_eligible_for_prediction={upcoming.height}"
        f" league_scope_column={league_scope_column}"
        f" league_scoped_for_run={upcoming.height}"
    )

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

    run_predictions_df = pl.DataFrame(predictions).select(MODEL_RUN_COLUMNS) if predictions else pl.DataFrame([])
    prediction_history_df = pl.DataFrame(prediction_markets)

    repo.write_df("curated_matches", matches)
    curated_future_with_odds = matches.filter(future_date_expr & has_price_expr).height
    print(
        "curated_matches retention:"
        f" future_rows={matches.filter(future_date_expr).height}"
        f" future_rows_written_to_curated_matches={matches.filter(future_date_expr).height}"
        f" future_rows_with_published_odds={curated_future_with_odds}"
        f" total_rows={matches.height}"
    )
    repo.write_df("elo_history", elos)
    _append_if_valid(
        repo,
        "model_runs",
        run_predictions_df,
        empty_message="no eligible fixtures produced run-level predictions.",
    )
    _append_if_valid(
        repo,
        "model_market_predictions",
        prediction_history_df,
        empty_message="no market-level predictions were generated.",
    )
    if snapshots:
        repo.upsert_benchmark_snapshots(pl.concat(snapshots))
    benchmark_snapshots = repo.read_table_or_empty("benchmark_snapshots")
    live_review = _ensure_schema(
        build_live_review_rows(prediction_history_df, benchmark_snapshots, matches),
        LIVE_REVIEW_SCHEMA,
    )
    live_summary = _ensure_schema(
        build_live_run_summary(run_predictions_df, prediction_history_df, live_review),
        LIVE_RUN_SUMMARY_SCHEMA,
    )
    if live_summary.is_empty():
        print(
            "No eligible predictions generated for this run; persisting zero-volume run summary for auditability."
        )
        live_summary = pl.DataFrame(
            [
                {
                    "live_run_id": live_run_id,
                    "run_timestamp_utc": run_timestamp,
                    "config_name": selected_config_name,
                    "config_version": live_cfg.version,
                    "fixtures_scored": 0,
                    "market_predictions": 0,
                    "review_rows": 0,
                    "pending_rows": 0,
                    "settled_rows": 0,
                    "value_rows": 0,
                    "settled_value_rows": 0,
                    "value_hit_rate": None,
                    "avg_clv": None,
                    "benchmark_coverage_rate": None,
                    "summary_created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            ],
            schema=LIVE_RUN_SUMMARY_SCHEMA,
        )

    review_history = repo.read_table_or_empty("live_review_history")
    review_for_alerts = pl.concat([review_history, live_review], how="vertical") if not review_history.is_empty() else live_review

    run_summary_history = repo.read_table_or_empty("live_run_summaries_history")
    runs_for_alerts = pl.concat([run_summary_history, live_summary], how="vertical") if not run_summary_history.is_empty() else live_summary

    alerts = _ensure_schema(
        detect_drift_alerts(
        review_history=review_for_alerts,
        run_summaries_history=runs_for_alerts,
        benchmark_snapshots=benchmark_snapshots,
        config_name=selected_config_name,
        config_version=live_cfg.version,
        run_timestamp_utc=run_timestamp,
        thresholds=cfg.drift_alerts.model_dump(),
        ),
        ALERT_SCHEMA,
    )
    alert_history = repo.read_table_or_empty("live_alert_history")
    alert_history_updated = pl.concat([alert_history, alerts], how="vertical") if not alert_history.is_empty() else alerts
    open_alerts = _ensure_schema(build_open_alerts(alert_history_updated), ALERT_SCHEMA)
    email_events = _ensure_schema(
        build_email_alert_events(alerts, enabled=cfg.drift_alerts.severe_email_enabled),
        EMAIL_ALERT_SCHEMA,
    )

    repo.append_df("live_run_summaries_history", live_summary)
    _append_if_valid(
        repo,
        "live_prediction_history",
        prediction_history_df,
        empty_message="run generated zero prediction rows.",
    )
    _append_if_valid(
        repo,
        "live_review_history",
        live_review,
        empty_message="no review rows were generated.",
    )
    _append_if_valid(
        repo,
        "live_alert_history",
        alerts,
        empty_message="no alerts were generated.",
    )
    repo.write_df("live_open_alerts", open_alerts)
    _append_if_valid(
        repo,
        "live_alert_notifications_history",
        email_events,
        empty_message="no severe alert notifications were generated.",
    )
    repo.write_df("live_model_review", live_review)
    repo.close()

    print(
        f"Wrote {len(predictions)} predictions using config={selected_config_name}"
        f" version={live_cfg.version} run_id={live_run_id}"
    )


if __name__ == "__main__":
    main()
