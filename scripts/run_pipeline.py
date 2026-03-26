from __future__ import annotations

from pathlib import Path

import polars as pl

from footballmodel.config.settings import load_app_config
from footballmodel.ingestion.clubelo import load_clubelo_csv
from footballmodel.ingestion.football_data import load_football_data_csv
from footballmodel.orchestration.pipeline import (
    build_pre_kickoff_benchmark_snapshots,
    build_prediction_time_benchmark_snapshots,
    run_fixture_prediction,
)
from footballmodel.storage.repository import DuckRepository


def main() -> None:
    cfg = load_app_config("config/runtime.yaml")
    repo = DuckRepository()

    matches_path = Path("data/raw/football_data.csv")
    elo_path = Path("data/raw/clubelo.csv")

    if not matches_path.exists() or not elo_path.exists():
        print("Raw files missing; expected data/raw/football_data.csv and data/raw/clubelo.csv")
        return

    matches = load_football_data_csv(matches_path)
    elos = load_clubelo_csv(elo_path)

    upcoming = matches.filter(pl.col("home_goals").is_null())
    history = matches.filter(pl.col("home_goals").is_not_null())

    predictions = []
    prediction_markets: list[dict[str, object]] = []
    snapshots: list[pl.DataFrame] = []
    for fixture in upcoming.iter_rows(named=True):
        pred = run_fixture_prediction(history, fixture, elos, cfg)
        predictions.append(pred)
        snapshots.append(build_prediction_time_benchmark_snapshots(fixture, pred["timestamp_utc"]))
        for row in pred["markets"]:
            prediction_markets.append(
                {
                    "fixture_id": pred["fixture_id"],
                    "prediction_timestamp_utc": pred["timestamp_utc"],
                    "market": row["market"],
                    "outcome": row["outcome"],
                    "line": float(str(row["outcome"]).split("_", maxsplit=1)[1]) if str(row["market"]) == "AH" else (2.5 if str(row["market"]) == "OU25" else None),
                    "model_probability": row["model_probability"],
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

    repo.write_df("curated_matches", matches)
    repo.write_df("elo_history", elos)
    repo.write_df("model_runs", pl.DataFrame(predictions))
    repo.write_df("model_market_predictions", pl.DataFrame(prediction_markets))
    if snapshots:
        repo.upsert_benchmark_snapshots(pl.concat(snapshots))
    if upcoming.height:
        repo.upsert_benchmark_snapshots(build_pre_kickoff_benchmark_snapshots(upcoming))
    repo.close()
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
