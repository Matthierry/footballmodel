from __future__ import annotations

from pathlib import Path

import polars as pl

from footballmodel.config.settings import load_app_config
from footballmodel.ingestion.clubelo import load_clubelo_csv
from footballmodel.ingestion.football_data import load_football_data_csv
from footballmodel.orchestration.pipeline import run_fixture_prediction
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
    for fixture in upcoming.iter_rows(named=True):
        pred = run_fixture_prediction(history, fixture, elos, cfg)
        predictions.append(pred)

    repo.write_df("curated_matches", matches)
    repo.write_df("elo_history", elos)
    repo.write_df("model_runs", pl.DataFrame(predictions))
    repo.close()
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
