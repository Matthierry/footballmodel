from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from footballmodel.config.settings import load_app_config
from footballmodel.features.feature_builder import build_match_features
from footballmodel.ingestion.clubelo import load_clubelo_csv
from footballmodel.ingestion.football_data import load_football_data_csv
from footballmodel.markets.derivation import derive_1x2, derive_ah, derive_btts, derive_correct_score_top5, derive_ou25
from footballmodel.orchestration.pipeline import run_fixture_prediction
from footballmodel.storage.repository import DuckRepository


def test_pipeline_end_to_end_prediction_and_persistence(tmp_path: Path):
    fixtures_csv = tmp_path / "fixtures.csv"
    fixtures_csv.write_text(
        "Date,Div,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,B365H,B365D,B365A,BFH,BFD,BFA\n"
        "01/08/2024,ENG1,Manchester City,Manchester United,2,1,14,9,8,4,1.70,3.90,5.20,1.75,4.00,5.40\n"
        "08/08/2024,ENG1,Manchester United,Chelsea,1,1,10,11,5,5,2.50,3.30,2.75,2.45,3.35,2.85\n"
        "15/08/2024,ENG1,Chelsea,Manchester City,0,2,8,13,3,7,3.20,3.45,2.10,3.10,3.55,2.20\n"
        "22/08/2024,ENG1,Manchester City,Chelsea,,,,,,,,,,\n",
        encoding="utf-8",
    )

    elo_csv = tmp_path / "elo.csv"
    elo_csv.write_text(
        "Date,Club,Country,Elo\n"
        "2024-08-01,Manchester City,ENG,1910\n"
        "2024-08-01,Manchester United,ENG,1760\n"
        "2024-08-01,Chelsea,ENG,1740\n",
        encoding="utf-8",
    )

    matches = load_football_data_csv(fixtures_csv)
    elo_history = load_clubelo_csv(elo_csv)

    played = matches.filter(pl.col("home_goals").is_not_null())
    fixture = matches.filter((pl.col("match_date") == date(2024, 8, 22)) & (pl.col("home_goals").is_null())).to_dicts()[0]

    features = build_match_features(played, lookback_days=180, half_life_days=60)
    assert {"team", "atk_strength", "def_concede"}.issubset(features.columns)
    assert features.height >= 2

    cfg = load_app_config("config/runtime.yaml")
    prediction = run_fixture_prediction(played, fixture, elo_history, cfg)

    score_matrix = np.array(prediction["score_matrix"], dtype=float)
    one_x_two = derive_1x2(score_matrix)
    ou25 = derive_ou25(score_matrix)
    btts = derive_btts(score_matrix)
    cs_top5 = derive_correct_score_top5(score_matrix)
    ah = derive_ah(score_matrix)

    assert prediction["fixture_id"] == fixture["fixture_id"]
    assert score_matrix.shape == (cfg.runtime.max_goals + 1, cfg.runtime.max_goals + 1)
    assert np.isclose(score_matrix.sum(), 1.0, atol=1e-12)
    assert np.isclose(sum(one_x_two.values()), 1.0, atol=1e-12)
    assert np.isclose(ou25["over_2_5"] + ou25["under_2_5"], 1.0, atol=1e-12)
    assert np.isclose(btts["btts_yes"] + btts["btts_no"], 1.0, atol=1e-12)
    assert len(cs_top5) == 5
    assert "recommended_line" in ah

    db_path = tmp_path / "footballmodel_test.duckdb"
    repo = DuckRepository(str(db_path))
    try:
        repo.write_df("predictions", pl.DataFrame([prediction]))
        repo.write_df("market_rows", pl.DataFrame(prediction["markets"]))

        persisted_predictions = repo.read_df("select fixture_id, home_team, away_team from predictions")
        persisted_markets = repo.read_df(
            "select market, outcome from market_rows order by market, outcome"
        )
    finally:
        repo.close()

    assert persisted_predictions.height == 1
    assert persisted_predictions["fixture_id"][0] == fixture["fixture_id"]
    assert persisted_markets.filter(pl.col("market") == "1X2").height == 3
    assert persisted_markets.filter(pl.col("market") == "OU25").height == 2
    assert persisted_markets.filter(pl.col("market") == "BTTS").height == 2
    assert persisted_markets.filter(pl.col("market") == "AH").height == 2
