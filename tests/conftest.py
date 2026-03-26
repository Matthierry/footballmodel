from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


@pytest.fixture
def football_data_csv(tmp_path: Path) -> Path:
    path = tmp_path / "football_data.csv"
    path.write_text(
        "Date,Div,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,B365H,B365D,B365A,BFH,BFD,BFA\n"
        "15/08/2024,ENG1,Man City,Man United,2,1,12,9,6,3,1.70,4.00,5.00,1.72,4.10,5.20\n"
        "22/08/2024,ENG1,Leeds,Derby,,,,,,,,,,\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def clubelo_csv(tmp_path: Path) -> Path:
    path = tmp_path / "clubelo.csv"
    path.write_text(
        "Date,Club,Country,Elo\n"
        "2024-08-10,Manchester City,ENG,1900\n"
        "2024-08-10,Manchester United,ENG,1780\n"
        "2024-08-20,Leeds United,ENG,1650\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_matches():
    import polars as pl
    return pl.DataFrame(
        {
            "fixture_id": ["f1", "f2", "f3", "f4"],
            "league": ["ENG1", "ENG1", "ENG1", "ENG1"],
            "match_date": [date(2024, 8, 1), date(2024, 8, 8), date(2024, 8, 15), date(2024, 8, 22)],
            "home_team": ["A", "B", "A", "C"],
            "away_team": ["B", "A", "C", "A"],
            "home_goals": [1, 2, 0, None],
            "away_goals": [0, 1, 1, None],
            "home_sot": [4, 5, 3, None],
            "away_sot": [2, 4, 2, None],
            "avg_home_odds": [2.0, 2.1, 2.2, 2.0],
            "avg_draw_odds": [3.3, 3.2, 3.1, 3.3],
            "avg_away_odds": [3.8, 3.7, 3.6, 3.8],
            "bf_home_odds": [2.0, 2.1, 2.2, 2.0],
            "bf_draw_odds": [3.3, 3.2, 3.1, 3.3],
            "bf_away_odds": [3.8, 3.7, 3.6, 3.8],
            "avg_over_2_5_odds": [1.95, 1.98, 2.0, 1.92],
            "avg_under_2_5_odds": [1.92, 1.88, 1.86, 1.94],
            "bf_over_2_5_odds": [1.98, 2.0, 2.04, 1.96],
            "bf_under_2_5_odds": [1.9, 1.85, 1.83, 1.91],
            "avg_btts_yes_odds": [1.8, 1.85, 1.9, 1.83],
            "avg_btts_no_odds": [2.05, 2.0, 1.98, 2.02],
            "bf_btts_yes_odds": [1.83, 1.88, 1.93, 1.86],
            "bf_btts_no_odds": [2.02, 1.97, 1.95, 1.99],
            "ah_line": [-0.5, 0.0, 0.5, -0.5],
            "avg_ah_home_odds": [1.9, 2.0, 1.95, 1.92],
            "avg_ah_away_odds": [2.0, 1.9, 1.96, 1.98],
            "bf_ah_home_odds": [1.92, 2.02, 1.98, 1.94],
            "bf_ah_away_odds": [1.98, 1.88, 1.93, 1.96],
        }
    )


@pytest.fixture
def deterministic_matrix():
    import numpy as np
    matrix = np.zeros((3, 3), dtype=float)
    matrix[0, 0] = 0.20
    matrix[1, 0] = 0.15
    matrix[0, 1] = 0.10
    matrix[1, 1] = 0.15
    matrix[2, 0] = 0.10
    matrix[0, 2] = 0.05
    matrix[2, 1] = 0.10
    matrix[1, 2] = 0.05
    matrix[2, 2] = 0.10
    return matrix
