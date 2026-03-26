import polars as pl

from footballmodel.ingestion.football_data import FOOTBALL_DATA_MAPPING


def test_source_mapping_has_required_fields():
    assert FOOTBALL_DATA_MAPPING["HomeTeam"] == "home_team"
    assert FOOTBALL_DATA_MAPPING["AwayTeam"] == "away_team"
    assert FOOTBALL_DATA_MAPPING["FTHG"] == "home_goals"


def test_mapping_columns_unique():
    vals = list(FOOTBALL_DATA_MAPPING.values())
    assert len(vals) == len(set(vals))
