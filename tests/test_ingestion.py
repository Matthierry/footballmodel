from __future__ import annotations

from pathlib import Path

import polars as pl

from footballmodel.ingestion.clubelo import elo_as_of, load_clubelo_csv
from footballmodel.ingestion.football_data import FOOTBALL_DATA_MAPPING, load_football_data_csv
from footballmodel.ingestion.snapshot import run_guarded_ingestion, sha256_file, should_rerun


def test_football_data_ingestion_parses_expected_schema(football_data_csv: Path):
    df = load_football_data_csv(football_data_csv)
    assert {"fixture_id", "match_date", "home_team", "away_team", "home_goals", "away_goals"}.issubset(df.columns)
    assert df["home_team"].to_list()[0] == "Man City"
    assert str(df["match_date"][0]) == "2024-08-15"


def test_clubelo_ingestion_parses_expected_schema(clubelo_csv: Path):
    df = load_clubelo_csv(clubelo_csv)
    assert df.columns == ["elo_date", "team", "country", "elo"]
    assert float(df.filter(pl.col("team") == "Manchester City")["elo"][0]) == 1900.0


def test_source_mapping_columns_unique_and_core_fields_present():
    vals = list(FOOTBALL_DATA_MAPPING.values())
    assert len(vals) == len(set(vals))
    assert FOOTBALL_DATA_MAPPING["HomeTeam"] == "home_team"
    assert FOOTBALL_DATA_MAPPING["AwayTeam"] == "away_team"
    assert FOOTBALL_DATA_MAPPING["FTHG"] == "home_goals"


def test_hash_detection_skips_unchanged_source(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("abc", encoding="utf-8")
    last_hash = tmp_path / "last_hash.txt"
    first = sha256_file(source)
    last_hash.write_text(first, encoding="utf-8")

    assert should_rerun(last_hash, first) is False


def test_guarded_ingestion_runs_and_persists_hash(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("payload-v1", encoding="utf-8")
    last_hash = tmp_path / "last_hash.txt"
    touched = {"ran": False}

    def _runner():
        touched["ran"] = True

    status = run_guarded_ingestion(source, last_hash, _runner)

    assert touched["ran"] is True
    assert status.status == "ran"
    assert last_hash.exists()


def test_guarded_ingestion_failure_does_not_overwrite_last_good_hash(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("payload-v2", encoding="utf-8")
    last_hash = tmp_path / "last_hash.txt"
    last_hash.write_text("known-good-hash", encoding="utf-8")

    def _runner():
        raise RuntimeError("upstream source unavailable")

    status = run_guarded_ingestion(source, last_hash, _runner)

    assert status.status == "failed"
    assert status.alert is True
    assert "unavailable" in status.detail
    assert last_hash.read_text(encoding="utf-8") == "known-good-hash"


def test_elo_as_of_returns_historical_value_and_default(clubelo_csv: Path):
    elo_df = load_clubelo_csv(clubelo_csv)
    assert elo_as_of(elo_df, "Manchester City", "2024-08-11") == 1900.0
    assert elo_as_of(elo_df, "Expansion Team", "2024-08-11") == 1500.0
