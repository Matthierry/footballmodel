from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from footballmodel.ingestion.clubelo import elo_as_of, load_clubelo_csv
from footballmodel.ingestion.football_data import (
    FOOTBALL_DATA_MAPPING,
    build_football_data_raw_file,
    load_football_data_config,
    load_football_data_csv,
)
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


def test_load_football_data_config_requires_non_empty_sources(tmp_path: Path):
    cfg_path = tmp_path / "football_data_sources.yaml"
    cfg_path.write_text("seasons: ['2526']\nsources: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_football_data_config(cfg_path)


def test_build_football_data_raw_file_merges_sources_and_deduplicates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "football_data_sources.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seasons: ['2526']",
                "fail_fast: false",
                "persist_snapshots: true",
                "url_template: 'https://example.test/{season_code}/{csv_code}.csv'",
                "sources:",
                "  - league_code: ENG1",
                "    csv_code: E0",
                "  - league_code: GER1",
                "    csv_code: D1",
            ]
        ),
        encoding="utf-8",
    )

    payloads = {
        "https://example.test/2526/E0.csv": (
            "Date,Div,HomeTeam,AwayTeam,FTHG,FTAG\n"
            "15/08/2025,E0,Man City,Arsenal,2,1\n"
            "15/08/2025,E0,Man City,Arsenal,2,1\n"
        ),
        "https://example.test/2526/D1.csv": "Date,Div,HomeTeam,AwayTeam,FTHG,FTAG\n15/08/2025,D1,Bayern,Dortmund,3,2\n",
    }

    monkeypatch.setattr(
        "footballmodel.ingestion.football_data._fetch_source_csv",
        lambda url, timeout_seconds=20: payloads[url],
    )

    output_path = tmp_path / "raw" / "football_data.csv"
    snapshots_dir = tmp_path / "raw" / "snapshots"
    result = build_football_data_raw_file(config_path=cfg_path, output_path=output_path, snapshots_dir=snapshots_dir)
    merged = load_football_data_csv(output_path)

    assert result.rows_before_dedup == 3
    assert result.rows_after_dedup == 2
    assert result.failed_sources == []
    assert merged.filter(pl.col("league_code") == "ENG1").height == 1
    assert {"league_code", "season_code", "source_url", "fetched_at_utc", "fixture_id"}.issubset(merged.columns)
    assert (snapshots_dir / "2526_ENG1.csv").exists()
    assert (snapshots_dir / "2526_GER1.csv").exists()


def test_build_football_data_raw_file_fails_when_all_sources_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "football_data_sources.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "seasons: ['2526']",
                "sources:",
                "  - league_code: ENG1",
                "    csv_code: E0",
            ]
        ),
        encoding="utf-8",
    )

    def _raise(_url: str, timeout_seconds: int = 20) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr("footballmodel.ingestion.football_data._fetch_source_csv", _raise)

    with pytest.raises(RuntimeError, match="failed for all configured sources"):
        build_football_data_raw_file(
            config_path=cfg_path,
            output_path=tmp_path / "football_data.csv",
            snapshots_dir=tmp_path / "snapshots",
        )
