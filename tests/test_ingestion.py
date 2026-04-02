from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from footballmodel.ingestion.clubelo import elo_as_of, load_clubelo_csv
from footballmodel.ingestion.clubelo import build_clubelo_raw_file, load_clubelo_config
from footballmodel.ingestion.football_data import (
    FOOTBALL_DATA_MAPPING,
    _normalize_football_data_df,
    _parse_upcoming_fixtures_payload,
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


def test_load_football_data_csv_sanitizes_bom_prefixed_historical_div_header(tmp_path: Path):
    csv_path = tmp_path / "historical_with_bom.csv"
    csv_path.write_text(
        "\ufeffDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,15/08/2025,Man City,Arsenal,2,1\n",
        encoding="utf-8",
    )
    df = load_football_data_csv(csv_path)
    row = df.row(0, named=True)
    assert row["source_div"] == "E0"
    assert row["league_code"] == "E0"


def test_load_football_data_csv_maps_source_div_to_league_code_for_canonical_rows(tmp_path: Path):
    csv_path = tmp_path / "canonical_upcoming.csv"
    csv_path.write_text(
        "\n".join(
            [
                "match_date,home_team,away_team,avg_home_odds,avg_draw_odds,avg_away_odds,source_div,season_code,source_url,fetched_at_utc",
                "2026-08-24,Chelsea,Liverpool,2.3,3.4,2.9,E0,,https://example.test/fixtures.csv,2026-04-02T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    df = load_football_data_csv(csv_path, csv_to_league={"E0": "ENG1"})
    row = df.row(0, named=True)
    assert row["source_div"] == "E0"
    assert row["league_code"] == "ENG1"
    assert row["league"] == "ENG1"
    assert row["fixture_id"].startswith("ENG1_")


def test_normalize_football_data_df_does_not_require_league_code_before_mapping():
    raw_df = pl.DataFrame(
        {
            "Date": ["24/08/2026"],
            "HomeTeam": ["Chelsea"],
            "AwayTeam": ["Liverpool"],
            "B365H": [2.3],
            "B365D": [3.4],
            "B365A": [2.9],
        }
    )

    normalized = _normalize_football_data_df(
        raw_df,
        source_url="https://example.test/fixtures.csv",
        fetched_at_utc="2026-04-02T00:00:00+00:00",
    )
    row = normalized.row(0, named=True)
    assert row["source_div"] is None
    assert row["league_code"] is None
    assert row["fixture_id"].endswith("_Chelsea_Liverpool")


def test_clubelo_ingestion_parses_expected_schema(clubelo_csv: Path):
    df = load_clubelo_csv(clubelo_csv)
    assert {"elo_date", "team", "country", "elo"}.issubset(df.columns)
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
                "upcoming_fixtures_url: 'https://example.test/fixtures.csv'",
                "include_upcoming_fixtures: true",
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
        "https://example.test/fixtures.csv": (
            "Date,Div,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
            "24/08/2026,E0,Chelsea,Liverpool,2.3,3.4,2.9\n"
        ),
    }

    monkeypatch.setattr(
        "footballmodel.ingestion.football_data._fetch_source_csv",
        lambda url, timeout_seconds=20: payloads[url],
    )

    output_path = tmp_path / "raw" / "football_data.csv"
    snapshots_dir = tmp_path / "raw" / "snapshots"
    result = build_football_data_raw_file(config_path=cfg_path, output_path=output_path, snapshots_dir=snapshots_dir)
    merged = load_football_data_csv(output_path)

    assert result.future_fixtures_rows_fetched == 1
    assert result.rows_before_dedup == 4
    assert result.rows_after_dedup == 3
    assert result.future_fixtures_fetched == 1
    assert result.future_fixtures_after_normalization == 1
    assert result.future_fixtures_after_dedup == 1
    assert result.future_fixtures_with_published_odds == 1
    assert result.future_fixtures_with_league_code_after_normalization == 1
    assert result.future_fixtures_with_league_code_after_dedup == 1
    assert result.source_div_column_found is True
    assert result.league_code_created_from_source_div is True
    assert result.failed_sources == []
    assert "upcoming:fixtures_csv" in result.fetched_sources
    assert merged.filter((pl.col("league_code") == "ENG1") & (pl.col("source_dataset") == "historical_league_csv")).height == 1
    assert {"league_code", "season_code", "source_url", "fetched_at_utc", "fixture_id"}.issubset(merged.columns)
    assert merged.filter(pl.col("source_dataset") == "upcoming_fixtures_csv").height == 1
    upcoming_row = merged.filter(pl.col("source_dataset") == "upcoming_fixtures_csv").row(0, named=True)
    assert upcoming_row["source_div"] == "E0"
    assert upcoming_row["league_code"] == "ENG1"
    assert upcoming_row["league"] == "ENG1"
    assert (snapshots_dir / "2526_ENG1.csv").exists()
    assert (snapshots_dir / "2526_GER1.csv").exists()
    assert (snapshots_dir / "upcoming_matches.csv").exists()


def test_parse_upcoming_fixtures_payload_normalizes_rows_and_marks_future():
    fixture_day = (date.today() + timedelta(days=3)).strftime("%d/%m/%Y")
    payload = f"Date,Div,HomeTeam,AwayTeam,B365H,B365D,B365A\n{fixture_day},E0,Man City,Arsenal,1.8,3.7,4.5\n"
    parsed, diagnostics = _parse_upcoming_fixtures_payload(
        payload,
        source_url="https://example.test/fixtures.csv",
        fetched_at_utc="2026-03-27T00:00:00+00:00",
        csv_to_league={"E0": "ENG1"},
    )

    row = parsed.row(0, named=True)
    assert row["league"] == "ENG1"
    assert row["league_code"] == "ENG1"
    assert row["home_team"] == "Man City"
    assert row["is_future_fixture"] is True
    assert row["fixture_status"] == "upcoming"
    assert row["odds_capture_type"] == "published_at_source_fetch"
    assert diagnostics.fetched_rows == 1
    assert diagnostics.fetched_future_rows == 1
    assert diagnostics.future_rows == 1
    assert diagnostics.future_rows_with_published_odds == 1
    assert diagnostics.future_rows_with_league_code == 1
    assert diagnostics.raw_div_column_found is True
    assert diagnostics.source_div_populated_rows == 1
    assert diagnostics.league_code_populated_rows == 1
    assert diagnostics.mapped_league_code_rows == 1


def test_parse_upcoming_fixtures_payload_parses_datetime_dates():
    fixture_date = date.today() + timedelta(days=4)
    payload = (
        "Date,Div,HomeTeam,AwayTeam,B365H,B365D,B365A\n"
        f"{fixture_date.isoformat()} 19:45,E0,Man City,Arsenal,1.8,3.7,4.5\n"
    )
    parsed, diagnostics = _parse_upcoming_fixtures_payload(
        payload,
        source_url="https://example.test/fixtures.csv",
        fetched_at_utc="2026-03-27T00:00:00+00:00",
        csv_to_league={"E0": "ENG1"},
    )
    assert parsed.height == 1
    assert str(parsed.row(0, named=True)["match_date"]) == fixture_date.isoformat()
    assert diagnostics.future_rows == 1


def test_parse_upcoming_fixtures_payload_sanitizes_bom_prefixed_div_header():
    fixture_day = (date.today() + timedelta(days=2)).strftime("%d/%m/%Y")
    payload = f"\ufeffDiv,Date,HomeTeam,AwayTeam,B365H,B365D,B365A\nE0,{fixture_day},Wigan,Reading,2.4,3.3,2.9\n"
    parsed, diagnostics = _parse_upcoming_fixtures_payload(
        payload,
        source_url="https://example.test/fixtures.csv",
        fetched_at_utc="2026-04-01T00:00:00+00:00",
        csv_to_league={"E0": "ENG1"},
    )

    row = parsed.row(0, named=True)
    assert row["source_div"] == "E0"
    assert row["league_code"] == "ENG1"
    assert row["league"] == "ENG1"
    assert diagnostics.bom_header_sanitized is True
    assert diagnostics.sanitized_header_count == 1
    assert diagnostics.raw_div_column_found is True
    assert diagnostics.source_div_populated_rows == 1
    assert diagnostics.league_code_populated_rows == 1


def test_parse_upcoming_fixtures_payload_handles_schema_change_gracefully():
    payload = "Kickoff,Division,Home,Away\n2026-03-28 19:45,E0,Man City,Arsenal\n"
    with pytest.raises(RuntimeError, match="fixtures.csv schema changed"):
        _parse_upcoming_fixtures_payload(
            payload,
            source_url="https://example.test/fixtures.csv",
            fetched_at_utc="2026-03-27T00:00:00+00:00",
            csv_to_league={"E0": "ENG1"},
        )


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


def test_load_clubelo_config_rejects_unknown_frequency(tmp_path: Path):
    cfg_path = tmp_path / "clubelo_sources.yaml"
    cfg_path.write_text("date_frequency: hourly\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date_frequency"):
        load_clubelo_config(cfg_path)


def test_build_clubelo_raw_file_builds_canonical_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    matches_path = tmp_path / "football_data.csv"
    matches_path.write_text(
        "\n".join(
            [
                "Date,Div,HomeTeam,AwayTeam,FTHG,FTAG",
                "15/08/2024,ENG1,Manchester City,Manchester United,2,1",
                "20/08/2024,ENG1,Leeds United,Derby County,1,1",
            ]
        ),
        encoding="utf-8",
    )
    cfg_path = tmp_path / "clubelo_sources.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "url_template: 'https://elo.test/{date}.csv'",
                "date_format: '%Y-%m-%d'",
                "date_frequency: 'matchdays'",
                "start_date: '2024-08-01'",
                "end_date: '2024-08-31'",
                "include_match_dates_from_football_data: true",
                "include_today: false",
                "persist_snapshots: true",
                "fail_fast: true",
                "leagues: ['ENG1']",
            ]
        ),
        encoding="utf-8",
    )
    payloads = {
        "https://elo.test/2024-08-15.csv": "Rank,Club,Country,Level,Elo,From,To\n1,Manchester City,ENG,1,1900,2024-08-15,2024-08-16\n2,Manchester United,ENG,1,1780,2024-08-15,2024-08-16\n",
        "https://elo.test/2024-08-20.csv": "Rank,Club,Country,Level,Elo,From,To\n1,Leeds United,ENG,2,1650,2024-08-20,2024-08-21\n2,Derby County,ENG,2,1600,2024-08-20,2024-08-21\n",
    }
    monkeypatch.setattr("footballmodel.ingestion.clubelo._fetch_source_csv", lambda url, timeout_seconds=20: payloads[url])

    output_path = tmp_path / "raw" / "clubelo.csv"
    snapshots_dir = tmp_path / "raw" / "clubelo_sources"
    result = build_clubelo_raw_file(
        config_path=cfg_path,
        output_path=output_path,
        football_data_path=matches_path,
        snapshots_dir=snapshots_dir,
    )
    df = load_clubelo_csv(output_path)

    assert result.failed_dates == []
    assert set(result.fetched_dates) == {"2024-08-15", "2024-08-20"}
    assert {"elo_date", "team", "country", "elo", "source_url", "fetched_at_utc"}.issubset(df.columns)
    assert df.filter((pl.col("elo_date") == date(2024, 8, 15)) & (pl.col("team") == "Manchester City")).height == 1
    assert df.filter((pl.col("elo_date") == date(2024, 8, 20)) & (pl.col("team") == "Leeds United")).height == 1
    assert (snapshots_dir / "2024-08-15.csv").exists()
    assert (snapshots_dir / "2024-08-20.csv").exists()


def test_build_clubelo_raw_file_fails_when_all_dates_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    matches_path = tmp_path / "football_data.csv"
    matches_path.write_text(
        "Date,Div,HomeTeam,AwayTeam\n15/08/2024,ENG1,Manchester City,Manchester United\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "clubelo_sources.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "url_template: 'https://elo.test/{date}.csv'",
                "date_frequency: 'matchdays'",
                "include_match_dates_from_football_data: true",
                "include_today: false",
                "fail_fast: false",
            ]
        ),
        encoding="utf-8",
    )

    def _raise(_url: str, timeout_seconds: int = 20) -> str:
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr("footballmodel.ingestion.clubelo._fetch_source_csv", _raise)

    with pytest.raises(RuntimeError, match="failed for all configured dates"):
        build_clubelo_raw_file(
            config_path=cfg_path,
            output_path=tmp_path / "raw" / "clubelo.csv",
            football_data_path=matches_path,
            snapshots_dir=tmp_path / "raw" / "clubelo_sources",
        )
