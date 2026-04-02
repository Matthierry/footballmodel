from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import importlib.util

from footballmodel.storage.repository import DuckRepository

_RUN_PIPELINE_SPEC = importlib.util.spec_from_file_location("run_pipeline", Path("scripts/run_pipeline.py"))
assert _RUN_PIPELINE_SPEC and _RUN_PIPELINE_SPEC.loader
run_pipeline = importlib.util.module_from_spec(_RUN_PIPELINE_SPEC)
_RUN_PIPELINE_SPEC.loader.exec_module(run_pipeline)


class _FakeConfig:
    def __init__(self) -> None:
        self.drift_alerts = SimpleNamespace(
            severe_email_enabled=True,
            model_dump=lambda: {
                "windows_days": [7],
                "baseline_days": 30,
                "min_settled_rows": 10,
                "min_value_rows": 5,
                "min_calibration_rows": 5,
                "min_concentration_rows": 5,
            },
        )

    def resolve_live_config(self, _name: str | None):
        live_cfg = SimpleNamespace(
            leagues=["ENG1"],
            version="test-v1",
            calibration=SimpleNamespace(enabled=True),
        )
        return "champion_v1", live_cfg


class _FakeRepo:
    def __init__(self) -> None:
        self.append_calls: list[tuple[str, pl.DataFrame]] = []
        self.write_calls: list[tuple[str, pl.DataFrame]] = []

    def write_df(self, table: str, df: pl.DataFrame) -> None:
        self.write_calls.append((table, df))

    def ensure_optional_tables(self, _tables: list[str] | None = None) -> None:
        return

    def append_df(self, table: str, df: pl.DataFrame) -> None:
        self.append_calls.append((table, df))

    def upsert_benchmark_snapshots(self, _df: pl.DataFrame) -> None:
        return

    def read_table_or_empty(self, table: str, *, order_by: str | None = None, limit: int | None = None) -> pl.DataFrame:
        _ = (order_by, limit)
        if table == "benchmark_snapshots":
            return pl.DataFrame(
                schema={
                    "fixture_id": pl.Utf8,
                    "market": pl.Utf8,
                    "outcome": pl.Utf8,
                    "line": pl.Float64,
                    "benchmark_price": pl.Float64,
                    "benchmark_source": pl.Utf8,
                    "snapshot_type": pl.Utf8,
                    "snapshot_timestamp_utc": pl.Utf8,
                }
            )
        if table == "live_review_history":
            return pl.DataFrame(schema=run_pipeline.LIVE_REVIEW_SCHEMA)
        if table == "live_run_summaries_history":
            return pl.DataFrame(schema=run_pipeline.LIVE_RUN_SUMMARY_SCHEMA)
        if table == "live_alert_history":
            return pl.DataFrame(schema=run_pipeline.ALERT_SCHEMA)
        return pl.DataFrame([])

    def close(self) -> None:
        return


def test_pipeline_no_predictions_persists_seed_state_without_crashing(monkeypatch, tmp_path: Path):
    matches_path = tmp_path / "football_data.csv"
    elo_path = tmp_path / "clubelo.csv"
    matches_path.write_text("seed", encoding="utf-8")
    elo_path.write_text("seed", encoding="utf-8")

    fake_repo = _FakeRepo()

    monkeypatch.setattr(
        run_pipeline,
        "parse_args",
        lambda: SimpleNamespace(
            config_name=None,
            football_data_config="config/football_data_sources.yaml",
            skip_football_data_fetch=True,
            clubelo_config="config/clubelo_sources.yaml",
            refresh_clubelo=False,
            skip_clubelo_fetch=True,
        ),
    )
    monkeypatch.setattr(run_pipeline, "load_app_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(run_pipeline, "DuckRepository", lambda: fake_repo)
    monkeypatch.setattr(run_pipeline, "resolve_raw_data_paths", lambda: (matches_path, elo_path))
    monkeypatch.setattr(
        run_pipeline,
        "load_football_data_csv",
        lambda _path: pl.DataFrame(
            {
                "fixture_id": ["f_hist", "f_upcoming"],
                "match_date": ["2026-03-20", "2026-03-30"],
                "league": ["ENG1", "ESP1"],
                "home_team": ["A", "C"],
                "away_team": ["B", "D"],
                "home_goals": [1, None],
                "away_goals": [0, None],
            }
        ).with_columns(pl.col("match_date").cast(pl.Date, strict=False)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "load_clubelo_csv",
        lambda _path: pl.DataFrame(
            {
                "club": ["A", "B"],
                "country": ["ENG", "ENG"],
                "date": ["2026-03-19", "2026-03-19"],
                "elo": [1600.0, 1500.0],
            }
        ),
    )

    run_pipeline.main()

    appended_tables = [table for table, _ in fake_repo.append_calls]
    assert "model_runs" not in appended_tables
    assert "model_market_predictions" not in appended_tables
    assert "live_run_summaries_history" in appended_tables

    live_summary = [df for table, df in fake_repo.append_calls if table == "live_run_summaries_history"][0]
    assert live_summary.height == 1
    assert live_summary.row(0, named=True)["market_predictions"] == 0

    written_tables = [table for table, _ in fake_repo.write_calls]
    assert "curated_matches" in written_tables
    assert "elo_history" in written_tables
    assert "live_model_review" in written_tables


def test_pipeline_bootstrap_with_real_repository_handles_empty_optional_tables(monkeypatch, tmp_path: Path):
    matches_path = tmp_path / "football_data.csv"
    elo_path = tmp_path / "clubelo.csv"
    db_path = tmp_path / "pipeline_bootstrap.duckdb"
    matches_path.write_text("seed", encoding="utf-8")
    elo_path.write_text("seed", encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline,
        "parse_args",
        lambda: SimpleNamespace(
            config_name=None,
            football_data_config="config/football_data_sources.yaml",
            skip_football_data_fetch=True,
            clubelo_config="config/clubelo_sources.yaml",
            refresh_clubelo=False,
            skip_clubelo_fetch=True,
        ),
    )
    monkeypatch.setattr(run_pipeline, "load_app_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(run_pipeline, "resolve_raw_data_paths", lambda: (matches_path, elo_path))
    monkeypatch.setattr(
        run_pipeline,
        "DuckRepository",
        lambda: DuckRepository(str(db_path)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "load_football_data_csv",
        lambda _path: pl.DataFrame(
            {
                "fixture_id": ["f_hist", "f_upcoming"],
                "match_date": ["2026-03-20", "2026-03-30"],
                "league": ["ENG1", "ESP1"],
                "home_team": ["A", "C"],
                "away_team": ["B", "D"],
                "home_goals": [1, None],
                "away_goals": [0, None],
            }
        ).with_columns(pl.col("match_date").cast(pl.Date, strict=False)),
    )
    monkeypatch.setattr(
        run_pipeline,
        "load_clubelo_csv",
        lambda _path: pl.DataFrame(
            {
                "club": ["A", "B"],
                "country": ["ENG", "ENG"],
                "date": ["2026-03-19", "2026-03-19"],
                "elo": [1600.0, 1500.0],
            }
        ),
    )

    run_pipeline.main()

    repo = DuckRepository(str(db_path))
    try:
        snapshots = repo.read_table_or_empty("benchmark_snapshots")
        run_summaries = repo.read_table_or_empty("live_run_summaries_history")
        live_review = repo.read_table_or_empty("live_model_review")
    finally:
        repo.close()

    assert snapshots.is_empty()
    assert snapshots.columns == [
        "fixture_id",
        "market",
        "outcome",
        "line",
        "benchmark_price",
        "benchmark_source",
        "snapshot_type",
        "snapshot_timestamp_utc",
    ]
    assert run_summaries.height == 1
    assert run_summaries.row(0, named=True)["market_predictions"] == 0
    assert live_review.is_empty()


def test_pipeline_only_scores_future_fixtures_with_published_odds(monkeypatch, tmp_path: Path):
    matches_path = tmp_path / "football_data.csv"
    elo_path = tmp_path / "clubelo.csv"
    matches_path.write_text("seed", encoding="utf-8")
    elo_path.write_text("seed", encoding="utf-8")
    fake_repo = _FakeRepo()

    monkeypatch.setattr(
        run_pipeline,
        "parse_args",
        lambda: SimpleNamespace(
            config_name=None,
            football_data_config="config/football_data_sources.yaml",
            skip_football_data_fetch=True,
            clubelo_config="config/clubelo_sources.yaml",
            refresh_clubelo=False,
            skip_clubelo_fetch=True,
        ),
    )
    monkeypatch.setattr(run_pipeline, "load_app_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(run_pipeline, "DuckRepository", lambda: fake_repo)
    monkeypatch.setattr(run_pipeline, "resolve_raw_data_paths", lambda: (matches_path, elo_path))
    today = date.today()
    monkeypatch.setattr(
        run_pipeline,
        "load_football_data_csv",
        lambda _path: pl.DataFrame(
            {
                "fixture_id": ["f_hist", "f_past_unplayed", "f_no_odds", "f_has_odds"],
                "match_date": [
                    (today - timedelta(days=2)).isoformat(),
                    (today - timedelta(days=1)).isoformat(),
                    (today + timedelta(days=2)).isoformat(),
                    (today + timedelta(days=3)).isoformat(),
                ],
                "league": ["ENG1", "E0", "E0", "E0"],
                "league_code": ["ENG1", "ENG1", "ENG1", "ENG1"],
                "home_team": ["A", "P", "C", "E"],
                "away_team": ["B", "Q", "D", "F"],
                "home_goals": [1, None, None, None],
                "away_goals": [0, None, None, None],
                "is_future_fixture": [False, True, True, True],
                "avg_home_odds": [2.0, 2.2, None, 2.4],
                "avg_draw_odds": [3.0, 3.3, None, 3.2],
                "avg_away_odds": [4.0, 3.1, None, 2.8],
            }
        ).with_columns(pl.col("match_date").cast(pl.Date, strict=False)),
    )
    monkeypatch.setattr(run_pipeline, "load_clubelo_csv", lambda _path: pl.DataFrame({"team": ["A"], "elo": [1500.0]}))
    monkeypatch.setattr(
        run_pipeline,
        "run_fixture_prediction",
        lambda _hist, fixture, _elo, _cfg: {
            "fixture_id": fixture["fixture_id"],
            "timestamp_utc": "2026-03-27T00:00:00+00:00",
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "expected_home_goals": 1.0,
            "expected_away_goals": 1.0,
            "markets": [],
        },
    )

    run_pipeline.main()

    model_runs_appends = [df for table, df in fake_repo.append_calls if table == "model_runs"]
    assert len(model_runs_appends) == 1
    assert model_runs_appends[0].height == 1
    assert model_runs_appends[0]["fixture_id"].to_list() == ["f_has_odds"]
