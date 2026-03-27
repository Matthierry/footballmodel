from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import importlib.util


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

    def append_df(self, table: str, df: pl.DataFrame) -> None:
        self.append_calls.append((table, df))

    def upsert_benchmark_snapshots(self, _df: pl.DataFrame) -> None:
        return

    def read_df(self, query: str) -> pl.DataFrame:
        if "benchmark_snapshots" in query:
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
        raise RuntimeError("table missing")

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
