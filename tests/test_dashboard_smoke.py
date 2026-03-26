from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


class FakeStreamlit:
    def __init__(self):
        self.secrets = {"APP_PASSWORD": "secret"}
        self.sidebar = self

    def set_page_config(self, **_kwargs):
        return None

    def text_input(self, *_args, **_kwargs):
        return "secret"

    def warning(self, *_args, **_kwargs):
        return None

    def stop(self):
        return None

    def title(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def metric(self, *_args, **_kwargs):
        return None

    def dataframe(self, *_args, **_kwargs):
        return None

    def caption(self, *_args, **_kwargs):
        return None


class FakeRepo:
    def read_df(self, _query: str):
        return pl.DataFrame(
            {
                "fixture_id": ["f1"],
                "home_team": ["A"],
                "away_team": ["B"],
                "expected_home_goals": [1.2],
                "expected_away_goals": [0.9],
                "timestamp_utc": ["2026-01-01T00:00:00"],
            }
        )

    def close(self):
        return None


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dashboard_pages_smoke_load(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    from footballmodel.storage import repository

    monkeypatch.setattr(repository, "DuckRepository", FakeRepo)

    overview = _load_module(Path("app/Overview.py"), "Overview")
    assert overview is not None

    app_entry = _load_module(Path("app/app.py"), "footballmodel_app")
    assert app_entry is not None

    page_paths = [
        Path("app/pages/1X2.py"),
        Path("app/pages/Over_Under_2_5.py"),
        Path("app/pages/BTTS.py"),
        Path("app/pages/Correct_Score.py"),
        Path("app/pages/Asian_Handicap.py"),
        Path("app/pages/Fixture_Drilldown.py"),
        Path("app/pages/Backtest_Lab.py"),
        Path("app/pages/Experiments.py"),
        Path("app/pages/Run_Control.py"),
        Path("app/pages/History.py"),
    ]

    for idx, path in enumerate(page_paths):
        module = _load_module(path, f"page_{idx}")
        assert module is not None
