from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl


class FakeStreamlit:
    def __init__(self):
        self.secrets = {"APP_PASSWORD": "secret"}
        self.sidebar = self
        self.session_state = {}

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

    def columns(self, n: int):
        return [self for _ in range(n)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def date_input(self, *_args, **kwargs):
        return kwargs.get("value")

    def multiselect(self, *_args, **kwargs):
        return kwargs.get("default", [])

    def number_input(self, *_args, **kwargs):
        return kwargs.get("value", 1.0)

    def button(self, *_args, **_kwargs):
        return False

    def selectbox(self, _label, options, index=0, **_kwargs):
        return options[index] if options else None

    def success(self, *_args, **_kwargs):
        return None

    def subheader(self, *_args, **_kwargs):
        return None

    def markdown(self, *_args, **_kwargs):
        return None


class FakeRepo:
    def read_df(self, _query: str):
        if "live_review_history" in _query:
            return pl.DataFrame(
                {
                    "live_run_id": ["live_1", "live_1"],
                    "run_timestamp_utc": ["2026-01-01T00:00:00", "2026-01-01T00:00:00"],
                    "config_name": ["champion_v1", "champion_v1"],
                    "config_version": ["2026.03.1", "2026.03.1"],
                    "fixture_id": ["f1", "f1"],
                    "match_date": [date(2026, 1, 1), date(2026, 1, 1)],
                    "league": ["ENG1", "ENG1"],
                    "home_team": ["A", "A"],
                    "away_team": ["B", "B"],
                    "market": ["1X2", "1X2"],
                    "outcome": ["home", "away"],
                    "prediction_benchmark_source": ["exchange", "exchange"],
                    "prediction_benchmark_price": [2.1, 3.2],
                    "later_benchmark_price": [2.0, 3.4],
                    "settlement_status": ["settled", "settled"],
                    "result_status": ["won", "lost"],
                    "clv": [0.02, -0.01],
                    "value_flag": [True, False],
                }
            )
        if "live_run_summaries_history" in _query:
            return pl.DataFrame(
                {
                    "live_run_id": ["live_1"],
                    "run_timestamp_utc": ["2026-01-01T00:00:00"],
                    "config_name": ["champion_v1"],
                    "config_version": ["2026.03.1"],
                    "fixtures_scored": [1],
                    "market_predictions": [2],
                    "review_rows": [2],
                }
            )
        if "live_open_alerts" in _query:
            return pl.DataFrame(
                {
                    "alert_id": ["ALRT-1"],
                    "alert_timestamp_utc": ["2026-01-01T00:00:00"],
                    "alert_type": ["clv_deterioration"],
                    "severity": ["warning"],
                    "market": ["1X2"],
                    "league": ["ENG1"],
                    "status": ["open"],
                }
            )
        if "live_alert_history" in _query:
            return pl.DataFrame(
                {
                    "alert_id": ["ALRT-1"],
                    "alert_timestamp_utc": ["2026-01-01T00:00:00"],
                    "alert_type": ["clv_deterioration"],
                    "severity": ["warning"],
                    "market": ["1X2"],
                    "league": ["ENG1"],
                    "status": ["open"],
                }
            )
        if "curated_matches" in _query:
            return pl.DataFrame(
                {
                    "fixture_id": ["f1"],
                    "league": ["ENG1"],
                    "match_date": [date(2026, 1, 1)],
                    "home_team": ["A"],
                    "away_team": ["B"],
                    "home_goals": [1],
                    "away_goals": [0],
                }
            )
        if "elo_history" in _query:
            return pl.DataFrame({"elo_date": [date(2025, 1, 1)], "team": ["A"], "country": ["ENG"], "elo": [1600.0]})
        if "backtest_runs" in _query:
            return pl.DataFrame({"run_id": ["bt_1"], "created_at": ["2026-01-01T00:00:00"]})
        if "backtest_metrics" in _query:
            return pl.DataFrame(
                {
                    "run_id": ["bt_1"],
                    "breakdown": ["overall"],
                    "group_key": ["all"],
                    "raw_log_loss": [0.55],
                    "calibrated_log_loss": [0.5],
                    "raw_brier_score": [0.22],
                    "calibrated_brier_score": [0.2],
                    "calibration_error": [0.1],
                    "raw_calibration_error": [0.12],
                    "calibrated_calibration_error": [0.1],
                    "avg_clv": [0.01],
                    "share_beating_close": [0.55],
                    "flat_stake_roi": [0.03],
                    "strike_rate": [0.52],
                    "max_drawdown": [-1.0],
                    "bets": [10],
                    "robustness_score": [0.8],
                }
            )
        if "experiment_rankings" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "created_at": ["2026-01-01T00:00:00"]})
        if "experiment_sweep_metadata" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "created_at": ["2026-01-01T00:00:00"], "run_count": [1]})
        if "experiment_champion_view" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "run_id": ["bt_1"], "selection_role": ["champion"], "ranking_score": [1111]})
        if "experiment_calibration_buckets" in _query:
            return pl.DataFrame(
                {
                    "sweep_id": ["sw_1"],
                    "run_id": ["bt_1"],
                    "market": ["1X2"],
                    "league": ["ENG1"],
                    "probability_bucket": [0.5],
                    "samples": [10],
                }
            )
        if "experiment_clv_segments" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "run_id": ["bt_1"], "market": ["1X2"], "league": ["ENG1"], "avg_clv": [0.01]})
        if "experiment_value_flag_hit_rate" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "run_id": ["bt_1"], "market": ["1X2"], "league": ["ENG1"], "bets": [4]})
        if "experiment_false_positive_zones" in _query:
            return pl.DataFrame({"sweep_id": ["sw_1"], "run_id": ["bt_1"], "false_positives": [2]})
        if "benchmark_snapshots" in _query:
            return pl.DataFrame(
                {
                    "fixture_id": ["f1"],
                    "market": ["1X2"],
                    "outcome": ["home"],
                    "line": [None],
                    "benchmark_price": [2.1],
                    "benchmark_source": ["exchange"],
                    "snapshot_type": ["prediction_time"],
                    "snapshot_timestamp_utc": ["2026-01-01T00:00:00"],
                }
            )
        return pl.DataFrame(
            {
                "fixture_id": ["f1"],
                "home_team": ["A"],
                "away_team": ["B"],
                "expected_home_goals": [1.2],
                "expected_away_goals": [0.9],
                "timestamp_utc": ["2026-01-01T00:00:00"],
                "run_timestamp_utc": ["2026-01-01T00:00:00"],
                "config_name": ["champion_v1"],
                "config_version": ["2026.03.1"],
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
