from __future__ import annotations

from pathlib import Path

from footballmodel.config.runtime_env import get_app_password, resolve_duckdb_path, resolve_raw_data_paths


class _MissingSecretsStreamlit:
    @property
    def secrets(self):
        raise FileNotFoundError("No secrets file")


def test_get_app_password_prefers_env_without_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "render-secret")
    assert get_app_password(streamlit_module=_MissingSecretsStreamlit()) == "render-secret"


def test_get_app_password_falls_back_when_secrets_absent(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert get_app_password(streamlit_module=_MissingSecretsStreamlit(), default="fallback") == "fallback"


def test_storage_path_resolution_from_env(monkeypatch):
    monkeypatch.setenv("FOOTBALLMODEL_STORAGE_DIR", "/var/data/football")
    monkeypatch.delenv("FOOTBALLMODEL_DB_PATH", raising=False)
    monkeypatch.delenv("FOOTBALLMODEL_RAW_DIR", raising=False)

    assert resolve_duckdb_path() == Path("/var/data/football/footballmodel.duckdb")
    assert resolve_raw_data_paths() == (
        Path("/var/data/football/raw/football_data.csv"),
        Path("/var/data/football/raw/clubelo.csv"),
    )
