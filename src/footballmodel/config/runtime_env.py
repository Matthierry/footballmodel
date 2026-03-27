from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _streamlit_secret(key: str, streamlit_module: Any | None = None) -> str | None:
    """Return a Streamlit secret if available without crashing when secrets are absent."""
    st = streamlit_module
    if st is None:
        try:
            import streamlit as st  # type: ignore
        except Exception:
            return None

    try:
        secrets = getattr(st, "secrets")
    except Exception:
        return None

    try:
        if hasattr(secrets, "get"):
            value = secrets.get(key)
        else:
            value = secrets[key]
    except Exception:
        return None

    return str(value) if value is not None else None


def get_app_password(streamlit_module: Any | None = None, default: str = "changeme") -> str:
    env_value = os.getenv("APP_PASSWORD")
    if env_value:
        return env_value

    secret_value = _streamlit_secret("APP_PASSWORD", streamlit_module=streamlit_module)
    if secret_value:
        return secret_value

    return default


def get_storage_dir(default: str = "data") -> Path:
    configured = os.getenv("FOOTBALLMODEL_STORAGE_DIR")
    return Path(configured) if configured else Path(default)


def resolve_duckdb_path(default_name: str = "footballmodel.duckdb") -> Path:
    explicit = os.getenv("FOOTBALLMODEL_DB_PATH")
    if explicit:
        return Path(explicit)
    return get_storage_dir() / default_name


def resolve_raw_data_paths() -> tuple[Path, Path]:
    raw_dir = os.getenv("FOOTBALLMODEL_RAW_DIR")
    if raw_dir:
        base = Path(raw_dir)
    else:
        base = get_storage_dir() / "raw"
    return base / "football_data.csv", base / "clubelo.csv"
