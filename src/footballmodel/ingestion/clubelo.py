from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import polars as pl
import requests
import yaml

from footballmodel.ingestion.football_data import load_football_data_csv

CLUBELO_REQUIRED_COLUMNS = ("elo_date", "team", "country", "elo")


@dataclass(slots=True)
class ClubEloConfig:
    url_template: str = "http://api.clubelo.com/{date}"
    date_format: str = "%Y-%m-%d"
    leagues: list[str] | None = None
    explicit_dates: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    date_frequency: str = "matchdays"
    include_today: bool = True
    include_match_dates_from_football_data: bool = True
    persist_snapshots: bool = True
    fail_fast: bool = False


@dataclass(slots=True)
class ClubEloIngestionResult:
    output_path: Path
    fetched_dates: list[str]
    failed_dates: list[str]
    rows_written: int


def _safe_date_parse_expr(column: str) -> pl.Expr:
    col_utf8 = pl.col(column).cast(pl.Utf8, strict=False)
    return pl.coalesce(
        pl.col(column).cast(pl.Date, strict=False),
        col_utf8.str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        col_utf8.str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
    )


def load_clubelo_config(path: str | Path) -> ClubEloConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    frequency = str(payload.get("date_frequency", "matchdays")).lower()
    if frequency not in {"matchdays", "daily"}:
        raise ValueError("clubelo date_frequency must be one of: matchdays, daily")

    leagues_payload = payload.get("leagues")
    leagues = [str(v) for v in leagues_payload] if leagues_payload else None

    explicit_dates_payload = payload.get("explicit_dates")
    explicit_dates = [str(v) for v in explicit_dates_payload] if explicit_dates_payload else None

    return ClubEloConfig(
        url_template=str(payload.get("url_template") or ClubEloConfig.url_template),
        date_format=str(payload.get("date_format") or ClubEloConfig.date_format),
        leagues=leagues,
        explicit_dates=explicit_dates,
        start_date=str(payload["start_date"]) if payload.get("start_date") else None,
        end_date=str(payload["end_date"]) if payload.get("end_date") else None,
        date_frequency=frequency,
        include_today=bool(payload.get("include_today", True)),
        include_match_dates_from_football_data=bool(payload.get("include_match_dates_from_football_data", True)),
        persist_snapshots=bool(payload.get("persist_snapshots", True)),
        fail_fast=bool(payload.get("fail_fast", False)),
    )


def _fetch_source_csv(url: str, timeout_seconds: int = 20) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _build_date_range(start_dt: date, end_dt: date) -> list[date]:
    if end_dt < start_dt:
        raise ValueError("clubelo start_date must be <= end_date")
    days = (end_dt - start_dt).days
    return [start_dt + timedelta(days=i) for i in range(days + 1)]


def _normalize_clubelo_snapshot_df(
    raw_df: pl.DataFrame,
    *,
    snapshot_date: date,
    source_url: str | None = None,
    fetched_at_utc: str | None = None,
    teams_filter: set[str] | None = None,
) -> pl.DataFrame:
    alias_map = {
        "Date": "elo_date",
        "Club": "team",
        "Country": "country",
        "Elo": "elo",
        "date": "elo_date",
        "club": "team",
        "country": "country",
        "elo": "elo",
    }
    renamed = raw_df.rename({column: alias_map[column] for column in raw_df.columns if column in alias_map})
    if "elo_date" not in renamed.columns:
        renamed = renamed.with_columns(pl.lit(snapshot_date).alias("elo_date"))

    missing = [col for col in ("team", "country", "elo") if col not in renamed.columns]
    if missing:
        raise ValueError(f"ClubElo snapshot missing required columns: {missing}")

    df = renamed.with_columns(
        _safe_date_parse_expr("elo_date").alias("elo_date"),
        pl.col("team").cast(pl.Utf8, strict=False).str.strip_chars().alias("team"),
        pl.col("country").cast(pl.Utf8, strict=False).str.strip_chars().alias("country"),
        pl.col("elo").cast(pl.Float64, strict=False).alias("elo"),
    )
    if teams_filter:
        df = df.filter(pl.col("team").is_in(sorted(teams_filter)))
    if source_url:
        df = df.with_columns(pl.lit(source_url).alias("source_url"))
    if fetched_at_utc:
        df = df.with_columns(pl.lit(fetched_at_utc).alias("fetched_at_utc"))

    cleaned = df.filter(
        pl.col("elo_date").is_not_null() & pl.col("team").is_not_null() & pl.col("country").is_not_null() & pl.col("elo").is_not_null()
    )
    if cleaned.is_empty():
        raise ValueError(f"ClubElo snapshot for {snapshot_date.isoformat()} had no valid rows after normalization")
    return cleaned


def _collect_target_teams_and_dates(
    *,
    cfg: ClubEloConfig,
    football_data_path: str | Path | None,
) -> tuple[set[str], list[date]]:
    dates: set[date] = set()
    teams: set[str] = set()
    if cfg.explicit_dates:
        dates.update(date.fromisoformat(d) for d in cfg.explicit_dates)

    if cfg.start_date and cfg.end_date and cfg.date_frequency == "daily":
        dates.update(_build_date_range(date.fromisoformat(cfg.start_date), date.fromisoformat(cfg.end_date)))

    if football_data_path and cfg.include_match_dates_from_football_data:
        matches = load_football_data_csv(football_data_path)
        if cfg.leagues:
            matches = matches.filter(pl.col("league").is_in(cfg.leagues))
        match_dates = matches.select(_safe_date_parse_expr("match_date").alias("match_date")).drop_nulls()
        dates.update(match_dates.get_column("match_date").to_list())
        teams.update(matches.get_column("home_team").drop_nulls().cast(pl.Utf8).to_list())
        teams.update(matches.get_column("away_team").drop_nulls().cast(pl.Utf8).to_list())

    if cfg.start_date and cfg.end_date and cfg.date_frequency == "matchdays":
        start_dt = date.fromisoformat(cfg.start_date)
        end_dt = date.fromisoformat(cfg.end_date)
        dates = {d for d in dates if start_dt <= d <= end_dt}

    if cfg.include_today:
        dates.add(datetime.now(timezone.utc).date())

    ordered_dates = sorted(dates)
    if not ordered_dates:
        raise RuntimeError("ClubElo ingestion has no target dates. Check clubelo config/date inputs.")
    return teams, ordered_dates


def load_clubelo_csv(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    if set(CLUBELO_REQUIRED_COLUMNS).issubset(df.columns):
        out = df
    else:
        expected = {
        "Date": "elo_date",
        "Club": "team",
        "Country": "country",
        "Elo": "elo",
        }
        keep = [k for k in expected if k in df.columns]
        out = df.select(keep).rename({k: expected[k] for k in keep})
    return out.with_columns(
        _safe_date_parse_expr("elo_date").alias("elo_date"),
        pl.col("elo").cast(pl.Float64, strict=False).alias("elo"),
    )


def elo_as_of(elo_history: pl.DataFrame, team: str, dt: str) -> float:
    rows = (
        elo_history.filter((pl.col("team") == team) & (pl.col("elo_date") <= pl.lit(dt).str.strptime(pl.Date)))
        .sort("elo_date")
        .tail(1)
    )
    if rows.is_empty():
        return 1500.0
    return float(rows["elo"][0])


def build_clubelo_raw_file(
    *,
    config_path: str | Path,
    output_path: str | Path,
    football_data_path: str | Path | None = None,
    snapshots_dir: str | Path | None = None,
    request_timeout_seconds: int = 20,
) -> ClubEloIngestionResult:
    cfg = load_clubelo_config(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshots_root = Path(snapshots_dir) if snapshots_dir else output.parent / "clubelo_sources"
    if cfg.persist_snapshots:
        snapshots_root.mkdir(parents=True, exist_ok=True)

    teams_filter, target_dates = _collect_target_teams_and_dates(cfg=cfg, football_data_path=football_data_path)
    fetched_dates: list[str] = []
    failed_dates: list[str] = []
    frames: list[pl.DataFrame] = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for target_date in target_dates:
        date_token = target_date.strftime(cfg.date_format)
        url = cfg.url_template.format(date=date_token)
        try:
            csv_text = _fetch_source_csv(url, timeout_seconds=request_timeout_seconds)
            raw_df = pl.read_csv(BytesIO(csv_text.encode("utf-8")), ignore_errors=True)
            normalized = _normalize_clubelo_snapshot_df(
                raw_df,
                snapshot_date=target_date,
                source_url=url,
                fetched_at_utc=fetched_at,
                teams_filter=teams_filter if teams_filter else None,
            )
            frames.append(normalized)
            fetched_dates.append(date_token)
            if cfg.persist_snapshots:
                (snapshots_root / f"{date_token}.csv").write_text(csv_text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            message = f"{date_token} ({url}) failed: {exc}"
            failed_dates.append(message)
            if cfg.fail_fast:
                raise RuntimeError(message) from exc

    if not frames:
        raise RuntimeError("ClubElo ingestion failed for all configured dates")

    merged = (
        pl.concat(frames, how="vertical_relaxed")
        .sort(["elo_date", "team", "fetched_at_utc"])
        .unique(subset=["elo_date", "team"], keep="last")
    )
    merged.write_csv(output)

    if failed_dates:
        print("ClubElo ingestion completed with date failures:\n" + "\n".join(f"- {entry}" for entry in failed_dates))

    return ClubEloIngestionResult(
        output_path=output,
        fetched_dates=fetched_dates,
        failed_dates=failed_dates,
        rows_written=merged.height,
    )
