from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
import pandas as pd
import polars as pl
import requests
import yaml

FOOTBALL_DATA_MAPPING = {
    "Date": "match_date",
    "Div": "league",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "B365H": "avg_home_odds",
    "B365D": "avg_draw_odds",
    "B365A": "avg_away_odds",
    "BFH": "bf_home_odds",
    "BFD": "bf_draw_odds",
    "BFA": "bf_away_odds",
    "AHh": "ah_line",
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "avg_over_2_5_odds": ("B365>2.5", "B365O2.5"),
    "avg_under_2_5_odds": ("B365<2.5", "B365U2.5"),
    "bf_over_2_5_odds": ("P>2.5", "PO2.5"),
    "bf_under_2_5_odds": ("P<2.5", "PU2.5"),
    "avg_btts_yes_odds": ("B365BTTS", "B365BTS"),
    "avg_btts_no_odds": ("B365NBTS", "B365BTTS_No"),
    "bf_btts_yes_odds": ("PBBTS", "PBTTS"),
    "bf_btts_no_odds": ("PBNBTS", "PBTTS_No"),
    "avg_ah_home_odds": ("B365AHH",),
    "avg_ah_away_odds": ("B365AHA",),
    "bf_ah_home_odds": ("PAHH",),
    "bf_ah_away_odds": ("PAHA",),
}

REQUIRED_CANONICAL_COLUMNS = ("match_date", "home_team", "away_team")
DEDUPLICATION_KEY = ("league_code", "season_code", "match_date", "home_team", "away_team")


@dataclass(slots=True)
class FootballDataSource:
    league_code: str
    csv_code: str


@dataclass(slots=True)
class FootballDataConfig:
    seasons: list[str]
    sources: list[FootballDataSource]
    url_template: str = "https://www.football-data.co.uk/mmz4281/{season_code}/{csv_code}.csv"
    upcoming_fixtures_url: str | None = "https://www.football-data.co.uk/matches.php"
    include_upcoming_fixtures: bool = True
    fail_fast: bool = False
    persist_snapshots: bool = True


@dataclass(slots=True)
class FootballDataIngestionResult:
    output_path: Path
    fetched_sources: list[str]
    failed_sources: list[str]
    rows_before_dedup: int
    rows_after_dedup: int
    future_fixtures_fetched: int
    future_fixtures_after_normalization: int
    future_fixtures_after_dedup: int


@dataclass(slots=True)
class UpcomingFixturesParseDiagnostics:
    fetched_future_rows: int
    normalized_rows: int
    future_rows: int


def load_football_data_config(path: str | Path) -> FootballDataConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seasons = [str(s) for s in payload.get("seasons", [])]
    raw_sources = payload.get("sources", [])
    sources = [
        FootballDataSource(league_code=str(source["league_code"]), csv_code=str(source["csv_code"]))
        for source in raw_sources
    ]
    if not seasons:
        raise ValueError("football-data config must include at least one season code")
    if not sources:
        raise ValueError("football-data config must include at least one source")
    return FootballDataConfig(
        seasons=seasons,
        sources=sources,
        url_template=str(payload.get("url_template") or FootballDataConfig.url_template),
        upcoming_fixtures_url=(
            str(payload.get("upcoming_fixtures_url"))
            if payload.get("upcoming_fixtures_url") is not None
            else FootballDataConfig.upcoming_fixtures_url
        ),
        include_upcoming_fixtures=bool(payload.get("include_upcoming_fixtures", True)),
        fail_fast=bool(payload.get("fail_fast", False)),
        persist_snapshots=bool(payload.get("persist_snapshots", True)),
    )


def _safe_date_parse_expr(column: str) -> pl.Expr:
    as_text = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.coalesce(
        pl.col(column).cast(pl.Date, strict=False),
        as_text.str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
        as_text.str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
        as_text.str.strptime(pl.Date, format="%d/%m/%Y %H:%M", strict=False),
        as_text.str.strptime(pl.Date, format="%d/%m/%Y %H:%M:%S", strict=False),
        as_text.str.strptime(pl.Date, format="%Y-%m-%d %H:%M", strict=False),
        as_text.str.strptime(pl.Date, format="%Y-%m-%d %H:%M:%S", strict=False),
        as_text.str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False).dt.date(),
        as_text.str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False).dt.date(),
    )


def _build_fixture_id_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.coalesce(pl.col("league_code").cast(pl.Utf8, strict=False), pl.lit("")),
            pl.lit("_"),
            pl.coalesce(pl.col("season_code").cast(pl.Utf8, strict=False), pl.lit("")),
            pl.lit("_"),
            pl.coalesce(pl.col("match_date").cast(pl.Utf8, strict=False), pl.lit("")),
            pl.lit("_"),
            pl.coalesce(pl.col("home_team").cast(pl.Utf8, strict=False), pl.lit("")),
            pl.lit("_"),
            pl.coalesce(pl.col("away_team").cast(pl.Utf8, strict=False), pl.lit("")),
        ]
    ).alias("fixture_id")


def _normalize_football_data_df(
    raw_df: pl.DataFrame,
    *,
    league_code: str | None = None,
    season_code: str | None = None,
    source_url: str | None = None,
    fetched_at_utc: str | None = None,
) -> pl.DataFrame:
    keep_cols = [c for c in FOOTBALL_DATA_MAPPING if c in raw_df.columns]
    df = raw_df.select(keep_cols).rename({k: FOOTBALL_DATA_MAPPING[k] for k in keep_cols})

    for target, aliases in COLUMN_ALIASES.items():
        present_aliases = [alias for alias in aliases if alias in raw_df.columns]
        if present_aliases:
            alias_series = raw_df.select(
                pl.coalesce([pl.col(alias).cast(pl.Float64, strict=False) for alias in present_aliases]).alias(target)
            ).get_column(target)
            df = df.with_columns(alias_series)

    if "match_date" in df.columns:
        df = df.with_columns(_safe_date_parse_expr("match_date").alias("match_date"))

    if "league" in df.columns:
        df = df.with_columns(pl.col("league").cast(pl.Utf8, strict=False).alias("source_div"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("source_div"))

    if league_code:
        df = df.with_columns(
            pl.lit(league_code).alias("league"),
            pl.lit(league_code).alias("league_code"),
        )
    elif "league" in df.columns:
        df = df.with_columns(pl.col("league").cast(pl.Utf8, strict=False).alias("league_code"))

    if season_code:
        df = df.with_columns(pl.lit(season_code).alias("season_code"))
    elif "season_code" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("season_code"))

    if source_url:
        df = df.with_columns(pl.lit(source_url).alias("source_url"))
    elif "source_url" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("source_url"))

    if fetched_at_utc:
        df = df.with_columns(pl.lit(fetched_at_utc).alias("fetched_at_utc"))
    elif "fetched_at_utc" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("fetched_at_utc"))

    if "fixture_id" not in df.columns:
        df = df.with_columns(_build_fixture_id_expr())

    if "home_goals" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("home_goals"))
    if "away_goals" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("away_goals"))

    df = df.with_columns(
        (pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null()).alias("is_played"),
        (pl.col("home_goals").is_null() & pl.col("away_goals").is_null()).alias("is_future_fixture"),
        pl.when(pl.col("home_goals").is_not_null() & pl.col("away_goals").is_not_null())
        .then(pl.lit("played"))
        .otherwise(pl.lit("upcoming"))
        .alias("fixture_status"),
        pl.lit("historical_league_csv").alias("source_dataset"),
        pl.lit("published_at_source_fetch").alias("odds_capture_type"),
    )

    return df


def _normalize_upcoming_fixtures_df(
    raw_df: pl.DataFrame,
    *,
    csv_to_league: dict[str, str],
    source_url: str,
    fetched_at_utc: str,
) -> pl.DataFrame:
    normalized = _normalize_football_data_df(
        raw_df,
        source_url=source_url,
        fetched_at_utc=fetched_at_utc,
    )
    normalized = normalized.with_columns(
        pl.coalesce(
            pl.col("league").cast(pl.Utf8, strict=False),
            pl.col("source_div").cast(pl.Utf8, strict=False),
        ).alias("league"),
    )
    if "source_div" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("source_div")
            .cast(pl.Utf8, strict=False)
            .map_elements(lambda div: csv_to_league.get(div, div), return_dtype=pl.Utf8)
            .alias("league_code")
        )
    else:
        normalized = normalized.with_columns(pl.col("league").cast(pl.Utf8, strict=False).alias("league_code"))
    normalized = normalized.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("season_code"),
        pl.lit(False).alias("is_played"),
        pl.lit(True).alias("is_future_fixture"),
        pl.lit("upcoming").alias("fixture_status"),
        pl.lit("upcoming_fixtures_matches_php").alias("source_dataset"),
        pl.lit("published_at_source_fetch").alias("odds_capture_type"),
    )
    return normalized


def _is_canonical_frame(df: pl.DataFrame) -> bool:
    return set(REQUIRED_CANONICAL_COLUMNS).issubset(df.columns)


def load_football_data_csv(path: str | Path) -> pl.DataFrame:
    raw_df = pl.read_csv(path, ignore_errors=True)
    if _is_canonical_frame(raw_df):
        df = raw_df
        if "match_date" in df.columns:
            df = df.with_columns(_safe_date_parse_expr("match_date").alias("match_date"))
        if "fixture_id" not in df.columns:
            df = df.with_columns(_build_fixture_id_expr())
        return df
    return _normalize_football_data_df(raw_df)


def _fetch_source_csv(url: str, timeout_seconds: int = 20) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def _extract_html_tables(html: str) -> list[pl.DataFrame]:
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError):
        return []
    frames: list[pl.DataFrame] = []
    for frame in tables:
        if frame.empty:
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [str(col[-1]) if isinstance(col, tuple) else str(col) for col in frame.columns]
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append(pl.from_pandas(frame))
    return frames


def _parse_upcoming_fixtures_payload(
    payload: str,
    *,
    source_url: str,
    fetched_at_utc: str,
    csv_to_league: dict[str, str],
) -> tuple[pl.DataFrame, UpcomingFixturesParseDiagnostics]:
    candidate_frames: list[pl.DataFrame] = []
    try:
        candidate_frames.append(pl.read_csv(BytesIO(payload.encode("utf-8")), ignore_errors=True))
    except Exception:  # noqa: BLE001
        pass
    candidate_frames.extend(_extract_html_tables(payload))

    normalized_frames: list[pl.DataFrame] = []
    fetched_future_rows = 0
    today = datetime.now(timezone.utc).date()
    for frame in candidate_frames:
        lower_to_original = {str(col).strip().lower(): str(col) for col in frame.columns}
        rename_map: dict[str, str] = {}
        for canonical in ("Date", "Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG"):
            if canonical.lower() in lower_to_original:
                rename_map[lower_to_original[canonical.lower()]] = canonical

        aliases: dict[str, tuple[str, ...]] = {
            "B365H": ("b365h", "home", "home_odds"),
            "B365D": ("b365d", "draw", "draw_odds"),
            "B365A": ("b365a", "away", "away_odds"),
            "BFH": ("bfh", "pinh", "psh"),
            "BFD": ("bfd", "pind", "psd"),
            "BFA": ("bfa", "pina", "psa"),
        }
        for target, candidates in aliases.items():
            for alias in candidates:
                if alias in lower_to_original:
                    rename_map[lower_to_original[alias]] = target
                    break
        scoped = frame.rename(rename_map)
        if {"Date", "HomeTeam", "AwayTeam"}.issubset(set(scoped.columns)):
            fetched_future_rows += (
                scoped.with_columns(_safe_date_parse_expr("Date").alias("_parsed_date"))
                .filter(pl.col("_parsed_date") >= pl.lit(today))
                .height
            )
            normalized_frames.append(
                _normalize_upcoming_fixtures_df(
                    scoped,
                    csv_to_league=csv_to_league,
                    source_url=source_url,
                    fetched_at_utc=fetched_at_utc,
                )
            )

    if not normalized_frames:
        raise RuntimeError("Unable to parse upcoming fixtures payload from Football-Data matches source")
    merged = pl.concat(normalized_frames, how="diagonal_relaxed")
    cleaned = (
        merged.filter(pl.col("match_date").is_not_null())
        .filter(pl.col("home_team").is_not_null() & pl.col("away_team").is_not_null())
        .with_columns(_build_fixture_id_expr())
    )
    future_rows = cleaned.filter(pl.col("match_date") >= pl.lit(datetime.now(timezone.utc).date())).height
    diagnostics = UpcomingFixturesParseDiagnostics(
        fetched_future_rows=fetched_future_rows,
        normalized_rows=cleaned.height,
        future_rows=future_rows,
    )
    return cleaned, diagnostics


def build_football_data_raw_file(
    *,
    config_path: str | Path,
    output_path: str | Path,
    snapshots_dir: str | Path | None = None,
    request_timeout_seconds: int = 20,
) -> FootballDataIngestionResult:
    cfg = load_football_data_config(config_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot_root = Path(snapshots_dir) if snapshots_dir else output.parent / "football_data_sources"
    if cfg.persist_snapshots:
        snapshot_root.mkdir(parents=True, exist_ok=True)

    ingested_frames: list[pl.DataFrame] = []
    fetched_sources: list[str] = []
    failed_sources: list[str] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date()
    csv_to_league = {source.csv_code: source.league_code for source in cfg.sources}
    future_fixtures_fetched = 0
    future_fixtures_after_normalization = 0
    upcoming_rows_after_normalization = 0

    for season_code in cfg.seasons:
        for source in cfg.sources:
            url = cfg.url_template.format(season_code=season_code, csv_code=source.csv_code, league_code=source.league_code)
            source_id = f"{source.league_code}:{season_code}"
            try:
                csv_text = _fetch_source_csv(url, timeout_seconds=request_timeout_seconds)
                raw_df = pl.read_csv(BytesIO(csv_text.encode("utf-8")), ignore_errors=True)
                normalized = _normalize_football_data_df(
                    raw_df,
                    league_code=source.league_code,
                    season_code=season_code,
                    source_url=url,
                    fetched_at_utc=fetched_at,
                )
                ingested_frames.append(normalized)
                fetched_sources.append(source_id)
                if cfg.persist_snapshots:
                    snapshot_path = snapshot_root / f"{season_code}_{source.league_code}.csv"
                    snapshot_path.write_text(csv_text, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                message = f"{source_id} ({url}) failed: {exc}"
                failed_sources.append(message)
                if cfg.fail_fast:
                    raise RuntimeError(message) from exc

    if cfg.include_upcoming_fixtures and cfg.upcoming_fixtures_url:
        source_id = "upcoming:matches_php"
        try:
            fixtures_text = _fetch_source_csv(cfg.upcoming_fixtures_url, timeout_seconds=request_timeout_seconds)
            upcoming, upcoming_diagnostics = _parse_upcoming_fixtures_payload(
                fixtures_text,
                source_url=cfg.upcoming_fixtures_url,
                fetched_at_utc=fetched_at,
                csv_to_league=csv_to_league,
            )
            future_fixtures_fetched = upcoming_diagnostics.fetched_future_rows
            future_fixtures_after_normalization = upcoming_diagnostics.future_rows
            upcoming_rows_after_normalization = upcoming_diagnostics.normalized_rows
            if upcoming.height:
                ingested_frames.append(upcoming)
                fetched_sources.append(source_id)
                if cfg.persist_snapshots:
                    snapshot_path = snapshot_root / "upcoming_matches.csv"
                    upcoming.write_csv(snapshot_path)
        except Exception as exc:  # noqa: BLE001
            message = f"{source_id} ({cfg.upcoming_fixtures_url}) failed: {exc}"
            failed_sources.append(message)
            if cfg.fail_fast:
                raise RuntimeError(message) from exc

    if not ingested_frames:
        raise RuntimeError("Football-Data ingestion failed for all configured sources")

    merged = pl.concat(ingested_frames, how="diagonal_relaxed")
    rows_before_dedup = merged.height
    deduped = (
        merged.sort("fetched_at_utc")
        .unique(subset=[c for c in DEDUPLICATION_KEY if c in merged.columns], keep="last")
        .with_columns(_build_fixture_id_expr())
    )
    future_fixtures_after_dedup = (
        deduped
        .filter(pl.col("source_dataset") == "upcoming_fixtures_matches_php")
        .filter(pl.col("match_date") >= pl.lit(today))
        .height
        if "source_dataset" in deduped.columns
        else 0
    )
    deduped.write_csv(output)

    if failed_sources:
        print(
            "Football-Data ingestion completed with source failures:\n"
            + "\n".join(f"- {entry}" for entry in failed_sources)
        )
    if cfg.include_upcoming_fixtures:
        print(
            "Upcoming fixtures diagnostics:"
            f" fetched_future_rows={future_fixtures_fetched}"
            f" normalized_rows={upcoming_rows_after_normalization}"
            f" future_rows_after_normalization={future_fixtures_after_normalization}"
            f" future_rows_after_dedup={future_fixtures_after_dedup}"
        )

    return FootballDataIngestionResult(
        output_path=output,
        fetched_sources=fetched_sources,
        failed_sources=failed_sources,
        rows_before_dedup=rows_before_dedup,
        rows_after_dedup=deduped.height,
        future_fixtures_fetched=future_fixtures_fetched,
        future_fixtures_after_normalization=future_fixtures_after_normalization,
        future_fixtures_after_dedup=future_fixtures_after_dedup,
    )
