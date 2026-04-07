from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import unicodedata
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
    upcoming_fixtures_url: str | None = "https://www.football-data.co.uk/fixtures.csv"
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
    future_fixtures_rows_fetched: int
    future_fixtures_fetched: int
    future_fixtures_after_normalization: int
    future_fixtures_after_dedup: int
    future_fixtures_with_published_odds: int
    future_fixtures_with_league_code_after_normalization: int
    future_fixtures_with_league_code_after_dedup: int
    source_div_column_found: bool
    league_code_created_from_source_div: bool


@dataclass(slots=True)
class UpcomingFixturesParseDiagnostics:
    fetched_rows: int
    fetched_future_rows: int
    normalized_rows: int
    future_rows: int
    future_rows_with_published_odds: int
    raw_div_column_found: bool
    raw_div_populated_rows: int
    source_div_populated_rows: int
    league_populated_rows: int
    league_code_populated_rows: int
    mapped_league_code_rows: int
    future_rows_with_league_code: int
    bom_header_sanitized: bool
    sanitized_header_count: int


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

    if "league_code" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("league_code"))

    if "fixture_id" not in df.columns:
        source_div_exists = "source_div" in df.columns
        league_code_exists = "league_code" in df.columns
        mapping_from_source_div_applied = False
        if source_div_exists and league_code_exists:
            mapping_from_source_div_applied = (
                df.filter(
                    pl.col("source_div").is_not_null()
                    & pl.col("league_code").is_not_null()
                    & (pl.col("source_div").cast(pl.Utf8, strict=False) != pl.col("league_code").cast(pl.Utf8, strict=False))
                ).height
                > 0
            )
        print(
            "Fixture id build diagnostics:"
            f" columns={df.columns}"
            f" source_div_exists={source_div_exists}"
            f" league_code_exists={league_code_exists}"
            f" mapping_from_source_div_applied={mapping_from_source_div_applied}"
        )
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


def _sanitize_csv_headers(df: pl.DataFrame) -> tuple[pl.DataFrame, bool, int]:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        sanitized = str(column).lstrip("\ufeff").removeprefix("ï»¿").strip()
        if sanitized != column:
            rename_map[column] = sanitized
    if not rename_map:
        return df, False, 0
    return df.rename(rename_map), True, len(rename_map)


def _normalize_header_for_matching(column: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(column))
    normalized = (
        normalized.replace("\ufeff", "")
        .removeprefix("ï»¿")
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\u2060", "")
        .strip()
        .lower()
    )
    return "".join(ch for ch in normalized if ch.isalnum())


def _map_div_to_league_code_expr(column: str, *, csv_to_league: dict[str, str]) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .map_elements(lambda value: csv_to_league.get(value, value) if value else None, return_dtype=pl.Utf8)
    )


def _populated_count(df: pl.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return (
        df.filter(
            pl.col(column).cast(pl.Utf8, strict=False).is_not_null()
            & (pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars() != "")
        ).height
    )


def _ensure_canonical_league_fields(
    df: pl.DataFrame,
    *,
    csv_to_league: dict[str, str] | None = None,
    prefer_source_div: bool = False,
) -> pl.DataFrame:
    csv_to_league = csv_to_league or {}
    candidate_exprs: list[pl.Expr] = []
    if "league_code" in df.columns:
        candidate_exprs.append(pl.col("league_code").cast(pl.Utf8, strict=False).str.strip_chars())
    if "source_div" in df.columns:
        candidate_exprs.append(_map_div_to_league_code_expr("source_div", csv_to_league=csv_to_league))
    if not prefer_source_div and "league" in df.columns:
        candidate_exprs.append(_map_div_to_league_code_expr("league", csv_to_league=csv_to_league))
    elif "league" in df.columns:
        candidate_exprs.append(_map_div_to_league_code_expr("league", csv_to_league=csv_to_league))

    if candidate_exprs:
        league_code_expr = pl.coalesce(candidate_exprs).alias("league_code")
    else:
        league_code_expr = pl.lit(None, dtype=pl.Utf8).alias("league_code")

    league_candidates: list[pl.Expr] = []
    league_candidates.append(pl.col("league_code").cast(pl.Utf8, strict=False))
    if "league" in df.columns:
        league_candidates.append(pl.col("league").cast(pl.Utf8, strict=False).str.strip_chars())
    return df.with_columns(league_code_expr).with_columns(pl.coalesce(league_candidates).alias("league"))


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
    if "Div" in raw_df.columns:
        raw_div = raw_df.select(
            pl.col("Div")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .replace("", None)
            .alias("_raw_div")
        ).get_column("_raw_div")
        normalized = normalized.with_columns(raw_div).with_columns(
            pl.coalesce(pl.col("source_div").cast(pl.Utf8, strict=False), pl.col("_raw_div").cast(pl.Utf8, strict=False)).alias("source_div")
        )
    normalized = normalized.with_columns(pl.lit(None, dtype=pl.Utf8).alias("league_code"))
    normalized = _ensure_canonical_league_fields(normalized, csv_to_league=csv_to_league, prefer_source_div=True)
    if "_raw_div" in normalized.columns:
        normalized = normalized.drop("_raw_div")
    normalized = normalized.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("season_code"),
        pl.lit(False).alias("is_played"),
        pl.lit(True).alias("is_future_fixture"),
        pl.lit("upcoming").alias("fixture_status"),
        pl.lit("upcoming_fixtures_csv").alias("source_dataset"),
        pl.lit("published_at_source_fetch").alias("odds_capture_type"),
    )
    return normalized


def _is_canonical_frame(df: pl.DataFrame) -> bool:
    return set(REQUIRED_CANONICAL_COLUMNS).issubset(df.columns)


def load_football_data_csv(path: str | Path, *, csv_to_league: dict[str, str] | None = None) -> pl.DataFrame:
    raw_df = pl.read_csv(path, ignore_errors=True)
    raw_df, _, _ = _sanitize_csv_headers(raw_df)
    if _is_canonical_frame(raw_df):
        df = _ensure_canonical_league_fields(raw_df, csv_to_league=csv_to_league, prefer_source_div=True)
        if "match_date" in df.columns:
            df = df.with_columns(_safe_date_parse_expr("match_date").alias("match_date"))
        if "fixture_id" not in df.columns:
            df = df.with_columns(_build_fixture_id_expr())
        return df
    return _normalize_football_data_df(raw_df)


def _fetch_source_csv(url: str, timeout_seconds: int = 20) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    try:
        return response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return response.text


def _parse_upcoming_fixtures_payload(
    payload: str,
    *,
    source_url: str,
    fetched_at_utc: str,
    csv_to_league: dict[str, str],
) -> tuple[pl.DataFrame, UpcomingFixturesParseDiagnostics]:
    first_line = payload.splitlines()[0] if payload else ""
    payload_has_bom_header = "\ufeff" in first_line or first_line.startswith("ï»¿")
    try:
        raw_frame = pl.read_csv(BytesIO(payload.encode("utf-8")), ignore_errors=True)
        raw_columns_exact = [str(column) for column in raw_frame.columns]
        raw_frame, bom_header_sanitized, sanitized_header_count = _sanitize_csv_headers(raw_frame)
        sanitized_columns = [str(column) for column in raw_frame.columns]
        if payload_has_bom_header and not bom_header_sanitized:
            bom_header_sanitized = True
            sanitized_header_count = max(sanitized_header_count, 1)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Unable to parse upcoming fixtures payload from Football-Data fixtures CSV") from exc

    if raw_frame.is_empty():
        return pl.DataFrame([]), UpcomingFixturesParseDiagnostics(
            fetched_rows=0,
            fetched_future_rows=0,
            normalized_rows=0,
            future_rows=0,
            future_rows_with_published_odds=0,
            raw_div_column_found=False,
            raw_div_populated_rows=0,
            source_div_populated_rows=0,
            league_populated_rows=0,
            league_code_populated_rows=0,
            mapped_league_code_rows=0,
            future_rows_with_league_code=0,
            bom_header_sanitized=bom_header_sanitized,
            sanitized_header_count=sanitized_header_count,
        )

    lower_to_original = {str(col).strip().lower(): str(col) for col in raw_frame.columns}
    normalized_to_original: dict[str, str] = {}
    for column in raw_frame.columns:
        normalized_key = _normalize_header_for_matching(str(column))
        normalized_to_original.setdefault(normalized_key, str(column))
    rename_map: dict[str, str] = {}
    for canonical in ("Date", "Div", "HomeTeam", "AwayTeam", "FTHG", "FTAG"):
        normalized_canonical = _normalize_header_for_matching(canonical)
        if normalized_canonical in normalized_to_original:
            rename_map[normalized_to_original[normalized_canonical]] = canonical

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
            normalized_alias = _normalize_header_for_matching(alias)
            if normalized_alias in normalized_to_original:
                rename_map[normalized_to_original[normalized_alias]] = target
                break

    scoped = raw_frame.rename(rename_map)
    selected_raw_div_column = normalized_to_original.get(_normalize_header_for_matching("Div"))
    selected_div_column = rename_map.get(selected_raw_div_column, selected_raw_div_column) if selected_raw_div_column else None
    if selected_div_column and selected_div_column in scoped.columns:
        scoped = scoped.with_columns(
            pl.col(selected_div_column).cast(pl.Utf8, strict=False).str.strip_chars().replace("", None).alias("source_div")
        )
    print(
        "Upcoming fixtures column diagnostics:"
        f" raw_frame.columns={raw_columns_exact}"
        f" sanitized_columns={sanitized_columns}"
        f" lower_to_original={lower_to_original}"
        f" rename_map={rename_map}"
        f" scoped.columns={scoped.columns}"
        f" selected_division_column={selected_div_column}"
    )
    if not {"Date", "HomeTeam", "AwayTeam"}.issubset(set(scoped.columns)):
        available = ", ".join(scoped.columns)
        raise RuntimeError(
            "fixtures.csv schema changed; missing required columns Date/HomeTeam/AwayTeam. "
            f"Available columns: {available}"
        )

    merged = _normalize_upcoming_fixtures_df(
        scoped,
        csv_to_league=csv_to_league,
        source_url=source_url,
        fetched_at_utc=fetched_at_utc,
    )
    print(
        "Upcoming fixtures stage=raw:"
        f" rows={scoped.height}"
        f" raw_div_populated_rows={_populated_count(scoped, 'Div')}"
        f" source_div_populated_rows={_populated_count(scoped, 'source_div')}"
        f" league_populated_rows={_populated_count(scoped, 'league')}"
        f" league_code_populated_rows={_populated_count(scoped, 'league_code')}"
    )
    print(
        "Upcoming fixtures stage=normalized:"
        f" rows={merged.height}"
        f" raw_div_populated_rows={_populated_count(merged, 'source_div')}"
        f" source_div_populated_rows={_populated_count(merged, 'source_div')}"
        f" league_populated_rows={_populated_count(merged, 'league')}"
        f" league_code_populated_rows={_populated_count(merged, 'league_code')}"
    )
    cleaned = (
        merged.filter(pl.col("match_date").is_not_null())
        .filter(pl.col("home_team").is_not_null() & pl.col("away_team").is_not_null())
        .with_columns(_build_fixture_id_expr())
    )
    print(
        "Upcoming fixtures stage=cleaned:"
        f" rows={cleaned.height}"
        f" raw_div_populated_rows={_populated_count(cleaned, 'source_div')}"
        f" source_div_populated_rows={_populated_count(cleaned, 'source_div')}"
        f" league_populated_rows={_populated_count(cleaned, 'league')}"
        f" league_code_populated_rows={_populated_count(cleaned, 'league_code')}"
    )
    today = datetime.now(timezone.utc).date()
    odds_cols = [c for c in ["avg_home_odds", "avg_draw_odds", "avg_away_odds", "bf_home_odds", "bf_draw_odds", "bf_away_odds"] if c in cleaned.columns]
    has_price_expr = pl.any_horizontal([pl.col(c).is_not_null() for c in odds_cols]) if odds_cols else pl.lit(False)
    future_rows = cleaned.filter(pl.col("match_date") >= pl.lit(today)).height
    future_rows_with_published_odds = cleaned.filter((pl.col("match_date") >= pl.lit(today)) & has_price_expr).height
    future_rows_with_league_code = (
        cleaned.filter((pl.col("match_date") >= pl.lit(today)) & pl.col("league_code").is_not_null()).height
        if "league_code" in cleaned.columns
        else 0
    )
    raw_div_rows = _populated_count(scoped, selected_div_column) if selected_div_column else 0
    source_div_rows = _populated_count(cleaned, "source_div")
    league_rows = _populated_count(cleaned, "league")
    league_code_rows = _populated_count(cleaned, "league_code")
    mapped_league_rows = (
        cleaned.filter(
            pl.col("source_div").is_not_null()
            & pl.col("league_code").is_not_null()
            & (pl.col("source_div").cast(pl.Utf8, strict=False) != pl.col("league_code").cast(pl.Utf8, strict=False))
        ).height
        if {"source_div", "league_code"}.issubset(set(cleaned.columns))
        else 0
    )
    diagnostics = UpcomingFixturesParseDiagnostics(
        fetched_rows=raw_frame.height,
        fetched_future_rows=(
            scoped.with_columns(_safe_date_parse_expr("Date").alias("_parsed_date"))
            .filter(pl.col("_parsed_date") >= pl.lit(today))
            .height
        ),
        normalized_rows=cleaned.height,
        future_rows=future_rows,
        future_rows_with_published_odds=future_rows_with_published_odds,
        raw_div_column_found=selected_div_column is not None,
        raw_div_populated_rows=raw_div_rows,
        source_div_populated_rows=source_div_rows,
        league_populated_rows=league_rows,
        league_code_populated_rows=league_code_rows,
        mapped_league_code_rows=mapped_league_rows,
        future_rows_with_league_code=future_rows_with_league_code,
        bom_header_sanitized=bom_header_sanitized,
        sanitized_header_count=sanitized_header_count,
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
    future_fixtures_rows_fetched = 0
    future_fixtures_fetched = 0
    future_fixtures_after_normalization = 0
    future_fixtures_with_published_odds = 0
    upcoming_rows_after_normalization = 0
    raw_div_column_found = False
    raw_div_populated_rows = 0
    source_div_populated_rows = 0
    league_populated_rows = 0
    league_code_populated_rows = 0
    mapped_league_code_rows = 0
    future_fixtures_with_league_code_after_normalization = 0
    bom_header_sanitized = False
    sanitized_header_count = 0

    for season_code in cfg.seasons:
        for source in cfg.sources:
            url = cfg.url_template.format(season_code=season_code, csv_code=source.csv_code, league_code=source.league_code)
            source_id = f"{source.league_code}:{season_code}"
            try:
                csv_text = _fetch_source_csv(url, timeout_seconds=request_timeout_seconds)
                raw_df = pl.read_csv(BytesIO(csv_text.encode("utf-8")), ignore_errors=True)
                raw_df, _, _ = _sanitize_csv_headers(raw_df)
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
        source_id = "upcoming:fixtures_csv"
        try:
            fixtures_text = _fetch_source_csv(cfg.upcoming_fixtures_url, timeout_seconds=request_timeout_seconds)
            upcoming, upcoming_diagnostics = _parse_upcoming_fixtures_payload(
                fixtures_text,
                source_url=cfg.upcoming_fixtures_url,
                fetched_at_utc=fetched_at,
                csv_to_league=csv_to_league,
            )
            future_fixtures_rows_fetched = upcoming_diagnostics.fetched_rows
            future_fixtures_fetched = upcoming_diagnostics.fetched_future_rows
            future_fixtures_after_normalization = upcoming_diagnostics.future_rows
            future_fixtures_with_published_odds = upcoming_diagnostics.future_rows_with_published_odds
            upcoming_rows_after_normalization = upcoming_diagnostics.normalized_rows
            raw_div_column_found = upcoming_diagnostics.raw_div_column_found
            raw_div_populated_rows = upcoming_diagnostics.raw_div_populated_rows
            source_div_populated_rows = upcoming_diagnostics.source_div_populated_rows
            league_populated_rows = upcoming_diagnostics.league_populated_rows
            league_code_populated_rows = upcoming_diagnostics.league_code_populated_rows
            mapped_league_code_rows = upcoming_diagnostics.mapped_league_code_rows
            future_fixtures_with_league_code_after_normalization = upcoming_diagnostics.future_rows_with_league_code
            bom_header_sanitized = upcoming_diagnostics.bom_header_sanitized
            sanitized_header_count = upcoming_diagnostics.sanitized_header_count
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
        .filter(pl.col("source_dataset") == "upcoming_fixtures_csv")
        .filter(pl.col("match_date") >= pl.lit(today))
        .height
        if "source_dataset" in deduped.columns
        else 0
    )
    future_fixtures_with_league_code_after_dedup = (
        deduped
        .filter(pl.col("source_dataset") == "upcoming_fixtures_csv")
        .filter(pl.col("match_date") >= pl.lit(today))
        .filter(pl.col("league_code").is_not_null())
        .height
        if {"source_dataset", "league_code"}.issubset(set(deduped.columns))
        else 0
    )
    dedup_upcoming = (
        deduped.filter(pl.col("source_dataset") == "upcoming_fixtures_csv")
        if "source_dataset" in deduped.columns
        else pl.DataFrame([])
    )
    print(
        "Upcoming fixtures stage=deduped:"
        f" rows={dedup_upcoming.height}"
        f" raw_div_populated_rows={_populated_count(dedup_upcoming, 'source_div')}"
        f" source_div_populated_rows={_populated_count(dedup_upcoming, 'source_div')}"
        f" league_populated_rows={_populated_count(dedup_upcoming, 'league')}"
        f" league_code_populated_rows={_populated_count(dedup_upcoming, 'league_code')}"
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
            f" rows_fetched={future_fixtures_rows_fetched}"
            f" fetched_future_rows={future_fixtures_fetched}"
            f" normalized_rows={upcoming_rows_after_normalization}"
            f" bom_header_sanitized={bom_header_sanitized}"
            f" sanitized_header_count={sanitized_header_count}"
            f" raw_div_column_found={raw_div_column_found}"
            f" raw_div_rows={raw_div_populated_rows}"
            f" source_div_rows={source_div_populated_rows}"
            f" league_rows={league_populated_rows}"
            f" league_code_rows={league_code_populated_rows}"
            f" mapped_league_code_rows={mapped_league_code_rows}"
            f" league_code_created_from_source_div={mapped_league_code_rows > 0}"
            f" future_rows_after_normalization={future_fixtures_after_normalization}"
            f" future_rows_with_league_code_after_normalization={future_fixtures_with_league_code_after_normalization}"
            f" future_rows_after_dedup={future_fixtures_after_dedup}"
            f" future_rows_with_league_code_after_dedup={future_fixtures_with_league_code_after_dedup}"
            f" future_rows_with_published_odds={future_fixtures_with_published_odds}"
        )

    return FootballDataIngestionResult(
        output_path=output,
        fetched_sources=fetched_sources,
        failed_sources=failed_sources,
        rows_before_dedup=rows_before_dedup,
        rows_after_dedup=deduped.height,
        future_fixtures_rows_fetched=future_fixtures_rows_fetched,
        future_fixtures_fetched=future_fixtures_fetched,
        future_fixtures_after_normalization=future_fixtures_after_normalization,
        future_fixtures_after_dedup=future_fixtures_after_dedup,
        future_fixtures_with_published_odds=future_fixtures_with_published_odds,
        future_fixtures_with_league_code_after_normalization=future_fixtures_with_league_code_after_normalization,
        future_fixtures_with_league_code_after_dedup=future_fixtures_with_league_code_after_dedup,
        source_div_column_found=raw_div_column_found,
        league_code_created_from_source_div=mapped_league_code_rows > 0,
    )
