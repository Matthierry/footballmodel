from __future__ import annotations

import re
import unicodedata
from collections import Counter

import polars as pl


class TeamMappingError(ValueError):
    """Raised when source-to-source team mapping is invalid."""


def normalize_team_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def validate_mapping(mapping: dict[str, str]) -> None:
    norm_keys = [normalize_team_name(k) for k in mapping]
    dups = [team for team, count in Counter(norm_keys).items() if count > 1]
    if dups:
        raise TeamMappingError(f"Duplicate source aliases after normalization: {dups}")

    targets = list(mapping.values())
    collisions = [team for team, count in Counter(targets).items() if count > 1]
    if collisions:
        raise TeamMappingError(f"Many-to-one mapping detected for targets: {collisions}")


def find_unmapped_teams(source_teams: list[str], mapping: dict[str, str]) -> list[str]:
    mapped_keys = {normalize_team_name(k) for k in mapping}
    missing = [team for team in source_teams if normalize_team_name(team) not in mapped_keys]
    return sorted(set(missing))


def apply_team_mapping(
    fixtures: pl.DataFrame,
    mapping: dict[str, str],
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pl.DataFrame:
    validate_mapping(mapping)

    alias_map = {normalize_team_name(k): v for k, v in mapping.items()}

    mapped = fixtures.with_columns(
        pl.col(home_col)
        .map_elements(lambda n: alias_map.get(normalize_team_name(str(n))), return_dtype=pl.String)
        .alias("home_team_elo"),
        pl.col(away_col)
        .map_elements(lambda n: alias_map.get(normalize_team_name(str(n))), return_dtype=pl.String)
        .alias("away_team_elo"),
    )

    missing_home = mapped.filter(pl.col("home_team_elo").is_null())[home_col].unique().to_list()
    missing_away = mapped.filter(pl.col("away_team_elo").is_null())[away_col].unique().to_list()
    missing = sorted(set(missing_home + missing_away))
    if missing:
        raise TeamMappingError(f"Unmapped teams detected: {missing}")

    return mapped
