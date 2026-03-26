from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from footballmodel.ingestion.team_mapping import (
    TeamMappingError,
    apply_team_mapping,
    find_unmapped_teams,
    normalize_team_name,
    validate_mapping,
)


def test_normalize_team_name_handles_alias_formatting():
    assert normalize_team_name("Paris Saint-Germain") == "paris saint germain"
    assert normalize_team_name("M'gladbach") == "m gladbach"
    assert normalize_team_name("Bayern  München") == "bayern munchen"


def test_apply_team_mapping_maps_known_aliases_across_leagues():
    fixtures = pl.DataFrame(
        {
            "home_team": ["Man City", "Inter"],
            "away_team": ["Man United", "Juventus"],
        }
    )
    mapping = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Inter": "Inter Milan",
        "Juventus": "Juventus",
    }

    mapped = apply_team_mapping(fixtures, mapping)
    assert mapped["home_team_elo"].to_list() == ["Manchester City", "Inter Milan"]
    assert mapped["away_team_elo"].to_list() == ["Manchester United", "Juventus"]


def test_unmapped_teams_are_reported_explicitly():
    missing = find_unmapped_teams(["Leicester", "Ipswich", "Man City"], {"Man City": "Manchester City"})
    assert missing == ["Ipswich", "Leicester"]


def test_duplicate_or_ambiguous_mapping_fails_fast():
    with pytest.raises(TeamMappingError, match="Many-to-one"):
        validate_mapping({"Man City": "Manchester City", "Manchester City": "Manchester City"})


def test_apply_mapping_raises_for_promoted_team_without_alias():
    fixtures = pl.DataFrame({"home_team": ["Leeds"], "away_team": ["Man City"]})
    mapping = {"Man City": "Manchester City"}

    with pytest.raises(TeamMappingError, match="Unmapped teams detected"):
        apply_team_mapping(fixtures, mapping)
