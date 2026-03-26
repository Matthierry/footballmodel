from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class FixtureRecord:
    fixture_id: str
    league: str
    match_date: date
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_sot: int | None = None
    away_sot: int | None = None
    avg_home_odds: float | None = None
    avg_draw_odds: float | None = None
    avg_away_odds: float | None = None
    bf_home_odds: float | None = None
    bf_draw_odds: float | None = None
    bf_away_odds: float | None = None
    avg_over_2_5_odds: float | None = None
    avg_under_2_5_odds: float | None = None
    bf_over_2_5_odds: float | None = None
    bf_under_2_5_odds: float | None = None
    avg_btts_yes_odds: float | None = None
    avg_btts_no_odds: float | None = None
    bf_btts_yes_odds: float | None = None
    bf_btts_no_odds: float | None = None
    ah_line: float | None = None
    avg_ah_home_odds: float | None = None
    avg_ah_away_odds: float | None = None
    bf_ah_home_odds: float | None = None
    bf_ah_away_odds: float | None = None
