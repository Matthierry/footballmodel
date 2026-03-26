from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkSelection:
    current_price: float | None
    benchmark_source: str | None


# Exchange columns first, then market average fallback.
BENCHMARK_COLUMN_CANDIDATES: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {
    ("1X2", "home"): (("bf_home_odds", "exchange"), ("avg_home_odds", "market_average")),
    ("1X2", "draw"): (("bf_draw_odds", "exchange"), ("avg_draw_odds", "market_average")),
    ("1X2", "away"): (("bf_away_odds", "exchange"), ("avg_away_odds", "market_average")),
    ("OU25", "over_2_5"): (("bf_over_2_5_odds", "exchange"), ("avg_over_2_5_odds", "market_average")),
    ("OU25", "under_2_5"): (("bf_under_2_5_odds", "exchange"), ("avg_under_2_5_odds", "market_average")),
    ("BTTS", "btts_yes"): (("bf_btts_yes_odds", "exchange"), ("avg_btts_yes_odds", "market_average")),
    ("BTTS", "btts_no"): (("bf_btts_no_odds", "exchange"), ("avg_btts_no_odds", "market_average")),
    ("AH", "home"): (("bf_ah_home_odds", "exchange"), ("avg_ah_home_odds", "market_average")),
    ("AH", "away"): (("bf_ah_away_odds", "exchange"), ("avg_ah_away_odds", "market_average")),
}


def resolve_benchmark_price(fixture: dict, market: str, outcome: str) -> BenchmarkSelection:
    """Resolve benchmark price for a market/outcome pair.

    For Asian Handicap we key on the side only (home/away), not line-specific outcome labels.
    """
    outcome_key = outcome
    if market == "AH":
        outcome_key = "home" if outcome.startswith("home_") else "away" if outcome.startswith("away_") else outcome

    candidates = BENCHMARK_COLUMN_CANDIDATES.get((market, outcome_key), ())
    for column, source in candidates:
        value = fixture.get(column)
        if value is not None:
            return BenchmarkSelection(current_price=float(value), benchmark_source=source)
    return BenchmarkSelection(current_price=None, benchmark_source="unavailable")
