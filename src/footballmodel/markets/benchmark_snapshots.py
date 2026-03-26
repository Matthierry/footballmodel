from __future__ import annotations

from datetime import datetime

import polars as pl

from footballmodel.markets.benchmark import resolve_benchmark_price


SNAPSHOT_TYPE_PREDICTION_TIME = "prediction_time"
SNAPSHOT_TYPE_PRE_KICKOFF_LATEST = "pre_kickoff_latest"
SNAPSHOT_TYPE_CLOSING = "closing"
SNAPSHOT_TYPE_CLOSING_SURROGATE = "closing_surrogate"

_MARKET_OUTCOMES: dict[str, tuple[str, ...]] = {
    "1X2": ("home", "draw", "away"),
    "OU25": ("over_2_5", "under_2_5"),
    "BTTS": ("btts_yes", "btts_no"),
    "AH": ("home_-1.5", "away_-1.5", "home_-1.0", "away_-1.0", "home_-0.5", "away_-0.5", "home_0.0", "away_0.0"),
}


def _line_for_market_outcome(market: str, outcome: str) -> float | None:
    if market == "OU25":
        return 2.5
    if market == "AH":
        try:
            return float(outcome.split("_", maxsplit=1)[1])
        except (ValueError, IndexError):
            return None
    return None


def benchmark_snapshot_rows_from_fixture(
    fixture: dict[str, object],
    snapshot_type: str,
    snapshot_timestamp_utc: str | None = None,
) -> pl.DataFrame:
    snapshot_ts = snapshot_timestamp_utc or datetime.utcnow().isoformat()
    rows: list[dict[str, object]] = []
    fixture_id = str(fixture["fixture_id"])
    match_date = fixture.get("match_date")
    for market, outcomes in _MARKET_OUTCOMES.items():
        for outcome in outcomes:
            bench = resolve_benchmark_price(fixture, market=market, outcome=outcome)
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "market": market,
                    "outcome": outcome,
                    "line": _line_for_market_outcome(market, outcome),
                    "benchmark_price": bench.current_price,
                    "benchmark_source": bench.benchmark_source,
                    "snapshot_type": snapshot_type,
                    "snapshot_timestamp_utc": snapshot_ts,
                    "match_date": match_date,
                }
            )
    return pl.DataFrame(rows)


def choose_later_snapshot(snapshot_rows: pl.DataFrame) -> dict[str, object] | None:
    if snapshot_rows.is_empty():
        return None
    for preferred in (
        SNAPSHOT_TYPE_PRE_KICKOFF_LATEST,
        SNAPSHOT_TYPE_CLOSING,
        SNAPSHOT_TYPE_CLOSING_SURROGATE,
    ):
        scoped = snapshot_rows.filter((pl.col("snapshot_type") == preferred) & pl.col("benchmark_price").is_not_null())
        if not scoped.is_empty():
            row = scoped.sort("snapshot_timestamp_utc", descending=True).row(0, named=True)
            return row
    return None
