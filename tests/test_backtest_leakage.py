from datetime import date

import polars as pl

from footballmodel.backtest.walkforward import BacktestRequest, run_walkforward


def test_walkforward_uses_past_only():
    matches = pl.DataFrame(
        {
            "fixture_id": ["a", "b", "c"],
            "league": ["ENG1", "ENG1", "ENG1"],
            "match_date": [date(2025, 1, 1), date(2025, 1, 8), date(2025, 1, 15)],
        }
    )

    seen_hist_sizes = []

    def predict_fn(hist, _fixture):
        seen_hist_sizes.append(hist.height)
        return {"pred": 1.0}

    req = BacktestRequest(start_date=date(2025, 1, 1), end_date=date(2025, 1, 31), leagues=["ENG1"])
    _ = run_walkforward(matches, predict_fn, req)
    assert seen_hist_sizes == [0, 1, 2]
