from __future__ import annotations

import numpy as np
import polars as pl


def derive_1x2(matrix: np.ndarray) -> dict[str, float]:
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return {"home": home, "draw": draw, "away": away}


def derive_ou25(matrix: np.ndarray) -> dict[str, float]:
    over = 0.0
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            if h + a > 2:
                over += matrix[h, a]
    return {"over_2_5": float(over), "under_2_5": float(1 - over)}


def derive_btts(matrix: np.ndarray) -> dict[str, float]:
    no = float(matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0])
    yes = 1 - no
    return {"btts_yes": yes, "btts_no": no}


def derive_correct_score_top5(matrix: np.ndarray) -> list[dict[str, float | str]]:
    cells: list[tuple[str, float]] = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            score = f"{h}-{'6+' if a == matrix.shape[1] - 1 else a}" if h == matrix.shape[0] - 1 else f"{h}-{a}"
            cells.append((score, float(matrix[h, a])))
    cells.sort(key=lambda x: x[1], reverse=True)
    return [{"score": s, "probability": p, "fair_odds": 1 / p if p > 0 else None} for s, p in cells[:5]]


def derive_ah(matrix: np.ndarray, lines: tuple[float, ...] = (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5)) -> dict[str, float]:
    goal_diff_probs = {}
    for line in lines:
        home_cover = 0.0
        for h in range(matrix.shape[0]):
            for a in range(matrix.shape[1]):
                if (h - a) > line:
                    home_cover += matrix[h, a]
        goal_diff_probs[f"home_{line:+.1f}"] = float(home_cover)
    best_line = min(lines, key=lambda ln: abs(goal_diff_probs[f"home_{ln:+.1f}"] - 0.5))
    goal_diff_probs["recommended_line"] = float(best_line)
    return goal_diff_probs


def matrix_to_market_table(fixture_id: str, matrix: np.ndarray) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for market, probs in {
        "1X2": derive_1x2(matrix),
        "OU25": derive_ou25(matrix),
        "BTTS": derive_btts(matrix),
    }.items():
        for outcome, p in probs.items():
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "market": market,
                    "outcome": outcome,
                    "model_probability": p,
                    "model_fair_odds": 1 / p if p > 0 else None,
                }
            )
    return pl.DataFrame(rows)
