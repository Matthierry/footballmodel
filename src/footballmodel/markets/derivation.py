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


def _ah_outcome_probabilities(matrix: np.ndarray, line: float) -> dict[str, float]:
    home_win = 0.0
    away_win = 0.0
    push = 0.0

    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            adjusted = (h - a) - line
            if adjusted > 0:
                home_win += matrix[h, a]
            elif adjusted < 0:
                away_win += matrix[h, a]
            else:
                push += matrix[h, a]

    return {
        "home_win": float(home_win),
        "away_win": float(away_win),
        "push": float(push),
    }


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

    ah = derive_ah(matrix)
    ah_line = float(ah["recommended_line"])
    ah_probs = _ah_outcome_probabilities(matrix, ah_line)
    non_push_mass = 1.0 - ah_probs["push"]

    for side in ("home", "away"):
        win_prob = ah_probs[f"{side}_win"]
        normalized_prob = win_prob / non_push_mass if non_push_mass > 0 else 0.0
        rows.append(
            {
                "fixture_id": fixture_id,
                "market": "AH",
                "outcome": f"{side}_{ah_line:+.1f}",
                "model_probability": float(normalized_prob),
                "model_fair_odds": (non_push_mass / win_prob) if win_prob > 0 else None,
                "market_line": ah_line,
                "push_probability": ah_probs["push"],
            }
        )

    return pl.DataFrame(rows)
