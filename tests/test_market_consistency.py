from __future__ import annotations

import numpy as np

from footballmodel.markets.derivation import derive_1x2, derive_ah, derive_btts, derive_correct_score_top5, derive_ou25


def test_1x2_exactly_matches_score_matrix_aggregation(deterministic_matrix: np.ndarray):
    matrix = deterministic_matrix
    one_x_two = derive_1x2(matrix)

    manual_home = sum(matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h > a)
    manual_draw = sum(matrix[d, d] for d in range(matrix.shape[0]))
    manual_away = sum(matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h < a)

    assert np.isclose(one_x_two["home"], manual_home)
    assert np.isclose(one_x_two["draw"], manual_draw)
    assert np.isclose(one_x_two["away"], manual_away)
    assert np.isclose(sum(one_x_two.values()), 1.0)


def test_ou25_and_btts_match_matrix_exactly(deterministic_matrix: np.ndarray):
    matrix = deterministic_matrix
    ou = derive_ou25(matrix)
    btts = derive_btts(matrix)

    manual_over = sum(matrix[h, a] for h in range(matrix.shape[0]) for a in range(matrix.shape[1]) if h + a > 2)
    manual_no = matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]

    assert np.isclose(ou["over_2_5"], manual_over)
    assert np.isclose(ou["over_2_5"] + ou["under_2_5"], 1.0)
    assert np.isclose(btts["btts_no"], manual_no)
    assert np.isclose(btts["btts_yes"] + btts["btts_no"], 1.0)


def test_correct_score_top5_preserves_original_probabilities(deterministic_matrix: np.ndarray):
    matrix = deterministic_matrix
    top5 = derive_correct_score_top5(matrix)

    for row in top5:
        h, a = row["score"].split("-")
        h_idx = int(h)
        a_idx = 2 if a == "6+" else int(a)
        assert np.isclose(row["probability"], matrix[h_idx, a_idx])


def test_asian_handicap_lines_and_recommended_line_are_coherent(deterministic_matrix: np.ndarray):
    lines = (-1.5, -0.5, 0.5, 1.5)
    derived = derive_ah(deterministic_matrix, lines=lines)

    for line in lines:
        key = f"home_{line:+.1f}"
        expected = sum(
            deterministic_matrix[h, a]
            for h in range(deterministic_matrix.shape[0])
            for a in range(deterministic_matrix.shape[1])
            if (h - a) > line
        )
        assert np.isclose(derived[key], expected)

    assert derived["recommended_line"] in lines
