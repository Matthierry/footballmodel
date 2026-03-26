from __future__ import annotations

import numpy as np

from footballmodel.model.score_engine import GoalModelInputs, UnifiedScoreEngine


def test_score_matrix_shape_and_probability_bounds():
    matrix = UnifiedScoreEngine(max_goals=6).score_matrix(GoalModelInputs(1.4, 1.1, dc_rho=-0.05))
    assert matrix.shape == (7, 7)
    assert np.all(matrix >= 0)
    assert np.all(matrix <= 1)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-12)


def test_tail_bucket_is_probability_conserving_before_and_after_adjustment():
    engine = UnifiedScoreEngine(max_goals=6)
    raw = engine.score_matrix(GoalModelInputs(2.2, 1.8, dc_rho=0.0))
    adjusted = engine.score_matrix(GoalModelInputs(2.2, 1.8, dc_rho=-0.08))
    assert np.isclose(raw.sum(), 1.0, atol=1e-12)
    assert np.isclose(adjusted.sum(), 1.0, atol=1e-12)
    assert adjusted[0, 0] != raw[0, 0]


def test_low_score_adjustment_preserves_non_negative_probabilities_with_extreme_rho():
    matrix = UnifiedScoreEngine(max_goals=6).score_matrix(GoalModelInputs(1.0, 1.0, dc_rho=0.95))
    assert np.all(matrix >= 0)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-12)
