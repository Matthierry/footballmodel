import numpy as np

from footballmodel.markets.derivation import derive_1x2, derive_btts, derive_ou25
from footballmodel.model.score_engine import GoalModelInputs, UnifiedScoreEngine


def test_score_matrix_sums_to_one():
    m = UnifiedScoreEngine(max_goals=6).score_matrix(GoalModelInputs(1.4, 1.1))
    assert np.isclose(m.sum(), 1.0)


def test_market_derivations_coherent():
    m = UnifiedScoreEngine(max_goals=6).score_matrix(GoalModelInputs(1.4, 1.1))
    one_x_two = derive_1x2(m)
    assert np.isclose(sum(one_x_two.values()), 1.0)
    ou = derive_ou25(m)
    assert np.isclose(ou["over_2_5"] + ou["under_2_5"], 1.0)
    btts = derive_btts(m)
    assert np.isclose(btts["btts_yes"] + btts["btts_no"], 1.0)
