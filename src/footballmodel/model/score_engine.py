from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass(slots=True)
class GoalModelInputs:
    home_xg: float
    away_xg: float
    dc_rho: float = -0.05
    max_goals: int = 6


class UnifiedScoreEngine:
    """Build one coherent score matrix and derive all markets from it."""

    def __init__(self, max_goals: int = 6):
        self.max_goals = max_goals

    def score_matrix(self, inp: GoalModelInputs) -> np.ndarray:
        g = np.arange(0, self.max_goals + 1)
        h_probs = poisson.pmf(g, inp.home_xg)
        a_probs = poisson.pmf(g, inp.away_xg)

        h_tail = max(0.0, 1 - h_probs.sum())
        a_tail = max(0.0, 1 - a_probs.sum())
        h_probs[-1] += h_tail
        a_probs[-1] += a_tail

        matrix = np.outer(h_probs, a_probs)
        matrix = self._apply_dc_adjustment(matrix, inp.dc_rho)
        return matrix / matrix.sum()

    @staticmethod
    def _apply_dc_adjustment(matrix: np.ndarray, rho: float) -> np.ndarray:
        adjusted = matrix.copy()
        adjusted[0, 0] *= 1 - rho
        adjusted[0, 1] *= 1 + rho
        adjusted[1, 0] *= 1 + rho
        adjusted[1, 1] *= 1 - rho
        return np.clip(adjusted, 0, None)
