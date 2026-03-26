from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SubModelSignal:
    home_xg: float
    away_xg: float


def blend_signals(
    dixon_coles: SubModelSignal,
    elo_prior: SubModelSignal,
    shot_adjustment: SubModelSignal,
    w_dc: float,
    w_elo: float,
    w_shot: float,
) -> SubModelSignal:
    total = w_dc + w_elo + w_shot
    if total <= 0:
        raise ValueError("weights must sum to positive value")

    return SubModelSignal(
        home_xg=(dixon_coles.home_xg * w_dc + elo_prior.home_xg * w_elo + shot_adjustment.home_xg * w_shot) / total,
        away_xg=(dixon_coles.away_xg * w_dc + elo_prior.away_xg * w_elo + shot_adjustment.away_xg * w_shot) / total,
    )


def elo_to_goal_prior(home_elo: float, away_elo: float) -> SubModelSignal:
    delta = (home_elo - away_elo) / 400
    home = 1.25 + 0.5 * delta
    away = 1.10 - 0.4 * delta
    return SubModelSignal(max(0.2, home), max(0.2, away))
