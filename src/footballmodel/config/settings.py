from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class WeightConfig(BaseModel):
    dixon_coles: float = 0.60
    elo_prior: float = 0.25
    shot_adjustment: float = 0.15


class HalfLifeConfig(BaseModel):
    team_form_days: int = 60
    league_prior_days: int = 365


class LookbackConfig(BaseModel):
    team_form_days: int = 180
    league_prior_seasons: int = 3


class RuntimeConfig(BaseModel):
    value_edge_threshold: float = 0.025
    credibility_threshold: float = 0.55
    max_goals: int = 6


class AppConfig(BaseModel):
    leagues: list[str]
    weights: WeightConfig = Field(default_factory=WeightConfig)
    half_life: HalfLifeConfig = Field(default_factory=HalfLifeConfig)
    lookback: LookbackConfig = Field(default_factory=LookbackConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config(path: str | Path = "config/runtime.yaml") -> AppConfig:
    payload = load_yaml(path)
    return AppConfig.model_validate(payload)
