from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


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


class CalibrationConfig(BaseModel):
    enabled: bool = True
    min_samples: int = 50


class RuntimeConfig(BaseModel):
    value_edge_threshold: float = 0.025
    credibility_threshold: float = 0.55
    max_goals: int = 6


class LiveModelConfig(BaseModel):
    version: str
    leagues: list[str]
    weights: WeightConfig = Field(default_factory=WeightConfig)
    half_life: HalfLifeConfig = Field(default_factory=HalfLifeConfig)
    lookback: LookbackConfig = Field(default_factory=LookbackConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class AppConfig(BaseModel):
    leagues: list[str]
    weights: WeightConfig = Field(default_factory=WeightConfig)
    half_life: HalfLifeConfig = Field(default_factory=HalfLifeConfig)
    lookback: LookbackConfig = Field(default_factory=LookbackConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    default_live_config: str = "champion_v1"
    live_configs: dict[str, LiveModelConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_default_config(self) -> AppConfig:
        if not self.live_configs:
            self.live_configs = {
                self.default_live_config: LiveModelConfig(
                    version="1.0.0",
                    leagues=self.leagues,
                    weights=self.weights,
                    half_life=self.half_life,
                    lookback=self.lookback,
                    runtime=self.runtime,
                    calibration=self.calibration,
                )
            }
        if self.default_live_config not in self.live_configs:
            raise ValueError(f"default_live_config '{self.default_live_config}' missing from live_configs")
        return self

    def resolve_live_config(self, config_name: str | None = None) -> tuple[str, LiveModelConfig]:
        selected = config_name or self.default_live_config
        if selected not in self.live_configs:
            raise KeyError(f"Unknown live config '{selected}'. Available: {sorted(self.live_configs)}")
        return selected, self.live_configs[selected]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config(path: str | Path = "config/runtime.yaml") -> AppConfig:
    payload = load_yaml(path)
    return AppConfig.model_validate(payload)
