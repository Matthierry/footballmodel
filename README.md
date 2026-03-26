# footballmodel

Private football prediction and value-detection platform.

## Architecture
- Unified score-distribution engine (0..6 + 6+ bucket) as the single source for all markets.
- Derived markets: 1X2, OU2.5, BTTS, Correct Score, Asian Handicap.
- Data sources: Football-Data + ClubElo.
- Storage: DuckDB + Parquet-ready data folders.
- App: password-protected Streamlit dashboard.

## Monorepo structure
- `app/` Streamlit UI
- `src/footballmodel/` core library
- `config/` YAML configs
- `data/` raw/curated outputs
- `scripts/` automation + pipeline scripts
- `tests/` pytest suite
- `.github/workflows/` CI/scheduled jobs

## v1 exclusions
- player-level injuries
- weather
- advanced ML/deep learning
- public site
- auto-betting
