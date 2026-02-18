# 🏏 CricOracle 2026

> **AI-powered T20 World Cup prediction platform** — Match outcomes, score forecasts, and squad optimisation using ensemble ML, LSTM sequence models, and genetic algorithms.

[![CI](https://github.com/Viraj97-SL/CricOracle2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Viraj97-SL/CricOracle2026/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-53%25-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CricOracle 2026 is a production-grade cricket analytics platform built for the ICC T20 World Cup 2026 (India & Sri Lanka). It transforms 2,519 T20I matches of ball-by-ball data (565,377 deliveries) into actionable predictions through three ML modules.

| Module | Model | Task | Status | Metric |
|--------|-------|------|--------|--------|
| **A — Win Predictor** | XGBoost Ensemble | Match outcome probability | ✅ Live | **AUC-ROC 0.916** |
| **B — Score Predictor** | XGBoost Regression | First innings score | ✅ Live | **MAE 25.1 runs** |
| **C — Squad Optimiser** | Genetic Algorithm (DEAP) | Optimal Playing XI | 🔄 In progress | — |

---

## 🏆 Model Performance

### Module A — Win Predictor

| Metric | Score |
|--------|-------|
| Test AUC-ROC | **0.9157** |
| Test Accuracy | **82.8%** |
| Test Log Loss | 0.390 |
| CV AUC (5-fold temporal) | **0.890 ± 0.024** |
| CV Accuracy | 79.8% ± 1.9% |

**Top 15 Features by Importance:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `team2_batting_depth` | 26.9% |
| 2 | `batting_power_diff` | 6.4% |
| 3 | `team1_batting_depth` | 3.2% |
| 4 | `venue_chase_win_pct` | 2.8% |
| 5 | `team2_avg_boundary_pct` | 2.3% |
| 6 | `team1_batting_power` | 2.0% |
| 7 | `team1_avg_boundary_pct` | 1.9% |
| 8 | `experience_diff` | 1.8% |
| 9 | `team2_spin_bowling_pct` | 1.8% |
| 10 | `team2_form_L10` | 1.7% |

### Module B — Score Predictor

| Metric | Score |
|--------|-------|
| MAE | **25.1 runs** |
| RMSE | 32.3 runs |
| R² | **0.457** |
| Mean Actual | 152.6 runs |
| Mean Predicted | 149.1 runs |

---

## Architecture

```
Data Layer      →  Feature Store          →  Model Layer        →  Inference Layer
(Cricsheet)        (61 features, 6 groups)   (2 live modules)      (FastAPI + React)
565K deliveries    Player · Venue · H2H      Win + Score           /predict/match
2,519 matches      Form · Context · Team     Predictor             /predict/score
2019–2026          Strength                  XGBoost               /squad/optimise
```

### Feature Engineering Pipeline (6 Steps)

```
Raw CSV → [1] Load & Clean → [2] Player Profiles → [3] Venue Features
       → [4] Match Context → [5] Team Features → [6] Match Player Features
       → 61-column match-level dataset
```

**Feature Groups:**

| Group | Features | Description |
|-------|----------|-------------|
| Date | 3 | Month, day of week, year |
| Toss | 2 | Toss winner, decision |
| Venue | 12 | Avg scores, RPO by phase, spin/pace split, chase rate |
| H2H | 2 | Head-to-head win rate, matches played |
| Form | 6 | Rolling win rate (L10), experience, differentials |
| **Player Batting** | **12** | Team batting power, top-3 SR (form), boundary %, depth |
| **Player Bowling** | **10** | Economy, dot ball %, bowling SR, spin %, differentials |

---

## Key Engineering Decisions

- **Entity-aware aggregation** — playing XI inferred from ball-by-ball evidence (who actually batted/bowled), not from squad lists
- **Data-driven bowler classification** — K-Means clustering (Spin/Pace) replaces hardcoded dictionaries of 150+ bowlers
- **Phase-aware features** — separate stats for Powerplay (1–6) / Middle (7–15) / Death (16–20) overs
- **Temporal validation** — `TimeSeriesSplit` only, never random splits (cricket form is temporal)
- **Fallback-safe profiles** — global median used for players below minimum ball threshold, no crashes on sparse data
- **Separate feature BOMs** — Win and Score predictors use distinct feature sets; score model prioritises batting team strength over pre-match context

---

## Project Structure

```
CricOracle2026/
├── src/
│   ├── data/
│   │   ├── loader.py              # CricketDataLoader — load, clean, derive, validate
│   │   └── weather.py             # OpenWeatherMap API client
│   ├── features/
│   │   ├── pipeline.py            # FeaturePipeline orchestrator (6-step)
│   │   ├── player.py              # PlayerFeatureEngine — batting/bowling profiles, K-Means styles
│   │   ├── team.py                # TeamFeatureEngine — H2H, form, experience, strength
│   │   ├── venue.py               # VenueFeatureEngine — scoring, spin ratio, chase rates
│   │   ├── match_context.py       # MatchContextEngine — toss, date, target variable
│   │   └── match_player_features.py  # MatchPlayerFeatureEngine — XI inference → team strength ⭐
│   ├── models/
│   │   ├── win_predictor.py       # WinPredictor — XGBoost + temporal CV (AUC 0.916)
│   │   ├── score_predictor.py     # ScorePredictor — XGBoost regression (MAE 25.1)
│   │   ├── score_lstm.py          # LSTM sequence model (in progress)
│   │   ├── squad_optimiser.py     # Genetic algorithm — Playing XI selection (in progress)
│   │   └── trainer.py             # ModelTrainer — feature BOM definitions, training orchestration
│   ├── evaluation/
│   │   └── metrics.py             # SHAP explainability, calibration (in progress)
│   └── utils/
│       ├── config.py              # Settings, paths, hyperparameters, constants
│       ├── logger.py              # Structured logging
│       └── validators.py          # Pydantic row-level validation
├── api/                           # FastAPI backend (in progress)
├── tests/
│   ├── conftest.py                # Shared fixtures + sys.path setup
│   ├── test_loader.py             # 10 tests — data loading & column derivation
│   ├── test_features.py           # 9 tests — player & venue feature engines
│   ├── test_models.py             # 6 tests — WinPredictor train/predict/save/load
│   └── test_match_player_features.py  # 38 tests — player aggregation, trainer, integration ⭐
├── scripts/
│   ├── train.py                   # CLI: python -m scripts.train --pipeline --model all
│   └── predict.py                 # CLI: match prediction (in progress)
├── data/
│   ├── raw/                       # t20_ball_by_ball_v2.csv (Cricsheet)
│   └── processed/                 # Parquet feature store
├── models/                        # Saved model artifacts (.pkl)
├── docker-compose.yml
└── pyproject.toml
```

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Viraj97-SL/CricOracle2026.git
cd CricOracle2026
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place Cricsheet T20I ball-by-ball CSV in data/raw/
cp /path/to/your/data.csv data/raw/t20_ball_by_ball_v2.csv

# Run full feature pipeline + train all models (~2 min)
python -m scripts.train --pipeline --model all

# Run test suite
pytest tests/ -v
```

---

## Data Requirements

- **Source:** [Cricsheet](https://cricsheet.org/) T20I ball-by-ball CSV
- **Format:** One row per delivery with columns: `match_id`, `date`, `venue`, `team1`, `team2`, `winner`, `batting_team`, `over`, `batter`, `bowler`, `runs_batter`, `runs_extra`, `runs_total`, `wicket_type`, `player_out`
- **Coverage:** 3,132 raw T20I matches, filtered to 2,519 modern-era matches (2019–2026)
- **Volume:** 706,342 raw deliveries, 565,377 after modern era filter

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| **ML/DL** | XGBoost · scikit-learn · PyTorch (LSTM) · DEAP (GA) · Optuna · SHAP |
| **Data** | pandas · NumPy · Cricsheet |
| **Backend** | FastAPI · Pydantic · uvicorn |
| **DevOps** | Docker · GitHub Actions · pytest · ruff |
| **Frontend** | React · Tailwind CSS · Recharts *(planned)* |

---

## Roadmap

- [x] Data loading & validation pipeline
- [x] Player feature engine (batting profiles, bowling profiles, K-Means style classification)
- [x] Venue feature engine (phase-wise RPO, spin/pace ratio, chase stats)
- [x] Team feature engine (H2H, rolling form, experience)
- [x] **Match player feature engine** (XI inference → team batting/bowling strength)
- [x] Win Predictor — XGBoost (AUC 0.916, Accuracy 82.8%)
- [x] Score Predictor — XGBoost regression (MAE 25.1 runs)
- [x] 63-test pytest suite with temporal CV validation
- [ ] SHAP explainability layer (`evaluation/metrics.py`)
- [ ] Score Predictor — LSTM sequence model (target MAE < 15 runs)
- [ ] Squad Optimiser — Genetic Algorithm (DEAP)
- [ ] FastAPI backend with `/predict/match`, `/predict/score`, `/squad/optimise`
- [ ] Optuna hyperparameter tuning
- [ ] Model calibration (Platt scaling)
- [ ] React frontend dashboard
- [ ] Weather API integration (dew point for evening T20s)

---

## API Endpoints *(planned)*

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/match` | POST | Win probability for both teams |
| `/predict/score` | POST | First innings score prediction |
| `/squad/optimise` | POST | Optimal Playing XI from 15-man squad |
| `/docs` | GET | Interactive Swagger documentation |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Viraj Bulugahapitiya](https://github.com/Viraj97-SL) | MSc Data Science
