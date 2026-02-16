# 🏏 CricOracle 2026

> **AI-powered T20 World Cup prediction platform** — Match outcomes, score forecasts, and squad optimisation using ensemble ML, LSTM sequence models, and genetic algorithms.

[![CI](https://github.com/YOUR_USERNAME/CricOracle2026/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/CricOracle2026/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

CricOracle 2026 is a production-grade cricket analytics platform built for the ICC T20 World Cup 2026 (India & Sri Lanka). It transforms 3,000+ T20I matches of ball-by-ball data into actionable predictions through three ML modules:

| Module | Model | Task | Target |
|--------|-------|------|--------|
| **A — Win Predictor** | XGBoost + LightGBM + CatBoost Ensemble | Match outcome probability | AUC-ROC > 0.70 |
| **B — Score Predictor** | XGBoost (tabular) + LSTM (sequence) | First innings & over-by-over scores | MAE < 15 runs |
| **C — Squad Optimiser** | Genetic Algorithm (DEAP) | Optimal Playing XI selection | Overlap > 7/11 |

## Architecture

```
Data Layer → Feature Store → Model Layer → Inference Layer
(Cricsheet)   (80+ features)  (3 modules)   (FastAPI + React)
```

**Feature Families:** Player Form (rolling windows) · Team Strength · Venue & Conditions · Match Context · Weather (OpenWeatherMap) · Sentiment (NLP)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/CricOracle2026.git
cd CricOracle2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place your Cricsheet CSV in data/raw/
cp /path/to/t20_ball_by_ball.csv data/raw/t20_ball_by_ball_v2.csv

# Run feature pipeline + train models
python scripts/train.py --pipeline --model all

# Start API
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

## Project Structure

```
CricOracle2026/
├── src/
│   ├── data/          # Data loading, weather API client
│   ├── features/      # Player, team, venue, context feature engines
│   ├── models/        # Win predictor, score LSTM, squad optimiser
│   ├── evaluation/    # Metrics, SHAP explainability, calibration
│   └── utils/         # Logger, validators, config
├── api/               # FastAPI backend with prediction endpoints
├── tests/             # Pytest suite with fixtures
├── scripts/           # CLI training and prediction scripts
├── notebooks/         # Exploration notebooks (non-production)
└── data/              # Raw, processed, external data
```

## Key Innovations

- **Entity Embeddings** for teams/venues (replaces label encoding)
- **Data-driven bowler classification** via K-Means clustering (replaces hardcoded dictionaries)
- **Phase-aware features** — separate stats for Powerplay/Middle/Death overs
- **Temporal validation** — time-based splits, never random (cricket form is temporal)
- **SHAP explainability** — every prediction comes with "why" factors
- **Weather integration** — dew point critically affects evening T20 matches

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/match` | POST | Win probability for both teams |
| `/predict/score` | POST | First innings score prediction |
| `/squad/optimise` | POST | Optimal Playing XI from 15-man squad |
| `/docs` | GET | Interactive Swagger documentation |

## Tech Stack

**ML/DL:** XGBoost · LightGBM · CatBoost · PyTorch (LSTM) · scikit-learn · Optuna · SHAP  
**Backend:** FastAPI · Pydantic · uvicorn  
**Data:** pandas · NumPy · Cricsheet  
**DevOps:** Docker · GitHub Actions · pytest · ruff  
**Frontend:** React · Tailwind CSS · Recharts (planned)

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Viraj Bulugahapitiya](https://github.com/YOUR_USERNAME) | MSc Data Science
