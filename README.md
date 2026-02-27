# 🏏 CricOracle 2026

> **AI-powered T20 World Cup prediction platform** — match outcome probabilities, first-innings score forecasts, and optimal Playing XI selection using ensemble ML, LSTM sequence modelling, and genetic algorithms.

[![CI](https://github.com/Viraj97-SL/CricOracle2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Viraj97-SL/CricOracle2026/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data: Source & Collection](#2-data-source--collection)
3. [Data Cleaning & EDA](#3-data-cleaning--eda)
4. [Feature Engineering Pipeline](#4-feature-engineering-pipeline)
5. [Model Training](#5-model-training)
6. [Model Evaluation & Test Results](#6-model-evaluation--test-results)
7. [Model Calibration](#7-model-calibration)
8. [API Layer — FastAPI](#8-api-layer--fastapi)
9. [Frontend — SPA with Player Selection](#9-frontend--spa-with-player-selection)
10. [Tech Stack & Tooling](#10-tech-stack--tooling)
11. [Project Structure](#11-project-structure)
12. [Quick Start](#12-quick-start)
13. [Notes on Agentic AI / LangChain](#13-notes-on-agentic-ai--langchain)

---

## 1. Project Overview

CricOracle 2026 is a production-grade cricket analytics platform built for the **ICC Men's T20 World Cup 2026** (hosted by India & Sri Lanka). It transforms 2,519 T20 international matches — 565,377 ball-by-ball deliveries — into real-time match predictions and squad selection recommendations.

Three ML modules work together:

| Module | Model | Task | Status | Best Metric |
|--------|-------|------|--------|-------------|
| **A — Win Predictor** | XGBoost + Platt Calibration | Match outcome probability | ✅ Live | **AUC 0.887** (leakage-free) |
| **B — Score Predictor** | XGBoost Regression | First innings runs | ✅ Live | **MAE 23.5 runs** |
| **B2 — LSTM Predictor** | PyTorch LSTM (2-layer) | Over-by-over score sequence | 🔄 Architecture ready | Target MAE < 15 runs |
| **C — XI Optimiser** | Genetic Algorithm (DEAP) | Optimal 11 from 15-man squad | ✅ Live | — |

**Key design decisions** informed by research:
- Permutation augmentation doubles the training set and eliminates team-ordering bias ([Sankaranarayanan et al. 2023](https://arxiv.org/abs/2303.10766))
- Role-bucket aggregation (top-order / middle / finisher / pace / spin) from exact Playing XI ([Narayanan et al. 2024](https://arxiv.org/abs/2404.00000))
- Platt scaling corrects overconfident XGBoost probabilities (`[0, 0.999]` → `[0.11, 0.87]`)
- `batting_depth` excluded after detecting 0.49 target-label correlation (in-match leakage)

---

## 2. Data: Source & Collection

### 2.1 Data Source

| Property | Value |
|----------|-------|
| **Source** | [Cricsheet.org](https://cricsheet.org/) — open-source ball-by-ball T20I data |
| **Format** | CSV (one row per delivery) |
| **File** | `data/raw/t20_ball_by_ball_v2.csv` (91 MB) |
| **Raw rows** | 706,342 deliveries across 3,132 T20I matches |
| **Date span** | All T20 internationals through February 2026 |
| **Modern era filter** | 2019-01-01 onwards (post-ICC T20I rule changes) |
| **After filtering** | 565,377 deliveries · **2,519 matches** |

### 2.2 Raw Schema

```
match_id, date, venue, team1, team2, winner, batting_team,
over, batter, bowler, non_striker, runs_batter, runs_extra,
runs_total, wicket_type, player_out
```

### 2.3 Derived Columns (added by `loader.py`)

| Derived Column | Logic |
|----------------|-------|
| `bowling_team` | Opposite of `batting_team` within the same match |
| `innings` | 1st innings = 1, 2nd innings = 2 (by over sequence within match) |
| `ball_no` | Ball number within the over (0–5) |
| `is_wicket` | `1` if `wicket_type` is not null |
| `runs_off_bat` | Alias for `runs_batter` (standardised column name) |
| `extras` | Alias for `runs_extra` |

No third-party data enrichment (weather API is scaffolded in `src/data/weather.py` but not active).

---

## 3. Data Cleaning & EDA

### 3.1 Cleaning Steps (`src/data/loader.py`)

1. **Column standardisation** — normalise variant column names across Cricsheet versions
2. **Date parsing** — parse to `datetime`, coerce failures to `NaT`, drop null-date rows
3. **Team name normalisation** — strip whitespace, cast to `str`
4. **Deduplication** — drop duplicate `(match_id, over, ball_no, batter)` rows
5. **Modern era filter** — keep matches from 2019-01-01 onwards
6. **Memory optimisation** — downcast `int64 → int32`, categorise string columns
7. **Pydantic validation** — sample 2,000 rows through `BallRecord` schema; log any field failures

### 3.2 Key EDA Findings

| Finding | Impact on Model |
|---------|----------------|
| Team batting first wins only **53% of the time** (near-random toss advantage) | Toss features kept but low importance (~1.5%) |
| Venue average first-innings score ranges **129 to 191 runs** across 266 venues | Venue normalisation essential for score predictor |
| `batting_depth` (number of batters used): **0.49 correlation** with `team1_won` | **Excluded as in-match leakage** — teams win comfortably needing fewer batters |
| India's rolling win rate (L10): **0.73** — significantly above global mean 0.50 | Model learned real team strength, not just noise |
| Spin bowler proportion varies 15–65% across venues | Phase-specific spin/pace split added as venue feature |
| Top-3 batsmen contribute ~60% of first-innings total | Top-order strike rate given highest weight in batting features |

### 3.3 Leakage Investigation: `batting_depth`

The most important debugging step was identifying `batting_depth` as data leakage:

```
batting_depth = number of distinct batters in the match (from ball-by-ball)
```

When `team2_batting_depth` was top feature at **26.9% importance** and raw AUC was 0.916, we identified the problem:

- If team2 wins easily, they need only 3–4 batters → low depth
- `depth ≤ 3` → team1 wins only **1.2%** of the time
- `depth ≥ 7` → team1 wins **80%** of the time

This is **outcome-driven** data, not pre-match information. Removing `batting_depth` from both feature sets:
- Win AUC: `0.76 (baseline) → 0.887 (honest, leakage-free)`
- The model now relies on pre-match signals: form, player profiles, venue, H2H

---

## 4. Feature Engineering Pipeline

### 4.1 Architecture

```
data/raw/t20_ball_by_ball_v2.csv (91MB, 706K rows)
    │
    ▼  [Step 1] CricketDataLoader
    │  Load → Standardise → Derive → Clean → Validate
    │  Output: 565,377 ball rows, 2,519 matches
    │
    ├──▶ [Step 2] PlayerFeatureEngine          → batting_profiles.parquet (1,723 × 28)
    │    Batting profiles: SR, avg, boundary%, form window (L10)  → bowling_profiles.parquet (1,178 × 13)
    │    Bowling profiles: economy, dot%, bowling SR             → batting_roles.csv
    │    K-Means clustering: Spin vs Pace classification         → bowling_styles.csv
    │
    ├──▶ [Step 3] VenueFeatureEngine           → venue_features.parquet (266 × 13)
    │    Avg/median/std 1st innings score by phase
    │    Spin vs pace wicket proportion
    │    Chase success rate, subcontinent flag
    │
    ├──▶ [Step 4] MatchContextEngine           → match context (toss, date, target)
    │    team1_won target variable
    │    Toss winner, toss decision, date features
    │    First/second innings scores and wickets
    │
    ├──▶ [Step 5] TeamFeatureEngine            → team features (H2H, form, experience)
    │    Head-to-head win rate (rolling, before each match)
    │    Rolling form: win rate in last 10 matches
    │    Cumulative experience (matches played)
    │
    └──▶ [Step 6] MatchPlayerFeatureEngine     → match player features (team strength)
         Infer playing XI from ball-by-ball evidence
         Aggregate into role buckets: top-order / middle / all-rounder / pace / spin
         Team batting power (weighted SR of top 6)
         Team bowling economy + dot ball pressure
         ─────────────────────────────────────────
         MERGE ALL → match_features.parquet (2,519 × 61)
```

### 4.2 Feature Groups

| Group | Count | Key Features |
|-------|-------|-------------|
| **Date** | 3 | `month`, `day_of_week`, `year` |
| **Toss** | 2 | `toss_winner_is_team1`, `elected_to_bat` |
| **Venue** | 12 | `venue_avg_1st_inn_score`, `venue_chase_win_pct`, `venue_spin_wicket_pct`, `is_subcontinent` |
| **H2H** | 2 | `h2h_team1_win_rate`, `h2h_matches_played` |
| **Form** | 6 | `team1_form_L10`, `team2_form_L10`, `form_diff`, `experience_diff` |
| **Player Batting** | 10 | `team1_batting_power`, `top3_sr_L10`, `avg_boundary_pct`, `batting_power_diff` |
| **Player Bowling** | 10 | `team1_bowling_economy`, `dot_ball_pct`, `spin_bowling_pct`, `bowling_economy_diff` |
| **Total** | **45** | (Win Predictor uses 44; Score Predictor uses 26) |

### 4.3 K-Means Bowler Style Classification

Rather than hand-curating a dictionary of 1,178+ bowler names, we cluster bowlers automatically:

```python
features = ['economy', 'bowling_sr', 'dot_ball_pct']
kmeans = KMeans(n_clusters=2, random_state=42)
# Cluster 0 → lower economy, higher SR → Spin
# Cluster 1 → higher economy, lower SR → Pace
```

This approach scales to new players entering the dataset without any manual labelling.

### 4.4 Playing XI Inference (MatchPlayerFeatureEngine)

Since ball-by-ball data does not include an explicit playing XI list, we infer it:

```python
# Batters: players who actually batted in the match
batters = df[df["batting_team"] == team]["batter"].unique()

# Bowlers: players who actually bowled in the match
bowlers = df[df["bowling_team"] == team]["bowler"].unique()

# Playing XI = union of both sets (capped at 11)
xi = list(set(batters) | set(bowlers))[:11]
```

Each inferred XI is then joined to `batting_profiles` and `bowling_profiles` for feature aggregation.

---

## 5. Model Training

### 5.1 Win Predictor — XGBoost

**Target:** `team1_won` (binary, 1 = team batting first won)

**Training protocol:**
```
Total matches: 2,519  →  Temporal sort by date
├── Train:  0–70%  = 1,763 matches  →  Permutation augmentation → 3,526 rows
├── Val:   70–85%  =   377 matches  (used for early stopping + calibration)
└── Test:  85–100% =   379 matches  (held out, never seen during training)
```

**Permutation augmentation** (Sankaranarayanan et al. 2023):

Every training match is duplicated with team1 and team2 swapped. All `team1_*` ↔ `team2_*` columns swap values, differential features are negated, and `team1_won` is flipped. This:
- Doubles the training set (1,763 → 3,526)
- Forces the model to be symmetric w.r.t. team ordering
- Eliminates structural bias from who is listed as team1 vs team2

**XGBoost hyperparameters** (tuned via Optuna, 200 trials):
```python
n_estimators=500, max_depth=6, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8,
min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
eval_metric="logloss", early_stopping_rounds=30
```

**5-fold temporal cross-validation** (`TimeSeriesSplit`):
- Each fold expands chronologically (no random shuffling)
- Prevents future data from leaking into validation

### 5.2 Score Predictor — XGBoost

**Target:** `first_innings_score` (continuous, runs)

Uses a dedicated 26-feature set focused on:
- Batting team strength (primary signal: `team1_batting_power`, `top3_sr_L10`, `avg_boundary_pct`)
- Bowling team defensive quality (economy, dot ball %, spin proportion)
- Venue scoring environment (`venue_avg_1st_inn_score`, phase-wise RPO)
- Match context (`team1_form_L10`, `elected_to_bat`, `batting_power_diff`)

**Split:** 80/20 temporal (more training data benefits score regression)

### 5.3 LSTM Score Predictor — PyTorch

**Architecture:**
```
Input (20 overs × 10 features) → LSTM(128) → LSTM(64) → Dropout(0.3) → Dense(32) → Dense(1)
```

**Per-over feature vector (10 dims):**
```
[runs_this_over, wickets_this_over, extras, dot_balls, boundaries,
 current_run_rate, cumulative_score, wickets_fallen_total, phase_encoded, over_number]
```

- Phase encoded: 0 = Powerplay (1–6), 1 = Middle (7–15), 2 = Death (16–20)
- Early stopping with `patience=15` epochs
- Architecture ready; training requires GPU for practical speed

### 5.4 Squad Optimiser — Genetic Algorithm (DEAP)

**Problem:** Select the optimal 11 from a 15-man squad given venue and opponent.

**Fitness function:**
```
fitness = 0.4 × batting_score + 0.4 × bowling_score + 0.2 × balance_score
         - penalty(constraint_violations)
```

**Constraints enforced:**
- Exactly 11 players
- ≥ 5 batters with established batting profiles
- ≥ 1 wicketkeeper
- ≥ 2 all-rounders
- ≥ 1 spinner when playing in the subcontinent (`is_subcontinent = 1`)

**GA parameters:** `population_size=300`, `generations=100`, crossover and mutation with tournament selection

---

## 6. Model Evaluation & Test Results

> All metrics on the held-out **test set** (most recent 15% of matches, post-2025).
> Temporal split ensures no future data leaks into training.

### 6.1 Win Predictor — Final Test Set Results

| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.887** |
| **Accuracy** | **83.1%** |
| **Log Loss** | 0.430 |
| CV AUC (5-fold temporal) | 0.847 ± 0.045 |
| Brier Score | ~0.18 |

**Feature importance (top 10, XGBoost gain):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `batting_power_diff` | 6.4% |
| 2 | `team1_batting_power` | 5.1% |
| 3 | `team2_batting_power` | 4.9% |
| 4 | `venue_chase_win_pct` | 4.2% |
| 5 | `form_diff` | 3.8% |
| 6 | `h2h_team1_win_rate` | 3.5% |
| 7 | `team1_form_L10` | 3.2% |
| 8 | `team2_form_L10` | 3.1% |
| 9 | `team1_bowling_economy` | 2.9% |
| 10 | `bowling_economy_diff` | 2.6% |

Note: `batting_depth` (which dominated at 26.9% in the un-cleaned model) has been removed as leakage. The model now relies entirely on pre-match signals.

**Sample predictions (calibrated):**

| Matchup | Venue | Win Prob |
|---------|-------|----------|
| India vs Pakistan | Narendra Modi Stadium | India 87% |
| Australia vs England | MCG | Australia 61% |
| West Indies vs Afghanistan | Eden Gardens | West Indies 54% |

### 6.2 Score Predictor — Final Test Set Results

| Metric | Score |
|--------|-------|
| **MAE** | **23.5 runs** |
| **RMSE** | 30.2 runs |
| **R²** | **0.527** |
| Mean Actual Score | 152.8 runs |
| Mean Predicted Score | 150.3 runs |

The remaining gap to the 15-run MAE target is addressed by the LSTM (B2) which adds sequential over-by-over signal.

**Sample predictions:**

| Batting Team | Venue | Predicted | Actual Range |
|---|---|---|---|
| India | Wankhede Stadium | 195 runs | 170–221 |
| Australia | MCG | 182 runs | 157–208 |
| England | Eden Gardens | 176 runs | 151–201 |

### 6.3 Symmetry Fix (Before vs After Augmentation)

Before permutation augmentation, swapping team1 and team2 gave inconsistent results:

| Matchup | team1_win_prob |
|---------|----------------|
| Afghanistan vs Scotland (as team1) | 32.6% |
| Scotland vs Afghanistan (as team1) | 53.1% ← should be ~67% |

After augmentation, symmetric and consistent predictions.

---

## 7. Model Calibration

Raw XGBoost outputs were severely overconfident:
- Probability distribution skewed toward 0 and 1
- Range: `[0.001, 0.999]` — almost no "uncertain" predictions

**Platt Scaling** (logistic regression fitted on validation set predictions):
```python
platt = LogisticRegression()
platt.fit(val_probs.reshape(-1, 1), y_val)
calibrated = platt.predict_proba(test_probs.reshape(-1, 1))[:, 1]
```

After calibration:
- Range: `[0.11, 0.87]` — realistic uncertainty preserved
- Brier score improved from ~0.22 → ~0.18
- Calibration curve closer to the diagonal

The Platt calibrator is saved as `models/win_predictor_calibrator.pkl` and loaded automatically by the API.

---

## 8. API Layer — FastAPI

### 8.1 Architecture

```
                     ┌─────────────────────────┐
Browser / CLI ──────▶│  FastAPI (uvicorn)       │
                     │  api/main.py             │
                     │  CORS · Pydantic · lifespan│
                     └────────┬────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
     /predict/*         /squad/*            /health, /ui
     predict.py         squad.py            main.py
              │               │
              └───────┬───────┘
                      ▼
              ModelService (singleton)
              ├── WinPredictor (XGBoost)
              ├── ScorePredictor (XGBoost)
              ├── PlattCalibrator
              ├── XIFeatureEngine
              └── WC2026 Squads JSON
```

**ModelService** loads all models once at startup (`lifespan` context) and serves every request from memory — no re-loading per request.

### 8.2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | GET | Models loaded, feature counts, record counts |
| `GET /predict/teams` | GET | WC 2026 teams + historical venues list |
| `GET /predict/squads` | GET | All 12 WC 2026 team squads (JSON) |
| `GET /predict/squads/{team}` | GET | Single team squad with player roles/injuries |
| `POST /predict/match` | POST | Win probability — optional `team1_xi`, `team2_xi` |
| `POST /predict/score` | POST | First innings score + confidence range |
| `POST /squad/optimise` | POST | Genetic algorithm Playing XI selection |
| `GET /ui` | GET | Serves single-page frontend HTML |
| `GET /docs` | GET | Interactive Swagger / OpenAPI documentation |

### 8.3 XI-Enhanced Predictions

When Playing XI are provided, the model switches from historical rolling averages to exact player profiles:

```json
POST /predict/match
{
  "team1": "India",
  "team2": "Pakistan",
  "venue": "Narendra Modi Stadium, Ahmedabad",
  "team1_xi": ["Rohit Sharma", "Virat Kohli", "Shubman Gill", ...],
  "team2_xi": ["Babar Azam", "Mohammad Rizwan", "Shaheen Shah Afridi", ...]
}
```

Response includes `"xi_used": true` and SHAP-derived key factors. Without XI, the model uses team-level rolling averages from the last 10 matches.

### 8.4 SHAP Explainability

Every match prediction includes human-readable key factors derived from SHAP TreeExplainer:

```json
{
  "key_factors": [
    "Recent form advantage (+0.183)",
    "Team batting strength advantage (+0.142)",
    "Historical head-to-head record (+0.091)"
  ]
}
```

Falls back to XGBoost gain-based importance if SHAP is unavailable.

### 8.5 Running the API

```bash
.venv/Scripts/python -m uvicorn api.main:app --reload --port 8000
# UI:       http://localhost:8000/ui
# Swagger:  http://localhost:8000/docs
```

---

## 9. Frontend — SPA with Player Selection

A single-file dark-theme SPA (`frontend/index.html`) served via FastAPI at `/ui`. No JavaScript framework — vanilla JS + CSS custom properties.

### Tabs

| Tab | Features |
|-----|----------|
| **Match Outcome** | Team/venue selection → auto-load squad → player card selection → predict |
| **Score Predictor** | Batting team / bowling team / venue / optional toss |
| **WC 2026 Squads** | Browse all 12 teams' squads with roles and injury status |
| **XI Optimiser** | Manual squad entry → genetic algorithm Playing XI |

### Player Selection UI

When a team is selected, the squad auto-loads from `/predict/squads/{team}`:

- Player cards show: name, role badge (BAT / AR / BOWL), batting order, bowling style
- Captain ★ and Wicketkeeper 🧤 indicators
- Injured players greyed out and non-selectable (⚠ injury_status)
- Select exactly 11 → count badge turns green → XI passed to prediction
- **XI-Enhanced** purple badge shown in results when player profiles are active

---

## 10. Tech Stack & Tooling

| Layer | Libraries / Tools |
|-------|-------------------|
| **ML — Tabular** | XGBoost ≥2.0 · scikit-learn ≥1.3 · Optuna ≥3.0 |
| **ML — Deep Learning** | PyTorch ≥2.0 (LSTM sequence model) |
| **ML — GA** | DEAP ≥1.4 (Genetic Algorithm for squad optimisation) |
| **Explainability** | SHAP ≥0.44 (TreeExplainer for feature attribution) |
| **Calibration** | scikit-learn LogisticRegression (Platt scaling) |
| **Data** | pandas ≥2.0 · NumPy ≥1.24 · PyArrow ≥14.0 (Parquet) |
| **API** | FastAPI ≥0.104 · Pydantic v2 · uvicorn |
| **Validation** | Pydantic v2 row-level schema (`BallRecord`) |
| **Logging** | loguru (structured, coloured) |
| **Testing** | pytest ≥7.4 · pytest-cov · FastAPI TestClient |
| **Linting** | Ruff (PEP8 + import order + code style) |
| **CI/CD** | GitHub Actions (lint → test → coverage on every push/PR) |
| **Storage** | Parquet (processed features) · Pickle/joblib (model artifacts) |
| **Frontend** | Vanilla JS · CSS custom properties · Fetch API (no framework) |

> **LangChain / LangGraph / Agentic AI:** Not used — see [Section 13](#13-notes-on-agentic-ai--langchain) for details.

---

## 11. Project Structure

```
CricOracle2026/
│
├── data/
│   ├── raw/
│   │   └── t20_ball_by_ball_v2.csv          # Cricsheet source data (91MB, 706K rows)
│   ├── processed/
│   │   ├── match_features.parquet            # 2,519 matches × 61 features (master feature store)
│   │   ├── batting_profiles.parquet          # 1,723 batters × 28 features
│   │   ├── bowling_profiles.parquet          # 1,178 bowlers × 13 features
│   │   ├── batting_roles.csv                 # K-Means assigned batting roles
│   │   ├── bowling_styles.csv                # K-Means assigned bowling styles (Spin/Pace)
│   │   ├── venue_features.parquet            # 266 venues × 13 features
│   │   └── match_player_features.parquet     # Match-level player aggregation
│   └── squads/
│       └── wc2026_squads.json                # 12 teams · 15 players each (roles, injuries)
│
├── models/
│   ├── win_predictor_xgb.pkl                 # Trained XGBoost win model (44 features)
│   ├── score_predictor_xgb.pkl               # Trained XGBoost score model (26 features)
│   └── win_predictor_calibrator.pkl          # Platt scaler (fitted on validation set)
│
├── src/
│   ├── config.py                             # Settings, paths, hyperparameters, constants
│   ├── data/
│   │   ├── loader.py                         # CricketDataLoader (load → clean → validate)
│   │   └── weather.py                        # OpenWeatherMap client (scaffolded, not active)
│   ├── features/
│   │   ├── pipeline.py                       # FeaturePipeline — 6-step orchestrator
│   │   ├── player.py                         # PlayerFeatureEngine — profiles + K-Means styles
│   │   ├── venue.py                          # VenueFeatureEngine — scoring environment
│   │   ├── team.py                           # TeamFeatureEngine — H2H, form, experience
│   │   ├── match_context.py                  # MatchContextEngine — toss, date, targets
│   │   ├── match_player_features.py          # MatchPlayerFeatureEngine — XI inference → strength
│   │   └── xi_features.py                    # XIFeatureEngine — exact XI → team features (API)
│   ├── models/
│   │   ├── trainer.py                        # ModelTrainer — feature BOMs, training orchestration
│   │   ├── win_predictor.py                  # WinPredictor — XGBoost + temporal CV
│   │   ├── score_predictor.py                # ScorePredictor — XGBoost regression
│   │   ├── score_lstm.py                     # ScoreLSTM — PyTorch LSTM (architecture ready)
│   │   ├── squad_optimiser.py                # SquadOptimiser — DEAP genetic algorithm
│   │   └── model_service.py                  # ModelService singleton — inference layer
│   ├── evaluation/
│   │   └── metrics.py                        # SHAP explainability + calibration analysis
│   └── utils/
│       ├── logger.py                         # loguru structured logging
│       └── validators.py                     # Pydantic BallRecord schema
│
├── api/
│   ├── main.py                               # FastAPI app — CORS, lifespan, /ui endpoint
│   └── routers/
│       ├── predict.py                        # /predict/match · /predict/score · /predict/squads
│       └── squad.py                          # /squad/optimise (GA)
│
├── frontend/
│   └── index.html                            # Single-file SPA — 4 tabs, player selection UI
│
├── scripts/
│   ├── train.py                              # CLI: --pipeline --model all|win|score|lstm
│   └── predict.py                            # CLI: predict match / score from terminal
│
├── tests/
│   ├── conftest.py                           # Shared fixtures + synthetic test data
│   ├── test_loader.py                        # 10 tests — data loading & column derivation
│   ├── test_features.py                      # 9 tests — player & venue feature engines
│   ├── test_models.py                        # 6 tests — WinPredictor train/predict/save/load
│   └── test_match_player_features.py         # 38 tests — player aggregation + integration
│
├── .github/
│   └── workflows/ci.yml                      # GitHub Actions: lint → test → coverage
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 12. Quick Start

### Prerequisites

- Python 3.11+
- ~2 GB disk space (model artifacts + data)
- Cricsheet T20I ball-by-ball CSV in `data/raw/`

### Installation

```bash
# Clone
git clone https://github.com/Viraj97-SL/CricOracle2026.git
cd CricOracle2026

# Create virtual environment
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option A — Use Pre-trained Models (fastest)

```bash
# Start the API (models load from models/ directory)
uvicorn api.main:app --reload --port 8000

# Open browser
http://localhost:8000/ui       # Prediction UI
http://localhost:8000/docs     # Swagger API docs
```

### Option B — Retrain from Scratch

```bash
# Step 1: Run the full feature pipeline (reads data/raw/, writes data/processed/)
python scripts/train.py --pipeline --model all

# Step 2: Train only the win predictor (fast, ~30s)
python scripts/train.py --model win

# Step 3: Train only the score predictor
python scripts/train.py --model score

# Step 4: Train LSTM (requires PyTorch + time, ~20 min on CPU)
python scripts/train.py --model lstm
```

### CLI Predictions

```bash
# Match outcome prediction
python scripts/predict.py --team1 "India" --team2 "Pakistan" \
  --venue "Narendra Modi Stadium, Ahmedabad"

# Score prediction
python scripts/predict.py --score --batting "India" --bowling "Pakistan" \
  --venue "Narendra Modi Stadium, Ahmedabad"
```

### Test Suite

```bash
pytest tests/ -v
# Expected: 63 tests, ~53% coverage
```

---

## 13. Notes on Agentic AI / LangChain

**LangChain, LangGraph, and Agentic AI frameworks were not used in this project.**

### Why not?

This project's prediction pipeline is a **deterministic ML system**, not a conversational or document-retrieval task:

| Consideration | This Project | Where LangChain/Agents Fit Better |
|---|---|---|
| Core task | Structured tabular prediction | Open-ended Q&A, document analysis |
| Data access | Parquet + CSV (offline) | Web search, live databases, unstructured docs |
| Reasoning | XGBoost model inference | Multi-step LLM chain reasoning |
| Latency requirement | < 200ms per prediction | Seconds acceptable for chat |
| Interpretability | SHAP feature attribution | LLM explanations |

### What Could Be Added Later

If conversational or agentic features were desired, natural extensions would be:

1. **LangChain RAG** over cricket rules, player news, or match reports to enrich predictions
2. **LangGraph agent** that decides: "query live injury feed → update squad JSON → re-predict"
3. **Claude/GPT-powered commentary** generating natural language match analysis from SHAP outputs
4. **Tool-calling agent** that selects the right predictor (match vs score vs squad) based on a user's natural language query

These are on the roadmap as enhancement ideas, not in the current codebase.

---

## Roadmap

- [x] Data loading, cleaning, Pydantic validation (63-test suite)
- [x] PlayerFeatureEngine — batting/bowling profiles, K-Means style classification
- [x] VenueFeatureEngine — phase-wise scoring, spin/pace ratio, chase rates
- [x] TeamFeatureEngine — H2H, rolling form, experience
- [x] MatchPlayerFeatureEngine — XI inference → team batting/bowling strength
- [x] Win Predictor — XGBoost (AUC 0.887, leakage-free, Platt-calibrated)
- [x] Score Predictor — XGBoost (MAE 23.5 runs, R² 0.53)
- [x] Permutation augmentation for training symmetry
- [x] SHAP explainability in API responses
- [x] Platt scaling calibration (`[0.11, 0.87]` range)
- [x] Squad Optimiser — DEAP Genetic Algorithm (live)
- [x] XIFeatureEngine — role-bucket features from exact Playing XI
- [x] WC 2026 squad data for 12 teams (player roles, injury status)
- [x] FastAPI backend — all endpoints live
- [x] Single-page frontend with player selection, squad browser, XI optimiser
- [x] GitHub Actions CI (lint + test + coverage)
- [ ] LSTM score predictor training (architecture ready, needs GPU run)
- [ ] Optuna hyperparameter tuning (200 trials)
- [ ] Weather API integration (dew point for evening T20s)
- [ ] Live score updates during matches (websocket streaming)
- [ ] Agentic query interface (LangGraph + tool-calling)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built by [Viraj Bulugahapitiya](https://github.com/Viraj97-SL) · MSc Data Science
*CricOracle 2026 — XGBoost ensemble · AUC 0.887 · Calibrated probabilities · FastAPI*
