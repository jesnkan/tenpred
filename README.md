# 🎾 tenpred: Production Quantitative Tennis Prediction Platform (ATP & WTA)

**tenpred** is a modular, production-grade machine learning and stochastic simulation engine designed to forecast outcomes across ATP and WTA professional tennis matches. Built strictly on an **open and free-data approach**, the framework covers surface-adjusted Elo ratings, Court Pace Index (CPI) speed adjustments, Dual GBDT Ensemble (LightGBM + XGBoost + SurfaceElo stacking), Closed-Form Markov Chain Simulator, Interactive Betslip, and Real-Time ESPN Results Tracking.

---

## 🎯 Project Goal

The objective of this project is to build an institutional-grade, quantitative forecasting engine for professional tennis matches (ATP Tour, WTA Tour, Grand Slams, Masters 1000/WTA 1000, and Challenger levels). 

Key goals include:
- **Accurate Probability Estimation:** Producing well-calibrated match-level and derivative probabilities rather than simple binary classifications.
- **Edge Detection Against the Market:** Benchmarking model performance against closing market odds (e.g., Pinnacle, Bet365) using log-loss, Brier scores, and simulated ROI.
- **Hierarchical Modeling:** Decomposing tennis into point-level, game-level, set-level, and match-level stochastic dynamics (Markov chain models and tree-based gradient boosting).

---

## 🆓 Free-Data-Only Philosophy

This system is engineered to deliver state-of-the-art predictive performance **without requiring expensive commercial sports data subscriptions**. It leverages high-quality, publicly accessible data sources:

1. **Jeff Sackmann's Tennis Repositories (`tennis_atp` & `tennis_wta`):**
   - Canonical historical match records, detailed match statistics (aces, double faults, break points, 1st/2nd serve win %), tournament metadata, and official weekly ranking archives dating back to 1968.
   - Point-by-point tracking data via the *Match Charting Project*.
2. **Historical Betting Odds (`tennis-data.co.uk`):**
   - Over two decades of historical closing market odds from major bookmakers (Pinnacle, Bet365, B&W) used exclusively for out-of-sample calibration, validation, and historical market efficiency benchmarking.
3. **Public Scoreboards & Open Endpoints:**
   - Free upcoming fixture calendars and live point-by-point feeds from public tournament trackers for automated daily inference.

---

## 📊 Target Markets Predicted

The system models both primary and derivative tennis wagering and performance markets:

| Market | Description | Modeling Approach |
|---|---|---|
| **Match Winner (Moneyline)** | Straight win/loss probability for Player A vs Player B | Calibrated LightGBM / XGBoost ensembles + Surface-specific Elo blend |
| **Set Handicap** | $+1.5$ / $-1.5$ sets (Best-of-3) or $+1.5, +2.5$ (Grand Slam Best-of-5) | Joint hierarchical model of set-win marginal probabilities |
| **Total Sets** | Over/Under 2.5 sets (Bo3) or Over/Under 3.5 / 4.5 sets (Bo5) | Poisson/Binomial set-distribution modeling conditioned on match competitiveness |
| **Game Handicap (Spread)** | Game differential between players (e.g., $-3.5$, $+4.5$ games) | Regression / Gaussian process modeling game dominance ratios |
| **Total Games (Over / Under)** | Total match games (e.g., Over 21.5 / Under 21.5 games) | Combined serve-hold probability simulation & Markov chain point simulations |
| **First Set Winner** | Player to win Set 1 | Fast-starter momentum metrics and opening set hold rates |

---

## 🏗️ Architecture & Project Structure

The project adheres to strict modular design principles, separating data pipelines, rating models, feature engineering, and inference routines.

```text
tennis/
├── configs/               # Centralized hyperparameters, market lines, and training configs
│   └── config.py          # Surface weights, Elo decay constants, rolling window definitions
│
├── data/
│   ├── raw/               # Immutable raw match files, CSV dumps, and rankings
│   ├── processed/         # Cleaned, standardized, and merged analytical datasets
│   └── live/              # Today's fixtures, current odds, and in-play telemetry
│
├── notebooks/             # Exploratory analysis, feature research, and calibration plots
│
├── scripts/               # CLI entry points for pipeline execution (download, train, evaluate)
│
├── src/
│   ├── __init__.py
│   ├── data/              # Ingestion scripts, schema validation, data alignment (ATP & WTA)
│   ├── ratings/           # Dynamic ratings: Surface-specific Elo, Glicko-2, fatigue decay
│   ├── features/          # Feature engineering: Serve/return dominance, rest, H2H, form
│   ├── models/            # LightGBM, XGBoost, Logistic/Calibrated classifiers, ensemble heads
│   ├── evaluation/        # Backtesting simulator, Brier score, Kelly criterion, ROI analysis
│   └── live/              # Real-time odds parsing, daily fixture prediction, automated runners
│
├── requirements.txt       # Core project dependencies
└── README.md              # Project documentation
```

### Module Responsibilities

- **`src/data`**: Cleans and normalizes discrepancies between ATP and WTA datasets (player name resolution, retirements/walkovers, surface tags), plus free ESPN live match fetching.
- **`src/ratings`**: Computes surface-specific dynamic Elo ratings (Hard, Clay, Grass, Overall) with tournament-tier $K$-factors and inactivity decay.
- **`src/features`**: Constructs 115+ leak-free rolling features (bayesian hold/break rates, fatigue, decayed H2H, physical biometrics).
- **`src/models`**: MatchWinnerModel (calibrated LightGBM classifier), SetGameModel (multi-target regressors/classifiers), and TennisMarkovModel (closed-form point-to-match Markov chain simulator).
- **`src/evaluation`**: Time-split backtester (Brier score, ECE, calibration curves, CLV) and TennisRiskManager (correlated Kelly portfolio staking and Monte Carlo simulation).
- **`src/live`**: TennisPredictionPipeline orchestrator and DailyRunner automation CLI.
- **`app.py`**: Interactive Streamlit Web Graphical Dashboard.

---

## 🚀 Running the Streamlit Web Dashboard

Launch the browser interface with a single command:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### Dashboard Features:
1. **🎯 Head-to-Head Simulator:** Select any two of 1,000+ professional ATP & WTA players, court surface, tournament level, and market odds to generate:
   - Win probability gauge meter
   - **Player to Win At Least One Set (Yes/No)** probabilities and fair odds
   - **Set Handicap Spreads (-1.5 / +1.5 sets)**
   - **Exact Set Score Betting (Correct Score)** interactive bar chart
   - **Total Match Games Over/Under Lines** (19.5, 20.5, 21.5, 22.5, 23.5)
   - **Game Handicap Spreads** (-4.5 to +4.5 games)
   - **First Set Winner & Tiebreak Probabilities**
   - Head-to-head surface Elo telemetry
2. **📊 Daily Value Bet Screener:** Filter today's matches by minimum positive edge (+EV), surface, and recommended Kelly stakes. One-click live ESPN tournament schedule fetcher.
3. **🏆 Surface Elo Leaderboards:** Interactive rankings across Hard, Clay, Grass, and Overall with search and visual bar charts.
4. **🧠 Model Architecture & Features:** Interactive feature importance chart (Information Gain) across all 115 engineered predictive indicators.
5. **🛡️ Portfolio Risk & Monte Carlo:** 1,000-path bankroll growth simulation, 95% Drawdown Value-at-Risk (VaR), and Risk-of-Ruin analysis.

---

## 💻 CLI Commands & Daily Automation

```bash
# View system status, Elo rankings, and feature importance
python scripts/inspect_system.py

# Run daily predictions on upcoming fixtures
python scripts/daily_run.py --mode predict

# Fetch new match records and retrain models
python scripts/daily_run.py --mode retrain

# Run all 11 automated test suites
python -c "import subprocess, sys; [subprocess.run([sys.executable, f'scripts/{s}'], check=True) for s in ['test_surface_elo.py', 'test_loader.py', 'test_features.py', 'test_pipeline_integration.py', 'test_evaluator.py', 'test_match_winner.py', 'test_set_game.py', 'test_pipeline.py', 'test_daily_runner.py', 'test_markov_sim.py', 'test_risk_manager.py']]"
```
- **`src/ratings`**: Implements dynamic, surface-dependent Elo rating engines with tournament-tier $K$-factor weighting (Grand Slam vs. 250s) and inactivity time-decay.
- **`src/features`**: Computes rolling exponential moving averages of:
  - Serve & return efficiency (1st/2nd serve win %, hold %, break point conversion).
  - Surface-specific performance splits (Hard, Clay, Grass, Indoor Hard).
  - Head-to-head match histories and matchup stylistic compatibility (e.g., big server vs. elite returner).
  - Physical fatigue metrics (minutes on court in last 7/14 days, travel, rest differential).
- **`src/models`**: Trains gradient-boosted decision trees and neural baselines with custom objectives optimized for probabilistic cross-entropy and margin spreads.
- **`src/evaluation`**: Rigorous walk-forward validation (never leaking future matches) evaluating calibration curves, Brier skill score against Pinnacle, and market value betting backtests.
- **`src/live`**: Scrapes daily schedules, applies trained models to upcoming matchups, and flags potential market inefficiencies.

---

## ⚡ Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- Recommended: Dedicated virtual environment (`venv` or `conda`)

### 2. Installation

Clone or navigate to the repository directory and set up your virtual environment:

```bash
# Navigate to directory
cd tennis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Workflow Overview

1. **Ingest Data:** Download Sackmann ATP & WTA historical records into `data/raw/`.
2. **Generate Ratings & Features:** Run feature engineering to populate `data/processed/`.
3. **Train & Tune:** Fit LightGBM / XGBoost models configured via `configs/config.py`.
4. **Evaluate:** Assess probability calibration curves and backtest market strategies in `notebooks/` or via `src/evaluation`.
5. **Live Predictions:** Execute daily inference scripts to score upcoming ATP & WTA fixtures.

---

## ⚖️ License & Attribution

- Match history and player stats courtesy of [Jeff Sackmann's tennis_atp / tennis_wta](https://github.com/JeffSackmann).
- Historical odds data courtesy of [Tennis-Data.co.uk](http://www.tennis-data.co.uk).
- Released under the [MIT License](LICENSE).

