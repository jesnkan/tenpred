"""Central configuration for tenpred: ATP & WTA Tennis Prediction System."""

from pathlib import Path

SYSTEM_NAME = "tenpred"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LIVE_DATA_DIR = DATA_DIR / "live"

# Date ranges for backtesting and training
TRAIN_START_YEAR = 2010
TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024

# Elo rating parameters
ELO_CONFIG = {
    "base_rating": 1500.0,
    # Tournament level K-factors
    "k_factors": {
        "grand_slam": 32.0,       # G / Grand Slam
        "masters_1000": 28.0,     # M / 1000 / Premier Mandatory
        "tour_500": 24.0,         # A / 500 / Premier 500
        "tour_250": 20.0,         # D / 250 / International
        "challenger": 16.0,       # C / Challenger
        "default": 20.0,
    },
    # Round multipliers (higher stakes in deep tournament rounds)
    "round_multipliers": {
        "F": 1.20,                # Finals
        "SF": 1.10,               # Semifinals
        "QF": 1.05,               # Quarterfinals
        "R16": 1.00,
        "R32": 1.00,
        "R64": 1.00,
        "R128": 1.00,
        "RR": 1.00,               # Round Robin (Year-End Finals)
        "default": 1.00,
    },
    # Blending & dynamics
    "surface_weight": 0.5,        # 50% surface Elo + 50% overall Elo for match prediction
    # Inactivity decay settings
    "inactivity_decay_enabled": True,
    "decay_threshold_days": 60,   # Days of inactivity before decay begins
    "decay_half_life_days": 365,  # Time for inactive rating deviation from base to halve
    # Margin-of-victory (MoV) scaling settings
    "margin_scaling_enabled": True,
    "margin_weight": 0.35,        # Scale factor for game dominance
    "margin_multiplier_min": 0.75,
    "margin_multiplier_max": 1.35,
}

# Rolling statistical windows (number of matches)
ROLLING_WINDOWS = [5, 10, 20, 50]

# Supported target markets
TARGET_MARKETS = [
    "match_winner",      # Moneyline (Win/Loss)
    "set_handicap",     # +/- 1.5 sets (Best-of-3/5)
    "total_sets",       # Over / Under 2.5 (Bo3) or 3.5 / 4.5 (Bo5)
    "game_handicap",    # Spread in total games won
    "total_games",      # Over / Under match games line
    "first_set_winner", # Winner of Set 1
]

# Court Pace Index (CPI) and Indoor / Environmental Configurations
COURT_PACE_CONFIG = {
    # Keywords identifying indoor tournaments from tournament names
    "indoor_keywords": [
        "indoor", "vienna", "erste bank", "basel", "swiss indoors",
        "paris", "bercy", "atp finals", "turin", "wta finals",
        "rotterdam", "abn amro", "stockholm", "metz", "moselle",
        "antwerp", "marseille", "montpellier", "dallas", "cluj",
        "linz", "sofia", "almaty", "astana", "lyon indoor"
    ],
    # Specific fast outdoor tournament keywords
    "fast_hard_keywords": [
        "cincinnati", "shanghai", "dubai", "brisbane", "doha", "tokyo"
    ],
    # Specific slow outdoor hard tournament keywords
    "slow_hard_keywords": [
        "indian wells", "acapulco", "los cabos"
    ],
    # Baseline surface speed attributes: (CPI, hold_boost, tiebreak_boost, games_shift)
    "surface_profiles": {
        "Grass": {"cpi": 43.0, "category": "Fast", "hold_boost": 0.035, "tiebreak_boost": 0.050, "games_shift": 0.8},
        "Carpet": {"cpi": 42.0, "category": "Fast", "hold_boost": 0.035, "tiebreak_boost": 0.050, "games_shift": 0.8},
        "Indoor Hard": {"cpi": 40.0, "category": "Fast", "hold_boost": 0.030, "tiebreak_boost": 0.040, "games_shift": 0.7},
        "Outdoor Hard (Fast)": {"cpi": 38.0, "category": "Medium-Fast", "hold_boost": 0.015, "tiebreak_boost": 0.020, "games_shift": 0.4},
        "Hard": {"cpi": 34.0, "category": "Medium", "hold_boost": 0.000, "tiebreak_boost": 0.000, "games_shift": 0.0},
        "Outdoor Hard (Slow)": {"cpi": 28.0, "category": "Medium-Slow", "hold_boost": -0.015, "tiebreak_boost": -0.020, "games_shift": -0.4},
        "Clay": {"cpi": 22.0, "category": "Slow", "hold_boost": -0.045, "tiebreak_boost": -0.060, "games_shift": -0.9},
    }
}


