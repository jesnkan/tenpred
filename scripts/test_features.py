"""Unit tests for TennisFeatureEngineer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import TennisFeatureEngineer
from src.ratings.surface_elo import SurfaceElo


def test_feature_engineering_pipeline():
    engineer = TennisFeatureEngineer()

    # Create dummy sequential matches
    # Match 1: Alcaraz beats Sinner on Hard on 2023-03-01
    # Match 2: Alcaraz beats Medvedev on Hard on 2023-03-03 (2 days later)
    # Match 3: Sinner beats Medvedev on Hard on 2023-03-05
    # Match 4: Sinner plays Alcaraz on Hard on 2023-03-10
    sample_matches = pd.DataFrame([
        {
            "winner_name": "Carlos Alcaraz",
            "loser_name": "Jannik Sinner",
            "surface": "Hard",
            "tourney_level": "M",
            "tourney_date": 20230301,
            "round": "SF",
            "score": "7-6(4) 6-3",
            "best_of": 3,
            "winner_sets_won": 2,
            "loser_sets_won": 0,
            "total_sets": 2,
            "winner_games_won": 13,
            "loser_games_won": 9,
            "total_games": 22,
            "has_tiebreak": True,
            "is_completed": True,
            "w_svpt": 70, "w_1stIn": 45, "w_1stWon": 35, "w_2ndWon": 15, "w_SvGms": 11, "w_bpSaved": 2, "w_bpFaced": 3,
            "l_svpt": 65, "l_1stIn": 40, "l_1stWon": 28, "l_2ndWon": 12, "l_SvGms": 11, "l_bpSaved": 1, "l_bpFaced": 3,
            "winner_age": 19.8, "loser_age": 21.5, "winner_ht": 183, "loser_ht": 188,
            "winner_hand": "R", "loser_hand": "R", "minutes": 105,
        },
        {
            "winner_name": "Carlos Alcaraz",
            "loser_name": "Daniil Medvedev",
            "surface": "Hard",
            "tourney_level": "M",
            "tourney_date": 20230303,
            "round": "F",
            "score": "6-3 6-2",
            "best_of": 3,
            "winner_sets_won": 2,
            "loser_sets_won": 0,
            "total_sets": 2,
            "winner_games_won": 12,
            "loser_games_won": 5,
            "total_games": 17,
            "has_tiebreak": False,
            "is_completed": True,
            "w_svpt": 55, "w_1stIn": 38, "w_1stWon": 30, "w_2ndWon": 10, "w_SvGms": 9, "w_bpSaved": 1, "w_bpFaced": 1,
            "l_svpt": 50, "l_1stIn": 32, "l_1stWon": 20, "l_2ndWon": 8, "l_SvGms": 8, "l_bpSaved": 1, "l_bpFaced": 5,
            "winner_age": 19.8, "loser_age": 27.0, "winner_ht": 183, "loser_ht": 198,
            "winner_hand": "R", "loser_hand": "R", "minutes": 72,
        },
        {
            "winner_name": "Jannik Sinner",
            "loser_name": "Daniil Medvedev",
            "surface": "Hard",
            "tourney_level": "M",
            "tourney_date": 20230305,
            "round": "R16",
            "score": "6-4 4-6 6-4",
            "best_of": 3,
            "winner_sets_won": 2,
            "loser_sets_won": 1,
            "total_sets": 3,
            "winner_games_won": 16,
            "loser_games_won": 14,
            "total_games": 30,
            "has_tiebreak": False,
            "is_completed": True,
            "w_svpt": 85, "w_1stIn": 55, "w_1stWon": 42, "w_2ndWon": 18, "w_SvGms": 15, "w_bpSaved": 4, "w_bpFaced": 6,
            "l_svpt": 88, "l_1stIn": 58, "l_1stWon": 44, "l_2ndWon": 15, "l_SvGms": 15, "l_bpSaved": 3, "l_bpFaced": 6,
            "winner_age": 21.5, "loser_age": 27.0, "winner_ht": 188, "loser_ht": 198,
            "winner_hand": "R", "loser_hand": "R", "minutes": 135,
        },
        {
            "winner_name": "Jannik Sinner",
            "loser_name": "Carlos Alcaraz",
            "surface": "Hard",
            "tourney_level": "M",
            "tourney_date": 20230310,
            "round": "SF",
            "score": "6-7(4) 6-4 6-2",
            "best_of": 3,
            "winner_sets_won": 2,
            "loser_sets_won": 1,
            "total_sets": 3,
            "winner_games_won": 18,
            "loser_games_won": 13,
            "total_games": 31,
            "has_tiebreak": True,
            "is_completed": True,
            "w_svpt": 90, "w_1stIn": 60, "w_1stWon": 46, "w_2ndWon": 18, "w_SvGms": 15, "w_bpSaved": 3, "w_bpFaced": 5,
            "l_svpt": 92, "l_1stIn": 58, "l_1stWon": 40, "l_2ndWon": 16, "l_SvGms": 16, "l_bpSaved": 4, "l_bpFaced": 8,
            "winner_age": 21.5, "loser_age": 19.8, "winner_ht": 188, "loser_ht": 183,
            "winner_hand": "R", "loser_hand": "R", "minutes": 140,
        }
    ])

    # Transform with double symmetry (2 rows per match: 8 rows total)
    df_features = engineer.transform_matches(sample_matches, symmetry_mode="double", progress_bar=False)
    assert len(df_features) == 8

    # 1. Check Match 1 (First two rows): Should have unrated baseline Elo 1500
    row_0 = df_features.iloc[0]
    assert row_0["p1_elo_overall"] == 1500.0
    assert row_0["p2_elo_overall"] == 1500.0
    assert row_0["target_p1_won"] == 1
    assert row_0["target_total_sets"] == 2
    assert row_0["target_total_games"] == 22
    assert row_0["target_set1_p1_won"] == 1

    row_1 = df_features.iloc[1]
    assert row_1["target_p1_won"] == 0
    assert row_1["target_game_diff"] == -row_0["target_game_diff"]

    # 2. Check Match 2 (Rows 2 & 3): Alcaraz now has higher Elo than Medvedev
    row_2 = df_features.iloc[2]
    assert row_2["player1"] == "Carlos Alcaraz"
    assert row_2["p1_elo_overall"] > 1500.0
    assert row_2["p2_elo_overall"] == 1500.0
    assert row_2["p1_streak"] == 1.0  # Won match 1
    assert row_2["p1_days_rest"] == 2.0  # March 3 - March 1 = 2 days
    assert row_2["p1_is_back_to_back"] == 0.0

    # 3. Check Match 4 (Rows 6 & 7): Sinner vs Alcaraz rematch!
    # H2H should show 1 previous match (Alcaraz won)
    row_6 = df_features.iloc[6]
    assert row_6["player1"] == "Jannik Sinner"
    assert row_6["player2"] == "Carlos Alcaraz"
    assert row_6["h2h_matches"] == 1.0
    assert row_6["h2h_p1_wins"] == 0.0  # Sinner lost the previous meeting
    assert row_6["h2h_p1_win_pct"] == 0.0

    row_7 = df_features.iloc[7]
    assert row_7["player1"] == "Carlos Alcaraz"
    assert row_7["player2"] == "Jannik Sinner"
    assert row_7["h2h_matches"] == 1.0
    assert row_7["h2h_p1_wins"] == 1.0
    assert row_7["h2h_p1_win_pct"] == 1.0


def test_single_match_feature_extraction():
    engineer = TennisFeatureEngineer()
    # Test scoring an arbitrary future fixture
    feats = engineer.extract_features_for_match(
        p1="Novak Djokovic",
        p2="Rafael Nadal",
        surface="Clay",
        tournament_level="Grand Slam",
        round_name="F",
        match_date="2024-06-09",
        best_of=5,
        p1_age=37.0,
        p2_age=38.0,
        p1_ht=188.0,
        p2_ht=185.0,
        p1_hand="R",
        p2_hand="L",
    )

    assert feats["player1"] == "Novak Djokovic"
    assert feats["player2"] == "Rafael Nadal"
    assert feats["surface"] == "Clay"
    assert feats["best_of"] == 5
    assert feats["is_lefty_vs_righty"] == 1.0
    assert "diff_elo_blended" in feats
    assert "expected_combined_hold" in feats
    assert "est_match_total_games" in feats
    assert feats["est_match_total_games"] > 30.0  # Best-of-5 clay match has high expected games


if __name__ == "__main__":
    test_feature_engineering_pipeline()
    test_single_match_feature_extraction()
    print("All TennisFeatureEngineer tests passed successfully!")

