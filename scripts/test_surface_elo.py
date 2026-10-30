"""Unit tests for SurfaceElo rating system."""

import sys
from pathlib import Path
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.ratings.surface_elo import SurfaceElo


def test_initial_ratings_and_surfaces():
    elo = SurfaceElo(base_rating=1500.0)
    # Default before any match
    assert elo.get_rating("Carlos Alcaraz") == 1500.0
    assert elo.get_rating("Carlos Alcaraz", surface="Hard") == 1500.0
    assert elo.get_rating("Carlos Alcaraz", surface="Clay") == 1500.0
    assert elo.get_rating("Carlos Alcaraz", surface="Grass") == 1500.0

    # Win probability between equal unrated players must be 0.5
    prob = elo.get_win_probability("Carlos Alcaraz", "Jannik Sinner", surface="Hard")
    assert round(prob, 4) == 0.5


def test_surface_specific_updates():
    elo = SurfaceElo(base_rating=1500.0, margin_scaling_enabled=False)
    # Nadal beats Federer on Clay in French Open Final
    elo.update_match(
        winner="Rafael Nadal",
        loser="Roger Federer",
        surface="Clay",
        tournament_level="Grand Slam",
        round="F",
        score="6-1 6-3 6-0",
    )

    # Clay ratings must change
    nadal_clay = elo.get_rating("Rafael Nadal", surface="Clay", blended=False)
    fed_clay = elo.get_rating("Roger Federer", surface="Clay", blended=False)
    assert nadal_clay > 1500.0
    assert fed_clay < 1500.0

    # Overall ratings must change
    nadal_overall = elo.get_rating("Rafael Nadal")
    fed_overall = elo.get_rating("Roger Federer")
    assert nadal_overall > 1500.0
    assert fed_overall < 1500.0

    # Hard & Grass surface-only ratings should still be unplayed (1500.0)
    assert elo.get_rating("Rafael Nadal", surface="Grass", blended=False) == 1500.0
    assert elo.get_rating("Roger Federer", surface="Hard", blended=False) == 1500.0

    # But blended ratings reflect their overall prowess even on unplayed surfaces
    assert elo.get_rating("Rafael Nadal", surface="Grass", blended=True) > 1500.0


def test_tournament_level_k_factors():
    # Grand Slam should yield higher rating delta than a 250 tournament
    elo_gs = SurfaceElo(base_rating=1500.0, margin_scaling_enabled=False)
    res_gs = elo_gs.update_match(
        winner="Player A",
        loser="Player B",
        surface="Hard",
        tournament_level="G",
        round="R32",
    )

    elo_250 = SurfaceElo(base_rating=1500.0, margin_scaling_enabled=False)
    res_250 = elo_250.update_match(
        winner="Player C",
        loser="Player D",
        surface="Hard",
        tournament_level="250",
        round="R32",
    )

    assert res_gs["effective_k"] > res_250["effective_k"]
    assert res_gs["delta_overall"] > res_250["delta_overall"]


def test_round_multiplier():
    # Finals must produce higher effective K than R32
    elo = SurfaceElo(base_rating=1500.0, margin_scaling_enabled=False)
    res_r32 = elo.update_match("P1", "P2", "Hard", "1000", "R32")
    res_final = elo.update_match("P3", "P4", "Hard", "1000", "F")

    assert res_final["effective_k"] > res_r32["effective_k"]


def test_margin_of_victory_scaling():
    elo = SurfaceElo(base_rating=1500.0, margin_scaling_enabled=True)
    # Dominant blowout: 6-0 6-1
    res_blowout = elo.update_match("W1", "L1", "Hard", "500", "R16", score="6-0 6-1")

    # Tight tiebreak match: 7-6 6-7 7-6
    res_tight = elo.update_match("W2", "L2", "Hard", "500", "R16", score="7-6 6-7 7-6")

    assert res_blowout["effective_k"] > res_tight["effective_k"]


def test_inactivity_decay():
    elo = SurfaceElo(base_rating=1500.0, decay_threshold_days=60, decay_half_life_days=180)

    # Establish high rating on 2023-01-01
    elo.update_match("Active Player", "Opponent", "Hard", "G", "F", date="2023-01-01")
    peak_rating = elo.get_rating("Active Player")
    assert peak_rating > 1500.0

    # Check rating 30 days later (under 60 day threshold) -> no decay
    r_30d = elo.get_rating("Active Player", date="2023-01-31")
    assert r_30d == peak_rating

    # Check rating 240 days later (60 threshold + 180 half-life = 1 half-life of decay)
    r_decayed = elo.get_rating("Active Player", date="2023-08-29")
    expected_dev = (peak_rating - 1500.0) * 0.5
    assert np.isclose(r_decayed - 1500.0, expected_dev, atol=1.0)
    assert r_decayed < peak_rating


def test_process_matches_chronologically():
    elo = SurfaceElo()
    matches = [
        # Match 2 chronologically
        {"winner_name": "Alcaraz", "loser_name": "Medvedev", "surface": "Hard", "tourney_level": "M", "round": "SF", "tourney_date": 20230318, "score": "6-3 6-3"},
        # Match 1 chronologically
        {"winner_name": "Alcaraz", "loser_name": "Auger-Aliassime", "surface": "Hard", "tourney_level": "M", "round": "QF", "tourney_date": 20230316, "score": "6-4 6-4"},
        # Match 3 chronologically
        {"winner_name": "Alcaraz", "loser_name": "Medvedev", "surface": "Hard", "tourney_level": "M", "round": "F", "tourney_date": 20230319, "score": "6-3 6-2"},
    ]
    df = pd.DataFrame(matches)
    enriched_df = elo.process_matches_chronologically(df)

    assert len(enriched_df) == 3
    # First match in enriched df must be QF on March 16
    assert enriched_df.iloc[0]["round"] == "QF"
    assert enriched_df.iloc[0]["winner_pre_elo_overall"] == 1500.0

    # Third match (Final on March 19): Alcaraz should have high pre-match rating
    assert enriched_df.iloc[2]["round"] == "F"
    assert enriched_df.iloc[2]["winner_pre_elo_overall"] > 1500.0
    assert "elo_prob_winner" in enriched_df.columns
    assert "elo_diff_blended" in enriched_df.columns


def test_save_and_load():
    elo = SurfaceElo(base_rating=1500.0)
    elo.update_match("Iga Swiatek", "Aryna Sabalenka", "Clay", "G", "F", score="6-2 6-3")

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_elo.joblib"
        elo.save(save_path)
        assert save_path.exists()

        loaded_elo = SurfaceElo.load(save_path)
        assert loaded_elo.get_rating("Iga Swiatek", surface="Clay") == elo.get_rating("Iga Swiatek", surface="Clay")
        assert loaded_elo.get_win_probability("Iga Swiatek", "Aryna Sabalenka", surface="Clay") == elo.get_win_probability("Iga Swiatek", "Aryna Sabalenka", surface="Clay")


if __name__ == "__main__":
    test_initial_ratings_and_surfaces()
    test_surface_specific_updates()
    test_tournament_level_k_factors()
    test_round_multiplier()
    test_margin_of_victory_scaling()
    test_inactivity_decay()
    test_process_matches_chronologically()
    test_save_and_load()
    print("All SurfaceElo tests passed successfully!")
