"""Unit tests for TennisDataLoader."""

import sys
from pathlib import Path
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import TennisDataLoader


def test_score_parser():
    loader = TennisDataLoader()

    # 1. Normal three-set match with tiebreak
    res1 = loader.parse_score_details("6-4 3-6 7-6(5)")
    assert res1["winner_sets_won"] == 2
    assert res1["loser_sets_won"] == 1
    assert res1["total_sets"] == 3
    assert res1["winner_games_won"] == 16
    assert res1["loser_games_won"] == 16
    assert res1["has_tiebreak"] == True
    assert res1["is_completed"] == True
    assert res1["is_retirement"] == False
    assert res1["is_walkover"] == False

    # 2. Retirement
    res2 = loader.parse_score_details("6-3 3-1 RET")
    assert res2["is_retirement"] == True
    assert res2["is_completed"] == False
    assert res2["winner_games_won"] == 9
    assert res2["loser_games_won"] == 4

    # 3. Walkover
    res3 = loader.parse_score_details("W/O")
    assert res3["is_walkover"] == True
    assert res3["is_completed"] == False
    assert res3["total_games"] == 0


def test_clean_matches():
    loader = TennisDataLoader()

    sample_data = pd.DataFrame([
        {
            "tourney_id": "2023-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20230116,
            "match_num": 127,
            "winner_name": "Novak Djokovic",
            "loser_name": "Stefanos Tsitsipas",
            "score": "6-3 7-6(4) 7-6(5)",
            "round": "F",
            "w_ace": 9,
            "w_df": 1,
            "w_svpt": 98,
            "w_1stIn": 66,
            "w_1stWon": 52,
            "w_2ndWon": 20,
            "w_SvGms": 17,
            "w_bpSaved": 2,
            "w_bpFaced": 3,
            "l_ace": 15,
            "l_df": 3,
            "l_svpt": 105,
            "l_1stIn": 69,
            "l_1stWon": 50,
            "l_2ndWon": 18,
            "l_SvGms": 17,
            "l_bpSaved": 2,
            "l_bpFaced": 4,
            "tour": "ATP",
        },
        {
            "tourney_id": "2023-800",
            "tourney_name": "Roland Garros",
            "surface": "Clay",
            "tourney_level": "Grand Slam",
            "tourney_date": "20230528",
            "match_num": 126,
            "winner_name": "Iga Swiatek",
            "loser_name": "Karolina Muchova",
            "score": "6-2 5-7 6-4",
            "round": "Final",
            "tour": "WTA",
        }
    ])

    cleaned = loader.clean_matches(sample_data)

    assert len(cleaned) == 2
    # Verify Australian Open (ATP Grand Slam)
    row_ao = cleaned.iloc[0]
    assert row_ao["year"] == 2023
    assert row_ao["month"] == 1
    assert row_ao["surface"] == "Hard"
    assert row_ao["tourney_level"] == "G"
    assert bool(row_ao["is_grand_slam"]) is True
    assert row_ao["best_of"] == 5
    assert row_ao["round"] == "F"
    assert row_ao["winner_sets_won"] == 3
    assert row_ao["loser_sets_won"] == 0
    assert row_ao["total_sets"] == 3
    assert bool(row_ao["has_tiebreak"]) is True
    assert bool(row_ao["is_completed"]) is True
    # Verify serve rates
    assert np.isclose(row_ao["w_1st_in_pct"], 66 / 98)
    assert np.isclose(row_ao["w_1st_won_pct"], 52 / 66)
    assert np.isclose(row_ao["w_hold_pct"], 16 / 17)

    # Verify Roland Garros (WTA Grand Slam)
    row_rg = cleaned.iloc[1]
    assert row_rg["surface"] == "Clay"
    assert row_rg["tourney_level"] == "G"
    assert bool(row_rg["is_grand_slam"]) is True
    assert row_rg["best_of"] == 3  # WTA Grand Slams are Best of 3!
    assert row_rg["winner_sets_won"] == 2
    assert row_rg["loser_sets_won"] == 1
    assert row_rg["total_sets"] == 3
    assert row_rg["winner_games_won"] == 17
    assert row_rg["loser_games_won"] == 13


def test_file_loading_and_saving():
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        processed_dir = Path(tmp_dir) / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        loader = TennisDataLoader(raw_dir=raw_dir, processed_dir=processed_dir)

        # Create dummy atp_matches_2023.csv
        sample_df = pd.DataFrame([{
            "tourney_id": "2023-001",
            "tourney_name": "Adelaide",
            "surface": "Hard",
            "tourney_level": "A",
            "tourney_date": 20230102,
            "match_num": 1,
            "winner_name": "Player A",
            "loser_name": "Player B",
            "score": "6-4 6-4",
            "round": "R32",
        }])
        file_path = raw_dir / "atp_matches_2023.csv"
        sample_df.to_csv(file_path, index=False)

        # Test load_file
        loaded = loader.load_file(file_path, clean=True)
        assert len(loaded) == 1
        assert loaded.iloc[0]["tour"] == "ATP"
        assert loaded.iloc[0]["is_completed"] == True

        # Test save_processed
        saved_path = loader.save_processed(loaded, "test_output.csv")
        assert saved_path.exists()
        reloaded = pd.read_csv(saved_path)
        assert len(reloaded) == 1


if __name__ == "__main__":
    test_score_parser()
    test_clean_matches()
    test_file_loading_and_saving()
    print("All TennisDataLoader tests passed successfully!")
