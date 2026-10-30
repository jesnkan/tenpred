"""Integration test verifying TennisDataLoader seamlessly feeds SurfaceElo."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.loader import TennisDataLoader
from src.ratings.surface_elo import SurfaceElo


def test_loader_to_elo_pipeline():
    raw_matches = pd.DataFrame([
        {
            "tourney_id": "2023-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20230116,
            "match_num": 1,
            "winner_name": "Novak Djokovic",
            "loser_name": "Roberto Carballes Baena",
            "score": "6-3 6-4 6-0",
            "round": "R128",
            "tour": "ATP"
        },
        {
            "tourney_id": "2023-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20230118,
            "match_num": 33,
            "winner_name": "Novak Djokovic",
            "loser_name": "Enzo Couacaud",
            "score": "6-1 6-7(5) 6-2 6-0",
            "round": "R64",
            "tour": "ATP"
        },
        {
            "tourney_id": "2023-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "tourney_level": "G",
            "tourney_date": 20230129,
            "match_num": 127,
            "winner_name": "Novak Djokovic",
            "loser_name": "Stefanos Tsitsipas",
            "score": "6-3 7-6(4) 7-6(5)",
            "round": "F",
            "tour": "ATP"
        }
    ])

    loader = TennisDataLoader()
    cleaned_df = loader.clean_matches(raw_matches)

    elo = SurfaceElo()
    enriched_df = elo.process_matches_chronologically(
        cleaned_df,
        winner_col="winner_name",
        loser_col="loser_name",
        surface_col="surface",
        tourney_level_col="tourney_level",
        round_col="round",
        score_col="score",
        date_col="match_date",
        match_num_col="match_num",
    )

    assert len(enriched_df) == 3
    # Check that Novak's Elo monotonically increased over these 3 wins
    assert enriched_df.iloc[0]["winner_pre_elo_overall"] == 1500.0
    assert enriched_df.iloc[1]["winner_pre_elo_overall"] > 1500.0
    assert enriched_df.iloc[2]["winner_pre_elo_overall"] > enriched_df.iloc[1]["winner_pre_elo_overall"]
    assert "elo_prob_winner" in enriched_df.columns
    print("Integration test passed successfully!")


if __name__ == "__main__":
    test_loader_to_elo_pipeline()

