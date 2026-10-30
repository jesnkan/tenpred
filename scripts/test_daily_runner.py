"""Unit tests for DailyRunner."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import TennisFeatureEngineer
from src.live.daily_runner import DailyRunner
from src.models.match_winner import MatchWinnerModel
from src.models.set_game import SetGameModel
from src.ratings.surface_elo import SurfaceElo


def test_daily_runner_workflow():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_dir = Path(tmp_dir)
        models_dir = base_dir / "models"
        live_dir = base_dir / "data" / "live"
        models_dir.mkdir(parents=True)
        live_dir.mkdir(parents=True)

        # 1. Setup miniature models and artifacts
        elo = SurfaceElo()
        fe = TennisFeatureEngineer(elo_model=elo)

        # Create miniature training data
        sample_df = pd.DataFrame([
            {
                "winner_name": "Novak Djokovic", "loser_name": "Daniil Medvedev", "surface": "Hard",
                "tourney_level": "G", "round": "F", "tourney_date": 20230129, "score": "6-3 7-6 7-6",
                "best_of": 5, "winner_sets_won": 3, "loser_sets_won": 0, "total_sets": 3,
                "winner_games_won": 20, "loser_games_won": 15, "total_games": 35,
                "is_completed": True, "has_tiebreak": True
            },
            {
                "winner_name": "Carlos Alcaraz", "loser_name": "Novak Djokovic", "surface": "Grass",
                "tourney_level": "G", "round": "F", "tourney_date": 20230716, "score": "1-6 7-6 6-1 3-6 6-4",
                "best_of": 5, "winner_sets_won": 3, "loser_sets_won": 2, "total_sets": 5,
                "winner_games_won": 23, "loser_games_won": 23, "total_games": 46,
                "is_completed": True, "has_tiebreak": True
            }
        ])

        feat_df = fe.transform_matches(sample_df, symmetry_mode="double", progress_bar=False)

        mw_model = MatchWinnerModel(model_type="lightgbm", random_state=42)
        mw_model.fit(feat_df, target_col="target_p1_won")

        sg_model = SetGameModel(model_type="lightgbm", random_state=42)
        sg_model.fit(feat_df)

        # Save to temporary models dir
        elo.save(models_dir / "surface_elo.joblib")
        mw_model.save(models_dir / "match_winner_lgbm.joblib")
        sg_model.save(models_dir / "set_game_model.joblib")

        # 2. Instantiate DailyRunner
        runner = DailyRunner(base_dir=base_dir, models_dir=models_dir, live_dir=live_dir)

        # 3. Test fixtures loader creating starter template
        fixtures_df = runner.load_fixtures(fixtures_input=None)
        assert not fixtures_df.empty
        assert (live_dir / "upcoming_fixtures.csv").exists()

        # 4. Run daily predictions
        predictions_df = runner.run_daily(
            fixtures_file=live_dir / "upcoming_fixtures.csv",
            update_data=False,
            retrain_models=False,
            min_edge_threshold=0.02,
        )

        assert not predictions_df.empty
        assert "p1_win_prob" in predictions_df.columns
        assert "p1_fair_odds" in predictions_df.columns
        assert "pred_total_games" in predictions_df.columns
        assert "recommended_bet" in predictions_df.columns
        assert "edge_pct" in predictions_df.columns

        # Verify output files generated
        csv_files = list(live_dir.glob("predictions_*.csv"))
        json_files = list(live_dir.glob("predictions_*.json"))
        assert len(csv_files) >= 1
        assert len(json_files) >= 1


if __name__ == "__main__":
    test_daily_runner_workflow()
    print("All DailyRunner tests passed successfully!")

