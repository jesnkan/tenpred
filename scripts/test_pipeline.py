"""Unit tests for TennisPredictionPipeline."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import TennisFeatureEngineer
from src.live.predictor import TennisPredictionPipeline
from src.models.match_winner import MatchWinnerModel
from src.models.set_game import SetGameModel
from src.ratings.surface_elo import SurfaceElo


def create_trained_pipeline() -> TennisPredictionPipeline:
    """Helper to train miniature models and return a fully wired pipeline."""
    rng = np.random.RandomState(42)
    n = 150

    # Historical matches to prime player states and train models
    sample_matches = pd.DataFrame({
        "winner_name": [f"Player_{i % 8}" for i in range(n)],
        "loser_name": [f"Player_{(i + 3) % 8}" for i in range(n)],
        "surface": rng.choice(["Hard", "Clay", "Grass"], n),
        "tourney_level": rng.choice(["G", "M", "A", "D"], n),
        "round": rng.choice(["F", "SF", "QF", "R16", "R32"], n),
        "tourney_date": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y%m%d").astype(int),
        "score": rng.choice(["6-4 6-3", "7-6(4) 6-4", "6-2 3-6 6-4", "6-3 6-2"], n),
        "best_of": 3,
        "winner_sets_won": 2,
        "loser_sets_won": rng.choice([0, 1], n),
        "total_sets": rng.choice([2, 3], n),
        "winner_games_won": 12,
        "loser_games_won": 7,
        "total_games": 19,
        "is_completed": True,
        "has_tiebreak": False,
    })

    elo = SurfaceElo()
    fe = TennisFeatureEngineer(elo_model=elo)
    df_features = fe.transform_matches(sample_matches, symmetry_mode="double", progress_bar=False)

    train_df = df_features.iloc[:200].copy()
    val_df = df_features.iloc[200:].copy()

    # Fit models
    mw_model = MatchWinnerModel(model_type="lightgbm", random_state=42)
    mw_model.fit(train_df, target_col="target_p1_won", val_df=val_df)

    sg_model = SetGameModel(model_type="lightgbm", random_state=42)
    sg_model.fit(train_df, val_df=val_df)

    return TennisPredictionPipeline(
        match_winner_model=mw_model,
        set_game_model=sg_model,
        feature_engineer=fe,
        elo_model=elo,
    )


def test_single_match_prediction():
    pipeline = create_trained_pipeline()

    pred = pipeline.predict_match(
        player1="Player_0",
        player2="Player_3",
        surface="Hard",
        tournament_level="M",
        round_name="F",
        best_of=3,
        p1_odds=1.85,
        p2_odds=2.05,
    )

    assert isinstance(pred, dict)
    assert "p1_win_prob" in pred
    assert "p2_win_prob" in pred
    assert 0.0 <= pred["p1_win_prob"] <= 1.0
    assert np.isclose(pred["p1_win_prob"] + pred["p2_win_prob"], 1.0, atol=1e-3)

    # Check derivative markets
    assert "pred_total_games" in pred
    assert "pred_game_diff" in pred
    assert "p1_first_set_prob" in pred
    assert "prob_over_2_5_sets" in pred

    # Check odds & value edge fields
    assert "p1_edge_pct" in pred
    assert "p1_expected_value" in pred
    assert "recommended_bet" in pred
    assert "kelly_stake_pct" in pred

    # Test visual card formatting
    card = pipeline.format_match_card(pred)
    assert "Player_0 vs Player_3" in card
    assert "MATCH WINNER (MONEYLINE)" in card
    assert "TOTAL MATCH GAMES" in card


def test_batch_match_prediction():
    pipeline = create_trained_pipeline()

    fixtures = [
        {"player1": "Player_1", "player2": "Player_2", "surface": "Clay", "tournament_level": "G", "round": "SF"},
        {"player1": "Player_4", "player2": "Player_5", "surface": "Grass", "tournament_level": "A", "round": "QF"},
    ]

    df_preds = pipeline.predict_batch(fixtures)
    assert len(df_preds) == 2
    assert "p1_win_prob" in df_preds.columns
    assert "pred_total_games" in df_preds.columns
    assert "prob_over_22_5" in df_preds.columns
    assert "prob_p1_cover_minus_3_5" in df_preds.columns

    # Test post-prediction market odds addition
    df_with_odds = pipeline.add_market_odds(
        df_preds,
        p1_odds=[1.70, 2.20],
        p2_odds=[2.15, 1.68],
        min_edge_threshold=0.02,
    )
    assert "p1_edge_pct" in df_with_odds.columns
    assert "recommended_bet" in df_with_odds.columns


def test_from_trained_artifacts():
    pipeline = create_trained_pipeline()

    with tempfile.TemporaryDirectory() as tmp_dir:
        mw_path = Path(tmp_dir) / "mw.joblib"
        sg_path = Path(tmp_dir) / "sg.joblib"
        elo_path = Path(tmp_dir) / "elo.joblib"

        pipeline.match_winner_model.save(mw_path)
        pipeline.set_game_model.save(sg_path)
        pipeline.elo.save(elo_path)

        loaded_pipeline = TennisPredictionPipeline.from_trained_artifacts(
            match_winner_path=mw_path,
            set_game_path=sg_path,
            elo_path=elo_path,
        )

        pred = loaded_pipeline.predict_match("Player_0", "Player_1", surface="Hard")
        assert 0.0 <= pred["p1_win_prob"] <= 1.0


if __name__ == "__main__":
    test_single_match_prediction()
    test_batch_match_prediction()
    test_from_trained_artifacts()
    print("All TennisPredictionPipeline tests passed successfully!")

