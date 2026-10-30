"""Unit tests for SetGameModel."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.set_game import SetGameModel
from src.evaluation.evaluator import TennisEvaluator


def create_synthetic_set_game_data(n_samples: int = 220, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic dataset with set & game target variables."""
    rng = np.random.RandomState(random_state)

    diff_elo = rng.normal(0, 140, n_samples)
    elo_prob = 1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))

    comb_hold = rng.uniform(0.65, 0.90, n_samples)
    diff_hold = rng.normal(0, 0.08, n_samples)
    tb_freq = rng.uniform(0.10, 0.35, n_samples)

    # Realistic continuous targets
    # High combined hold -> more games per set
    tot_games = np.round(18.0 + 10.0 * comb_hold + 6.0 * tb_freq + rng.normal(0, 2.5, n_samples)).astype(int)
    tot_games = np.clip(tot_games, 12, 45)

    # Game differential correlated with Elo diff
    game_diff = np.round(diff_elo / 35.0 + 12.0 * diff_hold + rng.normal(0, 3.0, n_samples)).astype(int)

    # Sets: 2 or 3
    is_3_sets = (rng.uniform(0, 1, n_samples) < (0.35 + 0.25 * (1.0 - abs(diff_elo) / 300.0))).astype(int)
    tot_sets = 2 + is_3_sets

    # First set winner
    p_s1 = 1.0 / (1.0 + np.exp(-(0.007 * diff_elo + 3.0 * diff_hold)))
    s1_won = (rng.uniform(0, 1, n_samples) < p_s1).astype(int)

    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")

    return pd.DataFrame({
        "match_date": dates,
        "player1": [f"Player_{i % 15}" for i in range(n_samples)],
        "player2": [f"Player_{(i + 5) % 15}" for i in range(n_samples)],
        "surface": rng.choice(["Hard", "Clay", "Grass"], n_samples),
        "tournament_level": rng.choice(["G", "M", "A", "D"], n_samples),
        "round": rng.choice(["F", "SF", "QF", "R16", "R32"], n_samples),
        "best_of": rng.choice([3, 5], n_samples),
        "diff_elo_blended": diff_elo,
        "elo_win_prob_p1": elo_prob,
        "expected_combined_hold": comb_hold,
        "expected_game_dominance": diff_hold,
        "avg_tiebreak_freq": tb_freq,
        "p1_avg_games_per_set": 9.5 + comb_hold,
        "p2_avg_games_per_set": 9.5 + comb_hold,
        "diff_hold_pct": diff_hold,
        "diff_break_pct": -diff_hold,
        "p1_days_rest": rng.uniform(1, 7, n_samples),
        "p2_days_rest": rng.uniform(1, 7, n_samples),
        "is_completed": True,
        # Targets
        "target_total_games": tot_games,
        "target_game_diff": game_diff,
        "target_total_sets": tot_sets,
        "target_set1_p1_won": s1_won,
    })


def test_set_game_model_fit_and_predict():
    df = create_synthetic_set_game_data(n_samples=200)

    train_df = df.iloc[:140].copy()
    val_df = df.iloc[140:].copy()

    model = SetGameModel(model_type="lightgbm", random_state=42)
    model.fit(train_df, val_df=val_df)

    assert model.is_fitted is True

    # Test predict()
    preds = model.predict(val_df)
    assert len(preds) == len(val_df)
    assert "pred_total_games" in preds.columns
    assert "pred_game_diff" in preds.columns
    assert "pred_total_sets" in preds.columns
    assert "prob_over_2_5_sets" in preds.columns
    assert "prob_first_set_p1" in preds.columns

    # Verify probability ranges
    assert np.all((preds["prob_over_2_5_sets"] >= 0.0) & (preds["prob_over_2_5_sets"] <= 1.0))
    assert np.all((preds["prob_first_set_p1"] >= 0.0) & (preds["prob_first_set_p1"] <= 1.0))

    # Test continuous line helper: Total Games Over Probability
    over_probs = model.predict_total_games_over_prob(val_df, line=22.5)
    assert len(over_probs) == len(val_df)
    assert np.all((over_probs >= 0.0) & (over_probs <= 1.0))

    # Higher line (e.g. 26.5) should have strictly lower over probability than lower line (e.g. 19.5)
    prob_high = model.predict_total_games_over_prob(val_df, line=26.5)
    prob_low = model.predict_total_games_over_prob(val_df, line=19.5)
    assert np.all(prob_low >= prob_high)

    # Test handicap cover probability
    cover_probs = model.predict_game_handicap_cover_prob(val_df, handicap=-3.5)
    assert len(cover_probs) == len(val_df)
    assert np.all((cover_probs >= 0.0) & (cover_probs <= 1.0))

    # Test feature importance for different targets
    imp_tg = model.feature_importance(target="total_games", top_n=5)
    assert not imp_tg.empty
    imp_s1 = model.feature_importance(target="first_set", top_n=5)
    assert not imp_s1.empty

    # Evaluate with TennisEvaluator
    evaluator = TennisEvaluator()
    reg_metrics = evaluator.evaluate_regression(val_df["target_total_games"], preds["pred_total_games"])
    assert reg_metrics.mae < 4.0

    s1_metrics = evaluator.evaluate_binary(val_df["target_set1_p1_won"], preds["prob_first_set_p1"])
    assert s1_metrics.brier_score < 0.25


def test_set_game_save_load():
    df = create_synthetic_set_game_data(n_samples=100)
    train_df = df.iloc[:70].copy()
    test_df = df.iloc[70:].copy()

    model = SetGameModel(model_type="lightgbm")
    model.fit(train_df)

    orig_preds = model.predict(test_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "set_game_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = SetGameModel.load(save_path)
        assert loaded_model.is_fitted is True

        loaded_preds = loaded_model.predict(test_df)
        assert np.allclose(orig_preds["pred_total_games"], loaded_preds["pred_total_games"], atol=1e-4)
        assert np.allclose(orig_preds["prob_first_set_p1"], loaded_preds["prob_first_set_p1"], atol=1e-4)


if __name__ == "__main__":
    test_set_game_model_fit_and_predict()
    test_set_game_save_load()
    print("All SetGameModel tests passed successfully!")

