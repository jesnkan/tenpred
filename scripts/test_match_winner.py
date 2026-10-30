"""Unit tests for MatchWinnerModel."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.models.match_winner import MatchWinnerModel
from src.evaluation.evaluator import TennisEvaluator


def create_synthetic_tennis_features(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic feature dataset for testing."""
    rng = np.random.RandomState(random_state)

    diff_elo = rng.normal(0, 150, n_samples)
    elo_prob = 1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))

    # True probability is a function of Elo difference and hold rate difference
    diff_hold = rng.normal(0, 0.08, n_samples)
    diff_form = rng.normal(0, 0.20, n_samples)

    true_logit = 0.008 * diff_elo + 4.0 * diff_hold + 1.5 * diff_form
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    targets = (rng.uniform(0, 1, n_samples) < true_prob).astype(int)

    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")

    df = pd.DataFrame({
        "match_date": dates,
        "player1": [f"Player_{i % 20}" for i in range(n_samples)],
        "player2": [f"Player_{(i + 7) % 20}" for i in range(n_samples)],
        "surface": rng.choice(["Hard", "Clay", "Grass"], n_samples),
        "tournament_level": rng.choice(["G", "M", "A", "D"], n_samples),
        "round": rng.choice(["F", "SF", "QF", "R16", "R32"], n_samples),
        "best_of": rng.choice([3, 5], n_samples),
        "diff_elo_blended": diff_elo,
        "p1_elo_blended": 1500.0 + diff_elo / 2,
        "p2_elo_blended": 1500.0 - diff_elo / 2,
        "elo_win_prob_p1": elo_prob,
        "diff_hold_pct": diff_hold,
        "p1_hold_pct": 0.78 + diff_hold / 2,
        "p2_hold_pct": 0.78 - diff_hold / 2,
        "diff_win_rate_10": diff_form,
        "p1_win_rate_10": 0.50 + diff_form / 2,
        "p2_win_rate_10": 0.50 - diff_form / 2,
        "p1_days_rest": rng.uniform(1, 10, n_samples),
        "p2_days_rest": rng.uniform(1, 10, n_samples),
        "diff_days_rest": rng.normal(0, 2, n_samples),
        "expected_combined_hold": rng.uniform(0.65, 0.90, n_samples),
        "target_p1_won": targets,
    })
    return df


def test_fit_and_predict_lightgbm():
    df = create_synthetic_tennis_features(n_samples=250)

    # Time-based split: First 180 for train, next 70 for validation
    train_df = df.iloc[:180].copy()
    val_df = df.iloc[180:].copy()

    model = MatchWinnerModel(model_type="lightgbm", random_state=42)
    model.fit(train_df, target_col="target_p1_won", val_df=val_df)

    assert model.is_fitted is True
    assert len(model.feature_names_) > 5

    # Test predict_proba
    probs = model.predict_proba(val_df)
    assert len(probs) == len(val_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test predict
    preds = model.predict(val_df)
    assert len(preds) == len(val_df)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test feature importance
    df_imp = model.feature_importance(top_n=10)
    assert not df_imp.empty
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert "importance_pct" in df_imp.columns

    # Evaluate with TennisEvaluator
    evaluator = TennisEvaluator()
    metrics = evaluator.evaluate_binary(val_df["target_p1_won"], probs)
    assert metrics.brier_score < 0.25
    assert metrics.accuracy > 0.50


def test_fit_and_predict_xgboost():
    df = create_synthetic_tennis_features(n_samples=150)
    train_df = df.iloc[:100].copy()
    val_df = df.iloc[100:].copy()

    model = MatchWinnerModel(model_type="xgboost", random_state=42)
    model.fit(train_df, target_col="target_p1_won", val_df=val_df)

    assert model.is_fitted is True
    probs = model.predict_proba(val_df)
    assert len(probs) == len(val_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_save_and_load():
    df = create_synthetic_tennis_features(n_samples=120)
    train_df = df.iloc[:80].copy()
    test_df = df.iloc[80:].copy()

    model = MatchWinnerModel(model_type="lightgbm")
    model.fit(train_df, target_col="target_p1_won")

    orig_probs = model.predict_proba(test_df)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_match_winner.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = MatchWinnerModel.load(save_path)
        assert loaded_model.is_fitted is True

        loaded_probs = loaded_model.predict_proba(test_df)
        assert np.allclose(orig_probs, loaded_probs, atol=1e-5)


if __name__ == "__main__":
    test_fit_and_predict_lightgbm()
    test_fit_and_predict_xgboost()
    test_save_and_load()
    print("All MatchWinnerModel tests passed successfully!")

