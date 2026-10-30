"""Unit tests for TennisEvaluator."""

import sys
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import TennisEvaluator


def test_time_series_split():
    evaluator = TennisEvaluator()
    dates = pd.date_range("2021-01-01", "2024-06-01", freq="MS")
    df = pd.DataFrame({
        "match_date": dates,
        "dummy_val": range(len(dates)),
    })

    train_df, val_df, test_df = evaluator.split_time_series(
        df,
        date_col="match_date",
        train_end="2022-12-31",
        val_end="2023-12-31",
    )

    # Verify strictly monotonic, non-overlapping partitions
    assert train_df["match_date"].max() <= pd.Timestamp("2022-12-31")
    assert val_df["match_date"].min() > pd.Timestamp("2022-12-31")
    assert val_df["match_date"].max() <= pd.Timestamp("2023-12-31")
    assert test_df["match_date"].min() > pd.Timestamp("2023-12-31")
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_binary_metrics_and_calibration():
    evaluator = TennisEvaluator()

    # 1. Perfectly calibrated probabilities
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.8, 0.75, 0.2, 0.15, 0.1, 0.05])

    res = evaluator.evaluate_binary(y_true, y_prob)

    assert res.n_samples == 8
    assert res.accuracy == 1.0
    assert res.roc_auc == 1.0
    assert res.brier_score < 0.05
    assert res.brier_skill_score > 0.80
    assert res.log_loss < 0.30
    assert res.expected_calibration_error < 0.25
    assert len(res.calibration_curve_data["prob_true"]) > 0


def test_regression_metrics():
    evaluator = TennisEvaluator()
    # Actual match total games vs predicted match total games
    y_true = np.array([21.0, 24.0, 18.0, 32.0, 22.0])
    y_pred = np.array([22.0, 23.0, 19.0, 30.0, 21.0])

    res = evaluator.evaluate_regression(y_true, y_pred, spread_line=22.5)

    assert res.n_samples == 5
    assert res.mae <= 1.5
    assert res.rmse <= 1.5
    assert res.pearson_r > 0.90
    assert 0.0 <= res.spread_accuracy <= 1.0


def test_segmented_breakdowns():
    evaluator = TennisEvaluator()

    # Generate synthetic matches with surfaces and tour tiers
    df = pd.DataFrame({
        "target": [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        "prob":   [0.8, 0.3, 0.7, 0.65, 0.4, 0.85, 0.2, 0.45, 0.75, 0.6, 0.35, 0.25],
        "surface": ["Hard", "Hard", "Clay", "Clay", "Grass", "Grass", "Hard", "Clay", "Grass", "Hard", "Clay", "Grass"],
        "tourney_level": ["G", "G", "M", "M", "A", "A", "G", "M", "A", "G", "M", "A"],
        "tour": ["ATP", "ATP", "WTA", "WTA", "ATP", "WTA", "ATP", "WTA", "ATP", "ATP", "WTA", "ATP"],
        "diff_elo_blended": [120, -80, 150, 40, -110, 220, -180, 20, 90, 140, -45, -70],
    })

    breakdowns = evaluator.evaluate_breakdowns(
        df,
        y_true_col="target",
        y_prob_col="prob",
        group_cols=["surface", "tour", "tourney_level"],
        elo_diff_col="diff_elo_blended",
        min_samples_per_group=2,
    )

    assert "surface" in breakdowns
    assert "tour" in breakdowns
    assert "elo_gap_bucket" in breakdowns
    # Verify non-empty summary tables
    assert not breakdowns["surface"].empty
    assert "accuracy" in breakdowns["surface"].columns
    assert "brier_score" in breakdowns["surface"].columns


def test_clv_and_market_disagreements():
    evaluator = TennisEvaluator()

    # Synthetic betting data
    # Pinnacle closing odds vs our model probabilities
    df_bets = pd.DataFrame({
        "target": [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "model_prob": [0.65, 0.72, 0.58, 0.80, 0.45, 0.60, 0.35, 0.75, 0.68, 0.50],
        "market_prob": [0.55, 0.60, 0.52, 0.70, 0.40, 0.52, 0.40, 0.62, 0.58, 0.52],
        "odds_entry": [1.90, 1.75, 2.05, 1.50, 2.60, 2.00, 2.70, 1.65, 1.80, 2.00],
        "odds_close": [1.82, 1.67, 1.95, 1.43, 2.50, 1.92, 2.50, 1.61, 1.72, 1.92],  # Consistent positive CLV
    })

    # Test CLV
    clv_res = evaluator.compute_clv(df_bets, odds_entry_col="odds_entry", odds_close_col="odds_close")
    assert clv_res["n_bets"] == 10
    assert clv_res["mean_clv_pct"] > 0.0  # Entry odds were higher than closing
    assert clv_res["beat_closing_line_pct"] == 100.0

    # Test Disagreements
    disagreements = evaluator.analyze_market_disagreements(
        df_bets,
        y_true_col="target",
        p_model_col="model_prob",
        odds_col="odds_entry",
        p_market_col="market_prob",
        edge_thresholds=[0.05, 0.08, 0.10],
        stake_mode="flat",
        flat_stake=100.0,
    )

    assert len(disagreements) == 3
    assert "roi_pct" in disagreements.columns
    assert "net_profit" in disagreements.columns
    assert disagreements.iloc[0]["n_bets"] > 0


def test_reporting_and_save():
    evaluator = TennisEvaluator()
    y_true = np.array([1, 0, 1, 1, 0])
    y_prob = np.array([0.7, 0.3, 0.8, 0.6, 0.4])

    metrics = evaluator.evaluate_binary(y_true, y_prob)

    # Test ASCII summary print (no exception)
    evaluator.print_summary(metrics, title="TEST EVALUATION")

    # Test saving report to JSON
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_file = Path(tmp_dir) / "eval_report.json"
        saved = evaluator.save_report(metrics, report_file, extra_metadata={"model": "LightGBM_v1"})
        assert saved.exists()


if __name__ == "__main__":
    test_time_series_split()
    test_binary_metrics_and_calibration()
    test_regression_metrics()
    test_segmented_breakdowns()
    test_clv_and_market_disagreements()
    test_reporting_and_save()
    print("All TennisEvaluator tests passed successfully!")
