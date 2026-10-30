"""Evaluation metrics, backtesting simulations, calibration analysis (Brier score, log-loss), and ROI trackers."""

from src.evaluation.evaluator import (
    BinaryEvaluationMetrics,
    RegressionEvaluationMetrics,
    TennisEvaluator,
)
from src.evaluation.risk_manager import (
    TennisRiskManager,
    BetRecommendation,
)

__all__ = [
    "TennisEvaluator",
    "BinaryEvaluationMetrics",
    "RegressionEvaluationMetrics",
    "TennisRiskManager",
    "BetRecommendation",
]
