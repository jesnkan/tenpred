"""State-of-the-Art Evaluation Framework for ATP and WTA Tennis Prediction.

Provides:
1. Time-series temporal splits (strictly leak-free).
2. Probabilistic and classification metrics (Log Loss, Brier Score, ROC-AUC, ECE).
3. Calibration analysis and reliability diagrams.
4. Segmented breakdown reporting by Surface, Elo Gap, Tournament Level, and Tour.
5. Closing Line Value (CLV) evaluation against market lines.
6. Model vs Market disagreement analysis (Edge buckets, simulated ROI, Kelly staking).
7. Regression evaluation for Set & Game market totals and spreads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

try:
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class BinaryEvaluationMetrics:
    """Core classification and probabilistic evaluation metrics."""
    n_samples: int
    accuracy: float
    brier_score: float
    brier_skill_score: float
    log_loss: float
    roc_auc: float
    expected_calibration_error: float
    max_calibration_error: float
    mean_predicted_prob: float
    observed_win_rate: float
    calibration_curve_data: Dict[str, List[float]]


@dataclass
class RegressionEvaluationMetrics:
    """Continuous evaluation metrics for game spreads and match game totals."""
    n_samples: int
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    spread_accuracy: float
    mean_error_bias: float


class TennisEvaluator:
    """Production-grade evaluation engine for tennis prediction pipelines.

    Ensures zero temporal leakage, deep market efficiency benchmarking,
    and granular error attribution across surfaces, tournament tiers, and odds buckets.
    """

    def __init__(self, eps: float = 1e-15) -> None:
        """Initialize TennisEvaluator.

        Args:
            eps: Numerical epsilon to clip probabilities and prevent infinite log loss.
        """
        self.eps = eps

    # =========================================================================
    # 1. Temporal Splits (Strictly Time-Based)
    # =========================================================================

    @staticmethod
    def split_time_series(
        df: pd.DataFrame,
        date_col: str = "match_date",
        train_end: Union[str, date, datetime, pd.Timestamp] = "2022-12-31",
        val_end: Optional[Union[str, date, datetime, pd.Timestamp]] = "2023-12-31",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Split DataFrame chronologically into train, validation, and test subsets.

        Args:
            df: Full engineered DataFrame.
            date_col: Date column name.
            train_end: Cutoff date for training set (inclusive).
            val_end: Cutoff date for validation set (inclusive). If None, splits into (train, test).

        Returns:
            Tuple of (train_df, val_df, test_df) or (train_df, test_df, None).
        """
        data = df.copy()
        data["_eval_date"] = pd.to_datetime(data[date_col].astype(str), errors="coerce")

        ts_train_end = pd.to_datetime(train_end)
        train_df = data[data["_eval_date"] <= ts_train_end].drop(columns=["_eval_date"]).reset_index(drop=True)

        if val_end is not None:
            ts_val_end = pd.to_datetime(val_end)
            val_df = data[(data["_eval_date"] > ts_train_end) & (data["_eval_date"] <= ts_val_end)].drop(columns=["_eval_date"]).reset_index(drop=True)
            test_df = data[data["_eval_date"] > ts_val_end].drop(columns=["_eval_date"]).reset_index(drop=True)
            logger.info(f"Time split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
            return train_df, val_df, test_df
        else:
            test_df = data[data["_eval_date"] > ts_train_end].drop(columns=["_eval_date"]).reset_index(drop=True)
            logger.info(f"Time split: Train={len(train_df):,}, Test={len(test_df):,}")
            return train_df, test_df, None

    @staticmethod
    def split_by_years(
        df: pd.DataFrame,
        train_years: List[int],
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        year_col: str = "year",
        date_col: str = "match_date",
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Split DataFrame by explicit calendar years."""
        data = df.copy()
        if year_col not in data.columns:
            data[year_col] = pd.to_datetime(data[date_col].astype(str), errors="coerce").dt.year

        train_df = data[data[year_col].isin(train_years)].reset_index(drop=True)
        val_df = data[data[year_col].isin(val_years)].reset_index(drop=True) if val_years else None
        test_df = data[data[year_col].isin(test_years)].reset_index(drop=True) if test_years else None

        return train_df, val_df, test_df

    # =========================================================================
    # 2. Binary Classification & Probabilistic Metrics
    # =========================================================================

    def evaluate_binary(
        self,
        y_true: Union[pd.Series, np.ndarray, List[int]],
        y_prob: Union[pd.Series, np.ndarray, List[float]],
        threshold: float = 0.5,
        n_bins: int = 10,
    ) -> BinaryEvaluationMetrics:
        """Compute complete probabilistic evaluation metrics for binary outcomes.

        Args:
            y_true: Ground truth binary targets (0 or 1).
            y_prob: Predicted probabilities for class 1 [0.0, 1.0].
            threshold: Decision boundary threshold for classification accuracy.
            n_bins: Number of bins for calibration curve and ECE calculation.

        Returns:
            BinaryEvaluationMetrics dataclass instance.
        """
        y_t = np.asarray(y_true, dtype=np.int32)
        y_p = np.clip(np.asarray(y_prob, dtype=np.float64), self.eps, 1.0 - self.eps)

        if len(y_t) == 0:
            raise ValueError("Evaluation arrays cannot be empty.")

        n_samples = len(y_t)
        y_pred = (y_p >= threshold).astype(int)

        acc = float(accuracy_score(y_t, y_pred))
        brier = float(brier_score_loss(y_t, y_p))

        # Baseline reference Brier score: p_base * (1 - p_base)
        base_rate = float(np.mean(y_t))
        brier_ref = base_rate * (1.0 - base_rate) if 0 < base_rate < 1 else 0.25
        brier_skill = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0

        loss = float(log_loss(y_t, y_p))

        try:
            auc = float(roc_auc_score(y_t, y_p))
        except ValueError:
            auc = 0.50

        # Calibration curve & Expected Calibration Error (ECE)
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=n_bins, strategy="uniform")

        # Compute bin counts and exact ECE / MCE
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(y_p, bin_edges) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

        ece = 0.0
        max_cal_err = 0.0
        bin_counts = []

        for b in range(n_bins):
            mask = bin_assignments == b
            count = int(np.sum(mask))
            bin_counts.append(count)
            if count > 0:
                bin_acc = float(np.mean(y_t[mask]))
                bin_conf = float(np.mean(y_p[mask]))
                err = abs(bin_acc - bin_conf)
                ece += (count / n_samples) * err
                if err > max_cal_err:
                    max_cal_err = err

        return BinaryEvaluationMetrics(
            n_samples=n_samples,
            accuracy=acc,
            brier_score=brier,
            brier_skill_score=brier_skill,
            log_loss=loss,
            roc_auc=auc,
            expected_calibration_error=float(ece),
            max_calibration_error=float(max_cal_err),
            mean_predicted_prob=float(np.mean(y_p)),
            observed_win_rate=base_rate,
            calibration_curve_data={
                "prob_true": [float(x) for x in prob_true],
                "prob_pred": [float(x) for x in prob_pred],
                "bin_counts": bin_counts,
            },
        )

    # =========================================================================
    # 3. Continuous / Regression Metrics (Totals & Spreads)
    # =========================================================================

    @staticmethod
    def evaluate_regression(
        y_true: Union[pd.Series, np.ndarray, List[float]],
        y_pred: Union[pd.Series, np.ndarray, List[float]],
        spread_line: float = 0.0,
    ) -> RegressionEvaluationMetrics:
        """Evaluate regression models for total match games and game handicap spreads."""
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)

        n_samples = len(y_t)
        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

        try:
            r2 = float(r2_score(y_t, y_p))
        except ValueError:
            r2 = 0.0

        if np.std(y_t) > 0 and np.std(y_p) > 0:
            pearson_r, _ = stats.pearsonr(y_t, y_p)
            pearson_r = float(pearson_r)
        else:
            pearson_r = 0.0

        # Spread directional accuracy: did prediction correctly identify sign relative to line?
        pred_side = y_p > spread_line
        true_side = y_t > spread_line
        spread_acc = float(np.mean(pred_side == true_side))

        bias = float(np.mean(y_p - y_t))

        return RegressionEvaluationMetrics(
            n_samples=n_samples,
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearson_r=pearson_r,
            spread_accuracy=spread_acc,
            mean_error_bias=bias,
        )

    # =========================================================================
    # 4. Granular Segmented Breakdowns
    # =========================================================================

    def evaluate_breakdowns(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        y_prob_col: str,
        group_cols: Optional[List[str]] = None,
        elo_diff_col: Optional[str] = "diff_elo_blended",
        min_samples_per_group: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Compute metric breakdowns across surfaces, tournament tiers, tours, and Elo gaps.

        Args:
            df: DataFrame containing targets, predictions, and grouping columns.
            y_true_col: Ground truth binary target column.
            y_prob_col: Predicted win probability column.
            group_cols: List of grouping columns. Defaults to ['surface', 'tourney_level', 'tour'].
            elo_diff_col: Column representing Elo rating difference between players.
            min_samples_per_group: Minimum matches required to evaluate a slice (default: 5).

        Returns:
            Dictionary mapping category names to summary DataFrames.
        """
        data = df.copy()
        if group_cols is None:
            group_cols = [c for c in ["surface", "tourney_level", "tour"] if c in data.columns]

        # Add Elo Gap buckets if column exists
        if elo_diff_col and elo_diff_col in data.columns:
            abs_diff = data[elo_diff_col].abs()
            bins = [0, 50, 150, 300, 10000]
            labels = ["Tight (0-50)", "Moderate (50-150)", "Large (150-300)", "Heavy (>300)"]
            data["elo_gap_bucket"] = pd.cut(abs_diff, bins=bins, labels=labels, right=False)
            if "elo_gap_bucket" not in group_cols:
                group_cols.append("elo_gap_bucket")

        breakdown_tables: Dict[str, pd.DataFrame] = {}

        for col in group_cols:
            if col not in data.columns:
                continue

            rows = []
            for val, group in data.groupby(col, observed=True):
                sub_y_t = group[y_true_col].dropna().values
                sub_y_p = group[y_prob_col].loc[group[y_true_col].dropna().index].values

                if len(sub_y_t) < min_samples_per_group:
                    continue

                metrics = self.evaluate_binary(sub_y_t, sub_y_p)
                rows.append({
                    col: str(val),
                    "matches": metrics.n_samples,
                    "accuracy": round(metrics.accuracy, 4),
                    "brier_score": round(metrics.brier_score, 4),
                    "log_loss": round(metrics.log_loss, 4),
                    "roc_auc": round(metrics.roc_auc, 4),
                    "ece": round(metrics.expected_calibration_error, 4),
                    "observed_win_rate": round(metrics.observed_win_rate, 4),
                })

            df_breakdown = pd.DataFrame(rows)
            if not df_breakdown.empty:
                df_breakdown = df_breakdown.sort_values(by="matches", ascending=False).reset_index(drop=True)
            breakdown_tables[col] = df_breakdown

        return breakdown_tables

    # =========================================================================
    # 5. Market Efficiency & Closing Line Value (CLV)
    # =========================================================================

    @staticmethod
    def compute_clv(
        df: pd.DataFrame,
        odds_entry_col: str,
        odds_close_col: str,
        bet_placed_mask: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Compute Closing Line Value (CLV) metrics.

        CLV = (Odds_Entry / Odds_Close) - 1.0. A positive CLV demonstrates that the model
        beats market movement and captures valuable liquidity before line contraction.

        Args:
            df: DataFrame containing odds columns.
            odds_entry_col: Decimal odds when the model placed the bet.
            odds_close_col: Decimal odds at market close (e.g. Pinnacle closing line).
            bet_placed_mask: Optional boolean Series selecting matches where a bet was made.

        Returns:
            Dictionary with mean CLV %, % of times beating closing line, and sample size.
        """
        data = df if bet_placed_mask is None else df[bet_placed_mask]

        valid_mask = (data[odds_entry_col] > 1.0) & (data[odds_close_col] > 1.0)
        sub = data[valid_mask]

        if len(sub) == 0:
            return {"n_bets": 0, "mean_clv_pct": 0.0, "beat_closing_line_pct": 0.0}

        entry_odds = sub[odds_entry_col].values
        close_odds = sub[odds_close_col].values

        clv_ratios = (entry_odds / close_odds) - 1.0
        beat_line = entry_odds > close_odds

        return {
            "n_bets": int(len(sub)),
            "mean_clv_pct": float(np.mean(clv_ratios) * 100.0),
            "median_clv_pct": float(np.median(clv_ratios) * 100.0),
            "beat_closing_line_pct": float(np.mean(beat_line) * 100.0),
            "clv_std_pct": float(np.std(clv_ratios) * 100.0),
        }

    # =========================================================================
    # 6. Model vs Market Disagreement Analysis
    # =========================================================================

    def analyze_market_disagreements(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        p_model_col: str,
        odds_col: str,
        p_market_col: Optional[str] = None,
        edge_thresholds: Optional[List[float]] = None,
        stake_mode: str = "flat",
        flat_stake: float = 100.0,
        kelly_fraction: float = 0.25,
    ) -> pd.DataFrame:
        """Analyze betting performance where the model strongly disagrees with market odds.

        Simulates wagering on selections where:
            Edge = P_model - P_market >= Edge_Threshold.

        Args:
            df: DataFrame containing targets, model probabilities, and decimal odds.
            y_true_col: Ground truth outcome (1 if won, 0 if lost).
            p_model_col: Model's predicted probability.
            odds_col: Available decimal odds (e.g., Pinnacle or market average).
            p_market_col: Optional market implied probability. If None, uses 1 / odds.
            edge_thresholds: List of edge cutoffs (e.g., [0.02, 0.04, 0.06, 0.08, 0.10]).
            stake_mode: 'flat' (fixed units) or 'kelly' (fractional Kelly staking).
            flat_stake: Currency units per flat bet.
            kelly_fraction: Kelly multiplier to control variance (default: 0.25 = quarter Kelly).

        Returns:
            DataFrame summarizing bet counts, total wagered, profit, ROI, and win rates across edge tiers.
        """
        data = df.copy()
        if edge_thresholds is None:
            edge_thresholds = [0.02, 0.04, 0.06, 0.08, 0.10]

        # Valid rows filter
        mask = (data[odds_col] > 1.0) & (data[p_model_col] > 0.0) & (data[y_true_col].isin([0, 1]))
        data = data[mask].copy()

        # Compute market implied probability
        if p_market_col and p_market_col in data.columns:
            p_market = data[p_market_col].values
        else:
            p_market = 1.0 / data[odds_col].values

        p_model = data[p_model_col].values
        edges = p_model - p_market
        data["_edge"] = edges

        y_true = data[y_true_col].values
        decimal_odds = data[odds_col].values

        results = []

        for threshold in edge_thresholds:
            bet_mask = edges >= threshold
            n_bets = int(np.sum(bet_mask))

            if n_bets == 0:
                results.append({
                    "edge_threshold": f">={threshold * 100:.1f}%",
                    "n_bets": 0,
                    "model_win_rate": 0.0,
                    "market_implied_win_rate": 0.0,
                    "total_staked": 0.0,
                    "net_profit": 0.0,
                    "roi_pct": 0.0,
                    "max_drawdown": 0.0,
                })
                continue

            sub_y = y_true[bet_mask]
            sub_odds = decimal_odds[bet_mask]
            sub_p_mod = p_model[bet_mask]
            sub_p_mkt = p_market[bet_mask]

            # Calculate stakes
            if stake_mode == "kelly":
                b = sub_odds - 1.0
                q = 1.0 - sub_p_mod
                raw_kelly = (b * sub_p_mod - q) / b
                stakes = np.clip(raw_kelly * kelly_fraction, 0.01, 0.10) * flat_stake * 10.0
            else:
                stakes = np.full(n_bets, flat_stake)

            # Profit per bet: win -> stake * (odds - 1), loss -> -stake
            profits = np.where(sub_y == 1, stakes * (sub_odds - 1.0), -stakes)

            total_staked = float(np.sum(stakes))
            net_profit = float(np.sum(profits))
            roi = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0

            # Cumulative drawdown
            cum_profit = np.cumsum(profits)
            running_max = np.maximum.accumulate(cum_profit)
            drawdowns = running_max - cum_profit
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

            results.append({
                "edge_threshold": f">={threshold * 100:.1f}%",
                "n_bets": n_bets,
                "model_win_rate": round(float(np.mean(sub_y)), 4),
                "market_implied_win_rate": round(float(np.mean(sub_p_mkt)), 4),
                "total_staked": round(total_staked, 2),
                "net_profit": round(net_profit, 2),
                "roi_pct": round(roi, 2),
                "max_drawdown": round(max_dd, 2),
            })

        return pd.DataFrame(results)

    # =========================================================================
    # 7. Comprehensive Reporting & Display
    # =========================================================================

    def print_summary(
        self,
        metrics: BinaryEvaluationMetrics,
        breakdowns: Optional[Dict[str, pd.DataFrame]] = None,
        disagreements: Optional[pd.DataFrame] = None,
        title: str = "TENNIS PREDICTION EVALUATION REPORT",
    ) -> None:
        """Format and print an executive ASCII summary report."""
        sep = "=" * 70
        sub_sep = "-" * 70

        print(f"\n{sep}")
        print(f" {title.center(68)} ")
        print(f"{sep}")
        print(f" Evaluated Matches : {metrics.n_samples:,}")
        print(f" Observed Base Win Rate: {metrics.observed_win_rate:.2%}")
        print(f" Mean Predicted Prob   : {metrics.mean_predicted_prob:.2%}")
        print(f"{sub_sep}")
        print(f" Accuracy              : {metrics.accuracy:.2%}")
        print(f" Brier Score           : {metrics.brier_score:.4f}  (Ref: 0.2500)")
        print(f" Brier Skill Score     : {metrics.brier_skill_score:+.2%}")
        print(f" Log Loss              : {metrics.log_loss:.4f}")
        print(f" ROC-AUC               : {metrics.roc_auc:.4f}")
        print(f" Expected Calib. Error : {metrics.expected_calibration_error:.4f} ({metrics.expected_calibration_error:.2%})")
        print(f" Max Calibration Error : {metrics.max_calibration_error:.4f}")
        print(f"{sep}")

        # Print Segmented Breakdowns
        if breakdowns:
            print("\n" + sub_sep)
            print(" SEGMENTED PERFORMANCE BREAKDOWNS")
            print(sub_sep)
            for group_name, df_sub in breakdowns.items():
                print(f"\n--- Breakdown by: {group_name.upper()} ---")
                print(df_sub.to_string(index=False))

        # Print Disagreements
        if disagreements is not None and not disagreements.empty:
            print("\n" + sub_sep)
            print(" MODEL VS MARKET DISAGREEMENT & EDGE SIMULATION")
            print(sub_sep)
            print(disagreements.to_string(index=False))
            print(sep + "\n")

    def save_report(
        self,
        metrics: BinaryEvaluationMetrics,
        output_path: Union[str, Path],
        breakdowns: Optional[Dict[str, pd.DataFrame]] = None,
        disagreements: Optional[pd.DataFrame] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize complete evaluation report to a JSON artifact.

        Args:
            metrics: BinaryEvaluationMetrics object.
            output_path: Destination JSON file path.
            breakdowns: Segmented breakdown DataFrames.
            disagreements: Disagreement simulation DataFrame.
            extra_metadata: Additional key-value pairs (e.g. model name, features used).

        Returns:
            Path to saved JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "core_metrics": asdict(metrics),
            "breakdowns": {},
            "disagreement_analysis": disagreements.to_dict(orient="records") if disagreements is not None else [],
            "metadata": extra_metadata or {},
        }

        if breakdowns:
            for k, v in breakdowns.items():
                report_payload["breakdowns"][k] = v.to_dict(orient="records")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        logger.info(f"Saved evaluation report to {path}")
        return path
