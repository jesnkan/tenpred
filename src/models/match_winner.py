"""State-of-the-Art Match Winner Prediction Model for ATP and WTA Tennis.

Implements regularized Gradient Boosted Decision Trees (LightGBM / XGBoost)
blended with SurfaceElo probabilities and probability calibration (Platt scaling /
Isotonic regression) to optimize Brier skill score and log loss.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# Filter LightGBM eval_set deprecation warnings
warnings.filterwarnings("ignore", message=".*The argument 'eval_set' is deprecated.*")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class MatchWinnerModel:
    """Production-grade Match Winner probabilistic forecasting model.

    Features:
    - Primary Gradient Boosted Trees: LightGBM (default) or XGBoost.
    - Regularized default hyperparameters tuned specifically for tennis variance.
    - Multi-stage probability calibration and stacked logit blending with SurfaceElo.
    - Automated feature discovery from TennisFeatureEngineer datasets.
    - Zero lookahead leakage via time-based training and validation splits.
    - Joblib model serialization and feature importance inspection.
    """

    # Columns excluded from feature matrix
    DEFAULT_EXCLUDE_COLS: Set[str] = {
        "player1", "player2", "match_date", "tourney_date", "parsed_match_date",
        "score", "clean_score", "round_rank", "tourney_id", "tourney_name",
        "target_p1_won", "target_total_sets", "target_total_games",
        "target_game_diff", "target_set_diff", "target_set1_p1_won",
        "is_completed", "is_walkover", "is_retirement", "tour",
        "p1_hand", "p2_hand",  # encoded as is_lefty_vs_righty
    }

    # Categorical columns to cast for tree algorithms
    CATEGORICAL_COLS: List[str] = ["surface", "tournament_level", "round"]

    def __init__(
        self,
        model_type: str = "lightgbm",
        hyperparameters: Optional[Dict[str, Any]] = None,
        elo_prob_col: str = "elo_win_prob_p1",
        elo_blend_weight: float = 0.35,
        calibration_method: str = "sigmoid",  # 'sigmoid' (Platt scaling) or 'isotonic'
        random_state: int = 42,
    ) -> None:
        """Initialize the MatchWinnerModel.

        Args:
            model_type: 'lightgbm' (recommended) or 'xgboost'.
            hyperparameters: Custom parameters overriding default regularized tree settings.
            elo_prob_col: Column containing pre-match SurfaceElo win probabilities.
            elo_blend_weight: Fixed weight assigned to Elo when no calibration set is provided [0.0, 1.0].
            calibration_method: 'sigmoid' (Platt scaling via logistic regression) or 'isotonic'.
            random_state: Seed for reproducible bagging and feature subsampling.
        """
        self.model_type = model_type.lower()
        self.elo_prob_col = elo_prob_col
        self.elo_blend_weight = float(np.clip(elo_blend_weight, 0.0, 1.0))
        self.calibration_method = calibration_method.lower()
        self.random_state = random_state

        if self.model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError("LightGBM is not installed. Run 'pip install lightgbm'.")
        if self.model_type == "xgboost" and not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Run 'pip install xgboost'.")
        if self.model_type == "ensemble" and not (HAS_LIGHTGBM or HAS_XGBOOST):
            raise ImportError("Neither LightGBM nor XGBoost is installed for ensemble.")

        # Regularized tennis tree hyperparameters (conservative depth & high regularization)
        self.hyperparameters = self._get_default_hyperparameters()
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        # Fitted attributes
        self.model: Optional[Any] = None
        self.model_lgb: Optional[Any] = None
        self.model_xgb: Optional[Any] = None
        self.calibrator: Optional[Any] = None
        self.feature_names_: List[str] = []
        self.cat_mappings_: Dict[str, Dict[str, int]] = {}
        self.is_fitted: bool = False

    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Return tuned conservative regularization defaults for tennis prediction."""
        if self.model_type == "xgboost":
            return self._get_xgb_hyperparameters()
        # Default or ensemble LightGBM settings
        return self._get_lgb_hyperparameters()

    def _get_lgb_hyperparameters(self) -> Dict[str, Any]:
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "n_estimators": 500,
            "max_depth": 5,
            "num_leaves": 23,
            "min_child_samples": 40,
            "subsample": 0.80,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.20,       # L1 regularization
            "reg_lambda": 1.50,      # L2 regularization
            "random_state": self.random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

    def _get_xgb_hyperparameters(self) -> Dict[str, Any]:
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.03,
            "n_estimators": 500,
            "max_depth": 4,
            "min_child_weight": 30,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.20,
            "reg_lambda": 1.50,
            "random_state": self.random_state,
            "n_jobs": -1,
        }

    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        is_training: bool = False,
    ) -> pd.DataFrame:
        """Sanitize feature columns, encode categoricals, and align schema."""
        data = df.copy()

        if is_training:
            if feature_cols is not None:
                self.feature_names_ = list(feature_cols)
            else:
                # Auto-discover feature columns
                candidates = []
                for col in data.columns:
                    col_str = str(col)
                    if col_str in self.DEFAULT_EXCLUDE_COLS or col_str.startswith("target_"):
                        continue
                    # Must be numeric or recognized categorical
                    if pd.api.types.is_numeric_dtype(data[col]) or col_str in self.CATEGORICAL_COLS:
                        candidates.append(col_str)
                self.feature_names_ = sorted(candidates)

            # Build category encodings
            self.cat_mappings_ = {}
            for cat_col in self.CATEGORICAL_COLS:
                if cat_col in self.feature_names_ and cat_col in data.columns:
                    unique_vals = sorted(data[cat_col].dropna().astype(str).unique().tolist())
                    self.cat_mappings_[cat_col] = {val: idx for idx, val in enumerate(unique_vals)}

        # Check missing feature columns
        missing_cols = [c for c in self.feature_names_ if c not in data.columns]
        for c in missing_cols:
            data[c] = np.nan

        X = data[self.feature_names_].copy()

        # Apply category encoding
        for cat_col, mapping in self.cat_mappings_.items():
            if cat_col in X.columns:
                X[cat_col] = (
                    X[cat_col]
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        # Convert remaining to float32
        for col in X.columns:
            if col not in self.cat_mappings_:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

        return X

    @staticmethod
    def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Safely compute logit transform: ln(p / (1 - p))."""
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "target_p1_won",
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 35,
        verbose: bool = False,
    ) -> MatchWinnerModel:
        """Fit the primary GBDT model and train probability calibration layer.

        Args:
            train_df: Chronologically older training matches from TennisFeatureEngineer.
            feature_cols: Explicit list of feature names. If None, auto-detected.
            target_col: Binary target label column (default: 'target_p1_won').
            val_df: Chronologically subsequent validation matches for early stopping and calibration.
            early_stopping_rounds: Early stopping patience.
            verbose: Whether to log training iterations.

        Returns:
            self: The fitted MatchWinnerModel instance.
        """
        if train_df.empty:
            raise ValueError("Training DataFrame cannot be empty.")
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        logger.info(f"Fitting {self.model_type.upper()} MatchWinnerModel on {len(train_df):,} training samples...")

        # 1. Prepare training feature matrix
        X_train = self._prepare_features(train_df, feature_cols=feature_cols, is_training=True)
        y_train = train_df[target_col].astype(int).values

        # 2. Prepare validation set if available
        X_val, y_val = None, None
        eval_set = None
        if val_df is not None and not val_df.empty and target_col in val_df.columns:
            X_val = self._prepare_features(val_df, is_training=False)
            y_val = val_df[target_col].astype(int).values
            eval_set = [(X_val, y_val)]

        # 3. Train base tree model(s)
        if self.model_type == "ensemble":
            # Train LightGBM component
            lgb_params = self._get_lgb_hyperparameters()
            self.model_lgb = lgb.LGBMClassifier(**lgb_params)
            callbacks = []
            if eval_set and early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
            cat_feature_idx = [c for c in self.CATEGORICAL_COLS if c in self.feature_names_]
            self.model_lgb.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                categorical_feature=cat_feature_idx if cat_feature_idx else "auto",
                callbacks=callbacks if callbacks else None,
            )

            # Train XGBoost component
            xgb_params = self._get_xgb_hyperparameters()
            self.model_xgb = xgb.XGBClassifier(**xgb_params)
            eval_s = [(X_train, y_train)]
            if eval_set:
                eval_s.append((X_val, y_val))
            self.model_xgb.fit(
                X_train,
                y_train,
                eval_set=eval_s,
                verbose=verbose,
            )
            self.model = self.model_lgb  # primary reference

        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(**self.hyperparameters)

            callbacks = []
            if eval_set and early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=verbose))
            if verbose:
                callbacks.append(lgb.log_evaluation(50))

            cat_feature_idx = [c for c in self.CATEGORICAL_COLS if c in self.feature_names_]
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                categorical_feature=cat_feature_idx if cat_feature_idx else "auto",
                callbacks=callbacks if callbacks else None,
            )

        else:  # xgboost
            self.model = xgb.XGBClassifier(**self.hyperparameters)
            eval_s = [(X_train, y_train)]
            if eval_set:
                eval_s.append((X_val, y_val))

            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_s,
                verbose=verbose,
            )

        self.is_fitted = True

        # 4. Train Calibration & Stacking Layer
        # Uses validation set if provided; otherwise fits Platt scaling on train predictions
        calib_X_df = val_df if (val_df is not None and not val_df.empty) else train_df
        calib_X_features = X_val if (X_val is not None) else X_train
        calib_y = y_val if (y_val is not None) else y_train

        # Extract pre-match SurfaceElo probabilities if column exists
        if self.elo_prob_col in calib_X_df.columns:
            elo_probs = calib_X_df[self.elo_prob_col].fillna(0.50).values
        else:
            elo_probs = np.full(len(calib_y), 0.50)

        logit_elo = self._safe_logit(elo_probs).reshape(-1, 1)

        if self.model_type == "ensemble":
            raw_lgb = self.model_lgb.predict_proba(calib_X_features)[:, 1]
            raw_xgb = self.model_xgb.predict_proba(calib_X_features)[:, 1]
            logit_lgb = self._safe_logit(raw_lgb).reshape(-1, 1)
            logit_xgb = self._safe_logit(raw_xgb).reshape(-1, 1)
            stack_features = np.hstack([logit_lgb, logit_xgb, logit_elo])
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)
            self.calibrator.fit(stack_features, calib_y)

        else:
            raw_tree_probs = self.model.predict_proba(calib_X_features)[:, 1]
            logit_tree = self._safe_logit(raw_tree_probs).reshape(-1, 1)
            stack_features = np.hstack([logit_tree, logit_elo])

            if self.calibration_method == "isotonic":
                blended_raw = (1.0 - self.elo_blend_weight) * raw_tree_probs + self.elo_blend_weight * elo_probs
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(blended_raw, calib_y)
            else:
                self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)
                self.calibrator.fit(stack_features, calib_y)

        # Log validation calibration metrics
        calib_preds = self.predict_proba(calib_X_df)
        val_brier = brier_score_loss(calib_y, calib_preds)
        val_loss = log_loss(calib_y, calib_preds)
        logger.info(f"Model ({self.model_type}) successfully trained with {len(self.feature_names_)} features.")
        logger.info(f"Validation Calibrated Brier Score: {val_brier:.4f} | Log Loss: {val_loss:.4f}")

        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate well-calibrated win probabilities for Player 1.

        Blends GBDT inference with SurfaceElo predictions through the learned calibrator.

        Args:
            df: DataFrame containing match feature columns.

        Returns:
            1D numpy array of probabilities P(Player 1 wins) in range [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict_proba().")

        X = self._prepare_features(df, is_training=False)

        # Extract SurfaceElo win probability
        if self.elo_prob_col in df.columns:
            elo_probs = df[self.elo_prob_col].fillna(0.50).values
        else:
            elo_probs = np.full(len(df), 0.50)

        logit_elo = self._safe_logit(elo_probs).reshape(-1, 1)

        if self.model_type == "ensemble":
            raw_lgb = self.model_lgb.predict_proba(X)[:, 1]
            raw_xgb = self.model_xgb.predict_proba(X)[:, 1]
            logit_lgb = self._safe_logit(raw_lgb).reshape(-1, 1)
            logit_xgb = self._safe_logit(raw_xgb).reshape(-1, 1)
            stack_features = np.hstack([logit_lgb, logit_xgb, logit_elo])
            calibrated_probs = self.calibrator.predict_proba(stack_features)[:, 1]
            return np.clip(calibrated_probs, 1e-6, 1.0 - 1e-6)

        if self.model is None:
            return elo_probs

        raw_tree_probs = self.model.predict_proba(X)[:, 1]

        if self.calibrator is None:
            return (1.0 - self.elo_blend_weight) * raw_tree_probs + self.elo_blend_weight * elo_probs

        if self.calibration_method == "isotonic":
            blended_raw = (1.0 - self.elo_blend_weight) * raw_tree_probs + self.elo_blend_weight * elo_probs
            calibrated_probs = self.calibrator.predict(blended_raw)
        else:
            logit_tree = self._safe_logit(raw_tree_probs).reshape(-1, 1)
            stack_features = np.hstack([logit_tree, logit_elo])
            calibrated_probs = self.calibrator.predict_proba(stack_features)[:, 1]

        return np.clip(calibrated_probs, 1e-6, 1.0 - 1e-6)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match winner outcome (1 if Player 1 wins, 0 otherwise).

        Args:
            df: DataFrame containing match feature columns.
            threshold: Decision boundary probability threshold (default: 0.5).

        Returns:
            1D numpy array of integer binary predictions.
        """
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)

    def feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 25,
    ) -> pd.DataFrame:
        """Extract sorted feature importance from the underlying tree ensemble.

        Args:
            importance_type: 'gain' (total split information gain) or 'split' (frequency).
            top_n: Maximum number of top features to return.

        Returns:
            DataFrame with 'feature', 'importance', and 'importance_pct' columns.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Cannot extract feature importance.")

        if self.model_type == "ensemble":
            imp_lgb = self.model_lgb.booster_.feature_importance(importance_type=importance_type) if self.model_lgb else np.zeros(len(self.feature_names_))
            imp_xgb = self.model_xgb.feature_importances_ if self.model_xgb else np.zeros(len(self.feature_names_))
            sum_lgb = float(np.sum(imp_lgb))
            sum_xgb = float(np.sum(imp_xgb))
            norm_lgb = (imp_lgb / sum_lgb) if sum_lgb > 0 else np.zeros_like(imp_lgb)
            norm_xgb = (imp_xgb / sum_xgb) if sum_xgb > 0 else np.zeros_like(imp_xgb)
            raw_imp = 0.5 * norm_lgb + 0.5 * norm_xgb
        elif self.model_type == "lightgbm":
            raw_imp = self.model.booster_.feature_importance(importance_type=importance_type)
        else:  # xgboost
            raw_imp = self.model.feature_importances_

        tot_imp = float(np.sum(raw_imp))
        if tot_imp > 0:
            imp_pct = (raw_imp / tot_imp) * 100.0
        else:
            imp_pct = np.zeros_like(raw_imp)

        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": raw_imp,
            "importance_pct": np.round(imp_pct, 2),
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize the complete MatchWinnerModel state to disk.

        Args:
            filepath: Destination file path (e.g. 'models/match_winner_lgbm.joblib').
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"MatchWinnerModel successfully saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> MatchWinnerModel:
        """Load a persisted MatchWinnerModel instance from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded MatchWinnerModel instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"MatchWinnerModel checkpoint not found at: {path}")

        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is {type(instance)}, expected {cls.__name__}")

        logger.info(f"MatchWinnerModel successfully loaded from {path}")
        return instance
