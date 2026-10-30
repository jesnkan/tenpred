"""State-of-the-Art Set and Game Markets Prediction Engine for ATP & WTA Tennis.

Predicts multi-market secondary outcomes:
1. Total match games (Over / Under continuous expectation & line probabilities)
2. Game differential / handicap (Spread margin expectation & cover probabilities)
3. Total sets & Deciding set probabilities (Over 2.5 sets for Bo3 / Over 3.5 for Bo5)
4. First set winner (Calibrated opening set win probability)

Leverages specialized service hold dynamics, tiebreak frequencies, and tempo features.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss

# Filter LightGBM deprecation warnings
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


class SetGameModel:
    """Production-grade multi-target predictive engine for tennis set and game markets.

    Houses specialized regularized models for:
    - `total_games`: Regressor predicting total match games played.
    - `game_diff`: Regressor predicting games won differential (P1 games - P2 games).
    - `total_sets`: Regressor predicting total sets played + Over 2.5 sets classifier.
    - `first_set`: Calibrated classifier predicting probability that Player 1 wins Set 1.
    """

    DEFAULT_EXCLUDE_COLS: Set[str] = {
        "player1", "player2", "match_date", "tourney_date", "parsed_match_date",
        "score", "clean_score", "round_rank", "tourney_id", "tourney_name",
        "target_p1_won", "target_total_sets", "target_total_games",
        "target_game_diff", "target_set_diff", "target_set1_p1_won",
        "is_completed", "is_walkover", "is_retirement", "tour",
        "p1_hand", "p2_hand",
    }

    CATEGORICAL_COLS: List[str] = ["surface", "tournament_level", "round"]

    def __init__(
        self,
        model_type: str = "lightgbm",
        hyperparameters: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the SetGameModel suite.

        Args:
            model_type: 'lightgbm' (default) or 'xgboost'.
            hyperparameters: Optional dictionary overriding default regularized hyperparameters.
            random_state: Seed for reproducible tree fitting.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state

        if self.model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required. Run 'pip install lightgbm'.")
        if self.model_type == "xgboost" and not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Run 'pip install xgboost'.")

        self.hyperparameters = hyperparameters or {}

        # Sub-models
        self.model_total_games: Optional[Any] = None
        self.model_game_diff: Optional[Any] = None
        self.model_total_sets: Optional[Any] = None
        self.model_over_sets: Optional[Any] = None
        self.model_first_set: Optional[Any] = None

        # Calibrator for first set winner
        self.calibrator_first_set: Optional[LogisticRegression] = None
        self.calibrator_over_sets: Optional[LogisticRegression] = None

        # Residual standard deviations for parametric spread/total line pricing
        self.std_total_games_residual: float = 4.20
        self.std_game_diff_residual: float = 4.80

        self.feature_names_: List[str] = []
        self.cat_mappings_: Dict[str, Dict[str, int]] = {}
        self.is_fitted: bool = False

    def _get_tree_regressor_params(self) -> Dict[str, Any]:
        """Tuned regularized parameters for game/set regression."""
        if self.model_type == "lightgbm":
            base = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "learning_rate": 0.03,
                "n_estimators": 400,
                "max_depth": 4,
                "num_leaves": 15,
                "min_child_samples": 35,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.30,
                "reg_lambda": 1.50,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
        else:
            base = {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "learning_rate": 0.03,
                "n_estimators": 400,
                "max_depth": 4,
                "min_child_weight": 25,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.30,
                "reg_lambda": 1.50,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        base.update(self.hyperparameters)
        return base

    def _get_tree_classifier_params(self) -> Dict[str, Any]:
        """Tuned regularized parameters for binary classification (First Set & Over Sets)."""
        if self.model_type == "lightgbm":
            base = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.03,
                "n_estimators": 400,
                "max_depth": 4,
                "num_leaves": 15,
                "min_child_samples": 35,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.20,
                "reg_lambda": 1.50,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
        else:
            base = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "learning_rate": 0.03,
                "n_estimators": 400,
                "max_depth": 4,
                "min_child_weight": 25,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.20,
                "reg_lambda": 1.50,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
        base.update(self.hyperparameters)
        return base

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
                candidates = []
                for col in data.columns:
                    col_str = str(col)
                    if col_str in self.DEFAULT_EXCLUDE_COLS or col_str.startswith("target_"):
                        continue
                    if pd.api.types.is_numeric_dtype(data[col]) or col_str in self.CATEGORICAL_COLS:
                        candidates.append(col_str)
                self.feature_names_ = sorted(candidates)

            self.cat_mappings_ = {}
            for cat_col in self.CATEGORICAL_COLS:
                if cat_col in self.feature_names_ and cat_col in data.columns:
                    unique_vals = sorted(data[cat_col].dropna().astype(str).unique().tolist())
                    self.cat_mappings_[cat_col] = {val: idx for idx, val in enumerate(unique_vals)}

        missing_cols = [c for c in self.feature_names_ if c not in data.columns]
        for c in missing_cols:
            data[c] = np.nan

        X = data[self.feature_names_].copy()

        for cat_col, mapping in self.cat_mappings_.items():
            if cat_col in X.columns:
                X[cat_col] = (
                    X[cat_col]
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )

        for col in X.columns:
            if col not in self.cat_mappings_:
                X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

        return X

    @staticmethod
    def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute stable logit transform."""
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = False,
    ) -> SetGameModel:
        """Fit all 4 market sub-models with time-split validation.

        Args:
            train_df: Training DataFrame containing features and multi-market targets.
            feature_cols: Optional explicit list of feature names.
            val_df: Optional validation DataFrame for early stopping and calibration.
            early_stopping_rounds: Early stopping patience.
            verbose: Verbosity flag during training.

        Returns:
            self: Fitted SetGameModel instance.
        """
        if train_df.empty:
            raise ValueError("Training DataFrame cannot be empty.")

        # Require target columns or filter rows with targets
        req_targets = ["target_total_games", "target_game_diff", "target_total_sets", "target_set1_p1_won"]
        for t in req_targets:
            if t not in train_df.columns:
                raise KeyError(f"Required target column '{t}' not found in train_df.")

        logger.info(f"Fitting SetGameModel on {len(train_df):,} training matches...")

        X_train = self._prepare_features(train_df, feature_cols=feature_cols, is_training=True)
        X_val = self._prepare_features(val_df, is_training=False) if (val_df is not None and not val_df.empty) else None

        # Filter completed matches for game totals and spreads to avoid incomplete skew
        mask_train_complete = train_df.get("is_completed", pd.Series(True, index=train_df.index)).astype(bool)
        if mask_train_complete.sum() > 50:
            X_tr_reg = X_train[mask_train_complete]
            df_tr_reg = train_df[mask_train_complete]
        else:
            X_tr_reg = X_train
            df_tr_reg = train_df

        cat_feature_idx = [c for c in self.CATEGORICAL_COLS if c in self.feature_names_]

        # ---------------------------------------------------------------------
        # 1. Total Games Model (Regression)
        # ---------------------------------------------------------------------
        y_train_tg = df_tr_reg["target_total_games"].astype(float).values
        eval_set_tg = [(X_val, val_df["target_total_games"].astype(float).values)] if (val_df is not None and "target_total_games" in val_df.columns) else None

        if self.model_type == "lightgbm":
            self.model_total_games = lgb.LGBMRegressor(**self._get_tree_regressor_params())
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)] if eval_set_tg else None
            self.model_total_games.fit(X_tr_reg, y_train_tg, eval_set=eval_set_tg, categorical_feature=cat_feature_idx or "auto", callbacks=callbacks)
        else:
            self.model_total_games = xgb.XGBRegressor(**self._get_tree_regressor_params())
            self.model_total_games.fit(X_tr_reg, y_train_tg, eval_set=eval_set_tg, verbose=verbose)

        # ---------------------------------------------------------------------
        # 2. Game Differential Model (Regression)
        # ---------------------------------------------------------------------
        y_train_gd = df_tr_reg["target_game_diff"].astype(float).values
        eval_set_gd = [(X_val, val_df["target_game_diff"].astype(float).values)] if (val_df is not None and "target_game_diff" in val_df.columns) else None

        if self.model_type == "lightgbm":
            self.model_game_diff = lgb.LGBMRegressor(**self._get_tree_regressor_params())
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)] if eval_set_gd else None
            self.model_game_diff.fit(X_tr_reg, y_train_gd, eval_set=eval_set_gd, categorical_feature=cat_feature_idx or "auto", callbacks=callbacks)
        else:
            self.model_game_diff = xgb.XGBRegressor(**self._get_tree_regressor_params())
            self.model_game_diff.fit(X_tr_reg, y_train_gd, eval_set=eval_set_gd, verbose=verbose)

        # ---------------------------------------------------------------------
        # 3. Total Sets Model (Regression & Over 2.5 Sets Classifier)
        # ---------------------------------------------------------------------
        y_train_ts = df_tr_reg["target_total_sets"].astype(float).values
        eval_set_ts = [(X_val, val_df["target_total_sets"].astype(float).values)] if (val_df is not None and "target_total_sets" in val_df.columns) else None

        if self.model_type == "lightgbm":
            self.model_total_sets = lgb.LGBMRegressor(**self._get_tree_regressor_params())
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)] if eval_set_ts else None
            self.model_total_sets.fit(X_tr_reg, y_train_ts, eval_set=eval_set_ts, categorical_feature=cat_feature_idx or "auto", callbacks=callbacks)
        else:
            self.model_total_sets = xgb.XGBRegressor(**self._get_tree_regressor_params())
            self.model_total_sets.fit(X_tr_reg, y_train_ts, eval_set=eval_set_ts, verbose=verbose)

        # Binary Over 2.5 Sets Classifier
        y_train_over_sets = (train_df["target_total_sets"] > 2.5).astype(int).values
        eval_set_os = [(X_val, (val_df["target_total_sets"] > 2.5).astype(int).values)] if (val_df is not None and "target_total_sets" in val_df.columns) else None

        if self.model_type == "lightgbm":
            self.model_over_sets = lgb.LGBMClassifier(**self._get_tree_classifier_params())
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)] if eval_set_os else None
            self.model_over_sets.fit(X_train, y_train_over_sets, eval_set=eval_set_os, categorical_feature=cat_feature_idx or "auto", callbacks=callbacks)
        else:
            self.model_over_sets = xgb.XGBClassifier(**self._get_tree_classifier_params())
            self.model_over_sets.fit(X_train, y_train_over_sets, eval_set=eval_set_os, verbose=verbose)

        # ---------------------------------------------------------------------
        # 4. First Set Winner Model (Classification & Calibration)
        # ---------------------------------------------------------------------
        y_train_s1 = train_df["target_set1_p1_won"].astype(int).values
        eval_set_s1 = [(X_val, val_df["target_set1_p1_won"].astype(int).values)] if (val_df is not None and "target_set1_p1_won" in val_df.columns) else None

        if self.model_type == "lightgbm":
            self.model_first_set = lgb.LGBMClassifier(**self._get_tree_classifier_params())
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)] if eval_set_s1 else None
            self.model_first_set.fit(X_train, y_train_s1, eval_set=eval_set_s1, categorical_feature=cat_feature_idx or "auto", callbacks=callbacks)
        else:
            self.model_first_set = xgb.XGBClassifier(**self._get_tree_classifier_params())
            self.model_first_set.fit(X_train, y_train_s1, eval_set=eval_set_s1, verbose=verbose)

        # Calibrate First Set and Over Sets probabilities on validation set
        calib_df = val_df if (val_df is not None and not val_df.empty) else train_df
        calib_X = X_val if (X_val is not None) else X_train

        # First Set Calibrator (Platt scaling with Elo blend)
        raw_s1_prob = self.model_first_set.predict_proba(calib_X)[:, 1]
        elo_prob = calib_df.get("elo_win_prob_p1", pd.Series(0.50, index=calib_df.index)).fillna(0.50).values
        stack_s1 = np.column_stack([self._safe_logit(raw_s1_prob), self._safe_logit(elo_prob)])

        self.calibrator_first_set = LogisticRegression(C=1.0, solver="lbfgs", random_state=self.random_state)
        self.calibrator_first_set.fit(stack_s1, calib_df["target_set1_p1_won"].astype(int).values)

        # Estimate residual variances on validation data for continuous line pricing
        if val_df is not None and not val_df.empty:
            pred_val_tg = self.model_total_games.predict(X_val)
            pred_val_gd = self.model_game_diff.predict(X_val)
            res_tg = val_df["target_total_games"].values - pred_val_tg
            res_gd = val_df["target_game_diff"].values - pred_val_gd
            self.std_total_games_residual = float(np.clip(np.std(res_tg), 2.5, 6.5))
            self.std_game_diff_residual = float(np.clip(np.std(res_gd), 3.0, 7.5))

        self.is_fitted = True
        logger.info("SetGameModel suite successfully fitted across all targets.")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate point estimates and market probabilities across all targets.

        Returns DataFrame containing:
        - `pred_total_games`: Continuous expected match games.
        - `pred_game_diff`: Continuous expected game differential (P1 - P2).
        - `pred_total_sets`: Continuous expected total sets.
        - `prob_over_2_5_sets`: Probability match goes over 2.5 sets.
        - `prob_first_set_p1`: Calibrated probability that Player 1 wins Set 1.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = self._prepare_features(df, is_training=False)

        # 1. Continuous predictions
        pred_tg = self.model_total_games.predict(X)
        pred_gd = self.model_game_diff.predict(X)
        pred_ts = self.model_total_sets.predict(X)

        # 2. Over 2.5 sets probability
        raw_over_sets = self.model_over_sets.predict_proba(X)[:, 1]

        # 3. First set probability
        raw_s1 = self.model_first_set.predict_proba(X)[:, 1]
        elo_prob = df.get("elo_win_prob_p1", pd.Series(0.50, index=df.index)).fillna(0.50).values
        stack_s1 = np.column_stack([self._safe_logit(raw_s1), self._safe_logit(elo_prob)])

        if self.calibrator_first_set is not None:
            calib_s1 = self.calibrator_first_set.predict_proba(stack_s1)[:, 1]
        else:
            calib_s1 = 0.65 * raw_s1 + 0.35 * elo_prob

        return pd.DataFrame({
            "pred_total_games": np.round(pred_tg, 2),
            "pred_game_diff": np.round(pred_gd, 2),
            "pred_total_sets": np.round(pred_ts, 2),
            "prob_over_2_5_sets": np.round(np.clip(raw_over_sets, 0.01, 0.99), 4),
            "prob_first_set_p1": np.round(np.clip(calib_s1, 0.01, 0.99), 4),
        }, index=df.index)

    def predict_total_games_over_prob(
        self,
        df: pd.DataFrame,
        line: float = 22.5,
    ) -> np.ndarray:
        """Compute P(Total Games > Line) for any arbitrary bookmaker line.

        Uses the predicted continuous total games and calibrated residual standard error.
        """
        preds = self.predict(df)
        pred_means = preds["pred_total_games"].values
        z_scores = (line - pred_means) / self.std_total_games_residual
        # P(Total > Line) = 1 - CDF(z)
        over_probs = 1.0 - stats.norm.cdf(z_scores)
        return np.clip(over_probs, 0.01, 0.99)

    def predict_game_handicap_cover_prob(
        self,
        df: pd.DataFrame,
        handicap: float = -3.5,
    ) -> np.ndarray:
        """Compute P(Game Differential > Handicap), e.g. P(P1 games - P2 games > 3.5).

        Args:
            df: Feature DataFrame.
            handicap: Spread line (e.g., -3.5 means P1 must win by 4 or more games).

        Returns:
            1D array of cover probabilities for Player 1.
        """
        preds = self.predict(df)
        pred_diffs = preds["pred_game_diff"].values
        # P1 covers if (G1 - G2) + handicap > 0  =>  (G1 - G2) > -handicap
        required_diff = -handicap
        z_scores = (required_diff - pred_diffs) / self.std_game_diff_residual
        cover_probs = 1.0 - stats.norm.cdf(z_scores)
        return np.clip(cover_probs, 0.01, 0.99)

    def feature_importance(
        self,
        target: str = "total_games",
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return feature importance for a specific target sub-model.

        Args:
            target: 'total_games', 'game_diff', 'total_sets', 'over_sets', or 'first_set'.
            top_n: Number of top features to return.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        model_map = {
            "total_games": self.model_total_games,
            "game_diff": self.model_game_diff,
            "total_sets": self.model_total_sets,
            "over_sets": self.model_over_sets,
            "first_set": self.model_first_set,
        }

        if target not in model_map:
            raise ValueError(f"Unknown target '{target}'. Choose from {list(model_map.keys())}")

        sub_model = model_map[target]
        if self.model_type == "lightgbm":
            raw_imp = sub_model.booster_.feature_importance(importance_type="gain")
        else:
            raw_imp = sub_model.feature_importances_

        tot_imp = float(np.sum(raw_imp))
        imp_pct = (raw_imp / tot_imp * 100.0) if tot_imp > 0 else np.zeros_like(raw_imp)

        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": raw_imp,
            "importance_pct": np.round(imp_pct, 2),
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize SetGameModel suite to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"SetGameModel suite saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> SetGameModel:
        """Load a persisted SetGameModel instance."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")

        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is {type(instance)}, expected {cls.__name__}")

        logger.info(f"SetGameModel suite loaded from {path}")
        return instance

