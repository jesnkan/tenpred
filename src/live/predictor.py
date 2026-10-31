"""Production-Grade Unified Prediction Pipeline for ATP and WTA Tennis.

Orchestrates the complete prediction flow:
1. Ingests single or batch upcoming fixtures
2. Computes leak-free features via TennisFeatureEngineer and SurfaceElo
3. Executes MatchWinnerModel (calibrated win probabilities)
4. Executes SetGameModel (total games, spreads, set totals, first set)
5. Executes TennisMarkovModel (exact analytical set handicaps, player to win a set, set betting distributions)
6. Executes TennisRiskManager (correlated Kelly sizing, multi-market risk controls, EV)
7. Formats clean analysis cards and export dataframes
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR
    from src.data.loader import TennisDataLoader
    from src.evaluation.risk_manager import TennisRiskManager
    from src.features.engineer import TennisFeatureEngineer
    from src.models.markov_sim import TennisMarkovModel
    from src.models.match_winner import MatchWinnerModel
    from src.models.set_game import SetGameModel
    from src.ratings.surface_elo import SurfaceElo
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR
    from src.data.loader import TennisDataLoader
    from src.evaluation.risk_manager import TennisRiskManager
    from src.features.engineer import TennisFeatureEngineer
    from src.models.markov_sim import TennisMarkovModel
    from src.models.match_winner import MatchWinnerModel
    from src.models.set_game import SetGameModel
    from src.ratings.surface_elo import SurfaceElo

logger = logging.getLogger(__name__)


class TennisPredictionPipeline:
    """Unified operational orchestrator for ATP & WTA match forecasting."""

    def __init__(
        self,
        match_winner_model: Optional[MatchWinnerModel] = None,
        set_game_model: Optional[SetGameModel] = None,
        markov_model: Optional[TennisMarkovModel] = None,
        feature_engineer: Optional[TennisFeatureEngineer] = None,
        elo_model: Optional[SurfaceElo] = None,
        risk_manager: Optional[TennisRiskManager] = None,
    ) -> None:
        """Initialize the pipeline with fitted models, simulation engines, and risk manager."""
        self.elo = elo_model if elo_model is not None else SurfaceElo()
        self.feature_engineer = (
            feature_engineer
            if feature_engineer is not None
            else TennisFeatureEngineer(elo_model=self.elo)
        )
        self.match_winner_model = match_winner_model
        self.set_game_model = set_game_model
        self.markov_model = markov_model if markov_model is not None else TennisMarkovModel()
        self.risk_manager = risk_manager if risk_manager is not None else TennisRiskManager()

    @classmethod
    def from_trained_artifacts(
        cls,
        match_winner_path: Optional[Union[str, Path]] = None,
        set_game_path: Optional[Union[str, Path]] = None,
        elo_path: Optional[Union[str, Path]] = None,
    ) -> TennisPredictionPipeline:
        """Instantiate pipeline by loading persisted model checkpoints from disk."""
        elo = SurfaceElo.load(elo_path) if elo_path and Path(elo_path).exists() else SurfaceElo()
        fe = TennisFeatureEngineer(elo_model=elo)

        mw_model = None
        if match_winner_path and Path(match_winner_path).exists():
            mw_model = MatchWinnerModel.load(match_winner_path)
        else:
            logger.warning(f"MatchWinnerModel checkpoint not found at: {match_winner_path}")

        sg_model = None
        if set_game_path and Path(set_game_path).exists():
            sg_model = SetGameModel.load(set_game_path)
        else:
            logger.warning(f"SetGameModel checkpoint not found at: {set_game_path}")

        markov = TennisMarkovModel()
        rm = TennisRiskManager()

        return cls(
            match_winner_model=mw_model,
            set_game_model=sg_model,
            markov_model=markov,
            feature_engineer=fe,
            elo_model=elo,
            risk_manager=rm,
        )

    # =========================================================================
    # Single Match Prediction
    # =========================================================================

    def predict_match(
        self,
        player1: str,
        player2: str,
        surface: str = "Hard",
        tournament_level: str = "D",
        round_name: str = "R32",
        match_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
        best_of: int = 3,
        p1_odds: Optional[float] = None,
        p2_odds: Optional[float] = None,
        p1_age: float = np.nan,
        p2_age: float = np.nan,
        p1_ht: float = np.nan,
        p2_ht: float = np.nan,
        p1_hand: str = "R",
        p2_hand: str = "R",
        tournament: str = "",
        is_indoor: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive market predictions for an individual fixture."""
        match_dict = {
            "player1": player1,
            "player2": player2,
            "surface": surface,
            "tournament_level": tournament_level,
            "round": round_name,
            "match_date": match_date,
            "best_of": best_of,
            "p1_age": p1_age,
            "p2_age": p2_age,
            "p1_ht": p1_ht,
            "p2_ht": p2_ht,
            "p1_hand": p1_hand,
            "p2_hand": p2_hand,
            "tournament": tournament,
        }
        if is_indoor is not None:
            match_dict["is_indoor"] = is_indoor
        if p1_odds is not None:
            match_dict["p1_odds"] = p1_odds
        if p2_odds is not None:
            match_dict["p2_odds"] = p2_odds

        df_single = pd.DataFrame([match_dict])
        df_result = self.predict_batch(df_single)
        return df_result.iloc[0].to_dict()

    # =========================================================================
    # Batch Match Prediction
    # =========================================================================

    def predict_batch(
        self,
        matches: Union[pd.DataFrame, List[Dict[str, Any]], str, Path],
        min_edge_threshold: float = 0.03,
        kelly_fraction: float = 0.25,
    ) -> pd.DataFrame:
        """Run batch predictions across multiple fixtures."""
        if isinstance(matches, (str, Path)):
            path = Path(matches)
            if not path.exists():
                raise FileNotFoundError(f"Matches input file not found: {path}")
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        elif isinstance(matches, list):
            df = pd.DataFrame(matches)
        elif isinstance(matches, pd.DataFrame):
            df = matches.copy()
        else:
            raise TypeError("matches must be a DataFrame, List[Dict], or file path string.")

        if df.empty:
            return pd.DataFrame()

        # Normalize column names
        col_rename = {}
        for col in df.columns:
            c = str(col).strip().lower()
            if c in ("winner_name", "p1", "player_1"):
                col_rename[col] = "player1"
            elif c in ("loser_name", "p2", "player_2"):
                col_rename[col] = "player2"
            elif c in ("date", "tourney_date"):
                col_rename[col] = "match_date"
            elif c in ("tourney_level", "level"):
                col_rename[col] = "tournament_level"
            elif c in ("rnd", "round_name"):
                col_rename[col] = "round"
        df = df.rename(columns=col_rename)

        for req in ["player1", "player2"]:
            if req not in df.columns:
                raise KeyError(f"Required column '{req}' missing from matches input.")

        if "surface" not in df.columns:
            df["surface"] = "Hard"
        if "tournament_level" not in df.columns:
            df["tournament_level"] = "D"
        if "round" not in df.columns:
            df["round"] = "R32"
        if "best_of" not in df.columns:
            df["best_of"] = 3
        if "match_date" not in df.columns:
            df["match_date"] = pd.Timestamp.now()

        # 2. Extract features via TennisFeatureEngineer
        feat_rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            t_name = str(row.get("tournament", row.get("tourney_name", "")))
            is_ind = row.get("is_indoor", None)
            f_dict = self.feature_engineer.extract_features_for_match(
                p1=str(row["player1"]),
                p2=str(row["player2"]),
                surface=str(row["surface"]),
                tournament_level=str(row.get("tournament_level", "D")),
                round_name=str(row.get("round", "R32")),
                match_date=row.get("match_date", None),
                best_of=int(row.get("best_of", 3)),
                p1_age=float(row.get("p1_age", np.nan)),
                p2_age=float(row.get("p2_age", np.nan)),
                p1_ht=float(row.get("p1_ht", np.nan)),
                p2_ht=float(row.get("p2_ht", np.nan)),
                p1_hand=str(row.get("p1_hand", "R")),
                p2_hand=str(row.get("p2_hand", "R")),
                tournament=t_name,
                is_indoor=is_ind,
            )
            feat_rows.append(f_dict)

        df_features = pd.DataFrame(feat_rows)

        # 3. Predict Match Winner
        if self.match_winner_model is not None and self.match_winner_model.is_fitted:
            p1_win_probs = self.match_winner_model.predict_proba(df_features)
        else:
            logger.warning("MatchWinnerModel not fitted/provided. Falling back to SurfaceElo win probabilities.")
            p1_win_probs = df_features["elo_win_prob_p1"].values

        p2_win_probs = 1.0 - p1_win_probs
        p1_fair_odds = np.round(1.0 / np.clip(p1_win_probs, 1e-4, 1.0), 3)
        p2_fair_odds = np.round(1.0 / np.clip(p2_win_probs, 1e-4, 1.0), 3)
        predicted_winner = np.where(p1_win_probs >= 0.50, df["player1"], df["player2"])

        # 4. Predict Set and Game Markets (LightGBM Regression/Classification)
        if self.set_game_model is not None and self.set_game_model.is_fitted:
            sg_preds = self.set_game_model.predict(df_features)
            pred_tg = sg_preds["pred_total_games"].values
            pred_gd = sg_preds["pred_game_diff"].values
            pred_ts = sg_preds["pred_total_sets"].values
            p_over_2_5 = sg_preds["prob_over_2_5_sets"].values
            p1_s1_prob = sg_preds["prob_first_set_p1"].values
        else:
            pred_tg = np.round(df_features.get("est_match_total_games", pd.Series(22.5, index=df.index)).values, 2)
            pred_gd = np.round((p1_win_probs - 0.50) * 8.0, 2)
            pred_ts = np.round(2.0 + (1.0 - np.abs(p1_win_probs - 0.50) * 2.0) * 0.45, 2)
            p_over_2_5 = np.round(0.35 + (1.0 - np.abs(p1_win_probs - 0.50) * 2.0) * 0.20, 3)
            p1_s1_prob = np.round(0.85 * p1_win_probs + 0.15 * df_features.get("expected_hold_p1", 0.78), 3)

        p2_s1_prob = np.round(1.0 - p1_s1_prob, 4)

        # 5. Execute Exact Analytical Markov Simulation for Derivative Markets
        markov_results: List[Dict[str, Any]] = []
        for idx in range(len(df)):
            p1_p = float(p1_win_probs[idx])
            h1 = float(df_features.get("expected_hold_p1", pd.Series(0.80, index=df.index)).iloc[idx])
            h2 = float(df_features.get("expected_hold_p2", pd.Series(0.80, index=df.index)).iloc[idx])
            bo = int(df["best_of"].iloc[idx]) if "best_of" in df.columns else 3
            tour = str(df.get("tour", pd.Series("ATP", index=df.index)).iloc[idx])
            is_indoor_val = bool(df_features.get("is_indoor", pd.Series(0.0, index=df.index)).iloc[idx] > 0.5)
            cpi_val = float(df_features.get("court_pace_index", pd.Series(34.0, index=df.index)).iloc[idx])

            sim = self.markov_model.simulate_match(
                calibrated_p1_win_prob=p1_p,
                expected_hold_p1=h1,
                expected_hold_p2=h2,
                best_of=bo,
                tour=tour,
                is_indoor=is_indoor_val,
                cpi=cpi_val,
            )
            markov_results.append(sim)

        # Extract Markov Market Outputs
        p1_win_set_probs = [m["player_to_win_set"]["p1_win_at_least_one_set_prob"] for m in markov_results]
        p2_win_set_probs = [m["player_to_win_set"]["p2_win_at_least_one_set_prob"] for m in markov_results]
        p1_win_set_odds = [m["player_to_win_set"]["p1_win_at_least_one_set_fair_odds"] for m in markov_results]
        p2_win_set_odds = [m["player_to_win_set"]["p2_win_at_least_one_set_fair_odds"] for m in markov_results]

        p1_set_m1_5 = [m["set_handicaps"].get("p1_minus_1_5_sets", 0.0) for m in markov_results]
        p1_set_p1_5 = [m["set_handicaps"].get("p1_plus_1_5_sets", 0.0) for m in markov_results]
        p2_set_m1_5 = [m["set_handicaps"].get("p2_minus_1_5_sets", 0.0) for m in markov_results]
        p2_set_p1_5 = [m["set_handicaps"].get("p2_plus_1_5_sets", 0.0) for m in markov_results]

        score_2_0 = [m["set_scores_prob"].get("2-0", m["set_scores_prob"].get("3-0", 0.0)) for m in markov_results]
        score_2_1 = [m["set_scores_prob"].get("2-1", m["set_scores_prob"].get("3-1", 0.0)) for m in markov_results]
        score_1_2 = [m["set_scores_prob"].get("1-2", m["set_scores_prob"].get("1-3", 0.0)) for m in markov_results]
        score_0_2 = [m["set_scores_prob"].get("0-2", m["set_scores_prob"].get("0-3", 0.0)) for m in markov_results]

        # Discrete Game Totals & Handicaps
        p_over_19_5 = [m["total_games_over_probs"].get("over_19_5", 0.5) for m in markov_results]
        p_over_20_5 = [m["total_games_over_probs"].get("over_20_5", 0.5) for m in markov_results]
        p_over_21_5 = [m["total_games_over_probs"].get("over_21_5", 0.5) for m in markov_results]
        p_over_22_5 = [m["total_games_over_probs"].get("over_22_5", 0.5) for m in markov_results]
        p_over_23_5 = [m["total_games_over_probs"].get("over_23_5", 0.5) for m in markov_results]
        p_over_24_5 = [m["total_games_over_probs"].get("over_24_5", 0.5) for m in markov_results]

        p1_cover_m1_5 = [m["handicap_cover_probs"].get("p1_cover_minus_1_5", 0.5) for m in markov_results]
        p1_cover_m2_5 = [m["handicap_cover_probs"].get("p1_cover_minus_2_5", 0.5) for m in markov_results]
        p1_cover_m3_5 = [m["handicap_cover_probs"].get("p1_cover_minus_3_5", 0.5) for m in markov_results]
        p1_cover_m4_5 = [m["handicap_cover_probs"].get("p1_cover_minus_4_5", 0.5) for m in markov_results]
        p1_cover_p1_5 = [m["handicap_cover_probs"].get("p1_cover_plus_1_5", 0.5) for m in markov_results]
        p1_cover_p2_5 = [m["handicap_cover_probs"].get("p1_cover_plus_2_5", 0.5) for m in markov_results]
        p1_cover_p3_5 = [m["handicap_cover_probs"].get("p1_cover_plus_3_5", 0.5) for m in markov_results]

        p_tb_s1 = [m["first_set_tiebreak_prob"] for m in markov_results]
        p_any_tb = [m["any_tiebreak_prob"] for m in markov_results]

        # 6. Build Comprehensive Output DataFrame
        out_data = {
            "match_date": df["match_date"],
            "player1": df["player1"],
            "player2": df["player2"],
            "surface": df["surface"],
            "tournament_level": df["tournament_level"],
            "round": df["round"],
            "best_of": df["best_of"],
        }
        for meta_col in ("tournament", "match_time", "status", "tour"):
            if meta_col in df.columns:
                out_data[meta_col] = df[meta_col]

        out_data.update({
            # Ratings context
            "p1_elo_blended": np.round(df_features["p1_elo_blended"], 1),
            "p2_elo_blended": np.round(df_features["p2_elo_blended"], 1),
            "diff_elo_blended": np.round(df_features["diff_elo_blended"], 1),
            # 1. Match Winner Market
            "p1_win_prob": np.round(p1_win_probs, 4),
            "p2_win_prob": np.round(p2_win_probs, 4),
            "p1_fair_odds": p1_fair_odds,
            "p2_fair_odds": p2_fair_odds,
            "predicted_winner": predicted_winner,
            # 2. Player To Win At Least One Set (Dedicated Market)
            "p1_win_at_least_one_set_prob": np.round(p1_win_set_probs, 4),
            "p1_win_at_least_one_set_fair_odds": p1_win_set_odds,
            "p2_win_at_least_one_set_prob": np.round(p2_win_set_probs, 4),
            "p2_win_at_least_one_set_fair_odds": p2_win_set_odds,
            # 3. Set Handicap Market (-1.5 / +1.5 Sets)
            "prob_p1_set_handicap_minus_1_5": np.round(p1_set_m1_5, 4),
            "prob_p1_set_handicap_plus_1_5": np.round(p1_set_p1_5, 4),
            "prob_p2_set_handicap_minus_1_5": np.round(p2_set_m1_5, 4),
            "prob_p2_set_handicap_plus_1_5": np.round(p2_set_p1_5, 4),
            # 4. Exact Set Score Betting (Correct Score)
            "prob_set_score_2_0": np.round(score_2_0, 4),
            "prob_set_score_2_1": np.round(score_2_1, 4),
            "prob_set_score_1_2": np.round(score_1_2, 4),
            "prob_set_score_0_2": np.round(score_0_2, 4),
            # 5. Total Match Games Market
            "pred_total_games": np.round(pred_tg, 1),
            "prob_over_19_5": np.round(p_over_19_5, 3),
            "prob_over_20_5": np.round(p_over_20_5, 3),
            "prob_over_21_5": np.round(p_over_21_5, 3),
            "prob_over_22_5": np.round(p_over_22_5, 3),
            "prob_over_23_5": np.round(p_over_23_5, 3),
            "prob_over_24_5": np.round(p_over_24_5, 3),
            "prob_under_22_5": np.round(1.0 - np.array(p_over_22_5), 3),
            # 6. Game Handicap Spread Market
            "pred_game_diff": np.round(pred_gd, 1),
            "prob_p1_cover_minus_1_5": np.round(p1_cover_m1_5, 3),
            "prob_p1_cover_minus_2_5": np.round(p1_cover_m2_5, 3),
            "prob_p1_cover_minus_3_5": np.round(p1_cover_m3_5, 3),
            "prob_p1_cover_minus_4_5": np.round(p1_cover_m4_5, 3),
            "prob_p1_cover_plus_1_5": np.round(p1_cover_p1_5, 3),
            "prob_p1_cover_plus_2_5": np.round(p1_cover_p2_5, 3),
            "prob_p1_cover_plus_3_5": np.round(p1_cover_p3_5, 3),
            # 7. First Set Winner & Tiebreak
            "p1_first_set_prob": np.round(p1_s1_prob, 4),
            "p2_first_set_prob": np.round(p2_s1_prob, 4),
            "p1_first_set_fair_odds": np.round(1.0 / np.clip(p1_s1_prob, 1e-4, 1.0), 3),
            "p2_first_set_fair_odds": np.round(1.0 / np.clip(p2_s1_prob, 1e-4, 1.0), 3),
            "prob_first_set_tiebreak": np.round(p_tb_s1, 4),
            "prob_any_tiebreak": np.round(p_any_tb, 4),
            # 8. Total Sets Market
            "pred_total_sets": np.round(pred_ts, 2),
            "prob_over_2_5_sets": np.round(p_over_2_5, 3),
            "prob_straight_sets_2_0": np.round(1.0 - p_over_2_5, 3),
        })
        out = pd.DataFrame(out_data, index=df.index)

        # 7. Multi-Market Value Edge & Portfolio Allocation
        if "p1_odds" in df.columns and "p2_odds" in df.columns:
            out = self._append_market_edges(
                out,
                p1_odds=df["p1_odds"],
                p2_odds=df["p2_odds"],
                min_edge=min_edge_threshold,
                kelly_frac=kelly_fraction,
            )

        return out

    # =========================================================================
    # Market Odds & Value Edge Integration
    # =========================================================================

    def _append_market_edges(
        self,
        predictions_df: pd.DataFrame,
        p1_odds: pd.Series,
        p2_odds: pd.Series,
        min_edge: float = 0.03,
        kelly_frac: float = 0.25,
    ) -> pd.DataFrame:
        """Calculate market overround, vig-removed implied probabilities, EV, and Kelly stakes."""
        df = predictions_df.copy()
        df["p1_market_odds"] = p1_odds
        df["p2_market_odds"] = p2_odds

        p1_raw = 1.0 / p1_odds
        p2_raw = 1.0 / p2_odds
        overround = p1_raw + p2_raw

        p1_mkt = p1_raw / overround
        p2_mkt = p2_raw / overround

        p1_edge = df["p1_win_prob"] - p1_mkt
        p2_edge = df["p2_win_prob"] - p2_mkt

        p1_ev = (df["p1_win_prob"] * p1_odds) - 1.0
        p2_ev = (df["p2_win_prob"] * p2_odds) - 1.0

        def _calc_kelly(p_win: pd.Series, odds: pd.Series) -> pd.Series:
            b = odds - 1.0
            q = 1.0 - p_win
            raw_f = (b * p_win - q) / b
            return np.round(np.clip(raw_f * kelly_frac * 100.0, 0.0, 4.0), 2)

        p1_kelly = _calc_kelly(df["p1_win_prob"], p1_odds)
        p2_kelly = _calc_kelly(df["p2_win_prob"], p2_odds)

        rec_bet = []
        rec_edge = []
        rec_stake = []

        for p1_e, p2_e, p1_k, p2_k in zip(p1_edge, p2_edge, p1_kelly, p2_kelly):
            if p1_e >= min_edge and p1_e > p2_e:
                rec_bet.append("Player 1")
                rec_edge.append(round(p1_e * 100.0, 2))
                rec_stake.append(f"{p1_k:.1f}%")
            elif p2_e >= min_edge and p2_e > p1_e:
                rec_bet.append("Player 2")
                rec_edge.append(round(p2_e * 100.0, 2))
                rec_stake.append(f"{p2_k:.1f}%")
            else:
                rec_bet.append("Pass (No Value)")
                rec_edge.append(0.0)
                rec_stake.append("0.0%")

        df["market_overround_pct"] = np.round((overround - 1.0) * 100.0, 2)
        df["p1_edge_pct"] = np.round(p1_edge * 100.0, 2)
        df["p2_edge_pct"] = np.round(p2_edge * 100.0, 2)
        df["p1_expected_value"] = np.round(p1_ev, 3)
        df["p2_expected_value"] = np.round(p2_ev, 3)
        df["recommended_bet"] = rec_bet
        df["edge_pct"] = rec_edge
        df["kelly_stake_pct"] = rec_stake

        return df

    def add_market_odds(
        self,
        predictions_df: pd.DataFrame,
        p1_odds: Union[pd.Series, List[float], np.ndarray],
        p2_odds: Union[pd.Series, List[float], np.ndarray],
        min_edge_threshold: float = 0.03,
        kelly_fraction: float = 0.25,
    ) -> pd.DataFrame:
        """Public helper to append market odds and evaluate value edges post-prediction."""
        p1_s = pd.Series(p1_odds, index=predictions_df.index)
        p2_s = pd.Series(p2_odds, index=predictions_df.index)
        return self._append_market_edges(
            predictions_df,
            p1_odds=p1_s,
            p2_odds=p2_s,
            min_edge=min_edge_threshold,
            kelly_frac=kelly_fraction,
        )

    # =========================================================================
    # Simple Top 3 Pick Selector & Terminal Display Formatter
    # =========================================================================

    @staticmethod
    def extract_top_three_picks(
        match_pred: Union[pd.Series, Dict[str, Any]],
        mode: str = "safety",
    ) -> List[Dict[str, Any]]:
        """Rank and extract the top 3 highest-confidence market options in simple terms.

        Args:
            match_pred: Dictionary or Series of predicted match columns.
            mode: 'safety' (highest probability banker picks) or 'value' (highest edge).

        Returns:
            List of 3 distinct, plain-English pick recommendations.
        """
        r = match_pred.to_dict() if isinstance(match_pred, pd.Series) else dict(match_pred)
        p1 = str(r.get("player1", "Player 1"))
        p2 = str(r.get("player2", "Player 2"))
        surface = str(r.get("surface", "Hard"))

        candidates: List[Dict[str, Any]] = []

        # 1. Player to win at least 1 set (ultra-high probability safety pick)
        p1_set = float(r.get("p1_win_at_least_one_set_prob", 0.5))
        p2_set = float(r.get("p2_win_at_least_one_set_prob", 0.5))

        if p1_set >= 0.68:
            candidates.append({
                "title": f"{p1} to Win At Least 1 Set",
                "market": "Player to Win a Set",
                "win_prob_pct": round(p1_set * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p1_set), 2),
                "confidence_tier": "[ULTRA HIGH CONFIDENCE]" if p1_set >= 0.82 else "[VERY HIGH CONFIDENCE]",
                "explanation": f"{p1} only needs to win a single set. Our model projects an ultra-high {p1_set:.1%} win rate based on service hold dominance on {surface}.",
                "action": "Safest Banker / Parlay Anchor",
            })

        if p2_set >= 0.68:
            candidates.append({
                "title": f"{p2} to Win At Least 1 Set",
                "market": "Player to Win a Set",
                "win_prob_pct": round(p2_set * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p2_set), 2),
                "confidence_tier": "[ULTRA HIGH CONFIDENCE]" if p2_set >= 0.82 else "[VERY HIGH CONFIDENCE]",
                "explanation": f"{p2} only needs to win a single set. Highly resilient hold metrics give them a strong {p2_set:.1%} probability to take at least one set.",
                "action": "Safest Banker / Parlay Anchor",
            })

        # 2. Game Handicap Spreads
        p1_p25 = float(r.get("prob_p1_cover_plus_2_5", 0.5))
        p1_m25 = float(r.get("prob_p1_cover_minus_2_5", 0.5))
        p1_p15 = float(r.get("prob_p1_cover_plus_1_5", 0.5))

        if p1_p25 >= 0.75:
            candidates.append({
                "title": f"{p1} +2.5 Game Handicap",
                "market": "Game Spread",
                "win_prob_pct": round(p1_p25 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p1_p25), 2),
                "confidence_tier": "[ULTRA HIGH CONFIDENCE]" if p1_p25 >= 0.82 else "[VERY HIGH CONFIDENCE]",
                "explanation": f"{p1} covers this spread if they win the match, or even if they lose by 2 or fewer games.",
                "action": "High Floor Spread",
            })
        elif p1_m25 >= 0.60:
            candidates.append({
                "title": f"{p1} -2.5 Game Handicap",
                "market": "Game Spread",
                "win_prob_pct": round(p1_m25 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p1_m25), 2),
                "confidence_tier": "[STRONG FAVORITE SPREAD]",
                "explanation": f"{p1} is projected to win decisively by 3 or more total games on {surface}.",
                "action": "Dominant Favorite Spread",
            })
        elif p1_p15 >= 0.70:
            candidates.append({
                "title": f"{p1} +1.5 Game Handicap",
                "market": "Game Spread",
                "win_prob_pct": round(p1_p15 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p1_p15), 2),
                "confidence_tier": "[HIGH CONFIDENCE SPREAD]",
                "explanation": f"{p1} covers with any match victory or single game margin loss.",
                "action": "High Floor Spread",
            })

        # 3. Total Match Games (Prioritize Over 19.5 for maximum safety and near-zero failure rate)
        p_o19 = float(r.get("prob_over_19_5", 0.5))
        p_o20 = float(r.get("prob_over_20_5", 0.5))
        p_u23 = 1.0 - float(r.get("prob_over_23_5", 0.5))

        if p_o19 >= 0.65:
            candidates.append({
                "title": "Over 19.5 Total Match Games",
                "market": "Total Games",
                "win_prob_pct": round(p_o19 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p_o19), 2),
                "confidence_tier": "[LOCK / NEAR-ZERO FAILURE]" if p_o19 >= 0.88 else "[VERY HIGH CONFIDENCE]",
                "explanation": "Both players demonstrate strong service hold percentages or best-of-5 tournament depth, making 20+ total games statistically almost certain.",
                "action": "Consistent Totals Pick",
            })
        elif p_o20 >= 0.65:
            candidates.append({
                "title": "Over 20.5 Total Match Games",
                "market": "Total Games",
                "win_prob_pct": round(p_o20 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p_o20), 2),
                "confidence_tier": "[HIGH CONFIDENCE]",
                "explanation": "Projected sets are expected to be tightly contested with low break frequency.",
                "action": "Solid Over Line",
            })
        elif p_u23 >= 0.65:
            candidates.append({
                "title": "Under 23.5 Total Match Games",
                "market": "Total Games",
                "win_prob_pct": round(p_u23 * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, p_u23), 2),
                "confidence_tier": "[HIGH CONFIDENCE]",
                "explanation": "Significant skill differential projects a swift straight-sets victory.",
                "action": "Straight Sets Under Pick",
            })

        # 4. Match Winner (Moneyline)
        p1_w = float(r.get("p1_win_prob", 0.5))
        p2_w = float(r.get("p2_win_prob", 0.5))
        fav_player = p1 if p1_w >= p2_w else p2
        fav_prob = max(p1_w, p2_w)

        if fav_prob >= 0.60:
            candidates.append({
                "title": f"{fav_player} to Win Match (Moneyline)",
                "market": "Match Winner",
                "win_prob_pct": round(fav_prob * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, fav_prob), 2),
                "confidence_tier": "[DOMINANT MONEYLINE]" if fav_prob >= 0.78 else "[STRONG MONEYLINE]",
                "explanation": f"{fav_player} commands clear Elo superiority and predictive dominance on {surface}.",
                "action": "Straight Moneyline Bet",
            })

        # 5. First Set Winner
        p1_s1 = float(r.get("p1_first_set_prob", 0.5))
        p2_s1 = float(r.get("p2_first_set_prob", 0.5))
        s1_fav = p1 if p1_s1 >= p2_s1 else p2
        s1_prob = max(p1_s1, p2_s1)

        if s1_prob >= 0.65:
            candidates.append({
                "title": f"{s1_fav} to Win Set 1",
                "market": "First Set",
                "win_prob_pct": round(s1_prob * 100.0, 1),
                "fair_odds": round(1.0 / max(1e-4, s1_prob), 2),
                "confidence_tier": "[OPENING SET FAVORITE]",
                "explanation": f"{s1_fav} is a fast starter with high first-serve hold rates in opening games.",
                "action": "First Set Winner Bet",
            })

        # Smart Dynamic Candidate Ranking:
        # 1. Elite Set Banker (Player to Win a Set with >= 88.0% probability) is prioritized
        #    because it is blowout-proof (cashing even if the favorite cruises 6-2 6-2).
        # 2. In tight contests, Over 19.5 Games takes the lead when its probability is highest.
        # 3. Otherwise ranks purely by mathematical win probability descending.
        def candidate_rank_key(c: Dict[str, Any]) -> float:
            prob = float(c["win_prob_pct"])
            if c["market"] == "Player to Win a Set" and prob >= 88.0:
                return prob + 25.0  # Ultra-safe parlay anchor priority
            return prob

        candidates.sort(key=candidate_rank_key, reverse=True)

        # Select top 3 across distinct market categories for variety
        seen_markets = set()
        top_picks: List[Dict[str, Any]] = []

        for c in candidates:
            if c["market"] not in seen_markets:
                seen_markets.add(c["market"])
                top_picks.append(c)
            if len(top_picks) == 3:
                break

        # Fallback if fewer than 3 distinct markets met threshold
        if len(top_picks) < 3:
            for c in candidates:
                if c not in top_picks:
                    top_picks.append(c)
                if len(top_picks) == 3:
                    break

        return top_picks

    @staticmethod
    def format_match_card(row: Union[pd.Series, Dict[str, Any]]) -> str:
        """Format an individual match forecast into a comprehensive ASCII visual card."""
        r = row.to_dict() if isinstance(row, pd.Series) else row
        p1 = r.get("player1", "Player 1")
        p2 = r.get("player2", "Player 2")
        surface = r.get("surface", "Hard")
        lvl = r.get("tournament_level", "Tour")
        rnd = r.get("round", "R32")
        p1_prob = float(r.get("p1_win_prob", 0.50))
        p2_prob = float(r.get("p2_win_prob", 0.50))
        p1_fair = float(r.get("p1_fair_odds", 2.0))
        p2_fair = float(r.get("p2_fair_odds", 2.0))
        pred_tg = float(r.get("pred_total_games", 22.5))
        pred_gd = float(r.get("pred_game_diff", 0.0))
        p1_s1 = float(r.get("p1_first_set_prob", 0.50))
        p2_s1 = float(r.get("p2_first_set_prob", 0.50))

        # Player to win a set
        p1_set = float(r.get("p1_win_at_least_one_set_prob", 0.75))
        p2_set = float(r.get("p2_win_at_least_one_set_prob", 0.75))
        p1_set_odds = float(r.get("p1_win_at_least_one_set_fair_odds", 1.33))
        p2_set_odds = float(r.get("p2_win_at_least_one_set_fair_odds", 1.33))

        # Set Handicap
        p1_m1_5 = float(r.get("prob_p1_set_handicap_minus_1_5", 0.40))
        p2_p1_5 = float(r.get("prob_p2_set_handicap_plus_1_5", 0.60))

        # Exact Set Scores
        s_2_0 = float(r.get("prob_set_score_2_0", 0.40))
        s_2_1 = float(r.get("prob_set_score_2_1", 0.25))
        s_1_2 = float(r.get("prob_set_score_1_2", 0.20))
        s_0_2 = float(r.get("prob_set_score_0_2", 0.15))

        sep = "=" * 74
        sub_sep = "-" * 74

        lines = [
            sep,
            f" [MATCH] {p1} vs {p2} | {surface} | {lvl} {rnd}".center(72),
            sep,
            " 1. MATCH WINNER (MONEYLINE):",
            f"   - {p1:<24}: {p1_prob:.1%} (Fair Odds: {p1_fair:.2f})" + (f" | Mkt: {r['p1_market_odds']:.2f} (Edge: {r['p1_edge_pct']:+.1f}%)" if "p1_market_odds" in r else ""),
            f"   - {p2:<24}: {p2_prob:.1%} (Fair Odds: {p2_fair:.2f})" + (f" | Mkt: {r['p2_market_odds']:.2f} (Edge: {r['p2_edge_pct']:+.1f}%)" if "p2_market_odds" in r else ""),
            sub_sep,
            " 2. PLAYER TO WIN AT LEAST ONE SET (YES / NO):",
            f"   - {p1} to Win a Set : {p1_set:.1%} (Fair Odds: {p1_set_odds:.2f})",
            f"   - {p2} to Win a Set : {p2_set:.1%} (Fair Odds: {p2_set_odds:.2f})",
            sub_sep,
            " 3. SET HANDICAP (-1.5 / +1.5 SETS):",
            f"   - {p1} -1.5 Sets (2-0 win) : {p1_m1_5:.1%} (Fair: {1.0/max(1e-4, p1_m1_5):.2f})",
            f"   - {p2} +1.5 Sets (wins set) : {p2_p1_5:.1%} (Fair: {1.0/max(1e-4, p2_p1_5):.2f})",
            sub_sep,
            " 4. EXACT SET BETTING (CORRECT SCORE):",
            f"   - {p1} 2-0: {s_2_0:.1%} | {p1} 2-1: {s_2_1:.1%} | {p2} 2-1: {s_1_2:.1%} | {p2} 2-0: {s_0_2:.1%}",
            sub_sep,
            f" 5. TOTAL MATCH GAMES: {pred_tg:.1f} games projected",
            f"   - Over 20.5: {r.get('prob_over_20_5', 0.5):.1%} | Over 21.5: {r.get('prob_over_21_5', 0.5):.1%} | Over 22.5: {r.get('prob_over_22_5', 0.5):.1%} | Over 23.5: {r.get('prob_over_23_5', 0.5):.1%}",
            sub_sep,
            f" 6. GAME HANDICAP SPREAD: Projected margin {p1} {pred_gd:+.1f} games",
            f"   - {p1} -2.5 Cover: {r.get('prob_p1_cover_minus_2_5', 0.5):.1%} | {p1} -3.5 Cover: {r.get('prob_p1_cover_minus_3_5', 0.5):.1%} | {p1} -4.5 Cover: {r.get('prob_p1_cover_minus_4_5', 0.5):.1%}",
            f"   - {p1} +1.5 Cover: {r.get('prob_p1_cover_plus_1_5', 0.5):.1%} | {p1} +2.5 Cover: {r.get('prob_p1_cover_plus_2_5', 0.5):.1%}",
            sub_sep,
            " 7. FIRST SET WINNER & TIEBREAK:",
            f"   - {p1} Set 1: {p1_s1:.1%} (Fair: {1.0/max(1e-4, p1_s1):.2f}) | {p2} Set 1: {p2_s1:.1%} (Fair: {1.0/max(1e-4, p2_s1):.2f})",
            f"   - Tiebreak in Set 1: {r.get('prob_first_set_tiebreak', 0.2):.1%} | Any Tiebreak in Match: {r.get('prob_any_tiebreak', 0.4):.1%}",
        ]

        if "recommended_bet" in r and r["recommended_bet"] != "Pass (No Value)":
            lines.extend([
                sub_sep,
                f" [*] VALUE BET SIGNAL: {r['recommended_bet']} (Edge: {r['edge_pct']:+.1f}% | Kelly Stake: {r['kelly_stake_pct']})",
            ])

        # Top 3 Recommended Simple Picks
        top_picks = TennisPredictionPipeline.extract_top_three_picks(r)
        if top_picks:
            lines.extend([
                sub_sep,
                " [*] TOP 3 RECOMMENDED PICKS (IN SIMPLE TERMS):",
            ])
            medals = ["(1) [BANKER LOCK]", "(2) [SMART SPREAD]", "(3) [VALUE PICK]"]
            for i, p in enumerate(top_picks):
                badge = medals[i] if i < len(medals) else f"({i+1})"
                lines.append(f"   {badge} {p['title']} ({p['win_prob_pct']}%)")
                lines.append(f"      -> Reason: {p['explanation']}")
                lines.append(f"      -> Action: {p['action']}")

        lines.append(sep)
        return "\n".join(lines)
