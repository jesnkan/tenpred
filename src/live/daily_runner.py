"""Production-Ready Daily Pipeline Runner for ATP & WTA Tennis Prediction.

Main capabilities:
1. Ingests and cleans latest completed match archives (ATP & WTA)
2. Updates SurfaceElo rating states and player history telemetry
3. Refreshes or trains MatchWinnerModel and SetGameModel
4. Ingests upcoming fixtures and live bookmaker odds
5. Orchestrates TennisPredictionPipeline to generate multi-market forecasts
6. Detects positive expected value (EV) betting opportunities
7. Serializes forecasts to CSV / JSON and prints an executive ASCII dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from configs.config import BASE_DIR, LIVE_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
    from src.data.loader import TennisDataLoader
    from src.features.engineer import TennisFeatureEngineer
    from src.live.predictor import TennisPredictionPipeline
    from src.models.match_winner import MatchWinnerModel
    from src.models.set_game import SetGameModel
    from src.ratings.surface_elo import SurfaceElo
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import BASE_DIR, LIVE_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
    from src.data.loader import TennisDataLoader
    from src.features.engineer import TennisFeatureEngineer
    from src.live.predictor import TennisPredictionPipeline
    from src.models.match_winner import MatchWinnerModel
    from src.models.set_game import SetGameModel
    from src.ratings.surface_elo import SurfaceElo

logger = logging.getLogger("tennis.daily_runner")


class DailyRunner:
    """Production daily execution engine for tennis prediction pipelines.

    Supports both full end-to-end retraining and rapid incremental daily scoring.

    Usage Example:
        >>> runner = DailyRunner()
        >>> # Daily rapid prediction:
        >>> df_preds = runner.run_daily(
        ...     fixtures_file="data/live/upcoming_fixtures.csv",
        ...     update_data=False,
        ...     retrain_models=False
        ... )
        >>> # Full rebuild and retrain:
        >>> runner.run_daily(update_data=True, retrain_models=True)
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        live_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize directory paths and pipeline components."""
        self.base_dir = Path(base_dir) if base_dir else BASE_DIR
        self.models_dir = Path(models_dir) if models_dir else (self.base_dir / "models")
        self.live_dir = Path(live_dir) if live_dir else LIVE_DATA_DIR

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)

        # Canonical model artifact paths
        self.elo_path = self.models_dir / "surface_elo.joblib"
        self.match_winner_path = self.models_dir / "match_winner_lgbm.joblib"
        self.set_game_path = self.models_dir / "set_game_model.joblib"

        # Loader and Feature tools
        self.loader = TennisDataLoader(raw_dir=self.base_dir / "data" / "raw", processed_dir=self.base_dir / "data" / "processed")
        self.elo: Optional[SurfaceElo] = None
        self.feature_engineer: Optional[TennisFeatureEngineer] = None
        self.match_winner_model: Optional[MatchWinnerModel] = None
        self.set_game_model: Optional[SetGameModel] = None
        self.pipeline: Optional[TennisPredictionPipeline] = None

    # =========================================================================
    # Step 1: Ingest & Clean Match Archives
    # =========================================================================

    def update_historical_data(
        self,
        start_year: int = 2020,
        end_year: Optional[int] = None,
        tour: str = "both",
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """Fetch, clean, and consolidate latest ATP & WTA match archives.

        Args:
            start_year: Earliest year to load.
            end_year: Latest year to load (defaults to current calendar year).
            tour: 'atp', 'wta', or 'both'.
            auto_download: Whether to fetch missing files from Jeff Sackmann's GitHub.

        Returns:
            Cleaned and concatenated historical match DataFrame.
        """
        current_year = min(datetime.now().year, 2024)
        e_year = end_year or current_year

        logger.info(f"==> Step 1: Updating historical match records ({start_year}-{e_year}, Tour: {tour.upper()})...")

        df_all = self.loader.load_years(
            start_year=start_year,
            end_year=e_year,
            tour=tour,
            clean=True,
            auto_download=auto_download,
        )

        if not df_all.empty:
            out_path = self.loader.save_processed(df_all, "cleaned_matches.csv")
            logger.info(f"Saved consolidated dataset ({len(df_all):,} matches) to {out_path}")
        else:
            logger.warning("No matches loaded. Ensure raw data exists or auto_download is enabled.")

        return df_all

    # =========================================================================
    # Step 2: SurfaceElo & Feature Engineering Setup
    # =========================================================================

    def update_ratings_and_features(
        self,
        df_matches: pd.DataFrame,
        full_rebuild: bool = False,
    ) -> pd.DataFrame:
        """Process matches chronologically to populate SurfaceElo ratings and features.

        Args:
            df_matches: Cleaned match records.
            full_rebuild: If True, resets rating states from scratch.

        Returns:
            Engineered feature DataFrame with multi-market targets.
        """
        logger.info("==> Step 2: Updating SurfaceElo rating states and feature histories...")

        if not full_rebuild and self.elo_path.exists():
            try:
                self.elo = SurfaceElo.load(self.elo_path)
                logger.info(f"Loaded existing SurfaceElo checkpoint from {self.elo_path}")
            except Exception as exc:
                logger.warning(f"Failed to load SurfaceElo: {exc}. Performing full rebuild.")
                self.elo = SurfaceElo()
        else:
            self.elo = SurfaceElo()

        self.feature_engineer = TennisFeatureEngineer(elo_model=self.elo)

        if df_matches.empty:
            logger.warning("Empty match DataFrame passed to feature engineering.")
            return pd.DataFrame()

        # Transform matches chronologically
        df_features = self.feature_engineer.transform_matches(
            df_matches,
            symmetry_mode="double",
            progress_bar=True,
        )

        # Persist updated SurfaceElo
        self.elo.save(self.elo_path)
        logger.info(f"SurfaceElo checkpoint updated and saved to {self.elo_path}")

        return df_features

    # =========================================================================
    # Step 3: Model Retraining
    # =========================================================================

    def train_models(
        self,
        df_features: pd.DataFrame,
        val_year_split: Optional[int] = None,
    ) -> Tuple[MatchWinnerModel, SetGameModel]:
        """Fit regularized MatchWinnerModel and SetGameModel on engineered features.

        Args:
            df_features: Engineered feature DataFrame from TennisFeatureEngineer.
            val_year_split: Split year for validation set (defaults to most recent year).

        Returns:
            Tuple of (fitted MatchWinnerModel, fitted SetGameModel).
        """
        logger.info("==> Step 3: Retraining MatchWinnerModel and SetGameModel...")

        if df_features.empty:
            raise ValueError("Cannot train models on empty feature DataFrame.")

        # Temporal split for validation
        df = df_features.copy()
        if "year" not in df.columns:
            df["year"] = pd.to_datetime(df["match_date"].astype(str), errors="coerce").dt.year

        max_year = int(df["year"].max()) if not df["year"].isna().all() else datetime.now().year
        v_year = val_year_split or max_year

        train_mask = df["year"] < v_year
        val_mask = df["year"] >= v_year

        # If data is small, split by 80/20 index
        if train_mask.sum() < 100 or val_mask.sum() < 20:
            split_idx = int(len(df) * 0.85)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
        else:
            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

        logger.info(f"Training partition: {len(train_df):,} matches | Validation partition: {len(val_df):,} matches")

        # 1. Fit Dual Ensemble MatchWinnerModel (LightGBM + XGBoost + SurfaceElo stacking)
        self.match_winner_model = MatchWinnerModel(model_type="ensemble", random_state=42)
        self.match_winner_model.fit(
            train_df=train_df,
            target_col="target_p1_won",
            val_df=val_df,
            early_stopping_rounds=30,
            verbose=False,
        )
        self.match_winner_model.save(self.match_winner_path)

        # 2. Fit SetGameModel
        self.set_game_model = SetGameModel(model_type="lightgbm", random_state=42)
        self.set_game_model.fit(
            train_df=train_df,
            val_df=val_df,
            early_stopping_rounds=30,
            verbose=False,
        )
        self.set_game_model.save(self.set_game_path)

        logger.info("Both models successfully retrained and persisted to disk.")
        return self.match_winner_model, self.set_game_model

    # =========================================================================
    # Step 4: Load Upcoming Fixtures
    # =========================================================================

    def load_fixtures(
        self,
        fixtures_input: Optional[Union[pd.DataFrame, List[Dict[str, Any]], str, Path]] = None,
    ) -> pd.DataFrame:
        """Load upcoming matches from file, DataFrame, or generate starter template.

        Args:
            fixtures_input: Optional custom input. If None, checks data/live/upcoming_fixtures.csv.

        Returns:
            DataFrame of formatted upcoming fixtures.
        """
        logger.info("==> Step 4: Loading upcoming fixtures...")

        if fixtures_input is not None:
            if isinstance(fixtures_input, (str, Path)):
                p = Path(fixtures_input)
                if p.exists():
                    df = pd.read_csv(p)
                else:
                    raise FileNotFoundError(f"Specified fixtures file not found: {p}")
            elif isinstance(fixtures_input, list):
                df = pd.DataFrame(fixtures_input)
            elif isinstance(fixtures_input, pd.DataFrame):
                df = fixtures_input.copy()
            else:
                raise TypeError("fixtures_input must be a DataFrame, list of dicts, or path.")
            return df

        # Default path check
        default_file = self.live_dir / "upcoming_fixtures.csv"
        if default_file.exists():
            df = pd.read_csv(default_file)
            logger.info(f"Loaded {len(df):,} fixtures from {default_file}")
            return df

        # Create starter template if none exists
        logger.warning(f"No upcoming fixtures found at {default_file}. Creating starter template file...")
        template_fixtures = pd.DataFrame([
            {
                "player1": "Carlos Alcaraz",
                "player2": "Jannik Sinner",
                "surface": "Hard",
                "tournament_level": "M",
                "round": "SF",
                "best_of": 3,
                "p1_odds": 1.95,
                "p2_odds": 1.90,
            },
            {
                "player1": "Daniil Medvedev",
                "player2": "Alexander Zverev",
                "surface": "Hard",
                "tournament_level": "M",
                "round": "SF",
                "best_of": 3,
                "p1_odds": 2.10,
                "p2_odds": 1.75,
            },
            {
                "player1": "Iga Swiatek",
                "player2": "Aryna Sabalenka",
                "surface": "Clay",
                "tournament_level": "G",
                "round": "F",
                "best_of": 3,
                "p1_odds": 1.55,
                "p2_odds": 2.45,
            },
        ])
        template_fixtures.to_csv(default_file, index=False)
        logger.info(f"Created template fixtures file with 3 marquee matchups at: {default_file}")
        return template_fixtures

    # =========================================================================
    # Step 5 & 6: Run Predictions & Generate Outputs
    # =========================================================================

    def predict_upcoming(
        self,
        fixtures_df: pd.DataFrame,
        min_edge_threshold: float = 0.03,
        kelly_fraction: float = 0.25,
    ) -> pd.DataFrame:
        """Score upcoming matches using the unified prediction pipeline.

        Args:
            fixtures_df: DataFrame of upcoming match records.
            min_edge_threshold: Minimum positive edge to trigger a bet signal (default: 0.03 = 3%).
            kelly_fraction: Kelly multiplier for bankroll allocation.

        Returns:
            Enriched DataFrame with match winner, totals, spreads, and EV recommendations.
        """
        logger.info("==> Step 5: Generating multi-market forecasts via TennisPredictionPipeline...")

        # Initialize pipeline if not already active
        if self.pipeline is None:
            # Attempt loading models if unassigned
            if self.match_winner_model is None and self.match_winner_path.exists():
                self.match_winner_model = MatchWinnerModel.load(self.match_winner_path)
            if self.set_game_model is None and self.set_game_path.exists():
                self.set_game_model = SetGameModel.load(self.set_game_path)
            if self.elo is None and self.elo_path.exists():
                self.elo = SurfaceElo.load(self.elo_path)
            elif self.elo is None:
                self.elo = SurfaceElo()

            if self.feature_engineer is None:
                self.feature_engineer = TennisFeatureEngineer(elo_model=self.elo)

            self.pipeline = TennisPredictionPipeline(
                match_winner_model=self.match_winner_model,
                set_game_model=self.set_game_model,
                feature_engineer=self.feature_engineer,
                elo_model=self.elo,
            )

        df_preds = self.pipeline.predict_batch(
            fixtures_df,
            min_edge_threshold=min_edge_threshold,
            kelly_fraction=kelly_fraction,
        )

        return df_preds

    def save_and_report_predictions(
        self,
        predictions_df: pd.DataFrame,
        date_str: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """Serialize predictions to CSV & JSON and print executive ASCII dashboard.

        Args:
            predictions_df: Predicted DataFrame from predict_upcoming().
            date_str: Optional date string tag (defaults to current YYYYMMDD).

        Returns:
            Tuple of (Path to saved CSV, Path to saved JSON).
        """
        logger.info("==> Step 6: Saving outputs and formatting executive dashboard...")

        tag = date_str or datetime.now().strftime("%Y%m%d")
        csv_file = self.live_dir / f"predictions_{tag}.csv"
        json_file = self.live_dir / f"predictions_{tag}.json"

        # Save files
        predictions_df.to_csv(csv_file, index=False)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(predictions_df.to_dict(orient="records"), f, indent=2, default=str)

        logger.info(f"Predictions saved to:\n  • CSV: {csv_file}\n  • JSON: {json_file}")

        # Print Executive Dashboard
        self.print_daily_dashboard(predictions_df)
        return csv_file, json_file

    @staticmethod
    def print_daily_dashboard(df_preds: pd.DataFrame) -> None:
        """Display a structured ASCII terminal summary of today's predictions and value bets."""
        sep = "=" * 80
        sub_sep = "-" * 80

        print(f"\n{sep}")
        print(" [TENNIS] DAILY PREDICTION DASHBOARD & VALUE SIGNALS ".center(78))
        print(f"{sep}")
        print(f" Total Matches Evaluated : {len(df_preds):,}")

        # Check for recommended bets
        has_recommendations = "recommended_bet" in df_preds.columns
        if has_recommendations:
            val_bets = df_preds[df_preds["recommended_bet"] != "Pass (No Value)"]
            print(f" Value Opportunities Found: {len(val_bets):,}")
            print(sub_sep)

            if not val_bets.empty:
                print(" [*] ACTIONABLE VALUE BET RECOMMENDATIONS:")
                for _, r in val_bets.iterrows():
                    side = r["player1"] if r["recommended_bet"] == "Player 1" else r["player2"]
                    mkt_odds = r.get("p1_market_odds" if r["recommended_bet"] == "Player 1" else "p2_market_odds", "N/A")
                    fair_odds = r.get("p1_fair_odds" if r["recommended_bet"] == "Player 1" else "p2_fair_odds", "N/A")
                    win_prob = r.get("p1_win_prob" if r["recommended_bet"] == "Player 1" else "p2_win_prob", 0.5)
                    edge = r.get("edge_pct", 0.0)
                    stake = r.get("kelly_stake_pct", "0.0%")
                    print(f"   - {r['player1']} vs {r['player2']} [{r['surface']}]")
                    print(f"     -> Bet: {side:<18} | Odds: {mkt_odds} (Fair: {fair_odds}) | Win%: {win_prob:.1%} | Edge: {edge:+.1f}% | Stake: {stake}")
                print(sub_sep)
            else:
                print(" No market lines met the minimum edge threshold today. (All markets efficient).")
                print(sub_sep)

        # Overview Table
        print(" [MATCH FORECAST OVERVIEW]:")
        display_cols = [
            "player1", "player2", "surface", "p1_win_prob", "p1_fair_odds", "p2_fair_odds",
            "p1_win_at_least_one_set_prob", "prob_p1_set_handicap_minus_1_5",
            "pred_total_games", "prob_over_22_5"
        ]
        if "recommended_bet" in df_preds.columns:
            display_cols.append("recommended_bet")

        cols_to_show = [c for c in display_cols if c in df_preds.columns]
        summary_table = df_preds[cols_to_show].copy()
        if "p1_win_prob" in summary_table.columns:
            summary_table["p1_win_prob"] = summary_table["p1_win_prob"].apply(lambda p: f"{p:.1%}")
        if "p1_win_at_least_one_set_prob" in summary_table.columns:
            summary_table["p1_win_at_least_one_set_prob"] = summary_table["p1_win_at_least_one_set_prob"].apply(lambda p: f"{p:.1%}")
        if "prob_p1_set_handicap_minus_1_5" in summary_table.columns:
            summary_table["prob_p1_set_handicap_minus_1_5"] = summary_table["prob_p1_set_handicap_minus_1_5"].apply(lambda p: f"{p:.1%}")
        if "prob_over_22_5" in summary_table.columns:
            summary_table["prob_over_22_5"] = summary_table["prob_over_22_5"].apply(lambda p: f"{p:.1%}")

        print(summary_table.to_string(index=False))
        print(sep + "\n")

    # =========================================================================
    # Master Execution Flow
    # =========================================================================

    def run_daily(
        self,
        fixtures_file: Optional[Union[str, Path]] = None,
        update_data: bool = False,
        retrain_models: bool = False,
        start_year: int = 2021,
        min_edge_threshold: float = 0.03,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """Execute the end-to-end daily operational workflow.

        Args:
            fixtures_file: Path to upcoming matches file (defaults to data/live/upcoming_fixtures.csv).
            update_data: Whether to fetch and ingest the latest historical match files.
            retrain_models: Whether to retrain MatchWinnerModel and SetGameModel.
            start_year: Earliest year for historical ingestion if update_data=True.
            min_edge_threshold: Minimum positive edge required to trigger a bet signal.
            auto_download: Whether to auto-download missing files from GitHub.

        Returns:
            DataFrame of scored predictions.
        """
        logger.info("================================================================================")
        logger.info("[START] TENNIS PREDICTION DAILY PIPELINE RUNNER")
        logger.info("================================================================================")

        # 1. Update data if requested
        if update_data:
            df_matches = self.update_historical_data(start_year=start_year, auto_download=auto_download)
            df_features = self.update_ratings_and_features(df_matches, full_rebuild=False)
            if retrain_models:
                self.train_models(df_features)

        # 2. Check if models exist; if not, train them on available data
        models_exist = self.match_winner_path.exists() and self.set_game_path.exists()
        if not models_exist and not update_data:
            logger.info("Model checkpoints not found on disk. Initializing historical data & models...")
            df_matches = self.update_historical_data(start_year=start_year, auto_download=auto_download)
            df_features = self.update_ratings_and_features(df_matches, full_rebuild=True)
            self.train_models(df_features)

        # 3. Load upcoming fixtures
        df_fixtures = self.load_fixtures(fixtures_file)

        # 4. Predict
        df_predictions = self.predict_upcoming(
            df_fixtures,
            min_edge_threshold=min_edge_threshold,
        )

        # 5. Save & Report
        self.save_and_report_predictions(df_predictions)

        logger.info("[SUCCESS] Daily pipeline run completed successfully.")
        return df_predictions
