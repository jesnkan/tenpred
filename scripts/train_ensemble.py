"""Script to train the Dual Ensemble (LightGBM + XGBoost + SurfaceElo) on historical matches."""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.features.engineer import TennisFeatureEngineer
from src.models.match_winner import MatchWinnerModel
from src.models.set_game import SetGameModel
from src.ratings.surface_elo import SurfaceElo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ensemble")


def main():
    data_path = Path("data/processed/cleaned_matches.csv")
    if not data_path.exists():
        logger.error(f"Cleaned matches not found at {data_path}")
        return

    logger.info(f"Loading cleaned matches from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} cleaned matches.")

    # 1. Update ratings and transform features
    elo = SurfaceElo()
    fe = TennisFeatureEngineer(elo_model=elo)
    t0 = time.time()
    logger.info("Transforming matches into leak-free feature vectors...")
    df_features = fe.transform_matches(df, symmetry_mode="double", progress_bar=False)
    logger.info(f"Feature engineering completed in {time.time() - t0:.1f}s. Shape: {df_features.shape}")

    # Split train / validation (85% / 15%)
    split_idx = int(len(df_features) * 0.85)
    train_df = df_features.iloc[:split_idx].copy()
    val_df = df_features.iloc[split_idx:].copy()
    logger.info(f"Train split: {len(train_df):,} | Val split: {len(val_df):,}")

    # 2. Fit Dual Ensemble MatchWinnerModel (LightGBM + XGBoost + SurfaceElo stacking)
    logger.info("Training Dual Ensemble MatchWinnerModel (LightGBM + XGBoost + SurfaceElo stacking)...")
    mw_ensemble = MatchWinnerModel(model_type="ensemble", random_state=42)
    mw_ensemble.fit(train_df=train_df, target_col="target_p1_won", val_df=val_df, early_stopping_rounds=30)

    # Save to models directory
    Path("models").mkdir(parents=True, exist_ok=True)
    mw_ensemble.save("models/match_winner_ensemble.joblib")
    mw_ensemble.save("models/match_winner_lgbm.joblib")
    elo.save("models/surface_elo.joblib")

    # 3. Fit SetGameModel
    logger.info("Training SetGameModel (Total Games, Spreads, Sets)...")
    sg_model = SetGameModel(model_type="lightgbm", random_state=42)
    sg_model.fit(train_df=train_df, val_df=val_df, early_stopping_rounds=30)
    sg_model.save("models/set_game_model.joblib")

    logger.info("Ensemble models successfully trained and serialized to models/!")


if __name__ == "__main__":
    main()

