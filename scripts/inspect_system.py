"""System Inspection & Quick-Tour Utility.

Run this script to immediately view:
1. Current SurfaceElo Player Leaderboards (Overall, Hard, Clay, Grass)
2. Most influential features driving match winner predictions
3. Today's live match forecasts and value bet signals
4. Detailed prediction card for any custom player matchup
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.live.predictor import TennisPredictionPipeline
from src.models.match_winner import MatchWinnerModel
from src.models.set_game import SetGameModel
from src.ratings.surface_elo import SurfaceElo


def view_elo_leaderboards(elo_path: str = "models/surface_elo.joblib", top_n: int = 10) -> None:
    """Print the top rated players overall and across surfaces."""
    path = Path(elo_path)
    if not path.exists():
        print(f"[!] Elo checkpoint not found at: {path}. Run scripts/daily_run.py first.")
        return

    elo = SurfaceElo.load(path)
    df = elo.to_dataframe()

    sep = "=" * 80
    print(f"\n{sep}")
    print(f" [ELO] TOP {top_n} PLAYERS BY SURFACE-SPECIFIC ELO RATING ".center(78))
    print(f"{sep}")

    print(f"\n--- OVERALL TOP {top_n} ---")
    print(df[["player", "elo_overall", "matches_played", "last_active"]].head(top_n).to_string(index=False))

    print(f"\n--- HARD COURT SPECIALISTS ---")
    df_hard = df[df["matches_hard"] >= 10].sort_values(by="elo_hard", ascending=False)
    print(df_hard[["player", "elo_hard", "elo_overall", "matches_hard"]].head(top_n).to_string(index=False))

    print(f"\n--- CLAY COURT SPECIALISTS ---")
    df_clay = df[df["matches_clay"] >= 10].sort_values(by="elo_clay", ascending=False)
    print(df_clay[["player", "elo_clay", "elo_overall", "matches_clay"]].head(top_n).to_string(index=False))


def view_feature_importance(model_path: str = "models/match_winner_lgbm.joblib", top_n: int = 15) -> None:
    """Print the most impactful features driving match predictions."""
    path = Path(model_path)
    if not path.exists():
        print(f"[!] Model checkpoint not found at: {path}.")
        return

    model = MatchWinnerModel.load(path)
    df_imp = model.feature_importance(importance_type="gain", top_n=top_n)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f" [FEATURES] TOP {top_n} FEATURES DRIVING MATCH WINNER PREDICTIONS (GAIN) ".center(78))
    print(f"{sep}")
    print(df_imp.to_string(index=False))


def view_today_predictions(predictions_csv: str = "data/live/predictions_20261030.csv") -> None:
    """View today's generated predictions and value betting signals."""
    live_dir = Path("data/live")
    files = sorted(live_dir.glob("predictions_*.csv"))
    if not files:
        print("[!] No predictions found. Run: python scripts/daily_run.py")
        return

    latest_file = files[-1]
    df = pd.read_csv(latest_file)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f" [PREDICTIONS] TODAY'S PREDICTIONS & VALUE BET SIGNALS ({latest_file.name}) ".center(78))
    print(f"{sep}")

    display_cols = [
        "player1", "player2", "surface", "p1_win_prob", "p1_fair_odds", "p2_fair_odds",
        "p1_win_at_least_one_set_prob", "prob_p1_set_handicap_minus_1_5",
        "pred_total_games", "prob_over_22_5", "recommended_bet", "edge_pct"
    ]
    show_cols = [c for c in display_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


def simulate_custom_match(p1: str, p2: str, surface: str = "Hard") -> None:
    """Simulate a match between any two players and print the full forecast card."""
    mw_path = "models/match_winner_lgbm.joblib"
    sg_path = "models/set_game_model.joblib"
    elo_path = "models/surface_elo.joblib"

    if not Path(mw_path).exists():
        print("[!] Models not trained yet. Run 'python scripts/daily_run.py' first.")
        return

    pipeline = TennisPredictionPipeline.from_trained_artifacts(
        match_winner_path=mw_path,
        set_game_path=sg_path,
        elo_path=elo_path,
    )

    pred = pipeline.predict_match(
        player1=p1,
        player2=p2,
        surface=surface,
        tournament_level="M",
        round_name="F",
        best_of=3,
    )

    print("\n" + pipeline.format_match_card(pred))


def main() -> None:
    print("================================================================================")
    print("               ATP & WTA TENNIS PREDICTION SYSTEM INSPECTOR                     ")
    print("================================================================================")
    view_elo_leaderboards(top_n=10)
    view_feature_importance(top_n=12)
    view_today_predictions()
    simulate_custom_match("Carlos Alcaraz", "Daniil Medvedev", surface="Hard")


if __name__ == "__main__":
    main()
