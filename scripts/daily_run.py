"""Command-Line Interface (CLI) Entry Point for the Tennis Daily Prediction Runner.

Usage:
    # 1. Quick daily prediction on upcoming matches:
    python scripts/daily_run.py

    # 2. Daily prediction with custom fixtures file:
    python scripts/daily_run.py --fixtures data/live/today_matches.csv

    # 3. Incremental update (fetch latest matches, update Elo, and predict):
    python scripts/daily_run.py --mode update

    # 4. Full historical rebuild, retrain models, and predict:
    python scripts/daily_run.py --mode full --start-year 2020
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live.daily_runner import DailyRunner


def setup_cli_logging(verbose: bool = False) -> None:
    """Configure terminal logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Parse CLI arguments and invoke the DailyRunner workflow."""
    parser = argparse.ArgumentParser(
        description="ATP & WTA Tennis Prediction System - Daily Operational Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["predict", "update", "retrain", "full"],
        default="predict",
        help="Pipeline execution mode: 'predict' (fast scoring), 'update' (ingest recent matches & Elo update), 'retrain' (retrain models), or 'full' (rebuild all).",
    )
    parser.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help="Path to CSV file with upcoming fixtures. If None, checks data/live/upcoming_fixtures.csv or creates a starter template.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="Earliest historical year to ingest when updating or rebuilding.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.03,
        help="Minimum edge threshold (e.g. 0.03 = 3%%) to flag an actionable value bet.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debug logs.",
    )

    args = parser.parse_args()
    setup_cli_logging(args.verbose)

    # Translate mode to flags
    update_data = args.mode in ("update", "retrain", "full")
    retrain_models = args.mode in ("retrain", "full")

    runner = DailyRunner()
    runner.run_daily(
        fixtures_file=args.fixtures,
        update_data=update_data,
        retrain_models=retrain_models,
        start_year=args.start_year,
        min_edge_threshold=args.min_edge,
    )


if __name__ == "__main__":
    main()

