"""Unit tests for the TennisResultsTracker module."""

import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.live.results_tracker import TennisResultsTracker


class TestTennisResultsTracker(unittest.TestCase):
    """Test suite for live match result fetching and prediction grading."""

    def setUp(self):
        self.tracker = TennisResultsTracker()

    def test_normalize_name(self):
        """Test robust player name normalization."""
        self.assertEqual(self.tracker._normalize_name("Carlos Alcaraz"), "carlos alcaraz")
        self.assertEqual(self.tracker._normalize_name("Alex de Minaur"), "alex de minaur")
        self.assertEqual(self.tracker._normalize_name("J.-L. Struff"), "j l struff")
        self.assertEqual(self.tracker._normalize_name(None), "")

    def test_grade_cashed_prediction(self):
        """Test that winning prediction grades as CASHED with profit."""
        pred_row = {
            "player1": "Carlos Alcaraz",
            "player2": "Tommy Paul",
            "p1_win_prob": 0.75,
            "p1_odds": 1.40,
            "p2_odds": 3.10,
        }
        live_result = {
            "status": "FINAL",
            "winner": "Carlos Alcaraz",
            "score": "6-4 6-3",
            "p1_sets": 2,
            "p2_sets": 0,
            "total_games": 19,
        }
        grade = self.tracker.grade_prediction(pred_row, live_result, unit_wager=1.0)
        self.assertEqual(grade["settlement"], "CASHED")
        self.assertTrue(grade["is_cashed"])
        self.assertFalse(grade["is_lost"])
        self.assertEqual(grade["profit_units"], 0.40)
        self.assertEqual(grade["status_color"], "#10b981")

    def test_grade_lost_prediction(self):
        """Test that losing prediction grades as LOST with loss."""
        pred_row = {
            "player1": "Carlos Alcaraz",
            "player2": "Tommy Paul",
            "p1_win_prob": 0.75,
            "p1_odds": 1.40,
            "p2_odds": 3.10,
        }
        live_result = {
            "status": "FINAL",
            "winner": "Tommy Paul",
            "score": "4-6 6-4 7-6",
            "p1_sets": 1,
            "p2_sets": 2,
            "total_games": 33,
        }
        grade = self.tracker.grade_prediction(pred_row, live_result, unit_wager=1.0)
        self.assertEqual(grade["settlement"], "LOST")
        self.assertTrue(grade["is_lost"])
        self.assertFalse(grade["is_cashed"])
        self.assertEqual(grade["profit_units"], -1.0)
        self.assertEqual(grade["status_color"], "#f43f5e")

    def test_grade_pending_prediction(self):
        """Test that unscheduled or unplayed prediction grades as PENDING."""
        pred_row = {
            "player1": "Jannik Sinner",
            "player2": "Daniil Medvedev",
            "p1_win_prob": 0.65,
        }
        live_result = {
            "status": "SCHEDULED",
            "winner": None,
            "score": "",
        }
        grade = self.tracker.grade_prediction(pred_row, live_result)
        self.assertEqual(grade["settlement"], "PENDING")
        self.assertTrue(grade["is_pending"])
        self.assertEqual(grade["profit_units"], 0.0)


if __name__ == "__main__":
    unittest.main()
