"""Unit tests for the TennisResultsTracker module."""

import sys
from pathlib import Path

# Ensure root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from src.live.results_tracker import TennisResultsTracker


class TestTennisResultsTracker(unittest.TestCase):
    """Test suite for live match result fetching and Banker Pick settlement grading."""

    def setUp(self):
        self.tracker = TennisResultsTracker()

    def test_normalize_name(self):
        """Test robust player name normalization."""
        self.assertEqual(self.tracker.normalize_name("Carlos Alcaraz"), "carlos alcaraz")
        self.assertEqual(self.tracker.normalize_name("Alex de Minaur"), "alex de minaur")
        self.assertEqual(self.tracker.normalize_name("J.-L. Struff"), "j l struff")
        self.assertEqual(self.tracker.normalize_name(None), "")

    def test_grade_cashed_banker_pick_set(self):
        """Test that winning banker pick (to win at least 1 set) grades as WON with profit."""
        banker_pick = {
            "title": "Carlos Alcaraz to Win At Least 1 Set",
            "market": "Set Betting",
            "fair_odds": 1.25,
            "win_prob_pct": 85.0,
        }
        live_result = {
            "status": "FINAL",
            "winner": "Carlos Alcaraz",
            "score": "6-4 6-3",
            "p1_norm": "carlos alcaraz",
            "p2_norm": "tommy paul",
            "p1_sets": 2,
            "p2_sets": 0,
            "p1_games": 12,
            "p2_games": 7,
            "total_games": 19,
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Carlos Alcaraz",
            player2="Tommy Paul",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "WON")
        self.assertTrue(grade["is_won"])
        self.assertFalse(grade["is_failed"])
        self.assertEqual(grade["profit_units"], 0.25)
        self.assertEqual(grade["status_color"], "#10b981")

    def test_grade_failed_banker_pick_straight_sets(self):
        """Test that losing banker pick grades as FAILED with loss."""
        banker_pick = {
            "title": "Carlos Alcaraz -1.5 Sets (Straight 2-0)",
            "market": "Set Handicap",
            "fair_odds": 1.70,
            "win_prob_pct": 60.0,
        }
        live_result = {
            "status": "FINAL",
            "winner": "Tommy Paul",
            "score": "4-6 6-4 7-6",
            "p1_norm": "carlos alcaraz",
            "p2_norm": "tommy paul",
            "p1_sets": 1,
            "p2_sets": 2,
            "p1_games": 17,
            "p2_games": 16,
            "total_games": 33,
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Carlos Alcaraz",
            player2="Tommy Paul",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "FAILED")
        self.assertTrue(grade["is_failed"])
        self.assertFalse(grade["is_won"])
        self.assertEqual(grade["profit_units"], -1.0)
        self.assertEqual(grade["status_color"], "#f43f5e")

    def test_grade_pending_banker_pick(self):
        """Test that unscheduled or unplayed prediction grades as PENDING."""
        banker_pick = {
            "title": "Jannik Sinner to Win Match",
            "market": "Moneyline",
            "fair_odds": 1.45,
            "win_prob_pct": 70.0,
        }
        live_result = {
            "status": "SCHEDULED",
            "winner": None,
            "score": "",
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Jannik Sinner",
            player2="Daniil Medvedev",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "PENDING")
        self.assertTrue(grade["is_pending"])
        self.assertEqual(grade["profit_units"], 0.0)
        self.assertEqual(grade["status_color"], "#38bdf8")

    def test_grade_in_play_banker_pick(self):
        """Test that in-progress match grades as IN_PLAY."""
        banker_pick = {
            "title": "Jannik Sinner to Win At Least 1 Set",
            "market": "Set Betting",
            "fair_odds": 1.20,
            "win_prob_pct": 88.0,
        }
        live_result = {
            "status": "IN_PROGRESS",
            "winner": None,
            "score": "6-4 3-2",
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Jannik Sinner",
            player2="Daniil Medvedev",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "IN_PLAY")
        self.assertTrue(grade["is_pending"])
        self.assertEqual(grade["status_color"], "#f59e0b")

    def test_grade_over_19_5_games_banker_pick(self):
        """Test that Over 19.5 Total Match Games grades correctly."""
        banker_pick = {
            "title": "Over 19.5 Total Match Games",
            "market": "Total Games",
            "fair_odds": 1.22,
            "win_prob_pct": 83.5,
        }
        # Win scenario (23 games > 19.5)
        live_result_win = {
            "status": "FINAL",
            "winner": "Carlos Alcaraz",
            "score": "7-6 6-4",
            "p1_norm": "carlos alcaraz",
            "p2_norm": "tommy paul",
            "p1_sets": 2,
            "p2_sets": 0,
            "p1_games": 13,
            "p2_games": 10,
            "total_games": 23,
        }
        grade_win = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Carlos Alcaraz",
            player2="Tommy Paul",
            live_result=live_result_win,
        )
        self.assertEqual(grade_win["settlement"], "WON")
        self.assertTrue(grade_win["is_won"])
        self.assertEqual(grade_win["profit_units"], 0.22)

        # Loss scenario (18 games <= 19.5)
        live_result_loss = {
            "status": "FINAL",
            "winner": "Carlos Alcaraz",
            "score": "6-3 6-3",
            "p1_norm": "carlos alcaraz",
            "p2_norm": "tommy paul",
            "p1_sets": 2,
            "p2_sets": 0,
            "p1_games": 12,
            "p2_games": 6,
            "total_games": 18,
        }
        grade_loss = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Carlos Alcaraz",
            player2="Tommy Paul",
            live_result=live_result_loss,
        )
        self.assertEqual(grade_loss["settlement"], "FAILED")
        self.assertTrue(grade_loss["is_failed"])
        self.assertEqual(grade_loss["profit_units"], -1.0)

    def test_grade_walkover_voided(self):
        """Test that walkover results strictly grade as VOID with 0 units profit."""
        banker_pick = {
            "title": "Elena Rybakina to Win At Least 1 Set",
            "market": "Set Betting",
            "fair_odds": 1.25,
            "win_prob_pct": 82.0,
        }
        live_result = {
            "status": "WALKOVER",
            "winner": "Leylah Fernandez",
            "score": "W/O (Walkover)",
            "p1_norm": "leylah fernandez",
            "p2_norm": "elena rybakina",
            "p1_sets": 0,
            "p2_sets": 0,
            "total_games": 0,
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Leylah Fernandez",
            player2="Elena Rybakina",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "VOID")
        self.assertTrue(grade["is_void"])
        self.assertFalse(grade["is_won"])
        self.assertFalse(grade["is_failed"])
        self.assertEqual(grade["profit_units"], 0.0)
        self.assertEqual(grade["badge"], "⚠️ VOIDED")

    def test_grade_retirement_incomplete_voided(self):
        """Test that uncompleted total games aborted by retirement grade as VOID."""
        banker_pick = {
            "title": "Over 19.5 Total Match Games",
            "market": "Total Games",
            "fair_odds": 1.30,
            "win_prob_pct": 77.0,
        }
        live_result = {
            "status": "RETIRED",
            "winner": "Taylor Fritz",
            "score": "2-6 0-2 ret",
            "p1_norm": "arthur fils",
            "p2_norm": "taylor fritz",
            "p1_sets": 0,
            "p2_sets": 1,
            "total_games": 10,
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Arthur Fils",
            player2="Taylor Fritz",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "VOID")
        self.assertTrue(grade["is_void"])
        self.assertEqual(grade["profit_units"], 0.0)

    def test_grade_retirement_completed_won(self):
        """Test that selection already achieved before retirement grades as WON."""
        banker_pick = {
            "title": "Taylor Fritz to Win At Least 1 Set",
            "market": "Set Betting",
            "fair_odds": 1.25,
            "win_prob_pct": 82.0,
        }
        live_result = {
            "status": "RETIRED",
            "winner": "Taylor Fritz",
            "score": "2-6 0-2 ret",
            "p1_norm": "arthur fils",
            "p2_norm": "taylor fritz",
            "p1_sets": 0,
            "p2_sets": 1,  # Fritz already won set 1 (6-2)
            "total_games": 10,
        }
        grade = self.tracker.grade_banker_pick(
            banker_pick=banker_pick,
            player1="Arthur Fils",
            player2="Taylor Fritz",
            live_result=live_result,
        )
        self.assertEqual(grade["settlement"], "WON")
        self.assertTrue(grade["is_won"])
        self.assertEqual(grade["profit_units"], 0.25)


if __name__ == "__main__":
    unittest.main()
