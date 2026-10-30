"""Unit tests for TennisMarkovModel."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.markov_sim import TennisMarkovModel


def test_game_hold_probability():
    print("Testing game hold probability...")
    m = TennisMarkovModel()
    
    # 50% server should hold 50% of the time
    h_50 = m.game_hold_probability(0.50)
    assert abs(h_50 - 0.50) < 1e-4, f"Expected 0.50, got {h_50}"
    
    # Strong server (65%) holds ~83%
    h_65 = m.game_hold_probability(0.65)
    assert 0.80 < h_65 < 0.86, f"Expected ~0.83, got {h_65}"
    
    # Inversion
    p_inv = m.invert_hold_to_point_prob(h_65)
    assert abs(p_inv - 0.65) < 1e-3, f"Expected 0.65, got {p_inv}"
    print("  -> Game hold math passed.")


def test_tiebreak_probability():
    print("Testing tiebreak probability...")
    m = TennisMarkovModel()
    
    # Symmetric servers have 50% tiebreak win rate
    tb_sym = m.tiebreak_probability(0.65, 0.65)
    assert abs(tb_sym - 0.50) < 1e-4, f"Expected 0.50, got {tb_sym}"
    
    # Better server has > 50% tiebreak win rate
    tb_fav = m.tiebreak_probability(0.68, 0.62)
    assert tb_fav > 0.55, f"Expected > 0.55, got {tb_fav}"
    print("  -> Tiebreak dynamic programming passed.")


def test_set_score_distribution():
    print("Testing set score distribution...")
    m = TennisMarkovModel()
    dist = m.set_score_distribution(g_a=0.82, g_b=0.78, tb_a=0.55)
    
    # Total probability must sum to exactly 1.0
    tot_prob = sum(dist.values())
    assert abs(tot_prob - 1.0) < 1e-5, f"Expected 1.0, got {tot_prob}"
    
    # Must contain all 14 scorelines
    expected_scores = {
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6),
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 7), (6, 7),
    }
    assert set(dist.keys()) == expected_scores, f"Missing scorelines: {expected_scores - set(dist.keys())}"
    print("  -> Set score distribution passed (sum=1.0, 14 scorelines).")


def test_full_match_simulation_bo3():
    print("Testing full match simulation (Best of 3)...")
    m = TennisMarkovModel()
    res = m.simulate_match(
        calibrated_p1_win_prob=0.70,
        expected_hold_p1=0.84,
        expected_hold_p2=0.78,
        best_of=3,
        tour="ATP",
    )
    
    # Match winner prob matches target
    assert abs(res["p1_match_prob"] - 0.70) < 1e-3
    
    # Set scores sum to 1.0
    set_sum = sum(res["set_scores_prob"].values())
    assert abs(set_sum - 1.0) < 1e-3, f"Expected 1.0, got {set_sum}"
    
    # Player to win at least 1 set
    p1_set = res["player_to_win_set"]["p1_win_at_least_one_set_prob"]
    p2_set = res["player_to_win_set"]["p2_win_at_least_one_set_prob"]
    assert p1_set > 0.70
    assert 0.40 < p2_set < 0.75
    
    # Set handicaps consistency
    # P1 -1.5 Sets + P2 +1.5 Sets == 1.0
    p1_m1_5 = res["set_handicaps"]["p1_minus_1_5_sets"]
    p2_p1_5 = res["set_handicaps"]["p2_plus_1_5_sets"]
    assert abs((p1_m1_5 + p2_p1_5) - 1.0) < 1e-3
    
    # Over lines monotonic decreasing
    overs = res["total_games_over_probs"]
    assert overs["over_20_5"] >= overs["over_21_5"] >= overs["over_22_5"] >= overs["over_23_5"]
    
    print("  -> Best of 3 match simulation passed perfectly.")


def test_full_match_simulation_bo5():
    print("Testing full match simulation (Best of 5)...")
    m = TennisMarkovModel()
    res = m.simulate_match(
        calibrated_p1_win_prob=0.85,
        expected_hold_p1=0.88,
        expected_hold_p2=0.78,
        best_of=5,
        tour="ATP",
    )
    
    set_sum = sum(res["set_scores_prob"].values())
    assert abs(set_sum - 1.0) < 1e-3, f"Expected 1.0, got {set_sum}"
    assert "3-0" in res["set_scores_prob"]
    assert "0-3" in res["set_scores_prob"]
    
    p1_set = res["player_to_win_set"]["p1_win_at_least_one_set_prob"]
    assert p1_set > 0.95
    print("  -> Best of 5 match simulation passed perfectly.")


if __name__ == "__main__":
    test_game_hold_probability()
    test_tiebreak_probability()
    test_set_score_distribution()
    test_full_match_simulation_bo3()
    test_full_match_simulation_bo5()
    print("\nALL MARKOV SIMULATION TESTS PASSED!")

