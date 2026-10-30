"""Unit tests for TennisRiskManager."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.risk_manager import TennisRiskManager


def test_single_kelly_calculation():
    print("Testing single Kelly calculation...")
    rm = TennisRiskManager(bankroll=10000.0, kelly_fraction=0.25, min_edge_threshold=0.03, max_single_bet_pct=2.5)
    
    # Positive edge: 60% win prob at odds 1.90
    res = rm.calculate_single_kelly(prob=0.60, decimal_odds=1.90, vig_free_prob=0.526)
    assert res["edge_pct"] > 7.0
    assert res["ev"] > 0.10
    assert 0.0 < res["kelly_pct"] <= 2.5
    
    # Negative edge should produce 0 stake
    res_neg = rm.calculate_single_kelly(prob=0.45, decimal_odds=1.90, vig_free_prob=0.526)
    assert res_neg["kelly_pct"] == 0.0
    print("  -> Single Kelly calculation passed.")


def test_portfolio_optimization_and_correlation():
    print("Testing multi-market portfolio optimization & correlation dampening...")
    rm = TennisRiskManager(
        bankroll=10000.0,
        kelly_fraction=0.25,
        max_single_bet_pct=2.5,
        max_match_exposure_pct=4.0,
        max_daily_exposure_pct=10.0,
        correlation_dampening_factor=0.70,
    )
    
    candidate_bets = [
        # Two bets on same match M1
        {"match_id": "M1", "player1": "Alcaraz", "player2": "Sinner", "market": "match_winner", "selection": "Sinner", "model_prob": 0.65, "market_odds": 1.90, "vig_free_prob": 0.52},
        {"match_id": "M1", "player1": "Alcaraz", "player2": "Sinner", "market": "game_handicap", "selection": "Sinner -2.5", "model_prob": 0.58, "market_odds": 1.95, "vig_free_prob": 0.51},
        # One bet on M2
        {"match_id": "M2", "player1": "Medvedev", "player2": "Zverev", "market": "player_win_set", "selection": "Zverev to win set", "model_prob": 0.80, "market_odds": 1.40, "vig_free_prob": 0.72},
    ]
    
    df_opt = rm.optimize_portfolio_bets(candidate_bets)
    assert len(df_opt) > 0
    
    # Check that aggregate M1 stake does not exceed 4.0%
    m1_bets = df_opt[df_opt["match_id"] == "M1"]
    total_m1_stake = m1_bets["final_stake_pct"].sum()
    assert total_m1_stake <= 4.0 + 1e-4, f"M1 stake {total_m1_stake} exceeded match cap 4.0"
    
    # Check that total portfolio exposure <= 10.0%
    tot_exp = df_opt["final_stake_pct"].sum()
    assert tot_exp <= 10.0 + 1e-4
    print("  -> Multi-market correlation dampening & exposure limits passed.")


def test_monte_carlo_bankroll_simulation():
    print("Testing Monte Carlo bankroll simulation...")
    rm = TennisRiskManager(bankroll=10000.0)
    sim = rm.simulate_bankroll(
        n_bets=100,
        avg_win_prob=0.58,
        avg_decimal_odds=1.85,
        stake_pct=1.5,
        n_simulations=300,
        random_seed=42,
    )
    
    assert sim["expected_final_bankroll"] > 10000.0
    assert 0.0 <= sim["prob_doubling_bankroll_pct"] <= 100.0
    assert 0.0 <= sim["worst_case_drawdown_95pct_var"] <= 100.0
    print(f"  -> Monte Carlo passed: Expected Final = ${sim['expected_final_bankroll']:,.2f}, Median DD = {sim['median_max_drawdown_pct']}%.")


def test_clv_calculation():
    print("Testing Closing Line Value (CLV)...")
    placed = [1.95, 2.10, 1.50]
    closing = [1.85, 2.10, 1.60]
    clvs = TennisRiskManager.calculate_clv(placed, closing)
    assert clvs[0] > 0.0  # 1.95 vs 1.85 is positive CLV (+5.41%)
    assert clvs[1] == 0.0  # Equal is 0.0%
    assert clvs[2] < 0.0  # 1.50 vs 1.60 is negative CLV (-6.25%)
    print("  -> CLV calculation passed.")


if __name__ == "__main__":
    test_single_kelly_calculation()
    test_portfolio_optimization_and_correlation()
    test_monte_carlo_bankroll_simulation()
    test_clv_calculation()
    print("\nALL RISK MANAGER TESTS PASSED!")

