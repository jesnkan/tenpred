"""Institutional Bankroll Risk Management & Multi-Market Portfolio Staking.

Features:
1. Multi-Market Correlated Kelly Staking: Adjusts stakes when betting multiple markets
   (Match Winner, Game Handicap, Total Games, Player to Win a Set) on the same match.
2. Rigorous Exposure Constraints:
   - Max single-bet bankroll exposure (e.g., 2.5%)
   - Max single-match aggregate exposure (e.g., 4.0%)
   - Max daily portfolio exposure (e.g., 15.0%)
3. Monte Carlo Bankroll Simulator:
   - 1,000+ path trajectory simulation
   - Maximum expected drawdown (95% Value-at-Risk)
   - Risk-of-Ruin (RoR) estimation
   - Probability of doubling bankroll
4. Closing Line Value (CLV) & Realized Edge Evaluation:
   - The gold standard measure of true mathematical edge over market makers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BetRecommendation:
    """Structure representing a single recommended value bet."""
    match_id: str
    player1: str
    player2: str
    market: str  # 'match_winner', 'game_handicap', 'total_games', 'player_win_set', 'set_handicap'
    selection: str
    model_prob: float
    market_odds: float
    market_implied_prob: float
    raw_edge_pct: float
    expected_value: float
    kelly_stake_pct: float
    recommended_stake_amount: float


class TennisRiskManager:
    """Institutional portfolio risk manager and Kelly sizing engine for tennis."""

    def __init__(
        self,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,  # Quarter-Kelly for sports variance damping
        min_edge_threshold: float = 0.03,  # Minimum 3.0% net edge
        max_single_bet_pct: float = 2.50,  # Max 2.5% bankroll on single bet
        max_match_exposure_pct: float = 4.00,  # Max 4.0% bankroll on single match
        max_daily_exposure_pct: float = 15.00,  # Max 15.0% bankroll in play simultaneously
        correlation_dampening_factor: float = 0.70,  # Haircut on secondary bets on same match
    ) -> None:
        """Initialize risk management parameters.

        Args:
            bankroll: Current total betting bankroll in currency units.
            kelly_fraction: Fraction of full Kelly criterion (0.25 = quarter Kelly).
            min_edge_threshold: Minimum positive edge required (e.g., 0.03 = 3%).
            max_single_bet_pct: Upper ceiling on a single wager as % of bankroll.
            max_match_exposure_pct: Upper ceiling across all wagers on a single match.
            max_daily_exposure_pct: Upper ceiling across all active daily bets.
            correlation_dampening_factor: Multiplier applied to subsequent bets on same match.
        """
        self.bankroll = float(bankroll)
        self.kelly_fraction = float(kelly_fraction)
        self.min_edge_threshold = float(min_edge_threshold)
        self.max_single_bet_pct = float(max_single_bet_pct)
        self.max_match_exposure_pct = float(max_match_exposure_pct)
        self.max_daily_exposure_pct = float(max_daily_exposure_pct)
        self.correlation_dampening_factor = float(correlation_dampening_factor)

    # =========================================================================
    # 1. Single Bet Sizing & Edge Calculations
    # =========================================================================

    def calculate_single_kelly(
        self,
        prob: float,
        decimal_odds: float,
        vig_free_prob: Optional[float] = None,
    ) -> Dict[str, float]:
        """Compute fractional Kelly stake, edge, and Expected Value (EV).

        Formula:
        b = decimal_odds - 1.0
        q = 1.0 - prob
        Full Kelly f* = (b * p - q) / b
        Fractional Kelly f_frac = f* * kelly_fraction
        """
        if decimal_odds <= 1.0 or prob <= 0.0:
            return {"edge_pct": 0.0, "ev": -1.0, "kelly_pct": 0.0}

        b = decimal_odds - 1.0
        q = 1.0 - prob
        raw_f = (b * prob - q) / b

        mkt_p = vig_free_prob if vig_free_prob is not None else (1.0 / decimal_odds)
        edge = prob - mkt_p
        ev = (prob * decimal_odds) - 1.0

        if edge < self.min_edge_threshold or raw_f <= 0:
            kelly_pct = 0.0
        else:
            unconstrained_pct = raw_f * self.kelly_fraction * 100.0
            kelly_pct = min(unconstrained_pct, self.max_single_bet_pct)

        return {
            "edge_pct": round(edge * 100.0, 2),
            "ev": round(ev, 3),
            "kelly_pct": round(kelly_pct, 2),
        }

    # =========================================================================
    # 2. Multi-Market Portfolio Optimization with Correlation Adjustment
    # =========================================================================

    def optimize_portfolio_bets(
        self,
        candidate_bets: List[Dict[str, Any]],
        current_bankroll: Optional[float] = None,
    ) -> pd.DataFrame:
        """Screen, rank, and allocate stakes across multi-market bet candidates.

        Handles correlation: If a match has multiple positive edge bets (e.g. Winner + Spread),
        the highest EV bet receives primary stake, and secondary bets receive dampened stakes.
        Total match stake is capped at `max_match_exposure_pct`.
        Total daily portfolio stake is capped at `max_daily_exposure_pct`.

        Args:
            candidate_bets: List of dictionaries with keys:
                'match_id', 'player1', 'player2', 'market', 'selection',
                'model_prob', 'market_odds', 'vig_free_prob'
            current_bankroll: Optional override of self.bankroll.

        Returns:
            DataFrame of approved bets with final allocated stakes ($ and %).
        """
        b_roll = current_bankroll if current_bankroll is not None else self.bankroll
        if not candidate_bets:
            return pd.DataFrame()

        # 1. Compute standalone metrics for each candidate
        processed: List[Dict[str, Any]] = []
        for bet in candidate_bets:
            p = float(bet["model_prob"])
            odds = float(bet["market_odds"])
            vf_p = bet.get("vig_free_prob", 1.0 / odds)

            metrics = self.calculate_single_kelly(p, odds, vig_free_prob=vf_p)
            if metrics["edge_pct"] >= (self.min_edge_threshold * 100.0) and metrics["kelly_pct"] > 0:
                item = dict(bet)
                item["raw_edge_pct"] = metrics["edge_pct"]
                item["expected_value"] = metrics["ev"]
                item["raw_kelly_pct"] = metrics["kelly_pct"]
                processed.append(item)

        if not processed:
            logger.info("No candidates met the minimum edge threshold.")
            return pd.DataFrame()

        # Sort candidate bets by Expected Value descending
        processed = sorted(processed, key=lambda x: x["expected_value"], reverse=True)

        # 2. Apply per-match correlation dampening and match ceiling
        match_exposures: Dict[str, float] = defaultdict(float)
        match_bet_counts: Dict[str, int] = defaultdict(int)
        allocated_bets: List[Dict[str, Any]] = []

        for b in processed:
            m_id = str(b.get("match_id", f"{b.get('player1')}_{b.get('player2')}"))
            current_match_exp = match_exposures[m_id]

            if current_match_exp >= self.max_match_exposure_pct:
                continue  # Match exposure exhausted

            proposed_stake = b["raw_kelly_pct"]

            # If not the first bet on this match, apply correlation dampening
            if match_bet_counts[m_id] > 0:
                proposed_stake *= self.correlation_dampening_factor

            # Cap at single bet limit and remaining match budget
            remaining_match_budget = self.max_match_exposure_pct - current_match_exp
            final_stake_pct = min(proposed_stake, self.max_single_bet_pct, remaining_match_budget)

            if final_stake_pct >= 0.20:  # Minimum bet size 0.20% of bankroll
                b["final_stake_pct"] = round(final_stake_pct, 2)
                b["stake_amount"] = round(b_roll * (final_stake_pct / 100.0), 2)
                match_exposures[m_id] += final_stake_pct
                match_bet_counts[m_id] += 1
                allocated_bets.append(b)

        if not allocated_bets:
            return pd.DataFrame()

        # 3. Apply global daily portfolio cap
        total_exp = sum(b["final_stake_pct"] for b in allocated_bets)
        if total_exp > self.max_daily_exposure_pct:
            scale_factor = self.max_daily_exposure_pct / total_exp
            logger.info(f"Daily portfolio exposure ({total_exp:.1f}%) exceeded limit ({self.max_daily_exposure_pct}%). Scaling down by factor {scale_factor:.3f}")
            for b in allocated_bets:
                b["final_stake_pct"] = round(b["final_stake_pct"] * scale_factor, 2)
                b["stake_amount"] = round(b_roll * (b["final_stake_pct"] / 100.0), 2)

        df_out = pd.DataFrame(allocated_bets)
        cols_order = [
            "player1", "player2", "market", "selection", "model_prob",
            "market_odds", "raw_edge_pct", "expected_value", "final_stake_pct", "stake_amount"
        ]
        available_cols = [c for c in cols_order if c in df_out.columns] + [
            c for c in df_out.columns if c not in cols_order
        ]
        return df_out[available_cols]

    # =========================================================================
    # 3. Monte Carlo Bankroll Simulation & Risk of Ruin
    # =========================================================================

    def simulate_bankroll(
        self,
        n_bets: int = 500,
        avg_win_prob: float = 0.58,
        avg_decimal_odds: float = 1.85,
        stake_pct: float = 1.25,
        n_simulations: int = 1000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Perform Monte Carlo simulation of bankroll growth, drawdowns, and ruin risk.

        Args:
            n_bets: Number of sequential bets per simulation run.
            avg_win_prob: Average model win probability across selected bets.
            avg_decimal_odds: Average bookmaker decimal odds taken.
            stake_pct: Stake size per bet as % of current bankroll.
            n_simulations: Number of simulated trajectories.
            random_seed: Reproducibility seed.

        Returns:
            Dictionary with drawdown statistics, probability of doubling, and risk of ruin.
        """
        np.random.seed(random_seed)
        net_multiplier = avg_decimal_odds - 1.0
        f_stake = stake_pct / 100.0

        trajectories = np.zeros((n_simulations, n_bets + 1), dtype=np.float64)
        trajectories[:, 0] = self.bankroll

        max_drawdowns = np.zeros(n_simulations, dtype=np.float64)
        doubled_count = 0
        ruin_count = 0  # Ruin defined as bankroll falling below 50% of starting bankroll

        for sim in range(n_simulations):
            b_current = self.bankroll
            peak = b_current
            max_dd = 0.0

            outcomes = np.random.binomial(1, avg_win_prob, size=n_bets)

            for step, won in enumerate(outcomes):
                stake = b_current * f_stake
                if won == 1:
                    b_current += stake * net_multiplier
                else:
                    b_current -= stake

                b_current = max(0.0, b_current)
                trajectories[sim, step + 1] = b_current

                if b_current > peak:
                    peak = b_current
                dd = (peak - b_current) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

            max_drawdowns[sim] = max_dd
            if trajectories[sim, -1] >= (2.0 * self.bankroll):
                doubled_count += 1
            if np.min(trajectories[sim, :]) <= (0.50 * self.bankroll):
                ruin_count += 1

        final_bankrolls = trajectories[:, -1]

        return {
            "initial_bankroll": self.bankroll,
            "n_simulations": n_simulations,
            "n_bets": n_bets,
            "median_final_bankroll": round(float(np.median(final_bankrolls)), 2),
            "expected_final_bankroll": round(float(np.mean(final_bankrolls)), 2),
            "p10_final_bankroll": round(float(np.percentile(final_bankrolls, 10)), 2),
            "p90_final_bankroll": round(float(np.percentile(final_bankrolls, 90)), 2),
            "prob_doubling_bankroll_pct": round(float(doubled_count / n_simulations * 100.0), 2),
            "risk_of_50pct_drawdown_pct": round(float(ruin_count / n_simulations * 100.0), 2),
            "median_max_drawdown_pct": round(float(np.median(max_drawdowns) * 100.0), 2),
            "worst_case_drawdown_95pct_var": round(float(np.percentile(max_drawdowns, 95) * 100.0), 2),
        }

    # =========================================================================
    # 4. Closing Line Value (CLV) Calculation
    # =========================================================================

    @staticmethod
    def calculate_clv(
        placed_odds: Union[float, np.ndarray, pd.Series],
        closing_odds: Union[float, np.ndarray, pd.Series],
    ) -> Union[float, np.ndarray, pd.Series]:
        """Compute Closing Line Value (CLV) percentage.

        CLV = (Placed Odds / Closing Odds - 1.0) * 100%
        Positive CLV indicates sustained market-beating edge over time.
        """
        p_odds = np.asarray(placed_odds, dtype=float)
        c_odds = np.asarray(closing_odds, dtype=float)
        clv = (p_odds / np.clip(c_odds, 1.01, None) - 1.0) * 100.0
        return np.round(clv, 2)
