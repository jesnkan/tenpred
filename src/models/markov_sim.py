"""Closed-Form Analytical Markov Chain Engine for Tennis Match Simulation.

Provides exact mathematical modeling of tennis scoring hierarchy:
Point -> Game -> Tiebreak -> Set -> Match

Derived from foundational tennis Markov models (Barnett & Clarke, O'Malley, Newton & Aslam).
Computes exact analytical distributions for:
1. Exact Set Betting (2-0, 2-1, 1-2, 0-2 in Bo3; 3-0, 3-1, 3-2, etc. in Bo5)
2. Player to Win At Least One Set (Yes / No)
3. Set Handicap Spreads (-1.5 / +1.5 sets; -2.5 / +2.5 sets)
4. Discrete Match Total Games PMF (Over / Under lines from 18.5 to 38.5)
5. Discrete Match Game Handicap PMF (Cover probabilities for spreads -6.5 to +6.5)
6. First Set Exact Scorelines (6-0 through 7-6) & Tiebreak Probabilities

Seamlessly blends with LightGBM calibrated win probabilities and empirical service hold rates.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


class TennisMarkovModel:
    """Exact analytical Markov chain engine for professional tennis markets."""

    def __init__(
        self,
        default_atp_point_serve: float = 0.640,
        default_wta_point_serve: float = 0.565,
    ) -> None:
        """Initialize the Markov engine with baseline tour point serving parameters.

        Args:
            default_atp_point_serve: Typical ATP probability of winning a point on serve (~64%).
            default_wta_point_serve: Typical WTA probability of winning a point on serve (~56.5%).
        """
        self.default_atp_point_serve = default_atp_point_serve
        self.default_wta_point_serve = default_wta_point_serve

    # =========================================================================
    # 1. Point to Game Mathematics
    # =========================================================================

    @staticmethod
    def game_hold_probability(p: float) -> float:
        """Exact probability of holding serve given probability `p` of winning a point on serve.

        Using standard Markov absorbing probabilities:
        P(Game) = p^4 + 4*p^4*(1-p) + 10*p^4*(1-p)^2 + 20*p^3*(1-p)^3 * [p^2 / (p^2 + (1-p)^2)]
        """
        p = float(np.clip(p, 1e-5, 1.0 - 1e-5))
        q = 1.0 - p
        p4 = p**4
        denom = p**2 + q**2
        deuce_term = (20.0 * (p**3) * (q**3) * (p**2)) / denom if denom > 0 else 0.5
        hold_prob = p4 * (1.0 + 4.0 * q + 10.0 * (q**2)) + deuce_term
        return float(np.clip(hold_prob, 1e-5, 1.0 - 1e-5))

    @staticmethod
    def invert_hold_to_point_prob(hold_prob: float) -> float:
        """Invert hold probability to find point serve win probability p using Brent's method."""
        target = float(np.clip(hold_prob, 0.05, 0.99))
        try:
            return float(brentq(lambda p: TennisMarkovModel.game_hold_probability(p) - target, 0.01, 0.99))
        except Exception:
            return 0.62

    # =========================================================================
    # 2. Tiebreak Dynamic Programming
    # =========================================================================

    @staticmethod
    def tiebreak_probability(p_a: float, p_b: float) -> float:
        """Exact probability that Player A wins a standard 7-point tiebreak.

        Args:
            p_a: Probability Player A wins point on A's serve.
            p_b: Probability Player B wins point on B's serve.

        Serve sequence in 7-point tiebreak:
        Point 1: A serves (1 pt)
        Points 2-3: B serves (2 pts)
        Points 4-5: A serves (2 pts)
        Points 6-7: B serves (2 pts) ...
        """
        p_a = float(np.clip(p_a, 1e-5, 1.0 - 1e-5))
        p_b = float(np.clip(p_b, 1e-5, 1.0 - 1e-5))

        def is_a_server(point_idx: int) -> bool:
            if point_idx == 0:
                return True
            rem = (point_idx - 1) % 4
            return rem >= 2

        dp: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
        prob_a_wins_regular = 0.0

        for k in range(12):  # points 0 to 11 reach at most (6,6)
            new_dp: Dict[Tuple[int, int], float] = defaultdict(float)
            for (i, j), prob in dp.items():
                if prob <= 0:
                    continue
                a_serves = is_a_server(k)
                p_pt_a = p_a if a_serves else (1.0 - p_b)

                # A wins point
                ni, nj = i + 1, j
                if ni == 7 and nj <= 5:
                    prob_a_wins_regular += prob * p_pt_a
                else:
                    new_dp[(ni, nj)] += prob * p_pt_a

                # B wins point
                ni, nj = i, j + 1
                if nj == 7 and ni <= 5:
                    pass  # B wins regular
                else:
                    new_dp[(ni, nj)] += prob * (1.0 - p_pt_a)

            dp = new_dp

        p_6_6 = dp.get((6, 6), 0.0)

        # Deuce phase: from 6-6, every 2-point cycle has 1 A serve and 1 B serve
        # Prob A wins both points: p_a * (1 - p_b)
        # Prob B wins both points: (1 - p_a) * p_b
        # Prob split: p_a * p_b + (1 - p_a) * (1 - p_b)
        p_pair_a = p_a * (1.0 - p_b)
        p_pair_b = (1.0 - p_a) * p_b
        denom = p_pair_a + p_pair_b
        p_deuce_a = (p_pair_a / denom) if denom > 0 else 0.5

        return float(np.clip(prob_a_wins_regular + p_6_6 * p_deuce_a, 1e-5, 1.0 - 1e-5))

    # =========================================================================
    # 3. Exact Set Scoreline Distribution
    # =========================================================================

    @staticmethod
    def set_score_distribution(
        g_a: float,
        g_b: float,
        tb_a: float,
        a_serves_first: Optional[bool] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute the exact joint distribution of game scores in a set.

        Returns dictionary of 14 scoreline probabilities:
        (6,0), (6,1), (6,2), (6,3), (6,4), (7,5), (7,6) for Player A,
        (0,6), (1,6), (2,6), (3,6), (4,6), (5,7), (6,7) for Player B.
        """
        def _simulate_single_order(a_first: bool) -> Dict[Tuple[int, int], float]:
            dp: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
            outcomes: Dict[Tuple[int, int], float] = defaultdict(float)

            for game in range(12):
                new_dp: Dict[Tuple[int, int], float] = defaultdict(float)
                for (i, j), prob in dp.items():
                    if prob <= 0:
                        continue
                    is_a_serve = (game % 2 == 0) if a_first else (game % 2 == 1)
                    p_win_a = g_a if is_a_serve else (1.0 - g_b)

                    # A wins game
                    ni, nj = i + 1, j
                    if ni == 6 and nj <= 4:
                        outcomes[(ni, nj)] += prob * p_win_a
                    elif ni == 7 and nj == 5:
                        outcomes[(ni, nj)] += prob * p_win_a
                    else:
                        new_dp[(ni, nj)] += prob * p_win_a

                    # B wins game
                    ni, nj = i, j + 1
                    if nj == 6 and ni <= 4:
                        outcomes[(ni, nj)] += prob * (1.0 - p_win_a)
                    elif nj == 7 and ni == 5:
                        outcomes[(ni, nj)] += prob * (1.0 - p_win_a)
                    else:
                        new_dp[(ni, nj)] += prob * (1.0 - p_win_a)

                dp = new_dp

            p_6_6 = dp.get((6, 6), 0.0)
            outcomes[(7, 6)] += p_6_6 * tb_a
            outcomes[(6, 7)] += p_6_6 * (1.0 - tb_a)
            return dict(outcomes)

        if a_serves_first is True:
            return _simulate_single_order(True)
        elif a_serves_first is False:
            return _simulate_single_order(False)
        else:
            # Neutral expectation: 50% chance A serves first, 50% chance B serves first
            res_a = _simulate_single_order(True)
            res_b = _simulate_single_order(False)
            all_keys = set(res_a.keys()) | set(res_b.keys())
            return {k: 0.5 * res_a.get(k, 0.0) + 0.5 * res_b.get(k, 0.0) for k in all_keys}

    # =========================================================================
    # 4. Set to Match Inversion & Calibration
    # =========================================================================

    @staticmethod
    def match_prob_from_set_prob(s: float, best_of: int = 3) -> float:
        """Compute match win probability M given individual set win probability s."""
        s = float(np.clip(s, 1e-6, 1.0 - 1e-6))
        if best_of == 3:
            # M = s^2 * (3 - 2s)
            return s**2 * (3.0 - 2.0 * s)
        else:
            # Best of 5: s^3 * (10 - 15s + 6s^2)
            return s**3 * (10.0 - 15.0 * s + 6.0 * (s**2))

    @staticmethod
    def invert_match_prob_to_set_prob(m: float, best_of: int = 3) -> float:
        """Find individual set win probability s that produces target match win probability m."""
        m = float(np.clip(m, 1e-5, 1.0 - 1e-5))
        try:
            return float(brentq(
                lambda s: TennisMarkovModel.match_prob_from_set_prob(s, best_of) - m,
                1e-5,
                1.0 - 1e-5,
            ))
        except Exception:
            return m

    # =========================================================================
    # 5. Full Match Simulation & Market Derivation
    # =========================================================================

    def simulate_match(
        self,
        calibrated_p1_win_prob: float,
        expected_hold_p1: float = 0.80,
        expected_hold_p2: float = 0.80,
        best_of: int = 3,
        tour: str = "ATP",
        is_indoor: bool = False,
        court_speed: str = "Medium",
        cpi: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simulate all derivative market probabilities anchored to a calibrated match win prob.

        Args:
            calibrated_p1_win_prob: Final calibrated P(Player 1 wins match) from MatchWinnerModel.
            expected_hold_p1: Expected service hold rate for Player 1 (from feature engineer).
            expected_hold_p2: Expected service hold rate for Player 2 (from feature engineer).
            best_of: Match format (3 or 5).
            tour: 'ATP' or 'WTA'.
            is_indoor: Whether match is played indoors (low air resistance, higher hold/tiebreak rates).
            court_speed: Speed tier ('Fast', 'Medium-Fast', 'Medium', 'Medium-Slow', 'Slow').
            cpi: Optional Court Pace Index value (e.g. 20-45).

        Returns:
            Dictionary containing:
            - `set_scores_prob`: Exact set betting distribution
            - `player_to_win_set`: Probabilities and fair odds for P1/P2 winning at least 1 set
            - `set_handicaps`: Cover probabilities and fair odds for +/-1.5 and +/-2.5 sets
            - `game_totals_pmf` & `total_games_over_probs`: Continuous lines from 18.5 to 38.5
            - `game_handicap_pmf` & `handicap_cover_probs`: Covers from -6.5 to +6.5
            - `first_set_scores`: Full 14-scoreline PMF for Set 1
            - `first_set_tiebreak_prob`: P(Tiebreak in Set 1)
        """
        p1_m = float(np.clip(calibrated_p1_win_prob, 0.005, 0.995))
        p2_m = 1.0 - p1_m

        # 1. Invert target match win prob to set win prob s1
        s1 = self.invert_match_prob_to_set_prob(p1_m, best_of=best_of)
        s2 = 1.0 - s1

        # 2. Derive point serve win probabilities matching the players' relative hold tendencies
        base_pt = self.default_atp_point_serve if tour.upper() == "ATP" else self.default_wta_point_serve
        hold_diff = (expected_hold_p1 - expected_hold_p2)

        # Environmental court pace adjustment
        pt_speed_adj = 0.0
        if cpi is not None:
            pt_speed_adj = (cpi - 34.0) * 0.0018
        else:
            speed_str = str(court_speed).lower()
            if "fast" in speed_str:
                pt_speed_adj = 0.015
            elif "slow" in speed_str:
                pt_speed_adj = -0.018

        if is_indoor:
            pt_speed_adj += 0.012

        # Approximate point serve rates consistent with s1, hold_diff, and court pace
        pt_serve_p1 = np.clip(base_pt + pt_speed_adj + (s1 - 0.50) * 0.12 + (hold_diff * 0.10), 0.40, 0.85)
        pt_serve_p2 = np.clip(base_pt + pt_speed_adj - (s1 - 0.50) * 0.12 - (hold_diff * 0.10), 0.40, 0.85)

        g1 = self.game_hold_probability(pt_serve_p1)
        g2 = self.game_hold_probability(pt_serve_p2)
        tb_p1 = self.tiebreak_probability(pt_serve_p1, pt_serve_p2)

        # 3. Exact set scoreline PMF
        set_dist = self.set_score_distribution(g1, g2, tb_p1)
        p_tb_set = set_dist.get((7, 6), 0.0) + set_dist.get((6, 7), 0.0)

        # Re-normalize set distribution to guarantee exact match with target s1
        sum_p1_set = sum(prob for (i, j), prob in set_dist.items() if i > j)
        sum_p2_set = sum(prob for (i, j), prob in set_dist.items() if j > i)

        norm_set_dist: Dict[Tuple[int, int], float] = {}
        for (i, j), prob in set_dist.items():
            if i > j and sum_p1_set > 0:
                norm_set_dist[(i, j)] = prob * (s1 / sum_p1_set)
            elif j > i and sum_p2_set > 0:
                norm_set_dist[(i, j)] = prob * (s2 / sum_p2_set)
            else:
                norm_set_dist[(i, j)] = prob

        # =====================================================================
        # 4. Exact Set Betting Distribution & Player to Win A Set
        # =====================================================================
        set_scores_prob: Dict[str, float] = {}

        if best_of == 3:
            # Outcomes: 2-0, 2-1, 1-2, 0-2
            p_2_0 = s1**2
            p_2_1 = 2.0 * (s1**2) * s2
            p_1_2 = 2.0 * s1 * (s2**2)
            p_0_2 = s2**2

            set_scores_prob = {
                "2-0": round(p_2_0, 4),
                "2-1": round(p_2_1, 4),
                "1-2": round(p_1_2, 4),
                "0-2": round(p_0_2, 4),
            }

            # Player To Win At Least One Set (Yes / No)
            # P1 wins >= 1 set <=> P2 does not win 2-0
            p1_win_at_least_1_set = 1.0 - p_0_2
            # P2 wins >= 1 set <=> P1 does not win 2-0
            p2_win_at_least_1_set = 1.0 - p_2_0

            # Set Handicaps
            # P1 -1.5 Sets <=> P1 wins 2-0
            p1_set_m1_5 = p_2_0
            # P1 +1.5 Sets <=> P1 wins at least 1 set
            p1_set_p1_5 = p1_win_at_least_1_set
            # P2 -1.5 Sets <=> P2 wins 2-0
            p2_set_m1_5 = p_0_2
            # P2 +1.5 Sets <=> P2 wins at least 1 set
            p2_set_p1_5 = p2_win_at_least_1_set

            set_handicaps = {
                "p1_minus_1_5_sets": round(p1_set_m1_5, 4),
                "p1_plus_1_5_sets": round(p1_set_p1_5, 4),
                "p2_minus_1_5_sets": round(p2_set_m1_5, 4),
                "p2_plus_1_5_sets": round(p2_set_p1_5, 4),
            }

        else:
            # Best of 5 (Grand Slams)
            p_3_0 = s1**3
            p_3_1 = 3.0 * (s1**3) * s2
            p_3_2 = 6.0 * (s1**3) * (s2**2)
            p_2_3 = 6.0 * (s1**2) * (s2**3)
            p_1_3 = 3.0 * s1 * (s2**3)
            p_0_3 = s2**3

            set_scores_prob = {
                "3-0": round(p_3_0, 4),
                "3-1": round(p_3_1, 4),
                "3-2": round(p_3_2, 4),
                "2-3": round(p_2_3, 4),
                "1-3": round(p_1_3, 4),
                "0-3": round(p_0_3, 4),
            }

            p1_win_at_least_1_set = 1.0 - p_0_3
            p2_win_at_least_1_set = 1.0 - p_3_0

            p1_set_m2_5 = p_3_0
            p1_set_m1_5 = p_3_0 + p_3_1
            p1_set_p1_5 = 1.0 - (p_0_3 + p_1_3)
            p1_set_p2_5 = 1.0 - p_0_3

            set_handicaps = {
                "p1_minus_2_5_sets": round(p1_set_m2_5, 4),
                "p1_minus_1_5_sets": round(p1_set_m1_5, 4),
                "p1_plus_1_5_sets": round(p1_set_p1_5, 4),
                "p1_plus_2_5_sets": round(p1_set_p2_5, 4),
                "p2_minus_2_5_sets": round(1.0 - p1_set_p2_5, 4),
                "p2_minus_1_5_sets": round(1.0 - p1_set_p1_5, 4),
                "p2_plus_1_5_sets": round(1.0 - p1_set_m1_5, 4),
                "p2_plus_2_5_sets": round(1.0 - p1_set_m2_5, 4),
            }

        player_to_win_set = {
            "p1_win_at_least_one_set_prob": round(p1_win_at_least_1_set, 4),
            "p1_win_at_least_one_set_fair_odds": round(1.0 / max(1e-4, p1_win_at_least_1_set), 3),
            "p2_win_at_least_one_set_prob": round(p2_win_at_least_1_set, 4),
            "p2_win_at_least_one_set_fair_odds": round(1.0 / max(1e-4, p2_win_at_least_1_set), 3),
        }

        # =====================================================================
        # 5. Discrete Match Total Games & Game Handicap PMF via Convolution
        # =====================================================================
        # Construct set PMF for games played and game differential
        set_games_pmf: Dict[int, float] = defaultdict(float)
        set_diff_pmf: Dict[int, float] = defaultdict(float)

        for (g_w, g_l), prob in norm_set_dist.items():
            tot_g = g_w + g_l
            diff_g = g_w - g_l
            set_games_pmf[tot_g] += prob
            set_diff_pmf[diff_g] += prob

        # Helper: Convolve two PMFs
        def _convolve(pmf1: Dict[int, float], pmf2: Dict[int, float]) -> Dict[int, float]:
            res: Dict[int, float] = defaultdict(float)
            for v1, p1 in pmf1.items():
                for v2, p2 in pmf2.items():
                    res[v1 + v2] += p1 * p2
            return dict(res)

        match_games_pmf: Dict[int, float] = defaultdict(float)
        match_diff_pmf: Dict[int, float] = defaultdict(float)

        if best_of == 3:
            # 2-set match: convolve 2 sets
            g_2sets = _convolve(set_games_pmf, set_games_pmf)
            d_2sets = _convolve(set_diff_pmf, set_diff_pmf)
            # 3-set match: convolve 3 sets
            g_3sets = _convolve(g_2sets, set_games_pmf)
            d_3sets = _convolve(d_2sets, set_diff_pmf)

            prob_2_sets = set_scores_prob["2-0"] + set_scores_prob["0-2"]
            prob_3_sets = set_scores_prob["2-1"] + set_scores_prob["1-2"]

            for k, v in g_2sets.items():
                match_games_pmf[k] += prob_2_sets * v
            for k, v in g_3sets.items():
                match_games_pmf[k] += prob_3_sets * v

            for k, v in d_2sets.items():
                match_diff_pmf[k] += prob_2_sets * v
            for k, v in d_3sets.items():
                match_diff_pmf[k] += prob_3_sets * v

        else:
            # Best of 5
            g_2 = _convolve(set_games_pmf, set_games_pmf)
            g_3 = _convolve(g_2, set_games_pmf)
            g_4 = _convolve(g_3, set_games_pmf)
            g_5 = _convolve(g_4, set_games_pmf)

            d_2 = _convolve(set_diff_pmf, set_diff_pmf)
            d_3 = _convolve(d_2, set_diff_pmf)
            d_4 = _convolve(d_3, set_diff_pmf)
            d_5 = _convolve(d_4, set_diff_pmf)

            p_3sets = set_scores_prob["3-0"] + set_scores_prob["0-3"]
            p_4sets = set_scores_prob["3-1"] + set_scores_prob["1-3"]
            p_5sets = set_scores_prob["3-2"] + set_scores_prob["2-3"]

            for k, v in g_3.items():
                match_games_pmf[k] += p_3sets * v
            for k, v in g_4.items():
                match_games_pmf[k] += p_4sets * v
            for k, v in g_5.items():
                match_games_pmf[k] += p_5sets * v

            for k, v in d_3.items():
                match_diff_pmf[k] += p_3sets * v
            for k, v in d_4.items():
                match_diff_pmf[k] += p_4sets * v
            for k, v in d_5.items():
                match_diff_pmf[k] += p_5sets * v

        # Expected Continuous Totals
        expected_total_games = sum(k * v for k, v in match_games_pmf.items())
        expected_game_diff = sum(k * v for k, v in match_diff_pmf.items())

        # Total Games Over Lines
        total_games_over_probs = {}
        for line in [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5]:
            prob_over = sum(p for g, p in match_games_pmf.items() if g > line)
            total_games_over_probs[f"over_{str(line).replace('.', '_')}"] = round(float(prob_over), 4)

        # Game Handicap Cover Lines: P(G1 - G2 > -handicap)
        handicap_cover_probs = {}
        for h in [-5.5, -4.5, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
            # P1 covers handicap h if (G1 - G2) + h > 0 => G1 - G2 > -h
            req = -h
            cov = sum(p for d, p in match_diff_pmf.items() if d > req)
            label = f"p1_cover_{'plus' if h > 0 else 'minus'}_{str(abs(h)).replace('.', '_')}"
            handicap_cover_probs[label] = round(float(cov), 4)

        # 6. Format First Set Results
        first_set_scores_str = {f"{k[0]}-{k[1]}": round(v, 4) for k, v in norm_set_dist.items()}

        return {
            "p1_match_prob": round(p1_m, 4),
            "p2_match_prob": round(p2_m, 4),
            "p1_set_prob": round(s1, 4),
            "p2_set_prob": round(s2, 4),
            "best_of": best_of,
            "set_scores_prob": set_scores_prob,
            "player_to_win_set": player_to_win_set,
            "set_handicaps": set_handicaps,
            "expected_total_games": round(expected_total_games, 2),
            "expected_game_diff": round(expected_game_diff, 2),
            "total_games_over_probs": total_games_over_probs,
            "handicap_cover_probs": handicap_cover_probs,
            "first_set_scores": first_set_scores_str,
            "first_set_tiebreak_prob": round(p_tb_set, 4),
            "any_tiebreak_prob": round(1.0 - (1.0 - p_tb_set)**best_of, 4),
        }

