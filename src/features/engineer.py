"""State-of-the-Art Feature Engineering Module for ATP and WTA Tennis.

Generates leak-free pre-match predictive features for:
1. Match Winner (Moneyline)
2. Set Markets (Set Handicap, Total Sets, First Set Winner)
3. Game Markets (Game Handicap, Total Games Over/Under)

Strictly respects chronological boundaries with zero future data leakage.
"""

from __future__ import annotations

import logging
import math
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from configs.config import BASE_DIR, ELO_CONFIG, ROLLING_WINDOWS, COURT_PACE_CONFIG
    from src.ratings.surface_elo import SurfaceElo
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import BASE_DIR, ELO_CONFIG, ROLLING_WINDOWS, COURT_PACE_CONFIG
    from src.ratings.surface_elo import SurfaceElo

logger = logging.getLogger(__name__)


@dataclass
class MatchHistoryEntry:
    """Historical record of a single completed match for a player."""
    date: pd.Timestamp
    surface: str
    won: bool
    opponent: str
    opp_elo: float
    score: str
    sets_won: int
    sets_lost: int
    games_won: int
    games_lost: int
    had_tiebreak: bool
    minutes: float
    # Serve / Return box stats (0 if unrecorded)
    svpt: int = 0
    first_in: int = 0
    first_won: int = 0
    second_won: int = 0
    sv_gms: int = 0
    bp_saved: int = 0
    bp_faced: int = 0
    opp_svpt: int = 0
    opp_first_won: int = 0
    opp_second_won: int = 0
    opp_sv_gms: int = 0
    opp_bp_saved: int = 0
    opp_bp_faced: int = 0
    aces: int = 0
    dfs: int = 0


class PlayerHistory:
    """Maintains player match history, Elo peaks, and streak telemetry."""

    def __init__(self, player_name: str, base_elo: float = 1500.0) -> None:
        self.player_name = player_name
        self.base_elo = base_elo
        self.matches: Deque[MatchHistoryEntry] = deque(maxlen=200)

        # Elo tracking
        self.peak_elo_overall: float = base_elo
        self.peak_elo_surface: Dict[str, float] = {
            "Hard": base_elo,
            "Clay": base_elo,
            "Grass": base_elo,
        }
        self.elo_overall_history: Deque[float] = deque(maxlen=50)

        # Streak: positive = win streak, negative = loss streak
        self.streak: int = 0
        self.last_match_date: Optional[pd.Timestamp] = None

        # Static / Slow-changing attributes
        self.hand: str = "R"
        self.height: float = np.nan

    def update_elo_peak(self, current_overall: float, current_surface: float, surface: str) -> None:
        """Update historical peak ratings."""
        if current_overall > self.peak_elo_overall:
            self.peak_elo_overall = current_overall
        if surface in self.peak_elo_surface and current_surface > self.peak_elo_surface[surface]:
            self.peak_elo_surface[surface] = current_surface
        self.elo_overall_history.append(current_overall)

    def record_match(self, entry: MatchHistoryEntry) -> None:
        """Append match record and update streak/date."""
        self.matches.append(entry)
        self.last_match_date = entry.date

        if entry.won:
            self.streak = self.streak + 1 if self.streak > 0 else 1
        else:
            self.streak = self.streak - 1 if self.streak < 0 else -1


class TennisFeatureEngineer:
    """Production-grade feature engineering engine for professional tennis prediction.

    Generates rich, leak-free feature vectors for pre-match inference and
    historical machine learning training datasets.
    """

    # Tour baseline averages for Bayesian shrinkage / unrated priors
    TOUR_DEFAULTS = {
        "1st_in_pct": 0.62,
        "1st_won_pct": 0.70,
        "2nd_won_pct": 0.51,
        "hold_pct": 0.78,
        "break_pct": 0.22,
        "bp_conv_pct": 0.40,
        "bp_save_pct": 0.60,
        "ace_rate": 0.07,
        "df_rate": 0.03,
        "games_per_set": 9.75,
        "tiebreak_freq": 0.18,
    }

    def __init__(
        self,
        elo_model: Optional[SurfaceElo] = None,
        rolling_windows: Optional[List[int]] = None,
        h2h_decay_half_life_days: float = 730.0,  # 2 years
    ) -> None:
        """Initialize the feature engineer.

        Args:
            elo_model: Instance of SurfaceElo (instantiated if None).
            rolling_windows: Match count windows for rolling form/stats (default: [5, 10, 20]).
            h2h_decay_half_life_days: Recency decay half-life for historical head-to-head meetings.
        """
        self.elo = elo_model if elo_model is not None else SurfaceElo()
        self.rolling_windows = rolling_windows or [5, 10, 20]
        self.h2h_decay_half_life_days = h2h_decay_half_life_days

        # Player telemetry state
        self.players: Dict[str, PlayerHistory] = {}
        # H2H tracking: frozenset({p1, p2}) -> list of (date, winner, surface, score)
        self.h2h_history: Dict[frozenset, List[Dict[str, Any]]] = defaultdict(list)

    def _get_player(self, player_name: str) -> PlayerHistory:
        """Get or create player state tracking object."""
        if player_name not in self.players:
            self.players[player_name] = PlayerHistory(player_name, base_elo=self.elo.base_rating)
        return self.players[player_name]

    # =========================================================================
    # 1. Elo Features
    # =========================================================================

    def _extract_elo_features(
        self,
        p1: str,
        p2: str,
        surface: str,
        match_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """Compute pre-match Elo ratings, differences, peaks, and recent trends."""
        p1_hist = self._get_player(p1)
        p2_hist = self._get_player(p2)

        norm_surface = SurfaceElo._normalize_surface(surface)

        # Pre-match ratings
        p1_overall = self.elo.get_rating(p1, surface="Overall", date=match_date, blended=False)
        p2_overall = self.elo.get_rating(p2, surface="Overall", date=match_date, blended=False)

        p1_surface = self.elo.get_rating(p1, surface=norm_surface, date=match_date, blended=False)
        p2_surface = self.elo.get_rating(p2, surface=norm_surface, date=match_date, blended=False)

        p1_blended = self.elo.get_rating(p1, surface=norm_surface, date=match_date, blended=True)
        p2_blended = self.elo.get_rating(p2, surface=norm_surface, date=match_date, blended=True)

        win_prob_p1 = self.elo.get_win_probability(p1, p2, surface=norm_surface, date=match_date)

        # Elo peaks and distance from peak (negative or zero)
        p1_peak_overall = p1_hist.peak_elo_overall
        p2_peak_overall = p2_hist.peak_elo_overall
        p1_dist_peak_overall = p1_overall - p1_peak_overall
        p2_dist_peak_overall = p2_overall - p2_peak_overall

        p1_peak_surf = p1_hist.peak_elo_surface.get(norm_surface, self.elo.base_rating)
        p2_peak_surf = p2_hist.peak_elo_surface.get(norm_surface, self.elo.base_rating)
        p1_dist_peak_surf = p1_surface - p1_peak_surf
        p2_dist_peak_surf = p2_surface - p2_peak_surf

        # Elo trends / slopes over last 5 and 10 matches
        def _get_trend(hist: PlayerHistory, n: int) -> float:
            if len(hist.elo_overall_history) >= n:
                return float(hist.elo_overall_history[-1] - hist.elo_overall_history[-n])
            elif len(hist.elo_overall_history) > 1:
                return float(hist.elo_overall_history[-1] - hist.elo_overall_history[0])
            return 0.0

        p1_trend_5 = _get_trend(p1_hist, 5)
        p2_trend_5 = _get_trend(p2_hist, 5)
        p1_trend_10 = _get_trend(p1_hist, 10)
        p2_trend_10 = _get_trend(p2_hist, 10)

        return {
            "p1_elo_overall": p1_overall,
            "p2_elo_overall": p2_overall,
            "diff_elo_overall": p1_overall - p2_overall,
            "p1_elo_surface": p1_surface,
            "p2_elo_surface": p2_surface,
            "diff_elo_surface": p1_surface - p2_surface,
            "p1_elo_blended": p1_blended,
            "p2_elo_blended": p2_blended,
            "diff_elo_blended": p1_blended - p2_blended,
            "elo_win_prob_p1": win_prob_p1,
            "p1_elo_peak_overall": p1_peak_overall,
            "p2_elo_peak_overall": p2_peak_overall,
            "p1_dist_peak_overall": p1_dist_peak_overall,
            "p2_dist_peak_overall": p2_dist_peak_overall,
            "p1_dist_peak_surf": p1_dist_peak_surf,
            "p2_dist_peak_surf": p2_dist_peak_surf,
            "p1_elo_trend_5": p1_trend_5,
            "p2_elo_trend_5": p2_trend_5,
            "diff_elo_trend_5": p1_trend_5 - p2_trend_5,
            "p1_elo_trend_10": p1_trend_10,
            "p2_trend_10": p2_trend_10,
            "diff_elo_trend_10": p1_trend_10 - p2_trend_10,
        }

    # =========================================================================
    # 2. Serve & Return Features (Rolling)
    # =========================================================================

    def _extract_serve_return_features(
        self,
        player_name: str,
        prefix: str,
        n_matches: int = 15,
    ) -> Dict[str, float]:
        """Compute rolling serve hold, break, and dominance metrics."""
        hist = self._get_player(player_name)
        recent_matches = list(hist.matches)[-n_matches:]

        if not recent_matches:
            # Return tour baseline priors
            return {
                f"{prefix}_1st_in_pct": self.TOUR_DEFAULTS["1st_in_pct"],
                f"{prefix}_1st_won_pct": self.TOUR_DEFAULTS["1st_won_pct"],
                f"{prefix}_2nd_won_pct": self.TOUR_DEFAULTS["2nd_won_pct"],
                f"{prefix}_hold_pct": self.TOUR_DEFAULTS["hold_pct"],
                f"{prefix}_break_pct": self.TOUR_DEFAULTS["break_pct"],
                f"{prefix}_bp_conv_pct": self.TOUR_DEFAULTS["bp_conv_pct"],
                f"{prefix}_bp_save_pct": self.TOUR_DEFAULTS["bp_save_pct"],
                f"{prefix}_ace_rate": self.TOUR_DEFAULTS["ace_rate"],
                f"{prefix}_df_rate": self.TOUR_DEFAULTS["df_rate"],
                f"{prefix}_dominance_ratio": 1.0,
            }

        tot_svpt = sum(m.svpt for m in recent_matches)
        tot_1st_in = sum(m.first_in for m in recent_matches)
        tot_1st_won = sum(m.first_won for m in recent_matches)
        tot_2nd_won = sum(m.second_won for m in recent_matches)
        tot_svgms = sum(m.sv_gms for m in recent_matches)
        tot_bp_saved = sum(m.bp_saved for m in recent_matches)
        tot_bp_faced = sum(m.bp_faced for m in recent_matches)
        tot_aces = sum(m.aces for m in recent_matches)
        tot_dfs = sum(m.dfs for m in recent_matches)

        # Return stats (opponent serve metrics faced)
        tot_opp_svpt = sum(m.opp_svpt for m in recent_matches)
        tot_opp_1st_won = sum(m.opp_first_won for m in recent_matches)
        tot_opp_2nd_won = sum(m.opp_second_won for m in recent_matches)
        tot_opp_svgms = sum(m.opp_sv_gms for m in recent_matches)
        tot_opp_bp_saved = sum(m.opp_bp_saved for m in recent_matches)
        tot_opp_bp_faced = sum(m.opp_bp_faced for m in recent_matches)

        # Calculate with smooth Bayesian priors
        first_in_pct = (tot_1st_in + 62) / (tot_svpt + 100) if tot_svpt > 0 else self.TOUR_DEFAULTS["1st_in_pct"]
        first_won_pct = (tot_1st_won + 70) / (tot_1st_in + 100) if tot_1st_in > 0 else self.TOUR_DEFAULTS["1st_won_pct"]
        second_in = tot_svpt - tot_1st_in
        second_won_pct = (tot_2nd_won + 25) / (second_in + 50) if second_in > 0 else self.TOUR_DEFAULTS["2nd_won_pct"]

        breaks_conceded = tot_bp_faced - tot_bp_saved
        hold_pct = (tot_svgms - breaks_conceded + 16) / (tot_svgms + 20) if tot_svgms > 0 else self.TOUR_DEFAULTS["hold_pct"]

        breaks_achieved = tot_opp_bp_faced - tot_opp_bp_saved
        break_pct = (breaks_achieved + 4) / (tot_opp_svgms + 20) if tot_opp_svgms > 0 else self.TOUR_DEFAULTS["break_pct"]

        bp_conv_pct = (breaks_achieved + 4) / (tot_opp_bp_faced + 10) if tot_opp_bp_faced > 0 else self.TOUR_DEFAULTS["bp_conv_pct"]
        bp_save_pct = (tot_bp_saved + 6) / (tot_bp_faced + 10) if tot_bp_faced > 0 else self.TOUR_DEFAULTS["bp_save_pct"]

        ace_rate = tot_aces / tot_svpt if tot_svpt > 0 else self.TOUR_DEFAULTS["ace_rate"]
        df_rate = tot_dfs / tot_svpt if tot_svpt > 0 else self.TOUR_DEFAULTS["df_rate"]

        # Dominance ratio: % return points won / % return points conceded
        opp_ret_pts_won = tot_svpt - (tot_1st_won + tot_2nd_won)
        my_ret_pts_won = tot_opp_svpt - (tot_opp_1st_won + tot_opp_2nd_won)
        pct_my_ret_won = my_ret_pts_won / tot_opp_svpt if tot_opp_svpt > 0 else 0.35
        pct_opp_ret_won = opp_ret_pts_won / tot_svpt if tot_svpt > 0 else 0.35
        dominance_ratio = float(np.clip(pct_my_ret_won / max(0.01, pct_opp_ret_won), 0.5, 2.5))

        return {
            f"{prefix}_1st_in_pct": float(first_in_pct),
            f"{prefix}_1st_won_pct": float(first_won_pct),
            f"{prefix}_2nd_won_pct": float(second_won_pct),
            f"{prefix}_hold_pct": float(hold_pct),
            f"{prefix}_break_pct": float(break_pct),
            f"{prefix}_bp_conv_pct": float(bp_conv_pct),
            f"{prefix}_bp_save_pct": float(bp_save_pct),
            f"{prefix}_ace_rate": float(ace_rate),
            f"{prefix}_df_rate": float(df_rate),
            f"{prefix}_dominance_ratio": dominance_ratio,
        }

    # =========================================================================
    # 3. Form & Momentum Features
    # =========================================================================

    def _extract_form_features(
        self,
        player_name: str,
        prefix: str,
        surface: str,
    ) -> Dict[str, float]:
        """Compute rolling win rates, surface form, quality-weighted form, and streaks."""
        hist = self._get_player(player_name)
        matches = list(hist.matches)
        norm_surface = SurfaceElo._normalize_surface(surface)

        features: Dict[str, float] = {}

        # Rolling win rates across configurable windows (5, 10, 20)
        for w in self.rolling_windows:
            recent_w = matches[-w:]
            if recent_w:
                win_rate = sum(1 for m in recent_w if m.won) / len(recent_w)
            else:
                win_rate = 0.50
            features[f"{prefix}_win_rate_{w}"] = float(win_rate)

        # Surface-specific form (last 10 surface matches)
        surface_matches = [m for m in matches if m.surface == norm_surface][-10:]
        if surface_matches:
            surf_win_rate = sum(1 for m in surface_matches if m.won) / len(surface_matches)
        else:
            surf_win_rate = 0.50
        features[f"{prefix}_surface_win_rate_10"] = float(surf_win_rate)

        # Quality-weighted form: wins weighted by opponent rating relative to base 1500
        recent_10 = matches[-10:]
        if recent_10:
            quality_sum = 0.0
            for m in recent_10:
                if m.won:
                    # Scaling factor: win over 2000 Elo is worth ~1.33x, win over 1200 is ~0.8x
                    factor = max(0.5, m.opp_elo / self.elo.base_rating)
                    quality_sum += factor
                else:
                    # Loss penalty: losing to high-rated player has milder penalty
                    factor = min(1.5, self.elo.base_rating / max(500.0, m.opp_elo))
                    quality_sum -= 0.5 * factor
            quality_form = float(np.clip(quality_sum / len(recent_10), -1.0, 1.5))
        else:
            quality_form = 0.0
        features[f"{prefix}_quality_form_10"] = quality_form

        # Win / Loss streak
        features[f"{prefix}_streak"] = float(hist.streak)
        features[f"{prefix}_total_prior_matches"] = float(len(matches))

        return features

    # =========================================================================
    # 4. Head-to-Head (H2H) Features
    # =========================================================================

    def _extract_h2h_features(
        self,
        p1: str,
        p2: str,
        surface: str,
        match_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """Compute all-time, surface-specific, and recency-decayed H2H stats."""
        norm_surface = SurfaceElo._normalize_surface(surface)
        pair_key = frozenset([p1, p2])
        meetings = self.h2h_history.get(pair_key, [])

        if not meetings:
            return {
                "h2h_matches": 0.0,
                "h2h_p1_wins": 0.0,
                "h2h_p1_win_pct": 0.50,  # Unbiased prior
                "h2h_surface_matches": 0.0,
                "h2h_surface_p1_wins": 0.0,
                "h2h_surface_p1_win_pct": 0.50,
                "h2h_recency_weighted_p1_win_pct": 0.50,
            }

        total_matches = len(meetings)
        p1_wins = sum(1 for m in meetings if m["winner"] == p1)
        p1_win_pct = p1_wins / total_matches

        # Surface-specific meetings
        surf_meetings = [m for m in meetings if m["surface"] == norm_surface]
        surf_matches = len(surf_meetings)
        surf_p1_wins = sum(1 for m in surf_meetings if m["winner"] == p1)
        surf_p1_win_pct = surf_p1_wins / surf_matches if surf_matches > 0 else 0.50

        # Recency-weighted H2H using exponential decay
        if match_date is not None:
            weights_sum = 0.0
            weighted_p1_wins = 0.0
            for m in meetings:
                diff_days = max(0, (match_date - m["date"]).days)
                weight = 0.5 ** (diff_days / self.h2h_decay_half_life_days)
                weights_sum += weight
                if m["winner"] == p1:
                    weighted_p1_wins += weight
            recency_p1_win_pct = weighted_p1_wins / weights_sum if weights_sum > 0 else p1_win_pct
        else:
            recency_p1_win_pct = p1_win_pct

        return {
            "h2h_matches": float(total_matches),
            "h2h_p1_wins": float(p1_wins),
            "h2h_p1_win_pct": float(p1_win_pct),
            "h2h_surface_matches": float(surf_matches),
            "h2h_surface_p1_wins": float(surf_p1_wins),
            "h2h_surface_p1_win_pct": float(surf_p1_win_pct),
            "h2h_recency_weighted_p1_win_pct": float(recency_p1_win_pct),
        }

    # =========================================================================
    # 5. Fatigue & Schedule Features
    # =========================================================================

    def _extract_fatigue_features(
        self,
        player_name: str,
        prefix: str,
        match_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, float]:
        """Compute rest days, back-to-back flags, and match load in last 7/14 days."""
        hist = self._get_player(player_name)
        matches = list(hist.matches)

        if not matches or match_date is None or pd.isna(match_date):
            return {
                f"{prefix}_days_rest": 7.0,  # Neutral rest prior
                f"{prefix}_is_back_to_back": 0.0,
                f"{prefix}_matches_last_7d": 0.0,
                f"{prefix}_matches_last_14d": 0.0,
                f"{prefix}_sets_last_7d": 0.0,
                f"{prefix}_sets_last_14d": 0.0,
                f"{prefix}_minutes_last_7d": 0.0,
            }

        last_match = matches[-1]
        days_rest = max(0, (match_date - last_match.date).days)
        is_back_to_back = 1.0 if days_rest <= 1 else 0.0

        # Load in last 7 and 14 days
        cutoff_7d = match_date - pd.Timedelta(days=7)
        cutoff_14d = match_date - pd.Timedelta(days=14)

        matches_7d = [m for m in matches if m.date >= cutoff_7d]
        matches_14d = [m for m in matches if m.date >= cutoff_14d]

        sets_7d = sum(m.sets_won + m.sets_lost for m in matches_7d)
        sets_14d = sum(m.sets_won + m.sets_lost for m in matches_14d)
        mins_7d = sum(m.minutes for m in matches_7d)

        return {
            f"{prefix}_days_rest": float(np.clip(days_rest, 0, 90)),
            f"{prefix}_is_back_to_back": is_back_to_back,
            f"{prefix}_matches_last_7d": float(len(matches_7d)),
            f"{prefix}_matches_last_14d": float(len(matches_14d)),
            f"{prefix}_sets_last_7d": float(sets_7d),
            f"{prefix}_sets_last_14d": float(sets_14d),
            f"{prefix}_minutes_last_7d": float(mins_7d),
        }

    # =========================================================================
    # 6. Set & Game Market Features
    # =========================================================================

    def _extract_set_game_features(
        self,
        p1: str,
        p2: str,
        p1_stats: Dict[str, float],
        p2_stats: Dict[str, float],
        best_of: int = 3,
    ) -> Dict[str, float]:
        """Compute expected hold rates, game dominance, and total game line estimators."""
        p1_hist = self._get_player(p1)
        p2_hist = self._get_player(p2)

        # Average games per set over last 15 matches
        def _get_games_per_set(hist: PlayerHistory) -> float:
            recent = list(hist.matches)[-15:]
            if not recent:
                return self.TOUR_DEFAULTS["games_per_set"]
            total_sets = sum(m.sets_won + m.sets_lost for m in recent)
            total_games = sum(m.games_won + m.games_lost for m in recent)
            return total_games / total_sets if total_sets > 0 else self.TOUR_DEFAULTS["games_per_set"]

        # Tiebreak frequency
        def _get_tb_freq(hist: PlayerHistory) -> float:
            recent = list(hist.matches)[-15:]
            if not recent:
                return self.TOUR_DEFAULTS["tiebreak_freq"]
            total_sets = sum(m.sets_won + m.sets_lost for m in recent)
            tb_sets = sum(1 for m in recent if m.had_tiebreak)
            return tb_sets / total_sets if total_sets > 0 else self.TOUR_DEFAULTS["tiebreak_freq"]

        p1_gps = _get_games_per_set(p1_hist)
        p2_gps = _get_games_per_set(p2_hist)
        p1_tb = _get_tb_freq(p1_hist)
        p2_tb = _get_tb_freq(p2_hist)

        # Interaction hold rates: P1 hold facing P2 return
        tour_hold = self.TOUR_DEFAULTS["hold_pct"]
        p1_hold = p1_stats.get("p1_hold_pct", tour_hold)
        p2_break = p2_stats.get("p2_break_pct", self.TOUR_DEFAULTS["break_pct"])
        p2_hold = p2_stats.get("p2_hold_pct", tour_hold)
        p1_break = p1_stats.get("p1_break_pct", self.TOUR_DEFAULTS["break_pct"])

        # Opponent-adjusted expected hold probabilities
        exp_hold_p1 = float(np.clip(p1_hold + (tour_hold - p2_break) * 0.5, 0.35, 0.98))
        exp_hold_p2 = float(np.clip(p2_hold + (tour_hold - p1_break) * 0.5, 0.35, 0.98))

        combined_hold_strength = (exp_hold_p1 + exp_hold_p2) / 2.0
        expected_game_dominance = exp_hold_p1 - exp_hold_p2

        # Expected games per set formula: high hold strength -> close sets (approx 10-12 games)
        # return-dominant -> breaks -> shorter sets (approx 8.5-9.5 games)
        est_gps = 6.0 + 5.5 * combined_hold_strength
        est_match_games = est_gps * (2.4 if best_of == 3 else 3.8)

        return {
            "p1_avg_games_per_set": p1_gps,
            "p2_avg_games_per_set": p2_gps,
            "diff_avg_games_per_set": p1_gps - p2_gps,
            "p1_tiebreak_freq": p1_tb,
            "p2_tiebreak_freq": p2_tb,
            "avg_tiebreak_freq": (p1_tb + p2_tb) / 2.0,
            "expected_hold_p1": exp_hold_p1,
            "expected_hold_p2": exp_hold_p2,
            "expected_combined_hold": combined_hold_strength,
            "expected_game_dominance": expected_game_dominance,
            "est_match_total_games": float(est_match_games),
        }

    @staticmethod
    def resolve_court_pace(
        surface: str,
        tournament_name: str = "",
        is_indoor: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Resolve court speed profile, CPI, and indoor status from tournament metadata."""
        norm_surf = SurfaceElo._normalize_surface(surface)
        t_lower = str(tournament_name).lower() if tournament_name else ""

        # Determine indoor status
        if is_indoor is not None:
            indoor_flag = bool(is_indoor)
        else:
            indoor_keywords = COURT_PACE_CONFIG.get("indoor_keywords", [])
            indoor_flag = any(kw in t_lower for kw in indoor_keywords)

        # Determine surface profile
        profiles = COURT_PACE_CONFIG.get("surface_profiles", {})
        if norm_surf == "Grass":
            prof = profiles.get("Grass", {"cpi": 43.0, "category": "Fast", "hold_boost": 0.035, "tiebreak_boost": 0.050, "games_shift": 0.8})
        elif norm_surf == "Clay":
            prof = profiles.get("Clay", {"cpi": 22.0, "category": "Slow", "hold_boost": -0.045, "tiebreak_boost": -0.060, "games_shift": -0.9})
        elif norm_surf == "Carpet":
            prof = profiles.get("Carpet", {"cpi": 42.0, "category": "Fast", "hold_boost": 0.035, "tiebreak_boost": 0.050, "games_shift": 0.8})
        else:  # Hard
            if indoor_flag:
                prof = profiles.get("Indoor Hard", {"cpi": 40.0, "category": "Fast", "hold_boost": 0.030, "tiebreak_boost": 0.040, "games_shift": 0.7})
            elif any(kw in t_lower for kw in COURT_PACE_CONFIG.get("fast_hard_keywords", [])):
                prof = profiles.get("Outdoor Hard (Fast)", {"cpi": 38.0, "category": "Medium-Fast", "hold_boost": 0.015, "tiebreak_boost": 0.020, "games_shift": 0.4})
            elif any(kw in t_lower for kw in COURT_PACE_CONFIG.get("slow_hard_keywords", [])):
                prof = profiles.get("Outdoor Hard (Slow)", {"cpi": 28.0, "category": "Medium-Slow", "hold_boost": -0.015, "tiebreak_boost": -0.020, "games_shift": -0.4})
            else:
                prof = profiles.get("Hard", {"cpi": 34.0, "category": "Medium", "hold_boost": 0.0, "tiebreak_boost": 0.0, "games_shift": 0.0})

        return {
            "is_indoor": 1.0 if indoor_flag else 0.0,
            "court_pace_index": float(prof["cpi"]),
            "court_speed_category": str(prof["category"]),
            "pace_hold_boost": float(prof["hold_boost"]),
            "pace_tiebreak_boost": float(prof["tiebreak_boost"]),
            "pace_games_shift": float(prof["games_shift"]),
        }

    # =========================================================================
    # Public API: Extract Single Pre-Match Feature Vector
    # =========================================================================

    def extract_features_for_match(
        self,
        p1: str,
        p2: str,
        surface: str,
        tournament_level: str = "D",
        round_name: str = "R32",
        match_date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
        best_of: int = 3,
        p1_age: float = np.nan,
        p2_age: float = np.nan,
        p1_ht: float = np.nan,
        p2_ht: float = np.nan,
        p1_hand: str = "R",
        p2_hand: str = "R",
        tournament: str = "",
        is_indoor: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Compute the complete pre-match feature dictionary for player1 vs player2.

        Used for both historical model training and live daily inference.
        """
        ts_date = pd.to_datetime(match_date) if match_date is not None else None
        norm_surface = SurfaceElo._normalize_surface(surface)

        # Environmental court pace resolution
        pace_info = self.resolve_court_pace(norm_surface, tournament_name=tournament, is_indoor=is_indoor)

        features: Dict[str, Any] = {
            "player1": p1,
            "player2": p2,
            "surface": norm_surface,
            "tournament_level": tournament_level,
            "round": round_name,
            "best_of": best_of,
            "is_indoor": pace_info["is_indoor"],
            "court_pace_index": pace_info["court_pace_index"],
            "pace_hold_boost": pace_info["pace_hold_boost"],
        }

        # 1. Elo features
        elo_feats = self._extract_elo_features(p1, p2, norm_surface, ts_date)
        features.update(elo_feats)

        # 2. Serve & Return features (rolling 15 matches)
        p1_srv_ret = self._extract_serve_return_features(p1, prefix="p1")
        p2_srv_ret = self._extract_serve_return_features(p2, prefix="p2")
        features.update(p1_srv_ret)
        features.update(p2_srv_ret)

        # Serve/Return differentials
        for stat in ["1st_in_pct", "1st_won_pct", "2nd_won_pct", "hold_pct", "break_pct", "ace_rate", "df_rate"]:
            features[f"diff_{stat}"] = p1_srv_ret[f"p1_{stat}"] - p2_srv_ret[f"p2_{stat}"]

        # 3. Form & Momentum features
        p1_form = self._extract_form_features(p1, prefix="p1", surface=norm_surface)
        p2_form = self._extract_form_features(p2, prefix="p2", surface=norm_surface)
        features.update(p1_form)
        features.update(p2_form)

        for w in self.rolling_windows:
            features[f"diff_win_rate_{w}"] = p1_form[f"p1_win_rate_{w}"] - p2_form[f"p2_win_rate_{w}"]
        features["diff_streak"] = p1_form["p1_streak"] - p2_form["p2_streak"]
        features["diff_surface_win_rate_10"] = p1_form["p1_surface_win_rate_10"] - p2_form["p2_surface_win_rate_10"]

        # 4. H2H features
        h2h_feats = self._extract_h2h_features(p1, p2, norm_surface, ts_date)
        features.update(h2h_feats)

        # 5. Fatigue features
        p1_fatigue = self._extract_fatigue_features(p1, prefix="p1", match_date=ts_date)
        p2_fatigue = self._extract_fatigue_features(p2, prefix="p2", match_date=ts_date)
        features.update(p1_fatigue)
        features.update(p2_fatigue)
        features["diff_days_rest"] = p1_fatigue["p1_days_rest"] - p2_fatigue["p2_days_rest"]
        features["diff_sets_last_7d"] = p1_fatigue["p1_sets_last_7d"] - p2_fatigue["p2_sets_last_7d"]

        # 6. Attributes & Specialization
        features["p1_age"] = p1_age
        features["p2_age"] = p2_age
        features["diff_age"] = p1_age - p2_age if not (np.isnan(p1_age) or np.isnan(p2_age)) else 0.0
        features["diff_height"] = p1_ht - p2_ht if not (np.isnan(p1_ht) or np.isnan(p2_ht)) else 0.0
        features["is_lefty_vs_righty"] = 1.0 if (p1_hand == "L" and p2_hand == "R") or (p1_hand == "R" and p2_hand == "L") else 0.0

        # Surface specialization index (Surface Elo minus Overall Elo)
        features["p1_surface_spec"] = elo_feats["p1_elo_surface"] - elo_feats["p1_elo_overall"]
        features["p2_surface_spec"] = elo_feats["p2_elo_surface"] - elo_feats["p2_elo_overall"]
        features["diff_surface_spec"] = features["p1_surface_spec"] - features["p2_surface_spec"]

        # 7. Set & Game Market Features with court pace environmental adjustments
        set_game_feats = self._extract_set_game_features(p1, p2, p1_srv_ret, p2_srv_ret, best_of=best_of)
        set_game_feats["expected_hold_p1"] = float(np.clip(set_game_feats["expected_hold_p1"] + pace_info["pace_hold_boost"], 0.45, 0.95))
        set_game_feats["expected_hold_p2"] = float(np.clip(set_game_feats["expected_hold_p2"] + pace_info["pace_hold_boost"], 0.45, 0.95))
        set_game_feats["est_match_total_games"] = float(np.clip(set_game_feats["est_match_total_games"] + pace_info["pace_games_shift"], 16.0, 48.0))
        features.update(set_game_feats)

        return features

    # =========================================================================
    # Internal State Update: Update Player Histories After Match
    # =========================================================================

    def _update_player_states_after_match(
        self,
        winner: str,
        loser: str,
        surface: str,
        score: str,
        match_date: Optional[pd.Timestamp],
        minutes: float,
        row_dict: Dict[str, Any],
    ) -> None:
        """Update internal player histories and Elo ratings post-match."""
        norm_surface = SurfaceElo._normalize_surface(surface)
        w_hist = self._get_player(winner)
        l_hist = self._get_player(loser)

        # Parse score details
        clean_score = str(score) if score and not pd.isna(score) else ""
        w_sets = int(row_dict.get("winner_sets_won", 0))
        l_sets = int(row_dict.get("loser_sets_won", 0))
        w_games = int(row_dict.get("winner_games_won", 0))
        l_games = int(row_dict.get("loser_games_won", 0))
        had_tb = bool(row_dict.get("has_tiebreak", False))

        # Safe box stats
        def _get_val(k: str) -> int:
            v = row_dict.get(k, 0)
            return int(v) if not pd.isna(v) and str(v).replace(".", "", 1).isdigit() else 0

        w_svpt = _get_val("w_svpt")
        w_1stin = _get_val("w_1stIn")
        w_1stwon = _get_val("w_1stWon")
        w_2ndwon = _get_val("w_2ndWon")
        w_svgms = _get_val("w_SvGms")
        w_bpsaved = _get_val("w_bpSaved")
        w_bpfaced = _get_val("w_bpFaced")
        w_aces = _get_val("w_ace")
        w_dfs = _get_val("w_df")

        l_svpt = _get_val("l_svpt")
        l_1stin = _get_val("l_1stIn")
        l_1stwon = _get_val("l_1stWon")
        l_2ndwon = _get_val("l_2ndWon")
        l_svgms = _get_val("l_SvGms")
        l_bpsaved = _get_val("l_bpSaved")
        l_bpfaced = _get_val("l_bpFaced")
        l_aces = _get_val("l_ace")
        l_dfs = _get_val("l_df")

        # Winner history entry
        w_entry = MatchHistoryEntry(
            date=match_date if match_date is not None else pd.Timestamp.now(),
            surface=norm_surface,
            won=True,
            opponent=loser,
            opp_elo=self.elo.get_rating(loser, surface=norm_surface),
            score=clean_score,
            sets_won=w_sets,
            sets_lost=l_sets,
            games_won=w_games,
            games_lost=l_games,
            had_tiebreak=had_tb,
            minutes=minutes if not pd.isna(minutes) else 0.0,
            svpt=w_svpt,
            first_in=w_1stin,
            first_won=w_1stwon,
            second_won=w_2ndwon,
            sv_gms=w_svgms,
            bp_saved=w_bpsaved,
            bp_faced=w_bpfaced,
            opp_svpt=l_svpt,
            opp_first_won=l_1stwon,
            opp_second_won=l_2ndwon,
            opp_sv_gms=l_svgms,
            opp_bp_saved=l_bpsaved,
            opp_bp_faced=l_bpfaced,
            aces=w_aces,
            dfs=w_dfs,
        )
        w_hist.record_match(w_entry)

        # Loser history entry
        l_entry = MatchHistoryEntry(
            date=match_date if match_date is not None else pd.Timestamp.now(),
            surface=norm_surface,
            won=False,
            opponent=winner,
            opp_elo=self.elo.get_rating(winner, surface=norm_surface),
            score=clean_score,
            sets_won=l_sets,
            sets_lost=w_sets,
            games_won=l_games,
            games_lost=w_games,
            had_tiebreak=had_tb,
            minutes=minutes if not pd.isna(minutes) else 0.0,
            svpt=l_svpt,
            first_in=l_1stin,
            first_won=l_1stwon,
            second_won=l_2ndwon,
            sv_gms=l_svgms,
            bp_saved=l_bpsaved,
            bp_faced=l_bpfaced,
            opp_svpt=w_svpt,
            opp_first_won=w_1stwon,
            opp_second_won=w_2ndwon,
            opp_sv_gms=w_svgms,
            opp_bp_saved=w_bpsaved,
            opp_bp_faced=w_bpfaced,
            aces=l_aces,
            dfs=l_dfs,
        )
        l_hist.record_match(l_entry)

        # Record H2H meeting
        pair_key = frozenset([winner, loser])
        self.h2h_history[pair_key].append({
            "date": match_date,
            "winner": winner,
            "surface": norm_surface,
            "score": clean_score,
        })

        # Update SurfaceElo ratings
        elo_res = self.elo.update_match(
            winner=winner,
            loser=loser,
            surface=norm_surface,
            tournament_level=row_dict.get("tourney_level", "D"),
            round=row_dict.get("round", "R32"),
            score=clean_score,
            date=match_date,
        )

        # Update peak tracking
        w_hist.update_elo_peak(elo_res["winner_post_overall"], elo_res["winner_post_surface"], norm_surface)
        l_hist.update_elo_peak(elo_res["loser_post_overall"], elo_res["loser_post_surface"], norm_surface)

    # =========================================================================
    # Batch Processing: Chronological Historical Dataset Transformation
    # =========================================================================

    def transform_matches(
        self,
        dataframe: pd.DataFrame,
        symmetry_mode: str = "double",
        progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Transform historical match dataset into leak-free training dataset.

        Args:
            dataframe: Cleaned match DataFrame from TennisDataLoader.
            symmetry_mode:
                - 'double': Each match creates 2 rows (P1=Winner, target=1) and (P1=Loser, target=0)
                  Guarantees perfect class balance without symmetry artifacts.
                - 'random': Single row per match with coin flip deciding P1 vs P2.
                - 'winner_first': Single row per match where P1 is always the winner (target=1).
            progress_bar: Whether to display a tqdm progress bar.

        Returns:
            Engineered DataFrame with rich predictive feature columns and multi-market targets.
        """
        if dataframe.empty:
            return pd.DataFrame()

        df = dataframe.copy()

        # Ensure chronological sort
        date_col = "match_date" if "match_date" in df.columns else "tourney_date"
        df["parsed_match_date"] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df["round_rank"] = df["round"].astype(str).str.upper().map(SurfaceElo.ROUND_ORDER).fillna(4)

        sort_cols = ["parsed_match_date", "round_rank"]
        if "match_num" in df.columns:
            sort_cols.append("match_num")
        df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

        rows_out: List[Dict[str, Any]] = []
        iterator = df.itertuples(index=False)
        if progress_bar:
            iterator = tqdm(iterator, total=len(df), desc="Engineering features")

        for row in iterator:
            r = row._asdict()
            winner = str(r.get("winner_name", "")).strip()
            loser = str(r.get("loser_name", "")).strip()
            if not winner or not loser:
                continue

            surface = r.get("surface", "Hard")
            tourney_lvl = r.get("tourney_level", "D")
            rnd = r.get("round", "R32")
            best_of = int(r.get("best_of", 3))
            m_date = r.get("parsed_match_date", None)
            if pd.isna(m_date):
                m_date = None

            w_age = float(r.get("winner_age", np.nan))
            l_age = float(r.get("loser_age", np.nan))
            w_ht = float(r.get("winner_ht", np.nan))
            l_ht = float(r.get("loser_ht", np.nan))
            w_hand = str(r.get("winner_hand", "R")).upper()
            l_hand = str(r.get("loser_hand", "R")).upper()
            minutes = float(r.get("minutes", np.nan))
            score = str(r.get("score", ""))

            # Targets
            w_sets = int(r.get("winner_sets_won", 0))
            l_sets = int(r.get("loser_sets_won", 0))
            tot_sets = int(r.get("total_sets", 0))
            w_games = int(r.get("winner_games_won", 0))
            l_games = int(r.get("loser_games_won", 0))
            tot_games = int(r.get("total_games", 0))
            game_diff_w = w_games - l_games
            set_diff_w = w_sets - l_sets

            # Set 1 winner determination
            # If winner won set 1, target_set1_winner = 1
            first_set_w_won = 1
            if score:
                m_tokens = score.split()
                if m_tokens:
                    set1_match = re.search(r"(\d+)-(\d+)", m_tokens[0])
                    if set1_match:
                        first_set_w_won = 1 if int(set1_match.group(1)) > int(set1_match.group(2)) else 0

            # -----------------------------------------------------------------
            # Extract Pre-Match Features (STRICTLY BEFORE UPDATING STATE)
            # -----------------------------------------------------------------
            t_name = str(r.get("tourney_name", r.get("tournament", "")))

            if symmetry_mode == "double":
                # Row A: P1 = Winner, P2 = Loser, Target = 1
                feat_a = self.extract_features_for_match(
                    p1=winner, p2=loser, surface=surface, tournament_level=tourney_lvl,
                    round_name=rnd, match_date=m_date, best_of=best_of,
                    p1_age=w_age, p2_age=l_age, p1_ht=w_ht, p2_ht=l_ht, p1_hand=w_hand, p2_hand=l_hand,
                    tournament=t_name,
                )
                feat_a.update({
                    "match_date": m_date,
                    "target_p1_won": 1,
                    "target_total_sets": tot_sets,
                    "target_total_games": tot_games,
                    "target_game_diff": game_diff_w,
                    "target_set_diff": set_diff_w,
                    "target_set1_p1_won": first_set_w_won,
                    "is_completed": bool(r.get("is_completed", True)),
                })
                rows_out.append(feat_a)

                # Row B: P1 = Loser, P2 = Winner, Target = 0
                feat_b = self.extract_features_for_match(
                    p1=loser, p2=winner, surface=surface, tournament_level=tourney_lvl,
                    round_name=rnd, match_date=m_date, best_of=best_of,
                    p1_age=l_age, p2_age=w_age, p1_ht=l_ht, p2_ht=w_ht, p1_hand=l_hand, p2_hand=w_hand,
                    tournament=t_name,
                )
                feat_b.update({
                    "match_date": m_date,
                    "target_p1_won": 0,
                    "target_total_sets": tot_sets,
                    "target_total_games": tot_games,
                    "target_game_diff": -game_diff_w,
                    "target_set_diff": -set_diff_w,
                    "target_set1_p1_won": 1 - first_set_w_won,
                    "is_completed": bool(r.get("is_completed", True)),
                })
                rows_out.append(feat_b)

            elif symmetry_mode == "random":
                # Coin flip: 50% chance winner is P1, 50% chance loser is P1
                winner_is_p1 = random.random() > 0.5
                if winner_is_p1:
                    p1_name, p2_name = winner, loser
                    p1_a, p2_a = w_age, l_age
                    p1_h, p2_h = w_ht, l_ht
                    p1_hd, p2_hd = w_hand, l_hand
                    target_won = 1
                    g_diff = game_diff_w
                    s_diff = set_diff_w
                    s1_won = first_set_w_won
                else:
                    p1_name, p2_name = loser, winner
                    p1_a, p2_a = l_age, w_age
                    p1_h, p2_h = l_ht, w_ht
                    p1_hd, p2_hd = l_hand, w_hand
                    target_won = 0
                    g_diff = -game_diff_w
                    s_diff = -set_diff_w
                    s1_won = 1 - first_set_w_won

                feat = self.extract_features_for_match(
                    p1=p1_name, p2=p2_name, surface=surface, tournament_level=tourney_lvl,
                    round_name=rnd, match_date=m_date, best_of=best_of,
                    p1_age=p1_a, p2_age=p2_a, p1_ht=p1_h, p2_ht=p2_h, p1_hand=p1_hd, p2_hand=p2_hd,
                    tournament=t_name,
                )
                feat.update({
                    "match_date": m_date,
                    "target_p1_won": target_won,
                    "target_total_sets": tot_sets,
                    "target_total_games": tot_games,
                    "target_game_diff": g_diff,
                    "target_set_diff": s_diff,
                    "target_set1_p1_won": s1_won,
                    "is_completed": bool(r.get("is_completed", True)),
                })
                rows_out.append(feat)
                rows_out.append(feat)

            else:  # winner_first
                feat = self.extract_features_for_match(
                    p1=winner, p2=loser, surface=surface, tournament_level=tourney_lvl,
                    round_name=rnd, match_date=m_date, best_of=best_of,
                    p1_age=w_age, p2_age=l_age, p1_ht=w_ht, p2_ht=l_ht, p1_hand=w_hand, p2_hand=l_hand
                )
                feat.update({
                    "match_date": m_date,
                    "target_p1_won": 1,
                    "target_total_sets": tot_sets,
                    "target_total_games": tot_games,
                    "target_game_diff": game_diff_w,
                    "target_set_diff": set_diff_w,
                    "target_set1_p1_won": first_set_w_won,
                    "is_completed": bool(r.get("is_completed", True)),
                })
                rows_out.append(feat)

            # -----------------------------------------------------------------
            # Update Internal State Post-Match
            # -----------------------------------------------------------------
            self._update_player_states_after_match(
                winner=winner,
                loser=loser,
                surface=surface,
                score=score,
                match_date=m_date,
                minutes=minutes,
                row_dict=r,
            )

        df_out = pd.DataFrame(rows_out)
        logger.info(f"Feature engineering complete. Generated {len(df_out):,} rows with {len(df_out.columns)} columns.")
        return df_out
