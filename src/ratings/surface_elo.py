"""Production-ready Surface-Specific Elo rating system for ATP and WTA tennis.

Maintains multi-surface ratings (Overall, Hard, Clay, Grass), dynamic K-factors,
round weights, inactivity regression, margin-of-victory scaling, and leak-free
chronological backfill capabilities.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

try:
    from configs.config import ELO_CONFIG
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import ELO_CONFIG

logger = logging.getLogger(__name__)


class SurfaceElo:
    """Surface-Specific Elo rating engine for ATP & WTA professional tennis.

    Features:
    - Independent tracking for Overall, Hard, Clay, and Grass surfaces.
    - Tournament-tier K-factors (Grand Slam, Masters 1000, 500, 250, Challenger).
    - Round multipliers (Finals, Semifinals, Quarterfinals receive higher weight).
    - Exponential inactivity decay toward the base rating after extended layoffs.
    - Robust margin-of-victory scaling based on game dominance ratio.
    - Chronological pipeline backtesting with strict zero-leakage pre-match features.
    - Persistence via joblib / pickle.
    """

    VALID_SURFACES = {"Overall", "Hard", "Clay", "Grass"}

    # Canonical round chronological ordering (earlier rounds processed first)
    ROUND_ORDER: Dict[str, int] = {
        "R128": 1,
        "R64": 2,
        "R32": 3,
        "R16": 4,
        "QF": 5,
        "SF": 6,
        "BR": 7,  # Bronze medal / 3rd place playoff
        "F": 8,   # Finals
        "RR": 2,  # Round Robin in Year-End Championships
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        base_rating: Optional[float] = None,
        surface_weight: Optional[float] = None,
        decay_half_life_days: Optional[float] = None,
        decay_threshold_days: Optional[int] = None,
        margin_scaling_enabled: Optional[bool] = None,
    ) -> None:
        """Initialize the SurfaceElo engine.

        Args:
            config: Optional config dict overriding ELO_CONFIG defaults.
            base_rating: Default rating for unrated players (default: 1500.0).
            surface_weight: Weight given to surface rating in blended probabilities [0, 1].
            decay_half_life_days: Number of inactive days for deviation from base to halve.
            decay_threshold_days: Days of inactivity before decay kicks in.
            margin_scaling_enabled: Whether to scale K-factors by margin of victory.
        """
        cfg = dict(ELO_CONFIG)
        if config:
            cfg.update(config)

        self.base_rating: float = float(base_rating if base_rating is not None else cfg.get("base_rating", 1500.0))
        self.surface_weight: float = float(surface_weight if surface_weight is not None else cfg.get("surface_weight", 0.5))
        self.k_factors: Dict[str, float] = cfg.get("k_factors", {
            "grand_slam": 32.0,
            "masters_1000": 28.0,
            "tour_500": 24.0,
            "tour_250": 20.0,
            "challenger": 16.0,
            "default": 20.0,
        })
        self.round_multipliers: Dict[str, float] = cfg.get("round_multipliers", {
            "F": 1.20,
            "SF": 1.10,
            "QF": 1.05,
            "R16": 1.00,
            "R32": 1.00,
            "R64": 1.00,
            "R128": 1.00,
            "RR": 1.00,
            "default": 1.00,
        })

        # Inactivity settings
        self.inactivity_decay_enabled: bool = bool(cfg.get("inactivity_decay_enabled", True))
        self.decay_threshold_days: int = int(decay_threshold_days if decay_threshold_days is not None else cfg.get("decay_threshold_days", 60))
        self.decay_half_life_days: float = float(decay_half_life_days if decay_half_life_days is not None else cfg.get("decay_half_life_days", 365.0))

        # Margin-of-victory settings
        self.margin_scaling_enabled: bool = bool(margin_scaling_enabled if margin_scaling_enabled is not None else cfg.get("margin_scaling_enabled", True))
        self.margin_weight: float = float(cfg.get("margin_weight", 0.35))
        self.margin_multiplier_min: float = float(cfg.get("margin_multiplier_min", 0.75))
        self.margin_multiplier_max: float = float(cfg.get("margin_multiplier_max", 1.35))

        # Storage structures
        # player -> {'Overall': float, 'Hard': float, 'Clay': float, 'Grass': float}
        self.ratings: Dict[str, Dict[str, float]] = {}
        # player -> pd.Timestamp of last played match
        self.last_active: Dict[str, pd.Timestamp] = {}
        # player -> {'Overall': int, 'Hard': int, 'Clay': int, 'Grass': int}
        self.match_count: Dict[str, Dict[str, int]] = {}

    def _init_player(self, player: str) -> None:
        """Initialize records for a new player if not present."""
        if player not in self.ratings:
            self.ratings[player] = {surf: self.base_rating for surf in self.VALID_SURFACES}
            self.match_count[player] = {surf: 0 for surf in self.VALID_SURFACES}

    @classmethod
    def _normalize_surface(cls, surface: Optional[str]) -> str:
        """Map surface string to canonical categories: 'Hard', 'Clay', 'Grass', or 'Overall'."""
        if not surface or not isinstance(surface, str):
            return "Hard"
        s = surface.strip().capitalize()
        if s in cls.VALID_SURFACES:
            return s
        if "Clay" in s:
            return "Clay"
        if "Grass" in s:
            return "Grass"
        if "Carpet" in s or "Hard" in s or "Acrylic" in s:
            return "Hard"
        return "Hard"

    def _get_k_factor(self, tournament_level: Optional[str]) -> float:
        """Determine base K-factor based on tournament tier."""
        if not tournament_level or not isinstance(tournament_level, str):
            return self.k_factors.get("default", 20.0)

        level = tournament_level.strip().upper()
        if level in {"G", "GRAND SLAM", "GS"}:
            return self.k_factors.get("grand_slam", 32.0)
        elif level in {"M", "PM", "MASTERS", "MASTERS 1000", "1000"}:
            return self.k_factors.get("masters_1000", 28.0)
        elif level in {"A", "500", "P", "PREMIER 500", "ATP 500", "WTA 500"}:
            return self.k_factors.get("tour_500", 24.0)
        elif level in {"D", "250", "I", "INTERNATIONAL", "ATP 250", "WTA 250"}:
            return self.k_factors.get("tour_250", 20.0)
        elif level in {"C", "CHALLENGER", "CH", "F", "FUTURES"}:
            return self.k_factors.get("challenger", 16.0)

        return self.k_factors.get("default", 20.0)

    def _get_round_multiplier(self, round_name: Optional[str]) -> float:
        """Return multiplier based on tournament round importance."""
        if not round_name or not isinstance(round_name, str):
            return self.round_multipliers.get("default", 1.00)

        r = round_name.strip().upper()
        if r in {"F", "FINAL", "FINALS"}:
            return self.round_multipliers.get("F", 1.20)
        elif r in {"SF", "SEMIFINAL", "SEMIS", "SEMIFINALS"}:
            return self.round_multipliers.get("SF", 1.10)
        elif r in {"QF", "QUARTERFINAL", "QUARTERS", "QUARTERFINALS"}:
            return self.round_multipliers.get("QF", 1.05)
        elif r in self.round_multipliers:
            return self.round_multipliers[r]

        return self.round_multipliers.get("default", 1.00)

    def _parse_score(self, score: Optional[str]) -> Tuple[int, int, bool]:
        """Parse match score string to games won by winner and loser.

        Args:
            score: Raw score string, e.g., '6-4 3-6 7-6(5)', '6-2 6-0', 'W/O', 'RET'.

        Returns:
            Tuple of (winner_games, loser_games, is_complete).
        """
        if not score or not isinstance(score, str):
            return 0, 0, False

        score_clean = score.strip()
        is_incomplete = any(term in score_clean.upper() for term in ("RET", "W/O", "DEF", "ABN"))

        # Extract set scores like '6-4', '7-6(5)'
        tokens = score_clean.split()
        winner_games = 0
        loser_games = 0

        for token in tokens:
            match = re.search(r"(\d+)-(\d+)", token)
            if match:
                w_g = int(match.group(1))
                l_g = int(match.group(2))
                winner_games += w_g
                loser_games += l_g

        is_complete = not is_incomplete and (winner_games + loser_games) > 0
        return winner_games, loser_games, is_complete

    def _calculate_margin_multiplier(self, score: Optional[str]) -> float:
        """Compute K-factor multiplier based on match game dominance."""
        if not self.margin_scaling_enabled:
            return 1.0

        winner_games, loser_games, is_complete = self._parse_score(score)
        total_games = winner_games + loser_games
        if not is_complete or total_games == 0:
            return 1.0

        # Dominance ratio: (W - L) / (W + L), typically ~0.20 for standard 6-4 6-4 win
        dominance = (winner_games - loser_games) / total_games
        multiplier = 1.0 + self.margin_weight * (dominance - 0.20)
        return float(np.clip(multiplier, self.margin_multiplier_min, self.margin_multiplier_max))

    def _apply_inactivity_decay(self, player: str, current_date: Optional[Union[str, date, datetime, pd.Timestamp]]) -> None:
        """Regress player ratings toward base rating if inactive beyond threshold."""
        if not self.inactivity_decay_enabled or current_date is None:
            return

        if player not in self.ratings or player not in self.last_active:
            return

        last_date = self.last_active[player]
        if pd.isna(last_date):
            return

        ts_current = pd.to_datetime(current_date)
        inactive_days = (ts_current - last_date).days

        if inactive_days > self.decay_threshold_days:
            decay_days = inactive_days - self.decay_threshold_days
            decay_factor = 0.5 ** (decay_days / self.decay_half_life_days)

            for surf in self.VALID_SURFACES:
                curr = self.ratings[player][surf]
                self.ratings[player][surf] = self.base_rating + (curr - self.base_rating) * decay_factor

            # Update last_active to avoid compounding decay twice for the same gap
            self.last_active[player] = ts_current

    def get_rating(
        self,
        player: str,
        surface: Optional[str] = None,
        date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
        blended: bool = True,
    ) -> float:
        """Retrieve pre-match rating for a player.

        Args:
            player: Player name or identifier.
            surface: 'Hard', 'Clay', 'Grass', or None/'Overall'.
            date: Match date (applies inactivity decay if specified).
            blended: If True and surface is specified, blends surface Elo with overall Elo.

        Returns:
            Pre-match rating as float.
        """
        self._init_player(player)

        if date is not None:
            self._apply_inactivity_decay(player, date)

        norm_surface = self._normalize_surface(surface) if surface else "Overall"

        if norm_surface == "Overall" or not blended:
            return float(self.ratings[player][norm_surface])

        # Blended rating: (1 - w) * Overall + w * Surface
        overall = self.ratings[player]["Overall"]
        surf_val = self.ratings[player][norm_surface]
        return float((1.0 - self.surface_weight) * overall + self.surface_weight * surf_val)

    def get_win_probability(
        self,
        player1: str,
        player2: str,
        surface: Optional[str] = None,
        date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
    ) -> float:
        """Compute expected win probability of player1 against player2.

        Uses blended surface ratings if surface is provided.

        Args:
            player1: First player.
            player2: Second player.
            surface: Surface for match context.
            date: Match date.

        Returns:
            Win probability in [0.0, 1.0].
        """
        r1 = self.get_rating(player1, surface=surface, date=date, blended=True)
        r2 = self.get_rating(player2, surface=surface, date=date, blended=True)
        return float(1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0)))

    def update_match(
        self,
        winner: str,
        loser: str,
        surface: Optional[str],
        tournament_level: Optional[str],
        round: Optional[str],
        score: Optional[str] = None,
        date: Optional[Union[str, date, datetime, pd.Timestamp]] = None,
    ) -> Dict[str, Any]:
        """Update ratings following match completion.

        Both Overall and Surface-specific ratings are updated simultaneously.

        Args:
            winner: Winner name/id.
            loser: Loser name/id.
            surface: Match surface ('Hard', 'Clay', 'Grass').
            tournament_level: Tournament tier ('G', 'M', 'A', 'D', 'C').
            round: Round name ('F', 'SF', 'QF', etc.).
            score: Optional match score string for margin scaling.
            date: Match date for timestamping and decay.

        Returns:
            Dictionary containing pre-match ratings, post-match ratings, and deltas.
        """
        self._init_player(winner)
        self._init_player(loser)

        norm_surface = self._normalize_surface(surface)

        # Apply inactivity decay before computing pre-match ratings
        if date is not None:
            self._apply_inactivity_decay(winner, date)
            self._apply_inactivity_decay(loser, date)

        # Pre-match ratings
        pre_w_overall = self.ratings[winner]["Overall"]
        pre_l_overall = self.ratings[loser]["Overall"]
        pre_w_surface = self.ratings[winner][norm_surface]
        pre_l_surface = self.ratings[loser][norm_surface]

        pre_w_blended = self.get_rating(winner, surface=norm_surface, blended=True)
        pre_l_blended = self.get_rating(loser, surface=norm_surface, blended=True)
        p_win_blended = 1.0 / (1.0 + 10.0 ** ((pre_l_blended - pre_w_blended) / 400.0))

        # Expected win probabilities for Overall and Surface
        exp_w_overall = 1.0 / (1.0 + 10.0 ** ((pre_l_overall - pre_w_overall) / 400.0))
        exp_w_surface = 1.0 / (1.0 + 10.0 ** ((pre_l_surface - pre_w_surface) / 400.0))

        # Dynamic multipliers
        base_k = self._get_k_factor(tournament_level)
        round_mult = self._get_round_multiplier(round)
        margin_mult = self._calculate_margin_multiplier(score)

        effective_k = base_k * round_mult * margin_mult

        # Rating adjustments
        delta_overall = effective_k * (1.0 - exp_w_overall)
        delta_surface = effective_k * (1.0 - exp_w_surface)

        # Apply updates
        self.ratings[winner]["Overall"] += delta_overall
        self.ratings[loser]["Overall"] -= delta_overall
        self.ratings[winner][norm_surface] += delta_surface
        self.ratings[loser][norm_surface] -= delta_surface

        # Update match counts
        self.match_count[winner]["Overall"] += 1
        self.match_count[loser]["Overall"] += 1
        self.match_count[winner][norm_surface] += 1
        self.match_count[loser][norm_surface] += 1

        # Update last active timestamps
        if date is not None:
            ts = pd.to_datetime(date)
            self.last_active[winner] = ts
            self.last_active[loser] = ts

        return {
            "winner": winner,
            "loser": loser,
            "surface": norm_surface,
            "winner_pre_overall": pre_w_overall,
            "loser_pre_overall": pre_l_overall,
            "winner_pre_surface": pre_w_surface,
            "loser_pre_surface": pre_l_surface,
            "winner_pre_blended": pre_w_blended,
            "loser_pre_blended": pre_l_blended,
            "winner_prob": p_win_blended,
            "effective_k": effective_k,
            "delta_overall": delta_overall,
            "delta_surface": delta_surface,
            "winner_post_overall": self.ratings[winner]["Overall"],
            "loser_post_overall": self.ratings[loser]["Overall"],
            "winner_post_surface": self.ratings[winner][norm_surface],
            "loser_post_surface": self.ratings[loser][norm_surface],
        }

    def process_matches_chronologically(
        self,
        dataframe: pd.DataFrame,
        winner_col: str = "winner_name",
        loser_col: str = "loser_name",
        surface_col: str = "surface",
        tourney_level_col: str = "tourney_level",
        round_col: str = "round",
        score_col: str = "score",
        date_col: str = "tourney_date",
        match_num_col: Optional[str] = "match_num",
    ) -> pd.DataFrame:
        """Process matches chronologically and append pre/post ratings and win probabilities.

        Ensures strict time ordering and round sequencing to avoid lookahead bias.

        Args:
            dataframe: DataFrame containing historical match rows.
            winner_col: Column name for winner.
            loser_col: Column name for loser.
            surface_col: Column name for surface.
            tourney_level_col: Column name for tournament tier.
            round_col: Column name for round.
            score_col: Column name for score.
            date_col: Column name for date (supports YYYYMMDD or ISO dates).
            match_num_col: Optional column for match number within tournament.

        Returns:
            Enriched DataFrame with pre-match and post-match Elo ratings and probability columns.
        """
        df = dataframe.copy()

        # Parse date column for reliable chronological sorting
        if date_col in df.columns:
            # Handle integers like 20230116 or strings
            df["parsed_match_date"] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        else:
            df["parsed_match_date"] = pd.NaT

        # Add round sequencing column for intra-tournament ordering
        if round_col in df.columns:
            df["round_rank"] = df[round_col].astype(str).str.upper().map(self.ROUND_ORDER).fillna(4)
        else:
            df["round_rank"] = 0

        # Sort criteria
        sort_cols = ["parsed_match_date", "round_rank"]
        if match_num_col and match_num_col in df.columns:
            sort_cols.append(match_num_col)

        df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

        # Arrays to collect feature columns
        n_matches = len(df)
        w_pre_overall = np.empty(n_matches, dtype=np.float64)
        l_pre_overall = np.empty(n_matches, dtype=np.float64)
        w_pre_surface = np.empty(n_matches, dtype=np.float64)
        l_pre_surface = np.empty(n_matches, dtype=np.float64)
        w_pre_blended = np.empty(n_matches, dtype=np.float64)
        l_pre_blended = np.empty(n_matches, dtype=np.float64)
        elo_prob_winner = np.empty(n_matches, dtype=np.float64)
        elo_diff_blended = np.empty(n_matches, dtype=np.float64)
        w_post_overall = np.empty(n_matches, dtype=np.float64)
        l_post_overall = np.empty(n_matches, dtype=np.float64)

        for i, row in enumerate(df.itertuples(index=False)):
            row_dict = row._asdict()
            winner = str(row_dict.get(winner_col, "")).strip()
            loser = str(row_dict.get(loser_col, "")).strip()
            surface = row_dict.get(surface_col, "Hard")
            tourney_lvl = row_dict.get(tourney_level_col, "default")
            rnd = row_dict.get(round_col, "default")
            score = row_dict.get(score_col, None)
            m_date = row_dict.get("parsed_match_date", None)

            if pd.isna(m_date):
                m_date = None

            res = self.update_match(
                winner=winner,
                loser=loser,
                surface=surface,
                tournament_level=tourney_lvl,
                round=rnd,
                score=score,
                date=m_date,
            )

            w_pre_overall[i] = res["winner_pre_overall"]
            l_pre_overall[i] = res["loser_pre_overall"]
            w_pre_surface[i] = res["winner_pre_surface"]
            l_pre_surface[i] = res["loser_pre_surface"]
            w_pre_blended[i] = res["winner_pre_blended"]
            l_pre_blended[i] = res["loser_pre_blended"]
            elo_prob_winner[i] = res["winner_prob"]
            elo_diff_blended[i] = res["winner_pre_blended"] - res["loser_pre_blended"]
            w_post_overall[i] = res["winner_post_overall"]
            l_post_overall[i] = res["loser_post_overall"]

        # Append computed features
        df["winner_pre_elo_overall"] = w_pre_overall
        df["loser_pre_elo_overall"] = l_pre_overall
        df["winner_pre_elo_surface"] = w_pre_surface
        df["loser_pre_elo_surface"] = l_pre_surface
        df["winner_pre_elo_blended"] = w_pre_blended
        df["loser_pre_elo_blended"] = l_pre_blended
        df["elo_prob_winner"] = elo_prob_winner
        df["elo_diff_blended"] = elo_diff_blended
        df["winner_post_elo_overall"] = w_post_overall
        df["loser_post_elo_overall"] = l_post_overall

        # Drop internal sorting columns
        df.drop(columns=["parsed_match_date", "round_rank"], inplace=True)
        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current player ratings into a structured DataFrame.

        Returns:
            DataFrame indexed by player with columns for Overall, Hard, Clay, Grass,
            match counts, and last active date.
        """
        rows = []
        for player, r_dict in self.ratings.items():
            last_date = self.last_active.get(player, pd.NaT)
            m_dict = self.match_count.get(player, {})
            rows.append({
                "player": player,
                "elo_overall": round(r_dict["Overall"], 1),
                "elo_hard": round(r_dict["Hard"], 1),
                "elo_clay": round(r_dict["Clay"], 1),
                "elo_grass": round(r_dict["Grass"], 1),
                "matches_played": m_dict.get("Overall", 0),
                "matches_hard": m_dict.get("Hard", 0),
                "matches_clay": m_dict.get("Clay", 0),
                "matches_grass": m_dict.get("Grass", 0),
                "last_active": last_date,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="elo_overall", ascending=False).reset_index(drop=True)
        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the SurfaceElo model state to disk using joblib.

        Args:
            filepath: Destination file path (e.g. 'models/surface_elo.joblib').
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"SurfaceElo model successfully saved to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> SurfaceElo:
        """Load a persisted SurfaceElo instance from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded SurfaceElo instance.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"SurfaceElo checkpoint not found at: {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is {type(instance)}, expected {cls.__name__}")
        logger.info(f"SurfaceElo model loaded successfully from {path}")
        return instance
