"""Automated Live Tennis Fixtures and Daily Schedule Fetcher.

Fetches real scheduled ATP and WTA professional tournament matches from open public feeds:
1. ESPN Tennis Scoreboard API: Live tournament draws (Erste Bank Open, Swiss Indoors Basel, Toray Pan Pacific, Guangzhou, etc.)
2. Automatic parsing of Singles groupings, rounds, and surface mappings
3. Integration with SurfaceElo to generate consensus baseline market odds for edge calculation
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Surface mapping for major world tournaments
TOURNAMENT_SURFACE_MAP = {
    "australian open": "Hard",
    "roland garros": "Clay",
    "french open": "Clay",
    "wimbledon": "Grass",
    "us open": "Hard",
    "indian wells": "Hard",
    "miami": "Hard",
    "monte carlo": "Clay",
    "madrid": "Clay",
    "rome": "Clay",
    "canada": "Hard",
    "cincinnati": "Hard",
    "shanghai": "Hard",
    "paris": "Hard",
    "basel": "Hard",
    "vienna": "Hard",
    "erste bank": "Hard",
    "swiss indoors": "Hard",
    "beijing": "Hard",
    "tokyo": "Hard",
    "pan pacific": "Hard",
    "guangzhou": "Hard",
    "curitiba": "Clay",
    "valle": "Hard",
    "queretaro": "Hard",
    "barcelona": "Clay",
    "queen": "Grass",
    "halle": "Grass",
}


class TennisLiveFetcher:
    """Fetcher for live upcoming ATP & WTA fixtures and schedules."""

    ESPN_ATP_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    ESPN_WTA_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard"

    def __init__(self, timeout: int = 10) -> None:
        """Initialize fetcher with request timeout."""
        self.timeout = timeout

    def fetch_espn_daily_fixtures(
        self,
        tour: str = "both",
        only_scheduled: bool = False,
        target_date: Optional[str] = None,
        date_range: Optional[List[str]] = None,
        attach_baseline_odds: bool = True,
    ) -> pd.DataFrame:
        """Fetch live matches from ESPN free public tennis scoreboard endpoints.

        Traverses tournament events and their singles groupings to extract actual active matches.

        Args:
            tour: 'atp', 'wta', or 'both'.
            only_scheduled: If True, filters for STATUS_SCHEDULED matches (upcoming).
            target_date: Optional date string 'YYYY-MM-DD' to filter specific day.
            date_range: Optional list of date strings to filter.
            attach_baseline_odds: If True, synthesizes standard market consensus odds with 5% overround.

        Returns:
            DataFrame of real scheduled fixtures with actual event dates and scheduled times.
        """
        urls = []
        if tour.lower() in ("atp", "both"):
            urls.append((self.ESPN_ATP_URL, "ATP"))
        if tour.lower() in ("wta", "both"):
            urls.append((self.ESPN_WTA_URL, "WTA"))

        all_fixtures: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        for endpoint, tour_name in urls:
            try:
                resp = requests.get(endpoint, headers=headers, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.warning(f"ESPN {tour_name} endpoint returned status {resp.status_code}")
                    continue

                data = resp.json()
                events = data.get("events", [])

                for ev in events:
                    t_name = ev.get("name", "Tour Event")
                    surface = self._infer_surface(t_name)
                    tier = self._infer_level(t_name)

                    # ESPN nests matches in 'groupings' to separate Singles and Doubles
                    groupings = ev.get("groupings", [])
                    for grp in groupings:
                        slug = grp.get("grouping", {}).get("slug", "").lower()
                        # Only take Singles competitions
                        if "singles" not in slug:
                            continue

                        competitions = grp.get("competitions", [])
                        for comp in competitions:
                            st_type = comp.get("status", {}).get("type", {})
                            st_name = st_type.get("name", "")
                            st_detail = st_type.get("detail", "")
                            st_short = st_type.get("shortDetail", "")

                            c_date_raw = comp.get("date", ev.get("date", ""))
                            comp_date_iso = str(c_date_raw)[:10] if c_date_raw else datetime.now().strftime("%Y-%m-%d")
                            time_str = st_detail or st_short or "Scheduled"

                            if target_date and comp_date_iso != target_date:
                                continue
                            if date_range and comp_date_iso not in date_range:
                                continue
                            if only_scheduled and st_name not in ("STATUS_SCHEDULED", "STATUS_IN_PROGRESS"):
                                continue

                            competitors = comp.get("competitors", [])
                            if len(competitors) >= 2:
                                p1 = competitors[0].get("athlete", {}).get("displayName", "").strip()
                                p2 = competitors[1].get("athlete", {}).get("displayName", "").strip()

                                # Validate player names
                                if not p1 or not p2 or p1 == "TBD" or p2 == "TBD" or p1 == p2:
                                    continue

                                rnd_info = comp.get("round", {}).get("displayName", "R32")
                                round_norm = self._normalize_round(rnd_info)

                                all_fixtures.append({
                                    "tournament": t_name,
                                    "player1": p1,
                                    "player2": p2,
                                    "surface": surface,
                                    "tournament_level": tier,
                                    "round": round_norm,
                                    "best_of": 5 if ("open" in t_name.lower() or "wimbledon" in t_name.lower()) and tour_name == "ATP" else 3,
                                    "match_date": comp_date_iso,
                                    "match_time": time_str,
                                    "status": st_name,
                                    "tour": tour_name,
                                })

            except Exception as e:
                logger.warning(f"Error fetching live fixtures from ESPN {tour_name}: {e}")

        if not all_fixtures:
            logger.info("No live scheduled fixtures found currently on ESPN scoreboard.")
            return pd.DataFrame()

        df = pd.DataFrame(all_fixtures).drop_duplicates(subset=["match_date", "player1", "player2"]).reset_index(drop=True)
        logger.info(f"Successfully fetched {len(df)} real matches from ESPN.")

        # Attach realistic market consensus odds (with standard 5.0% bookmaker margin)
        if attach_baseline_odds:
            df = self.enrich_with_market_odds(df)

        return df

    @staticmethod
    def _normalize_round(round_text: str) -> str:
        """Standardize round names from ESPN."""
        r = round_text.lower()
        if "final" in r and "semi" not in r and "quarter" not in r:
            return "F"
        if "semi" in r:
            return "SF"
        if "quarter" in r:
            return "QF"
        if "16" in r or "third" in r:
            return "R16"
        if "32" in r or "second" in r:
            return "R32"
        if "64" in r or "first" in r:
            return "R64"
        if "128" in r:
            return "R128"
        return "R32"

    @staticmethod
    def _infer_surface(tournament_name: str) -> str:
        """Infer playing surface from tournament name."""
        tn_lower = tournament_name.lower()
        for key, surf in TOURNAMENT_SURFACE_MAP.items():
            if key in tn_lower:
                return surf
        return "Hard"

    @staticmethod
    def _infer_level(tournament_name: str) -> str:
        """Infer tournament tier from tournament name."""
        tn_lower = tournament_name.lower()
        if any(gs in tn_lower for gs in ["australian open", "roland garros", "french open", "wimbledon", "us open"]):
            return "G"
        if any(m in tn_lower for m in ["masters", "indian wells", "miami", "madrid", "rome", "shanghai", "paris"]):
            return "M"
        return "A"

    def enrich_with_market_odds(self, df: pd.DataFrame, overround: float = 0.05) -> pd.DataFrame:
        """Attach baseline market odds with realistic bookmaker vig (overround).

        Uses surface Elo ratings when available, else balanced lines.
        """
        data = df.copy()
        try:
            from src.ratings.surface_elo import SurfaceElo
            elo_model = SurfaceElo.load("models/surface_elo.joblib")
        except Exception:
            elo_model = None

        p1_odds_list = []
        p2_odds_list = []

        for _, row in data.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            surf = row["surface"]

            if elo_model is not None:
                p1_elo = elo_model.get_rating(p1, surf)
                p2_elo = elo_model.get_rating(p2, surf)
                elo_diff = p1_elo - p2_elo
                p1_fair_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            else:
                p1_fair_prob = 0.50

            # Add slight market noise / bookmaker pricing inefficiency
            # Vig = 5% overround
            p1_implied = p1_fair_prob * (1.0 + overround / 2.0)
            p2_implied = (1.0 - p1_fair_prob) * (1.0 + overround / 2.0)

            p1_o = round(float(np.clip(1.0 / p1_implied, 1.05, 12.0)), 2)
            p2_o = round(float(np.clip(1.0 / p2_implied, 1.05, 12.0)), 2)

            p1_odds_list.append(p1_o)
            p2_odds_list.append(p2_o)

        data["p1_odds"] = p1_odds_list
        data["p2_odds"] = p2_odds_list
        return data

    def save_live_fixtures(
        self,
        df: pd.DataFrame,
        filepath: Union[str, Path] = "data/live/upcoming_fixtures.csv",
    ) -> Path:
        """Save fixtures DataFrame to live directory."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} live fixtures to {path}")
        return path
