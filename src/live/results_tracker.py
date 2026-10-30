"""Real-Time Results Tracker and Prediction Settlement Engine for ATP & WTA Tennis.

Tracks live tournament match completions from ESPN public scoreboards and
evaluates predictions across all markets:
- Strictly grades the system-selected #1 Banker Pick for each match.
- Ticks `✅ WON` when the banker pick cashes.
- Ticks `❌ FAILED` when the banker pick loses.
- Marks `⏳ SCHEDULED` or `🎾 IN-PLAY` with live set scores when awaiting outcome.
- Zero fake or injected data: evaluates only real model predictions on scheduled tournament matches.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

try:
    from configs.config import LIVE_DATA_DIR
    from src.live.predictor import TennisPredictionPipeline
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import LIVE_DATA_DIR
    from src.live.predictor import TennisPredictionPipeline

logger = logging.getLogger(__name__)


class TennisResultsTracker:
    """Live match results fetcher and prediction settlement auditor."""

    ESPN_ATP_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    ESPN_WTA_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard"

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    @staticmethod
    def normalize_name(name: str) -> str:
        """Sanitize player name for robust matching."""
        if not name or pd.isna(name):
            return ""
        n = str(name).lower().strip()
        n = re.sub(r"[.\-']", " ", n)
        n = " ".join(n.split())
        return n

    def fetch_live_match_results(self, tour: str = "both") -> List[Dict[str, Any]]:
        """Fetch all current and completed tournament matches from ESPN."""
        urls = []
        if tour.lower() in ("atp", "both"):
            urls.append((self.ESPN_ATP_URL, "ATP"))
        if tour.lower() in ("wta", "both"):
            urls.append((self.ESPN_WTA_URL, "WTA"))

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        matches_found: List[Dict[str, Any]] = []

        for endpoint, tour_name in urls:
            try:
                resp = requests.get(endpoint, headers=headers, timeout=self.timeout)
                if resp.status_code != 200:
                    continue

                data = resp.json()
                for ev in data.get("events", []):
                    t_name = ev.get("name", "Tour Event")
                    for grp in ev.get("groupings", []):
                        slug = grp.get("grouping", {}).get("slug", "").lower()
                        if "singles" not in slug:
                            continue

                        for comp in grp.get("competitions", []):
                            status_name = comp.get("status", {}).get("type", {}).get("name", "STATUS_SCHEDULED")
                            competitors = comp.get("competitors", [])
                            if len(competitors) < 2:
                                continue

                            c1, c2 = competitors[0], competitors[1]
                            p1_raw = c1.get("athlete", {}).get("displayName", "").strip()
                            p2_raw = c2.get("athlete", {}).get("displayName", "").strip()

                            if not p1_raw or not p2_raw or p1_raw == "TBD" or p2_raw == "TBD":
                                continue

                            w1 = bool(c1.get("winner", False))
                            w2 = bool(c2.get("winner", False))

                            ls1 = [int(ls.get("value", 0)) for ls in c1.get("linescores", []) if ls.get("value") is not None]
                            ls2 = [int(ls.get("value", 0)) for ls in c2.get("linescores", []) if ls.get("value") is not None]

                            score_tokens = []
                            p1_sets = 0
                            p2_sets = 0
                            p1_games = sum(ls1)
                            p2_games = sum(ls2)

                            for s1, s2 in zip(ls1, ls2):
                                score_tokens.append(f"{s1}-{s2}")
                                if s1 > s2:
                                    p1_sets += 1
                                elif s2 > s1:
                                    p2_sets += 1

                            clean_score = " ".join(score_tokens)
                            total_games = p1_games + p2_games

                            winner_name = None
                            if w1:
                                winner_name = p1_raw
                            elif w2:
                                winner_name = p2_raw
                            elif status_name == "STATUS_FINAL":
                                if p1_sets > p2_sets:
                                    winner_name = p1_raw
                                elif p2_sets > p1_sets:
                                    winner_name = p2_raw

                            if status_name == "STATUS_FINAL":
                                status_label = "FINAL"
                            elif status_name in ("STATUS_IN_PROGRESS", "STATUS_FIRST_HALF", "STATUS_SECOND_HALF"):
                                status_label = "IN_PROGRESS"
                            elif status_name == "STATUS_RETIRED":
                                status_label = "RETIRED"
                            elif status_name == "STATUS_WALKOVER":
                                status_label = "WALKOVER"
                            else:
                                status_label = "SCHEDULED"

                            matches_found.append({
                                "tournament": t_name,
                                "player1": p1_raw,
                                "player2": p2_raw,
                                "p1_norm": self.normalize_name(p1_raw),
                                "p2_norm": self.normalize_name(p2_raw),
                                "status": status_label,
                                "winner": winner_name,
                                "winner_norm": self.normalize_name(winner_name) if winner_name else None,
                                "score": clean_score,
                                "p1_sets": p1_sets,
                                "p2_sets": p2_sets,
                                "p1_games": p1_games,
                                "p2_games": p2_games,
                                "total_games": total_games,
                                "tour": tour_name,
                            })

            except Exception as e:
                logger.warning(f"Error fetching live results for {tour_name}: {e}")

        return matches_found

    def match_prediction_with_live_result(
        self,
        p1: str,
        p2: str,
        live_results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the corresponding live result for a predicted matchup."""
        p1_n = self.normalize_name(p1)
        p2_n = self.normalize_name(p2)

        # 1. Exact normalized match
        for res in live_results:
            rp1 = res["p1_norm"]
            rp2 = res["p2_norm"]
            if (p1_n == rp1 and p2_n == rp2) or (p1_n == rp2 and p2_n == rp1):
                return res

        # 2. Surname matching
        p1_last = p1_n.split()[-1] if p1_n else ""
        p2_last = p2_n.split()[-1] if p2_n else ""

        for res in live_results:
            rp1 = res["p1_norm"]
            rp2 = res["p2_norm"]
            if (p1_last in rp1 and p2_last in rp2) or (p1_last in rp2 and p2_last in rp1):
                return res

        return None

    def grade_banker_pick(
        self,
        banker_pick: Dict[str, Any],
        player1: str,
        player2: str,
        live_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Grade strictly the system's selected Banker Pick against official match results.

        Returns:
            Dictionary with settlement ('WON', 'FAILED', 'PENDING', 'IN_PLAY'),
            badge, actual score, and profit/loss in units.
        """
        pick_title = str(banker_pick.get("title", ""))
        pick_market = str(banker_pick.get("market", "Banker Market"))
        pick_odds = float(banker_pick.get("fair_odds", 1.25))
        pick_prob = float(banker_pick.get("win_prob_pct", 80.0))

        # 1. If match is not found or scheduled: PENDING
        if live_result is None or live_result["status"] == "SCHEDULED":
            return {
                "settlement": "PENDING",
                "badge": "⏳ SCHEDULED",
                "is_won": False,
                "is_failed": False,
                "is_pending": True,
                "actual_score": "Upcoming",
                "actual_winner": "Awaiting Match",
                "profit_units": 0.0,
                "status_color": "#38bdf8",
                "outcome_detail": "Match scheduled for today. Awaiting start time.",
            }

        # 2. If match is currently playing: IN_PLAY
        if live_result["status"] == "IN_PROGRESS":
            score_str = live_result.get("score", "In Play")
            return {
                "settlement": "IN_PLAY",
                "badge": f"🎾 IN-PLAY ({score_str})",
                "is_won": False,
                "is_failed": False,
                "is_pending": True,
                "actual_score": score_str,
                "actual_winner": "Live on Court",
                "profit_units": 0.0,
                "status_color": "#f59e0b",
                "outcome_detail": f"Match currently active: {score_str}",
            }

        # 3. Match is complete (FINAL or RETIRED): evaluate the Banker Pick
        actual_winner = live_result.get("winner", "")
        actual_score = live_result.get("score", "Final")
        p1_sets = live_result.get("p1_sets", 0)
        p2_sets = live_result.get("p2_sets", 0)
        total_games = live_result.get("total_games", 0)

        # Align p1 and p2 relative to live_result competitors
        p1_n = self.normalize_name(player1)
        res_p1_n = live_result.get("p1_norm", "")
        is_p1_competitor_1 = (p1_n == res_p1_n) or (p1_n.split()[-1] in res_p1_n)

        p1_actual_sets = p1_sets if is_p1_competitor_1 else p2_sets
        p2_actual_sets = p2_sets if is_p1_competitor_1 else p1_sets

        pick_lower = pick_title.lower()
        cashed = False

        # Evaluation Rule 1: Player to Win At Least 1 Set
        if "to win at least 1 set" in pick_lower or "to win >= 1 set" in pick_lower:
            if self.normalize_name(player1) in pick_lower:
                cashed = (p1_actual_sets >= 1)
            else:
                cashed = (p2_actual_sets >= 1)

        # Evaluation Rule 2: Set Handicap -1.5 Sets (Straight 2-0 / 3-0)
        elif "-1.5 sets" in pick_lower or "straight 2-0" in pick_lower:
            if self.normalize_name(player1) in pick_lower:
                cashed = (p1_actual_sets - p2_actual_sets >= 2)
            else:
                cashed = (p2_actual_sets - p1_actual_sets >= 2)

        # Evaluation Rule 3: Set Handicap +1.5 Sets (Wins Set)
        elif "+1.5 sets" in pick_lower:
            if self.normalize_name(player1) in pick_lower:
                cashed = (p1_actual_sets >= 1)
            else:
                cashed = (p2_actual_sets >= 1)

        # Evaluation Rule 4: Total Games Over Line
        elif "over " in pick_lower and "games" in pick_lower:
            m = re.search(r"over\s+([\d.]+)", pick_lower)
            if m:
                line_val = float(m.group(1))
                cashed = (total_games > line_val)
            else:
                cashed = (total_games >= 22)

        # Evaluation Rule 5: Straight Match Winner
        else:
            if self.normalize_name(player1) in pick_lower:
                cashed = (self.normalize_name(actual_winner) == self.normalize_name(player1))
            else:
                cashed = (self.normalize_name(actual_winner) == self.normalize_name(player2))

        # Result output
        if cashed:
            profit = round(pick_odds - 1.0, 2)
            return {
                "settlement": "WON",
                "badge": f"✅ WON (+{profit:.2f}u)",
                "is_won": True,
                "is_failed": False,
                "is_pending": False,
                "actual_score": actual_score,
                "actual_winner": actual_winner,
                "profit_units": profit,
                "status_color": "#10b981",
                "outcome_detail": f"Banker Pick Cashed! {pick_title}. Final: {actual_score} (Winner: {actual_winner})",
            }
        else:
            return {
                "settlement": "FAILED",
                "badge": "❌ FAILED (-1.00u)",
                "is_won": False,
                "is_failed": True,
                "is_pending": False,
                "actual_score": actual_score,
                "actual_winner": actual_winner,
                "profit_units": -1.0,
                "status_color": "#f43f5e",
                "outcome_detail": f"Banker Pick Failed. Final: {actual_score} (Winner: {actual_winner})",
            }

    def get_banker_picks_ledger(
        self,
        predictions_file: Union[str, Path],
        live_results: Optional[List[Dict[str, Any]]] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract and grade ONLY the system's selected Banker Picks for all scheduled matches.

        Guarantees strictly 100% genuine data tracking with zero injected past fixtures.
        """
        p_path = Path(predictions_file)
        if not p_path.exists():
            return pd.DataFrame(), {}

        df_preds = pd.read_csv(p_path)
        if df_preds.empty:
            return pd.DataFrame(), {}

        if live_results is None:
            live_results = self.fetch_live_match_results(tour="both")

        banker_rows: List[Dict[str, Any]] = []

        for _, row in df_preds.iterrows():
            row_dict = row.to_dict()
            p1 = str(row_dict.get("player1", ""))
            p2 = str(row_dict.get("player2", ""))
            tourney = str(row_dict.get("tournament", "Tour Event"))
            surf = str(row_dict.get("surface", "Hard"))
            rnd = str(row_dict.get("round", "R32"))

            # Extract the system's #1 Banker Pick
            top_picks = TennisPredictionPipeline.extract_top_three_picks(row_dict)
            if not top_picks:
                continue
            banker = top_picks[0]

            # Match against live results from ESPN
            m_res = self.match_prediction_with_live_result(p1, p2, live_results)
            grade = self.grade_banker_pick(banker, p1, p2, m_res)

            banker_rows.append({
                "match": f"{p1} vs {p2}",
                "player1": p1,
                "player2": p2,
                "tournament": tourney,
                "surface": surf,
                "round": rnd,
                "banker_pick": banker["title"],
                "market": banker["market"],
                "win_prob_pct": banker["win_prob_pct"],
                "fair_odds": banker["fair_odds"],
                "settlement": grade["settlement"],
                "status_badge": grade["badge"],
                "actual_score": grade["actual_score"],
                "actual_winner": grade["actual_winner"],
                "profit_units": grade["profit_units"],
                "status_color": grade["status_color"],
                "outcome_detail": grade["outcome_detail"],
            })

        df_ledger = pd.DataFrame(banker_rows)
        if df_ledger.empty:
            return pd.DataFrame(), {}

        # Performance Statistics
        total = len(df_ledger)
        won = int((df_ledger["settlement"] == "WON").sum())
        failed = int((df_ledger["settlement"] == "FAILED").sum())
        pending = int((df_ledger["settlement"].isin(["PENDING", "IN_PLAY"])).sum())
        settled = won + failed

        strike_rate = (won / settled * 100.0) if settled > 0 else 0.0
        net_units = float(df_ledger["profit_units"].sum())

        summary = {
            "total_picks": total,
            "won_picks": won,
            "failed_picks": failed,
            "pending_picks": pending,
            "settled_picks": settled,
            "strike_rate_pct": round(strike_rate, 1),
            "net_units": round(net_units, 2),
            "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_ledger.to_csv(out_p, index=False)

        return df_ledger, summary
