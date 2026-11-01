"""Real-Time Results Tracker and Prediction Settlement Engine for ATP & WTA Tennis.

Tracks live tournament match completions from ESPN public scoreboards and
evaluates predictions across all markets:
- Strictly grades the system-selected #1 Banker Pick for each match.
- Actively grades each of the top 3 predicted markets for every match.
- Ticks `✅ WON` when the banker pick or prediction cashes.
- Ticks `❌ FAILED` when the banker pick or prediction loses.
- Ticks `⚠️ VOIDED` when a match ends via Walkover (W/O) or uncompleted Retirement.
- Marks `⏳ SCHEDULED` or `🎾 IN-PLAY` with live set scores when awaiting outcome.
- Zero fake or injected data: evaluates only real model predictions on scheduled tournament matches.
- Supports multi-date historical segmentation and audit logs from system inception.
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
                            c_date_raw = comp.get("date", ev.get("date", ""))
                            comp_date_iso = str(c_date_raw)[:10] if c_date_raw else ""
                            status_type = comp.get("status", {}).get("type", {})
                            status_name = status_type.get("name", "STATUS_SCHEDULED")
                            status_desc = str(status_type.get("description", "")).lower()
                            status_detail = str(status_type.get("detail", "")).lower()
                            status_detail_orig = str(status_type.get("detail", "") or status_type.get("shortDetail", "")).strip()

                            notes = comp.get("notes", [])
                            note_text = " ".join([n.get("text", "") for n in notes if isinstance(n, dict)]).strip()
                            note_lower = note_text.lower()

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

                            # Walkover & Cancellation Detection
                            is_walkover = (
                                status_name in ("STATUS_WALKOVER", "STATUS_POSTPONED", "STATUS_CANCELLED", "STATUS_CANCELED") or
                                "walkover" in status_name.lower() or
                                "walkover" in status_desc or
                                "walkover" in status_detail or
                                "w/o" in note_lower or
                                "walkover" in note_lower
                            )

                            # Retirement Detection
                            is_retired = (
                                status_name == "STATUS_RETIRED" or
                                "retired" in status_name.lower() or
                                "retired" in status_desc or
                                "retired" in status_detail or
                                " ret" in note_lower or
                                "retired" in note_lower or
                                "ret." in clean_score.lower()
                            )

                            if is_walkover:
                                status_label = "WALKOVER"
                                if not clean_score:
                                    clean_score = "W/O (Walkover)"
                            elif is_retired:
                                status_label = "RETIRED"
                                if "ret" not in clean_score.lower():
                                    clean_score = (clean_score + " ret").strip()
                            elif status_name == "STATUS_FINAL":
                                status_label = "FINAL"
                            elif status_name in ("STATUS_IN_PROGRESS", "STATUS_FIRST_HALF", "STATUS_SECOND_HALF"):
                                status_label = "IN_PROGRESS"
                            else:
                                status_label = "SCHEDULED"

                            # Winner determination
                            winner_name = None
                            if w1:
                                winner_name = p1_raw
                            elif w2:
                                winner_name = p2_raw
                            elif status_label in ("FINAL", "RETIRED", "WALKOVER"):
                                if p1_sets > p2_sets:
                                    winner_name = p1_raw
                                elif p2_sets > p1_sets:
                                    winner_name = p2_raw
                                elif note_text and " bt " in note_text:
                                    w_part = note_text.split(" bt ")[0]
                                    w_norm = self.normalize_name(w_part)
                                    if self.normalize_name(p1_raw) in w_norm or w_norm in self.normalize_name(p1_raw):
                                        winner_name = p1_raw
                                    elif self.normalize_name(p2_raw) in w_norm or w_norm in self.normalize_name(p2_raw):
                                        winner_name = p2_raw

                            matches_found.append({
                                "tournament": t_name,
                                "player1": p1_raw,
                                "player2": p2_raw,
                                "p1_norm": self.normalize_name(p1_raw),
                                "p2_norm": self.normalize_name(p2_raw),
                                "match_date": comp_date_iso,
                                "match_time": status_detail_orig,
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
                                "notes": note_text,
                            })

            except Exception as e:
                logger.warning(f"Error fetching live results for {tour_name}: {e}")

        return matches_found

    def match_prediction_with_live_result(
        self,
        p1: str,
        p2: str,
        live_results: List[Dict[str, Any]],
        match_date: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the corresponding live result for a predicted matchup."""
        p1_n = self.normalize_name(p1)
        p2_n = self.normalize_name(p2)

        # 1. Exact normalized match with matching match_date
        if match_date:
            for res in live_results:
                if res.get("match_date") == match_date:
                    rp1 = res["p1_norm"]
                    rp2 = res["p2_norm"]
                    if (p1_n == rp1 and p2_n == rp2) or (p1_n == rp2 and p2_n == rp1):
                        return res

        # 2. Exact normalized match (any date)
        for res in live_results:
            rp1 = res["p1_norm"]
            rp2 = res["p2_norm"]
            if (p1_n == rp1 and p2_n == rp2) or (p1_n == rp2 and p2_n == rp1):
                return res

        # 3. Surname matching with matching match_date
        p1_last = p1_n.split()[-1] if p1_n else ""
        p2_last = p2_n.split()[-1] if p2_n else ""

        if match_date and p1_last and p2_last:
            for res in live_results:
                if res.get("match_date") == match_date:
                    rp1 = res["p1_norm"]
                    rp2 = res["p2_norm"]
                    if (p1_last in rp1 and p2_last in rp2) or (p1_last in rp2 and p2_last in rp1):
                        return res

        # 4. Surname matching (any date)
        if p1_last and p2_last:
            for res in live_results:
                rp1 = res["p1_norm"]
                rp2 = res["p2_norm"]
                if (p1_last in rp1 and p2_last in rp2) or (p1_last in rp2 and p2_last in rp1):
                    return res

        return None

    def grade_prediction_pick(
        self,
        pick: Dict[str, Any],
        player1: str,
        player2: str,
        live_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Grade ANY predicted selection against official match results.

        Accurately enforces international sportsbook settlement standards:
        - Walkovers (W/O) -> Strictly VOID (refunded, 0 units P&L, neither won nor lost).
        - Retirements (RET) -> Unconditionally won markets stand as WON; incomplete markets VOID.
        - Finished matches -> Graded as WON or FAILED based on final score.
        - In-progress -> Graded as WON if already cashed, else IN_PLAY.
        """
        pick_title = str(pick.get("title", ""))
        pick_market = str(pick.get("market", "Market"))
        pick_odds = float(pick.get("fair_odds", 1.25))
        pick_prob = float(pick.get("win_prob_pct", 80.0))

        # 1. Match not found or scheduled: PENDING
        if live_result is None or live_result.get("status") == "SCHEDULED":
            return {
                "settlement": "PENDING",
                "badge": "⏳ SCHEDULED",
                "is_won": False,
                "is_failed": False,
                "is_void": False,
                "is_pending": True,
                "actual_score": "Upcoming",
                "actual_winner": "Awaiting Match",
                "profit_units": 0.0,
                "status_color": "#38bdf8",
                "outcome_detail": "Match scheduled. Awaiting start time.",
            }

        status_lbl = live_result.get("status", "SCHEDULED")
        actual_winner = live_result.get("winner", "")
        actual_score = live_result.get("score", "")
        p1_sets = live_result.get("p1_sets", 0)
        p2_sets = live_result.get("p2_sets", 0)
        total_games = live_result.get("total_games", 0)

        # Align p1 and p2 relative to live_result competitors
        p1_n = self.normalize_name(player1)
        res_p1_n = live_result.get("p1_norm", "")
        is_p1_competitor_1 = (p1_n == res_p1_n) or (p1_n.split()[-1] in res_p1_n)

        p1_actual_sets = p1_sets if is_p1_competitor_1 else p2_sets
        p2_actual_sets = p2_sets if is_p1_competitor_1 else p1_sets
        p1_actual_games = live_result.get("p1_games", 0) if is_p1_competitor_1 else live_result.get("p2_games", 0)
        p2_actual_games = live_result.get("p2_games", 0) if is_p1_competitor_1 else live_result.get("p1_games", 0)

        pick_lower = pick_title.lower()

        # 2. Walkover / Cancelled Matches -> STRICTLY VOID
        is_walkover = (
            status_lbl in ("WALKOVER", "CANCELLED", "POSTPONED") or
            "walkover" in actual_score.lower() or
            "w/o" in actual_score.lower()
        )
        if is_walkover:
            adv_str = f" ({actual_winner} advanced)" if actual_winner else ""
            return {
                "settlement": "VOID",
                "badge": "⚠️ VOIDED",
                "is_won": False,
                "is_failed": False,
                "is_void": True,
                "is_pending": False,
                "actual_score": actual_score or "W/O",
                "actual_winner": actual_winner or "Walkover",
                "profit_units": 0.0,
                "status_color": "#94a3b8",
                "outcome_detail": f"Match ended via Walkover{adv_str}. Bet is Voided (Stake returned).",
            }

        # 3. In-Progress Matches
        if status_lbl == "IN_PROGRESS":
            # Check if unconditionally cashed live
            is_live_cashed = False
            if "to win at least 1 set" in pick_lower or "to win >= 1 set" in pick_lower:
                if self.normalize_name(player1) in pick_lower and p1_actual_sets >= 1:
                    is_live_cashed = True
                elif self.normalize_name(player2) in pick_lower and p2_actual_sets >= 1:
                    is_live_cashed = True
            elif "over " in pick_lower and "games" in pick_lower:
                m = re.search(r"over\s+([\d.]+)", pick_lower)
                l_val = float(m.group(1)) if m else 19.5
                if total_games > l_val:
                    is_live_cashed = True

            if is_live_cashed:
                profit = round(pick_odds - 1.0, 2)
                return {
                    "settlement": "WON",
                    "badge": f"✅ CASHED LIVE (+{profit:.2f}u)",
                    "is_won": True,
                    "is_failed": False,
                    "is_void": False,
                    "is_pending": False,
                    "actual_score": actual_score or "Live",
                    "actual_winner": "Live on Court",
                    "profit_units": profit,
                    "status_color": "#10b981",
                    "outcome_detail": f"Selection already cashed live on court ({actual_score})!",
                }
            else:
                score_str = actual_score or "In Play"
                return {
                    "settlement": "IN_PLAY",
                    "badge": f"🎾 IN-PLAY ({score_str})",
                    "is_won": False,
                    "is_failed": False,
                    "is_void": False,
                    "is_pending": True,
                    "actual_score": score_str,
                    "actual_winner": "Live on Court",
                    "profit_units": 0.0,
                    "status_color": "#f59e0b",
                    "outcome_detail": f"Match currently active: {score_str}",
                }

        # 4. Retired Matches
        is_retired = (status_lbl == "RETIRED" or "ret" in actual_score.lower())
        if is_retired:
            # Did the selection achieve its condition before the retirement occurred?
            cashed_pre_ret = False
            if "to win at least 1 set" in pick_lower or "to win >= 1 set" in pick_lower:
                if self.normalize_name(player1) in pick_lower and p1_actual_sets >= 1:
                    cashed_pre_ret = True
                elif self.normalize_name(player2) in pick_lower and p2_actual_sets >= 1:
                    cashed_pre_ret = True
            elif "over " in pick_lower and "games" in pick_lower:
                m = re.search(r"over\s+([\d.]+)", pick_lower)
                l_val = float(m.group(1)) if m else 19.5
                if total_games > l_val:
                    cashed_pre_ret = True

            if cashed_pre_ret:
                profit = round(pick_odds - 1.0, 2)
                return {
                    "settlement": "WON",
                    "badge": f"✅ WON (+{profit:.2f}u)",
                    "is_won": True,
                    "is_failed": False,
                    "is_void": False,
                    "is_pending": False,
                    "actual_score": actual_score,
                    "actual_winner": actual_winner,
                    "profit_units": profit,
                    "status_color": "#10b981",
                    "outcome_detail": f"Cashed before match retirement: {actual_score} (Winner: {actual_winner})",
                }
            else:
                # Uncompleted totals/spreads/set betting aborted by retirement -> VOID
                return {
                    "settlement": "VOID",
                    "badge": "⚠️ VOIDED (Retired)",
                    "is_won": False,
                    "is_failed": False,
                    "is_void": True,
                    "is_pending": False,
                    "actual_score": actual_score,
                    "actual_winner": actual_winner,
                    "profit_units": 0.0,
                    "status_color": "#94a3b8",
                    "outcome_detail": f"Match ended via Retirement ({actual_score}). Selection did not complete and is Voided.",
                }

        # 5. Completed Match (STATUS_FINAL)
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
                cashed = (total_games >= 20)

        # Evaluation Rule 5: Total Games Under Line
        elif "under " in pick_lower and "games" in pick_lower:
            m = re.search(r"under\s+([\d.]+)", pick_lower)
            if m:
                line_val = float(m.group(1))
                cashed = (total_games < line_val)
            else:
                cashed = (total_games <= 22)

        # Evaluation Rule 6: Game Handicap Spread
        elif "game handicap" in pick_lower or "game spread" in pick_lower:
            m = re.search(r"([+-]?\d+\.?\d*)\s*game", pick_lower)
            if m:
                hc = float(m.group(1))
                if self.normalize_name(player1) in pick_lower:
                    cashed = ((p1_actual_games - p2_actual_games + hc) > 0)
                else:
                    cashed = ((p2_actual_games - p1_actual_games + hc) > 0)
            else:
                cashed = (self.normalize_name(actual_winner) == self.normalize_name(player1))

        # Evaluation Rule 7: Straight Match Winner (Moneyline)
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
                "is_void": False,
                "is_pending": False,
                "actual_score": actual_score,
                "actual_winner": actual_winner,
                "profit_units": profit,
                "status_color": "#10b981",
                "outcome_detail": f"Selection Cashed! {pick_title}. Final: {actual_score} (Winner: {actual_winner})",
            }
        else:
            return {
                "settlement": "FAILED",
                "badge": "❌ FAILED (-1.00u)",
                "is_won": False,
                "is_failed": True,
                "is_void": False,
                "is_pending": False,
                "actual_score": actual_score,
                "actual_winner": actual_winner,
                "profit_units": -1.0,
                "status_color": "#f43f5e",
                "outcome_detail": f"Selection Failed. Final: {actual_score} (Winner: {actual_winner})",
            }

    def grade_banker_pick(
        self,
        banker_pick: Dict[str, Any],
        player1: str,
        player2: str,
        live_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Grade strictly the system's selected Banker Pick against official match results."""
        return self.grade_prediction_pick(
            pick=banker_pick,
            player1=player1,
            player2=player2,
            live_result=live_result,
        )

    def get_all_available_prediction_dates(self) -> List[str]:
        """Find all available prediction dates from the data/live directory."""
        dates = set()
        live_dir = Path(LIVE_DATA_DIR)
        if live_dir.exists():
            for p in live_dir.glob("predictions_*.csv"):
                m = re.search(r"predictions_(\d{4})(\d{2})(\d{2})\.csv", p.name)
                if m:
                    dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                try:
                    df_d = pd.read_csv(p, nrows=200)
                    if "match_date" in df_d.columns:
                        for d_val in df_d["match_date"].dropna().unique():
                            d_str = str(d_val).strip()
                            if re.match(r"^\d{4}-\d{2}-\d{2}$", d_str):
                                dates.add(d_str)
                except Exception:
                    pass
        hist_p = live_dir / "banker_settled_history.csv"
        if hist_p.exists():
            try:
                df_h = pd.read_csv(hist_p)
                if "date" in df_h.columns:
                    dates.update(df_h["date"].dropna().astype(str).unique().tolist())
            except Exception:
                pass
        return sorted(list(dates), reverse=True)

    def get_banker_picks_ledger(
        self,
        predictions_file: Optional[Union[str, Path]] = None,
        live_results: Optional[List[Dict[str, Any]]] = None,
        selected_date: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Extract and grade Banker Picks with date segmentation and historical tracking.

        Args:
            predictions_file: Path to specific prediction CSV (optional).
            live_results: Optional pre-fetched live ESPN match results list.
            selected_date: Specific date YYYY-MM-DD or 'all' for consolidated history.
            output_file: Path to save settled ledger (optional).

        Returns:
            Tuple of (DataFrame of graded banker picks, Performance summary dict).
        """
        live_dir = Path(LIVE_DATA_DIR)
        if live_results is None:
            live_results = self.fetch_live_match_results(tour="both")

        # Determine target files to process
        target_files: List[Tuple[str, Path]] = []
        if selected_date and selected_date.lower() in ("all", "all dates (consolidated)"):
            pred_files = sorted(live_dir.glob("predictions_*.csv"))
            for pf in pred_files:
                m = re.search(r"predictions_(\d{4})(\d{2})(\d{2})\.csv", pf.name)
                d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "Live"
                target_files.append((d_str, pf))
        elif selected_date:
            date_clean = selected_date.replace("-", "")
            specific_file = live_dir / f"predictions_{date_clean}.csv"
            if specific_file.exists():
                target_files.append((selected_date, specific_file))
            for pf in sorted(live_dir.glob("predictions_*.csv")):
                if pf != specific_file:
                    target_files.append((selected_date, pf))
        elif predictions_file:
            pf = Path(predictions_file)
            m = re.search(r"predictions_(\d{4})(\d{2})(\d{2})\.csv", pf.name)
            d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else datetime.now().strftime("%Y-%m-%d")
            target_files.append((d_str, pf))
        else:
            pred_files = sorted(live_dir.glob("predictions_*.csv"))
            if pred_files:
                m = re.search(r"predictions_(\d{4})(\d{2})(\d{2})\.csv", pred_files[-1].name)
                d_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else datetime.now().strftime("%Y-%m-%d")
                target_files.append((d_str, pred_files[-1]))

        if not target_files:
            return pd.DataFrame(), {}

        banker_rows: List[Dict[str, Any]] = []

        for d_str, pf in target_files:
            if not pf.exists():
                continue
            df_p = pd.read_csv(pf)
            if df_p.empty:
                continue

            for _, row in df_p.iterrows():
                row_dict = row.to_dict()
                p1 = str(row_dict.get("player1", ""))
                p2 = str(row_dict.get("player2", ""))
                tourney = str(row_dict.get("tournament", "Tour Event"))
                surf = str(row_dict.get("surface", "Hard"))
                rnd = str(row_dict.get("round", "R32"))
                m_date = str(row_dict.get("match_date", d_str))

                # Strict date segmentation: do not show matches from other dates when filtering by date
                if selected_date and selected_date.lower() not in ("all", "all dates (consolidated)"):
                    if m_date != selected_date:
                        continue

                banker = TennisPredictionPipeline.extract_best_pick(row_dict)
                if not banker:
                    continue

                # Match against live results with exact match date priority
                m_res = self.match_prediction_with_live_result(p1, p2, live_results, match_date=m_date)
                grade = self.grade_prediction_pick(banker, p1, p2, m_res)

                banker_rows.append({
                    "date": m_date,
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

        # De-duplicate identical matches within same date
        df_ledger = df_ledger.drop_duplicates(subset=["date", "match", "banker_pick"]).reset_index(drop=True)

        # Performance Statistics
        total = len(df_ledger)
        won = int((df_ledger["settlement"] == "WON").sum())
        failed = int((df_ledger["settlement"] == "FAILED").sum())
        voided = int((df_ledger["settlement"] == "VOID").sum())
        pending = int((df_ledger["settlement"].isin(["PENDING", "IN_PLAY"])).sum())
        settled = won + failed

        strike_rate = (won / settled * 100.0) if settled > 0 else 0.0
        net_units = float(df_ledger["profit_units"].sum())

        summary = {
            "total_picks": total,
            "won_picks": won,
            "failed_picks": failed,
            "void_picks": voided,
            "pending_picks": pending,
            "settled_picks": settled,
            "strike_rate_pct": round(strike_rate, 1),
            "net_units": round(net_units, 2),
            "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Update historical persistent ledger
        hist_p = live_dir / "banker_settled_history.csv"
        try:
            if hist_p.exists():
                df_old = pd.read_csv(hist_p)
                combined = pd.concat([df_old, df_ledger], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date", "match", "banker_pick"], keep="last")
                combined.to_csv(hist_p, index=False)
            else:
                df_ledger.to_csv(hist_p, index=False)
        except Exception as e:
            logger.warning(f"Could not persist banker history ledger: {e}")

        if output_file:
            out_p = Path(output_file)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_ledger.to_csv(out_p, index=False)

        return df_ledger, summary
