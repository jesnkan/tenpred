"""Tennis Data Loading, Cleaning, and Standardization Module.

Supports ATP and WTA match datasets (Jeff Sackmann open format), multi-year
batch loading, schema normalization, score parsing, retirement/walkover handling,
match statistics derivation, and live feed standardization.
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
from tqdm import tqdm

try:
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from configs.config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class TennisDataLoader:
    """Robust loader and preprocessor for ATP and WTA tennis match datasets.

    Handles Sackmann-style CSV archives, normalizes schemas, parses complex
    score strings, flags match interruptions (walkovers, retirements), derives
    granular game/set and serve/return statistics, and prepares clean records
    for rating models and machine learning pipelines.
    """

    # Sackmann GitHub raw data URLs and resilient community mirrors
    GITHUB_RAW_URLS: Dict[str, List[str]] = {
        "atp": [
            "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/atp/atp_matches_{year}.csv",
            "https://raw.githubusercontent.com/Kadantte/tennis_atp/master/atp_matches_{year}.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv",
        ],
        "wta": [
            "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/wta/wta_matches_{year}.csv",
            "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv",
        ],
    }

    # Column aliases for schema harmonization
    COLUMN_MAPPINGS: Dict[str, str] = {
        "tourney_id": "tourney_id",
        "tourney_name": "tourney_name",
        "tournament": "tourney_name",
        "surface": "surface",
        "court_surface": "surface",
        "draw_size": "draw_size",
        "tourney_level": "tourney_level",
        "tournament_level": "tourney_level",
        "tourney_date": "tourney_date",
        "date": "tourney_date",
        "match_date": "tourney_date",
        "match_num": "match_num",
        "winner_id": "winner_id",
        "winner_seed": "winner_seed",
        "winner_entry": "winner_entry",
        "winner_name": "winner_name",
        "winner": "winner_name",
        "w_name": "winner_name",
        "winner_hand": "winner_hand",
        "winner_ht": "winner_ht",
        "winner_ioc": "winner_ioc",
        "winner_age": "winner_age",
        "loser_id": "loser_id",
        "loser_seed": "loser_seed",
        "loser_entry": "loser_entry",
        "loser_name": "loser_name",
        "loser": "loser_name",
        "l_name": "loser_name",
        "loser_hand": "loser_hand",
        "loser_ht": "loser_ht",
        "loser_ioc": "loser_ioc",
        "loser_age": "loser_age",
        "score": "score",
        "result": "score",
        "best_of": "best_of",
        "round": "round",
        "rnd": "round",
        "minutes": "minutes",
        # Match statistics
        "w_ace": "w_ace", "w_df": "w_df", "w_svpt": "w_svpt",
        "w_1stin": "w_1stIn", "w_1stwon": "w_1stWon", "w_2ndwon": "w_2ndWon",
        "w_svgms": "w_SvGms", "w_bpsaved": "w_bpSaved", "w_bpfaced": "w_bpFaced",
        "l_ace": "l_ace", "l_df": "l_df", "l_svpt": "l_svpt",
        "l_1stin": "l_1stIn", "l_1stwon": "l_1stWon", "l_2ndwon": "l_2ndWon",
        "l_svgms": "l_SvGms", "l_bpsaved": "l_bpSaved", "l_bpfaced": "l_bpFaced",
        "winner_rank": "winner_rank", "winner_rank_points": "winner_rank_points",
        "loser_rank": "loser_rank", "loser_rank_points": "loser_rank_points",
    }

    # Canonical surfaces
    CANONICAL_SURFACES: Dict[str, str] = {
        "HARD": "Hard",
        "CLAY": "Clay",
        "GRASS": "Grass",
        "CARPET": "Carpet",
        "INDOOR HARD": "Hard",
        "ACRYLIC": "Hard",
    }

    # Tournament levels
    LEVEL_MAPPINGS: Dict[str, str] = {
        "G": "G", "GRAND SLAM": "G", "GS": "G",
        "M": "M", "MASTERS": "M", "1000": "M", "PM": "M", "PREMIER MANDATORY": "M",
        "A": "A", "500": "A", "P": "A", "PREMIER": "A",
        "D": "D", "250": "D", "I": "D", "INTERNATIONAL": "D",
        "C": "C", "CHALLENGER": "C", "CH": "C",
        "F": "F", "FUTURES": "F", "ITF": "F",
    }

    # Round names
    ROUND_MAPPINGS: Dict[str, str] = {
        "F": "F", "FINAL": "F", "FINALS": "F",
        "SF": "SF", "SEMIFINAL": "SF", "SEMIS": "SF", "SEMIFINALS": "SF",
        "QF": "QF", "QUARTERFINAL": "QF", "QUARTERS": "QF", "QUARTERFINALS": "QF",
        "R16": "R16", "4TH ROUND": "R16", "ROUND OF 16": "R16",
        "R32": "R32", "3RD ROUND": "R32", "ROUND OF 32": "R32",
        "R64": "R64", "2ND ROUND": "R64", "ROUND OF 64": "R64",
        "R128": "R128", "1ST ROUND": "R128", "ROUND OF 128": "R128",
        "RR": "RR", "ROUND ROBIN": "RR",
        "BR": "BR", "3RD PLACE": "BR", "BRONZE": "BR",
    }

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the TennisDataLoader.

        Args:
            raw_dir: Directory containing raw match CSVs. Defaults to data/raw.
            processed_dir: Directory to save cleaned output datasets. Defaults to data/processed.
        """
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_sackmann_data(
        self,
        tour: str = "both",
        years: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> List[Path]:
        """Download yearly CSV archives from Jeff Sackmann's repositories.

        Args:
            tour: 'atp', 'wta', or 'both'.
            years: List of integer years (e.g. range(2010, 2024)). Defaults to [2020..2024].
            overwrite: If False, skips existing downloaded files.

        Returns:
            List of downloaded file paths.
        """
        if years is None:
            years = list(range(2020, datetime.now().year + 1))

        tours = ["atp", "wta"] if tour.lower() == "both" else [tour.lower()]
        downloaded_files: List[Path] = []

        for t in tours:
            if t not in self.GITHUB_RAW_URLS:
                logger.warning(f"Unsupported tour '{t}' for download. Choose 'atp', 'wta', or 'both'.")
                continue

            for year in tqdm(years, desc=f"Downloading {t.upper()} data"):
                filename = f"{t}_matches_{year}.csv"
                dest_path = self.raw_dir / filename

                if dest_path.exists() and not overwrite:
                    logger.debug(f"File {filename} already exists. Skipping download.")
                    downloaded_files.append(dest_path)
                    continue

                candidate_urls = (
                    self.GITHUB_RAW_URLS[t]
                    if isinstance(self.GITHUB_RAW_URLS[t], list)
                    else [self.GITHUB_RAW_URLS[t]]
                )

                download_success = False
                for tmpl in candidate_urls:
                    url = tmpl.format(year=year)
                    try:
                        resp = requests.get(url, timeout=25)
                        if resp.status_code == 200 and len(resp.content) > 1000:
                            dest_path.write_bytes(resp.content)
                            logger.info(f"Downloaded {filename} ({len(resp.content) / 1024:.1f} KB) from {url}")
                            downloaded_files.append(dest_path)
                            download_success = True
                            break
                    except Exception as exc:
                        logger.debug(f"Attempt failed for {url}: {exc}")

                if not download_success:
                    logger.warning(f"File {filename} could not be downloaded from available mirrors.")

        return downloaded_files

    def load_file(
        self,
        file_path: Union[str, Path],
        tour: Optional[str] = None,
        clean: bool = True,
    ) -> pd.DataFrame:
        """Load and optionally clean a single tennis match CSV file.

        Args:
            file_path: Path to the CSV file.
            tour: 'ATP', 'WTA', or None (inferred from filename).
            clean: Whether to apply standardization and cleaning pipeline.

        Returns:
            DataFrame containing match records.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Tennis data file not found: {path}")

        # Infer tour if not provided
        if tour is None:
            name_lower = path.name.lower()
            if "wta" in name_lower:
                tour = "WTA"
            elif "atp" in name_lower:
                tour = "ATP"
            else:
                tour = "ATP"

        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin1")

        if clean:
            df = self.clean_matches(df, tour=tour)
        else:
            df["tour"] = tour

        return df

    def load_years(
        self,
        start_year: int,
        end_year: int,
        tour: str = "both",
        clean: bool = True,
        auto_download: bool = False,
    ) -> pd.DataFrame:
        """Load multiple years of ATP and/or WTA match files into a unified DataFrame.

        Args:
            start_year: Beginning year (inclusive).
            end_year: Ending year (inclusive).
            tour: 'atp', 'wta', or 'both'.
            clean: Whether to run the full cleaning pipeline.
            auto_download: Whether to automatically download missing files from GitHub.

        Returns:
            Concatenated and cleaned DataFrame sorted chronologically.
        """
        tours = ["atp", "wta"] if tour.lower() == "both" else [tour.lower()]
        years = list(range(start_year, end_year + 1))
        dfs: List[pd.DataFrame] = []

        for t in tours:
            for y in years:
                filename = f"{t}_matches_{y}.csv"
                file_path = self.raw_dir / filename

                if not file_path.exists():
                    if auto_download:
                        logger.info(f"{filename} not found locally. Attempting download...")
                        self.download_sackmann_data(tour=t, years=[y])
                    else:
                        logger.warning(f"File {file_path} not found. Skipping year {y}.")
                        continue

                if file_path.exists():
                    try:
                        df_year = self.load_file(file_path, tour=t.upper(), clean=clean)
                        if not df_year.empty:
                            dfs.append(df_year)
                    except Exception as exc:
                        logger.error(f"Failed to load {filename}: {exc}")

        if not dfs:
            logger.warning(f"No match files loaded for {tour} ({start_year}-{end_year}).")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        if "tourney_date" in combined.columns:
            combined = combined.sort_values(by=["tourney_date", "tourney_id", "match_num"], ascending=True).reset_index(drop=True)

        logger.info(f"Loaded {len(combined):,} total matches for {tour.upper()} ({start_year}-{end_year}).")
        return combined

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize raw column names across varying years and case variations."""
        col_rename = {}
        for col in df.columns:
            col_clean = str(col).strip().lower()
            if col_clean in self.COLUMN_MAPPINGS:
                col_rename[col] = self.COLUMN_MAPPINGS[col_clean]

        return df.rename(columns=col_rename)

    @classmethod
    def parse_score_details(cls, score_val: Any) -> Dict[str, Any]:
        """Parse raw score string into sets, games, and status indicators.

        Args:
            score_val: Score string (e.g. '6-4 3-6 7-6(5)', '6-2 3-0 RET', 'W/O').

        Returns:
            Dictionary with parsed score metrics and flags.
        """
        if pd.isna(score_val) or not isinstance(score_val, str):
            return {
                "clean_score": "",
                "is_retirement": False,
                "is_walkover": False,
                "is_completed": False,
                "winner_sets_won": 0,
                "loser_sets_won": 0,
                "total_sets": 0,
                "winner_games_won": 0,
                "loser_games_won": 0,
                "total_games": 0,
                "game_diff": 0,
                "set_diff": 0,
                "has_tiebreak": False,
            }

        s = str(score_val).strip()
        s_upper = s.upper()

        is_walkover = any(k in s_upper for k in ("W/O", "WALKOVER", "DEF", "DEFAULT", "ABN"))
        is_retirement = "RET" in s_upper or "RETIRED" in s_upper

        # Clean score by removing markers
        clean = re.sub(r"\b(RET|W/O|DEF|WALKOVER|RETIRED)\b", "", s, flags=re.IGNORECASE).strip()
        clean = re.sub(r"\s+", " ", clean)

        # Parse individual set scores
        tokens = clean.split()
        winner_sets = 0
        loser_sets = 0
        winner_games = 0
        loser_games = 0
        has_tiebreak = False

        for token in tokens:
            # Match scores like 6-4 or 7-6(5) or 12-10
            m = re.search(r"(\d+)-(\d+)", token)
            if m:
                w_g = int(m.group(1))
                l_g = int(m.group(2))
                winner_games += w_g
                loser_games += l_g

                if "(" in token or ")" in token or (w_g >= 7 and l_g >= 6) or (w_g >= 6 and l_g >= 7):
                    has_tiebreak = True

                if w_g > l_g:
                    winner_sets += 1
                elif l_g > w_g:
                    loser_sets += 1

        total_sets = winner_sets + loser_sets
        total_games = winner_games + loser_games
        game_diff = winner_games - loser_games
        set_diff = winner_sets - loser_sets
        is_completed = not is_walkover and not is_retirement and total_games > 0

        return {
            "clean_score": clean if clean else s,
            "is_retirement": is_retirement,
            "is_walkover": is_walkover,
            "is_completed": is_completed,
            "winner_sets_won": winner_sets,
            "loser_sets_won": loser_sets,
            "total_sets": total_sets,
            "winner_games_won": winner_games,
            "loser_games_won": loser_games,
            "total_games": total_games,
            "game_diff": game_diff,
            "set_diff": set_diff,
            "has_tiebreak": has_tiebreak,
        }

    def clean_matches(
        self,
        df: pd.DataFrame,
        tour: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run complete cleaning and enrichment pipeline on match DataFrame.

        Standardizes columns, cleans dates, surfaces, rounds, levels, parses
        scores, and computes serve/return percentages.

        Args:
            df: Input match DataFrame.
            tour: Tour identifier ('ATP' or 'WTA'). If None, retains existing or defaults to 'ATP'.

        Returns:
            Cleaned and enriched DataFrame.
        """
        if df.empty:
            return df

        data = df.copy()
        data = self.standardize_columns(data)

        # Set or normalize tour
        if tour:
            data["tour"] = tour.upper()
        elif "tour" not in data.columns:
            data["tour"] = "ATP"
        else:
            data["tour"] = data["tour"].astype(str).str.upper()

        # 1. Clean and validate essential player names
        for col in ["winner_name", "loser_name"]:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
                data = data[data[col].str.len() > 0]
                data = data[~data[col].str.lower().isin(["nan", "none", "unknown", ""])]
            else:
                logger.error(f"Required column '{col}' missing from dataset.")

        # 2. Date parsing and standard year/month extraction
        if "tourney_date" in data.columns:
            # Handles integers like 20230116 or strings '2023-01-16'
            raw_dates = data["tourney_date"].astype(str).str.replace(".0", "", regex=False).str.strip()
            data["match_date"] = pd.to_datetime(raw_dates, format="%Y%m%d", errors="coerce")
            # If some dates failed, try generic parser
            if data["match_date"].isna().any():
                mask = data["match_date"].isna()
                data.loc[mask, "match_date"] = pd.to_datetime(raw_dates[mask], errors="coerce")
            data["year"] = data["match_date"].dt.year
            data["month"] = data["match_date"].dt.month
        else:
            data["match_date"] = pd.NaT
            data["year"] = np.nan
            data["month"] = np.nan

        # 3. Clean surface
        if "surface" in data.columns:
            data["surface"] = (
                data["surface"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(lambda s: self.CANONICAL_SURFACES.get(s, "Hard"))
            )
        else:
            data["surface"] = "Hard"

        # 4. Clean tournament level
        if "tourney_level" in data.columns:
            data["tourney_level"] = (
                data["tourney_level"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(lambda l: self.LEVEL_MAPPINGS.get(l, "D"))
            )
        else:
            data["tourney_level"] = "D"

        # Tournament flags
        data["is_grand_slam"] = data["tourney_level"] == "G"
        data["is_masters"] = data["tourney_level"] == "M"

        # 5. Clean round
        if "round" in data.columns:
            data["round"] = (
                data["round"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(lambda r: self.ROUND_MAPPINGS.get(r, "R32"))
            )
        else:
            data["round"] = "R32"

        # 6. Clean best_of (infer if missing)
        if "best_of" in data.columns:
            data["best_of"] = pd.to_numeric(data["best_of"], errors="coerce")
        else:
            data["best_of"] = np.nan

        # Grand Slam Men is best of 5; all WTA and standard ATP is best of 3
        is_gs_men = (data["tour"] == "ATP") & (data["is_grand_slam"])
        data.loc[is_gs_men & data["best_of"].isna(), "best_of"] = 5
        data["best_of"] = data["best_of"].fillna(3).astype(int)

        # 7. Parse match scores
        score_series = data["score"] if "score" in data.columns else pd.Series([""] * len(data), index=data.index)
        parsed_records = [self.parse_score_details(sc) for sc in score_series]
        df_parsed = pd.DataFrame(parsed_records, index=data.index)

        for col in df_parsed.columns:
            data[col] = df_parsed[col]

        # 8. Clean match statistics and derive rate features
        data = self._derive_match_rate_stats(data)

        # 9. Clean numeric ranks and ages
        for rank_col in ["winner_rank", "loser_rank", "winner_age", "loser_age", "minutes"]:
            if rank_col in data.columns:
                data[rank_col] = pd.to_numeric(data[rank_col], errors="coerce")

        return data.reset_index(drop=True)

    def _derive_match_rate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safely calculate serve hold, return, break point, and dominance ratios."""
        stat_cols = [
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"
        ]

        # If key serve columns exist, convert to numeric
        present_cols = [c for c in stat_cols if c in df.columns]
        for col in present_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "w_svpt" in df.columns and "w_1stIn" in df.columns:
            # Winner serve stats
            df["w_1st_in_pct"] = np.where(df["w_svpt"] > 0, df["w_1stIn"] / df["w_svpt"], np.nan)
            df["w_1st_won_pct"] = np.where(df["w_1stIn"] > 0, df["w_1stWon"] / df["w_1stIn"], np.nan)
            w_2nd_in = df["w_svpt"] - df["w_1stIn"]
            df["w_2nd_won_pct"] = np.where(w_2nd_in > 0, df["w_2ndWon"] / w_2nd_in, np.nan)
            df["w_bp_saved_pct"] = np.where(df["w_bpFaced"] > 0, df["w_bpSaved"] / df["w_bpFaced"], 1.0)

            # Loser serve stats
            df["l_1st_in_pct"] = np.where(df["l_svpt"] > 0, df["l_1stIn"] / df["l_svpt"], np.nan)
            df["l_1st_won_pct"] = np.where(df["l_1stIn"] > 0, df["l_1stWon"] / df["l_1stIn"], np.nan)
            l_2nd_in = df["l_svpt"] - df["l_1stIn"]
            df["l_2nd_won_pct"] = np.where(l_2nd_in > 0, df["l_2ndWon"] / l_2nd_in, np.nan)
            df["l_bp_saved_pct"] = np.where(df["l_bpFaced"] > 0, df["l_bpSaved"] / df["l_bpFaced"], 1.0)

            # Hold percentages
            if "w_SvGms" in df.columns and "l_SvGms" in df.columns:
                w_breaks_conceded = df["w_bpFaced"] - df["w_bpSaved"]
                l_breaks_conceded = df["l_bpFaced"] - df["l_bpSaved"]
                df["w_hold_pct"] = np.where(df["w_SvGms"] > 0, (df["w_SvGms"] - w_breaks_conceded) / df["w_SvGms"], np.nan)
                df["l_hold_pct"] = np.where(df["l_SvGms"] > 0, (df["l_SvGms"] - l_breaks_conceded) / df["l_SvGms"], np.nan)

                # Return break point conversion percentage
                df["w_bp_conv_pct"] = np.where(df["l_bpFaced"] > 0, l_breaks_conceded / df["l_bpFaced"], np.nan)
                df["l_bp_conv_pct"] = np.where(df["w_bpFaced"] > 0, w_breaks_conceded / df["w_bpFaced"], np.nan)

                # Dominance ratio: (% return points won) / (% return points conceded)
                w_ret_pts_won = df["l_svpt"] - (df["l_1stWon"] + df["l_2ndWon"])
                l_ret_pts_won = df["w_svpt"] - (df["w_1stWon"] + df["w_2ndWon"])
                w_ret_pct = np.where(df["l_svpt"] > 0, w_ret_pts_won / df["l_svpt"], np.nan)
                l_ret_pct = np.where(df["w_svpt"] > 0, l_ret_pts_won / df["w_svpt"], np.nan)
                df["w_dominance_ratio"] = np.where(l_ret_pct > 0, w_ret_pct / l_ret_pct, np.nan)

        return df

    def save_processed(
        self,
        df: pd.DataFrame,
        filename: str = "cleaned_matches.parquet",
    ) -> Path:
        """Save cleaned dataset to the processed data directory.

        Supports Parquet, Feather, or CSV based on file extension.

        Args:
            df: Cleaned DataFrame.
            filename: Output filename (e.g. 'matches_2010_2023.parquet' or '.csv').

        Returns:
            Path to saved file.
        """
        dest_path = self.processed_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if filename.endswith(".parquet"):
            try:
                df.to_parquet(dest_path, index=False)
            except (ImportError, ModuleNotFoundError):
                csv_path = dest_path.with_suffix(".csv")
                df.to_csv(csv_path, index=False)
                logger.info(f"Parquet engine not found. Saved as CSV: {csv_path}")
                return csv_path
        elif filename.endswith(".feather"):
            df.to_feather(dest_path)
        else:
            df.to_csv(dest_path, index=False)

        logger.info(f"Saved cleaned dataset ({len(df):,} rows) to {dest_path}")
        return dest_path
