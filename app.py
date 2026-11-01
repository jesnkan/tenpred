"""tenpred • Tennis Match Intelligence & Betting Analytics Platform.

Institutional-grade ATP & WTA forecasting terminal featuring:
- Match winner & derivative market predictions
- Closed-form point-to-match scoreline simulator
- Court surface and environmental adjustments
- Interactive Betslip & Parlay Builder with Kelly Staking
- Real-time ESPN tournament fixture settlement
- Clean, responsive institutional light interface
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def get_icon(name: str, size: int = 16, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """Return crisp vector SVG icons styled after modern Lucide/Feather icon systems."""
    svgs = {
        "tennis": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/><path d="M12 22a10 10 0 0 1-10-10"/>'
            '</svg>'
        ),
        "check": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width + 0.5}" stroke-linecap="round" stroke-linejoin="round">'
            '<polyline points="20 6 9 17 4 12"/>'
            '</svg>'
        ),
        "cross": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width + 0.5}" stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>'
            '</svg>'
        ),
        "void": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/>'
            '</svg>'
        ),
        "clock": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'
            '</svg>'
        ),
        "calendar": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/>'
            '</svg>'
        ),
        "shield": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            '</svg>'
        ),
        "pulse": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
            '</svg>'
        ),
        "trending_up": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>'
            '</svg>'
        ),
        "refresh": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M21 2v6h-6"/><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M3 22v-6h6"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>'
            '</svg>'
        ),
        "plus": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width + 0.5}" stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'
            '</svg>'
        ),
        "trash": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>'
            '</svg>'
        ),
        "trophy": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2z"/>'
            '</svg>'
        ),
        "layers": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
            '</svg>'
        ),
        "sliders": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>'
            '</svg>'
        ),
        "arrow_up_right": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/>'
            '</svg>'
        ),
        "target": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>'
            '</svg>'
        ),
        "wallet": (
            f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"/><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"/><path d="M18 12a2 2 0 0 0 0 4h4v-4z"/>'
            '</svg>'
        ),
    }
    return svgs.get(name, "")


# Configure wide page layout and title
st.set_page_config(
    page_title="tenpred • Tennis Match Intelligence & Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Clean White Institutional Sports Analytics Theme
# =============================================================================
st.markdown("""
<style>
    /* Clean Light Institutional Theme */
    :root {
        --bg-main: #f8fafc;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --card-border-hover: #cbd5e1;
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --accent-primary: #0f172a;
        --accent-emerald: #059669;
        --accent-rose: #e11d48;
        --accent-amber: #d97706;
    }

    /* Base Body & App Canvas */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* Force light backgrounds for containers and sidebars */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    /* Monospace precision for odds, stats, numbers */
    .mono-num {
        font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
        letter-spacing: -0.3px;
    }

    /* Custom Metric Cards */
    .quant-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04), 0 1px 2px -1px rgba(15, 23, 42, 0.04);
    }
    .quant-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
    }

    .quant-label {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 6px;
    }

    .quant-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        margin-bottom: 4px;
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        letter-spacing: -0.02em;
    }

    .quant-sub {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
    }

    /* Badge Pills with Icon Support */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        line-height: 1.2;
    }
    .badge-emerald {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .badge-slate, .badge-blue, .badge-dark {
        background: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
    }
    .badge-amber {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .badge-rose {
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
    }

    /* Probability Distribution Meter Bar */
    .prob-meter-container {
        height: 6px;
        width: 100%;
        background-color: #f1f5f9;
        border-radius: 9999px;
        overflow: hidden;
        display: flex;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
    }
    .prob-meter-fill {
        height: 100%;
        background-color: #0f172a;
        border-radius: 9999px;
        transition: width 0.3s ease;
    }

    /* Odds Display Chip */
    .odds-chip {
        display: inline-flex;
        align-items: center;
        padding: 2px 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Buttons in sleek black and dark gray */
    button[kind="primary"] {
        background-color: #0f172a !important;
        border: 1px solid #0f172a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
    }
    button[kind="primary"]:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.1) !important;
    }
    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #0f172a !important;
    }

    /* Betslip Drawer */
    .betslip-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .betslip-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease;
    }
    .betslip-item:hover {
        border-color: #cbd5e1;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.88rem;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
        background-color: rgba(255, 255, 255, 0.5) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
    }

    /* Streamlit widget tweaks for clean light look */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }
    div[data-testid="stExpander"] details {
        border: none !important;
    }

    /* Mobile Responsive Media Queries */
    @media (max-width: 768px) {
        .quant-val {
            font-size: 1.45rem !important;
        }
        .quant-card {
            padding: 14px 16px !important;
            margin-bottom: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px !important;
            font-size: 0.78rem !important;
        }
        .block-container {
            padding-left: 12px !important;
            padding-right: 12px !important;
            padding-top: 2rem !important;
        }
        button[kind="primary"], button[kind="secondary"] {
            min-height: 42px !important;
        }
    }
    @media (max-width: 480px) {
        .quant-val {
            font-size: 1.25rem !important;
        }
        .quant-card {
            padding: 12px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Imports from backend package
try:
    from src.data.live_scraper import TennisLiveFetcher
    from src.evaluation.risk_manager import TennisRiskManager
    from src.live.daily_runner import DailyRunner
    from src.live.predictor import TennisPredictionPipeline
    from src.live.results_tracker import TennisResultsTracker
    from src.models.markov_sim import TennisMarkovModel
    from src.models.match_winner import MatchWinnerModel
    from src.ratings.surface_elo import SurfaceElo
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.data.live_scraper import TennisLiveFetcher
    from src.evaluation.risk_manager import TennisRiskManager
    from src.live.daily_runner import DailyRunner
    from src.live.predictor import TennisPredictionPipeline
    from src.live.results_tracker import TennisResultsTracker
    from src.models.markov_sim import TennisMarkovModel
    from src.models.match_winner import MatchWinnerModel
    from src.ratings.surface_elo import SurfaceElo


# =============================================================================
# Session State: Betslip & Portfolio Tracker
# =============================================================================
if "betslip" not in st.session_state:
    st.session_state["betslip"] = []


def add_to_betslip(match: str, market: str, pick: str, odds: float, prob: float) -> None:
    """Add a selected bet leg to the interactive betslip session."""
    leg_id = f"{match}_{market}_{pick}".replace(" ", "_")
    existing_ids = [b["id"] for b in st.session_state["betslip"]]
    if leg_id not in existing_ids:
        st.session_state["betslip"].append({
            "id": leg_id,
            "match": match,
            "market": market,
            "pick": pick,
            "odds": float(odds),
            "prob": float(prob),
        })
        st.toast(f"Added to Betslip: {pick} @ {odds:.2f}")
    else:
        st.toast(f"Already in Betslip: {pick}")


def remove_from_betslip(idx: int) -> None:
    """Remove a single leg from the betslip."""
    if 0 <= idx < len(st.session_state["betslip"]):
        removed = st.session_state["betslip"].pop(idx)
        st.toast(f"Removed: {removed['pick']}")


def clear_betslip() -> None:
    """Clear all legs from the betslip."""
    st.session_state["betslip"] = []
    st.toast("Betslip cleared")


# =============================================================================
# Cached Artifact Loaders
# =============================================================================

@st.cache_resource
def load_pipeline() -> Optional[TennisPredictionPipeline]:
    """Load persisted models and initialize prediction pipeline."""
    mw_path = Path("models/match_winner_ensemble.joblib")
    if not mw_path.exists():
        mw_path = Path("models/match_winner_lgbm.joblib")
    sg_path = Path("models/set_game_model.joblib")
    elo_path = Path("models/surface_elo.joblib")

    if not mw_path.exists() or not elo_path.exists():
        return None

    return TennisPredictionPipeline.from_trained_artifacts(
        match_winner_path=mw_path,
        set_game_path=sg_path,
        elo_path=elo_path,
    )


@st.cache_data
def load_elo_database() -> pd.DataFrame:
    """Load player ratings dataframe."""
    elo_path = Path("models/surface_elo.joblib")
    if not elo_path.exists():
        return pd.DataFrame()
    elo = SurfaceElo.load(elo_path)
    return elo.to_dataframe()


@st.cache_data
def load_feature_importance_data() -> pd.DataFrame:
    """Load feature importances from trained MatchWinnerModel."""
    mw_path = Path("models/match_winner_ensemble.joblib")
    if not mw_path.exists():
        mw_path = Path("models/match_winner_lgbm.joblib")
    if not mw_path.exists():
        return pd.DataFrame()
    mw = MatchWinnerModel.load(mw_path)
    return mw.feature_importance(importance_type="gain", top_n=25)


pipeline = load_pipeline()
df_elo = load_elo_database()

# Player list sorted by Elo
if not df_elo.empty:
    all_players = df_elo["player"].tolist()
else:
    all_players = [
        "Jannik Sinner", "Carlos Alcaraz", "Novak Djokovic", "Alexander Zverev",
        "Daniil Medvedev", "Iga Swiatek", "Aryna Sabalenka", "Coco Gauff"
    ]


# =============================================================================
# Sidebar: System Telemetry, Betslip & Bankroll Settings
# =============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0;">
        <div style="background: #0f172a; width: 36px; height: 36px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(15,23,42,0.12);">
            {get_icon('tennis', 20, '#ffffff')}
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.25rem; color: #0f172a; letter-spacing: -0.4px; line-height: 1.1;">tenpred</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;">ATP & WTA Analytics</div>
        </div>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px;">
        <span class="badge-pill badge-dark">{get_icon('trophy', 12, '#0f172a')} ATP Tour</span>
        <span class="badge-pill badge-emerald">{get_icon('trophy', 12, '#047857')} WTA Tour</span>
        <span class="badge-pill badge-slate">{get_icon('check', 12, '#334155')} Live Feed</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE BETSLIP & PARLAY BUILDER (SIDEBAR DRAWER)
    # -------------------------------------------------------------------------
    st.markdown("---")
    betslip = st.session_state["betslip"]
    n_legs = len(betslip)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-weight: 700; font-size: 0.95rem; color: #0f172a; display: inline-flex; align-items: center; gap: 6px;">
            {get_icon('wallet', 16, '#0f172a')} Active Betslip
        </span>
        <span class="badge-pill {'badge-emerald' if n_legs > 0 else 'badge-slate'}">{n_legs} Leg{'s' if n_legs != 1 else ''}</span>
    </div>
    """, unsafe_allow_html=True)

    if n_legs == 0:
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 18px 12px; text-align: center; color: #64748b; font-size: 0.85rem;">
            <div style="margin-bottom: 8px; display: flex; justify-content: center;">{get_icon('layers', 24, '#94a3b8')}</div>
            <div style="font-weight: 700; color: #334155; margin-bottom: 4px;">Betslip is empty</div>
            <span style="font-size: 0.8rem; color: #64748b;">Click <b style="color: #0f172a;">+ Add to Slip</b> on any match pick to build single or parlay wagers.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display selected legs with delete buttons
        total_odds = 1.0
        joint_prob = 1.0

        for idx, leg in enumerate(betslip):
            total_odds *= leg["odds"]
            joint_prob *= leg["prob"]

            st.markdown(f"""
            <div class="betslip-item">
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 2px;">{leg['match']}</div>
                <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a;">{leg['pick']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                    <span class="mono-num" style="color: #059669; font-weight: 800;">@{leg['odds']:.2f}</span>
                    <span style="font-size: 0.78rem; color: #475569; font-weight: 600;">Prob: {leg['prob']:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Remove Leg", key=f"del_leg_{idx}", use_container_width=True):
                remove_from_betslip(idx)
                st.rerun()

        # Conservative correlation hair-cut for multi-leg (5% hair-cut per additional leg)
        if n_legs > 1:
            joint_prob = joint_prob * (0.95 ** (n_legs - 1))

        # Parlay Metrics Summary
        parlay_ev = (total_odds * joint_prob) - 1.0
        ev_color = "#059669" if parlay_ev > 0 else "#e11d48"

        st.markdown(f"""
        <div class="betslip-box">
            <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                {'Single Wager' if n_legs == 1 else f'{n_legs}-Leg Parlay Summary'}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 4px 0;">
                <span style="font-size: 0.88rem; color: #475569;">Cumulative Odds:</span>
                <span class="mono-num" style="font-size: 1.35rem; font-weight: 800; color: #0f172a;">@{total_odds:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span style="font-size: 0.85rem; color: #475569;">Win Probability:</span>
                <span class="mono-num" style="font-size: 0.92rem; font-weight: 700; color: #0f172a;">{joint_prob:.1%}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #475569;">Expected Value (EV):</span>
                <span class="mono-num" style="font-size: 0.92rem; font-weight: 700; color: {ev_color};">{parlay_ev:+.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick-Stake Selection Chips
        if "wager_stake" not in st.session_state:
            st.session_state["wager_stake"] = 50.0

        st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; margin: 8px 0 4px 0;'>Quick Stake</div>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        if q1.button("$10", key="qs_10", use_container_width=True):
            st.session_state["wager_stake"] = 10.0
            st.rerun()
        if q2.button("$25", key="qs_25", use_container_width=True):
            st.session_state["wager_stake"] = 25.0
            st.rerun()
        if q3.button("$50", key="qs_50", use_container_width=True):
            st.session_state["wager_stake"] = 50.0
            st.rerun()
        if q4.button("$100", key="qs_100", use_container_width=True):
            st.session_state["wager_stake"] = 100.0
            st.rerun()

        # Interactive Wager Calculator
        wager_amount = st.number_input(
            "Wager Stake ($)",
            min_value=1.0,
            max_value=50000.0,
            value=float(st.session_state["wager_stake"]),
            step=10.0,
            key="wager_input_box"
        )
        st.session_state["wager_stake"] = wager_amount

        potential_payout = wager_amount * total_odds
        net_profit = potential_payout - wager_amount

        st.markdown(f"""
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.82rem; color: #065f46; font-weight: 600;">Potential Payout:</span>
                <span class="mono-num" style="font-weight: 700; color: #0f172a;">${potential_payout:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                <span style="font-size: 0.82rem; color: #065f46; font-weight: 600;">Net Profit:</span>
                <span class="mono-num" style="font-weight: 800; color: #059669;">+${net_profit:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Clear Betslip", use_container_width=True):
            clear_betslip()
            st.rerun()

    # -------------------------------------------------------------------------
    # Bankroll Management Settings
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"""
    <span style='font-weight: 700; font-size: 0.95rem; color: #0f172a; display: inline-flex; align-items: center; gap: 6px;'>
        {get_icon('sliders', 16, '#0f172a')} Bankroll & Stake Settings
    </span>
    """, unsafe_allow_html=True)
    user_bankroll = st.number_input("Total Bankroll ($)", min_value=100.0, value=10000.0, step=500.0)
    user_kelly = st.slider("Kelly Fraction", min_value=0.10, max_value=0.50, value=0.25, step=0.05,
                           help="0.25 = Quarter Kelly (recommended for variance damping)")
    user_min_edge = st.slider("Min Edge Threshold (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    user_max_single_bet = st.slider("Max Single Bet (%)", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
    user_max_match_exp = st.slider("Max Match Exposure (%)", min_value=2.0, max_value=8.0, value=4.0, step=0.5)

    risk_manager = TennisRiskManager(
        bankroll=user_bankroll,
        kelly_fraction=user_kelly,
        min_edge_threshold=user_min_edge / 100.0,
        max_single_bet_pct=user_max_single_bet,
        max_match_exposure_pct=user_max_match_exp,
    )

    st.markdown("---")
    st.caption("Active Records: 11,561 Historical Match Archives | Clean Decimal Odds")


# =============================================================================
# Main Header
# =============================================================================

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; margin-bottom: 20px; border-bottom: 1px solid #e2e8f0; flex-wrap: wrap; gap: 12px;">
    <div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #0f172a; width: 38px; height: 38px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(15,23,42,0.15);">
                {get_icon('tennis', 22, '#ffffff')}
            </div>
            <span style="font-size: 1.85rem; font-weight: 800; letter-spacing: -0.6px; color: #0f172a;">tenpred</span>
            <span style="background: #f1f5f9; color: #0f172a; font-weight: 700; font-size: 0.78rem; padding: 4px 10px; border-radius: 6px; border: 1px solid #cbd5e1; letter-spacing: 0.4px;">ATP & WTA MATCH INTELLIGENCE</span>
        </div>
        <p style="color: #64748b; font-size: 0.92rem; margin: 6px 0 0 0;">
            Institutional tennis analytics, consensus market odds, and real-time score settlement.
        </p>
    </div>
    <div style="text-align: right; display: flex; align-items: center; gap: 8px;">
        <span class="badge-pill badge-emerald" style="display: inline-flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 8px; height: 8px; background: #059669; border-radius: 50%; box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.2);"></span>
            Live ESPN Feed
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Navigation Tabs (Clean & Professional, No Emojis, No AI Gimmicks)
# =============================================================================

tab_names = [
    "Match Predictions",
    "Results Tracker",
    "Head-to-Head Simulator",
    "Player Rankings",
    "Bankroll Calculator",
]
default_tab = st.session_state.pop("nav_tab_default", "Match Predictions")
if default_tab not in tab_names:
    default_tab = "Match Predictions"

tab_today, tab_banker_tracker, tab_sim, tab_elo, tab_risk = st.tabs(
    tab_names,
    default=default_tab,
)


# Shared Results Tracker & Live ESPN Feed Cache
if "demo_settlements" not in st.session_state:
    st.session_state["demo_settlements"] = {}

tracker = TennisResultsTracker()
live_espn_results = tracker.fetch_live_match_results(tour="both")

# Apply demo simulation overrides if any exist
for (demo_p1, demo_p2), demo_data in st.session_state["demo_settlements"].items():
    found = False
    for res in live_espn_results:
        if (res["p1_norm"] == tracker.normalize_name(demo_p1) and res["p2_norm"] == tracker.normalize_name(demo_p2)) or \
           (res["p2_norm"] == tracker.normalize_name(demo_p1) and res["p1_norm"] == tracker.normalize_name(demo_p2)):
            res.update(demo_data)
            found = True
            break
    if not found:
        live_espn_results.append({
            "tournament": demo_data.get("tournament", "Tour Event"),
            "player1": demo_p1,
            "player2": demo_p2,
            "p1_norm": tracker.normalize_name(demo_p1),
            "p2_norm": tracker.normalize_name(demo_p2),
            "status": demo_data.get("status", "FINAL"),
            "winner": demo_data.get("winner"),
            "score": demo_data.get("score"),
            "p1_sets": demo_data.get("p1_sets", 2),
            "p2_sets": demo_data.get("p2_sets", 0),
            "p1_games": demo_data.get("p1_games", 12),
            "p2_games": demo_data.get("p2_games", 7),
            "total_games": demo_data.get("total_games", 19),
            "tour": "ATP",
        })


# =============================================================================
# TAB 1: TODAY'S MATCHES & PREDICTIONS
# =============================================================================

with tab_today:
    st.subheader("Match Predictions & Consensus Market Odds")
    st.markdown("Live scan of active ATP & WTA tournament fixtures with consensus market odds and automated match settlement.")

    # Refresh Button
    c_btn1, c_btn2 = st.columns([3, 3])
    with c_btn1:
        if st.button("Fetch Daily Fixtures from ESPN", type="primary", use_container_width=True):
            with st.spinner("Fetching latest live scheduled fixtures and updating predictions..."):
                fetcher = TennisLiveFetcher()
                runner = DailyRunner()
                live_df = fetcher.fetch_espn_daily_fixtures(tour="both", only_scheduled=False)
                if not live_df.empty:
                    dates_in_feed = live_df["match_date"].dropna().unique()
                    total_saved = 0
                    for d_val in dates_in_feed:
                        df_d = live_df[live_df["match_date"] == d_val].copy()
                        if not df_d.empty:
                            preds_d = runner.predict_upcoming(df_d)
                            tag = str(d_val).replace("-", "")
                            preds_d.to_csv(f"data/live/predictions_{tag}.csv", index=False)
                            total_saved += len(preds_d)
                    
                    # Persist upcoming scheduled matches
                    sched_df = live_df[live_df["status"].isin(["STATUS_SCHEDULED", "STATUS_IN_PROGRESS"])]
                    if not sched_df.empty:
                        fetcher.save_live_fixtures(sched_df, "data/live/upcoming_fixtures.csv")

                    st.success(f"Fetched, partitioned, and evaluated {total_saved} matches across {len(dates_in_feed)} dates!")
                    st.rerun()
                else:
                    st.warning("No new matches found currently on ESPN live board.")

    with c_btn2:
        if st.button("Sync Live Scores from ESPN", use_container_width=True):
            with st.spinner("Updating live scoreboards..."):
                st.rerun()

    # Load Predictions with Date Segmentation
    live_dir = Path("data/live")
    today_iso = datetime.now().strftime("%Y-%m-%d")
    tomorrow_iso = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    avail_dates = tracker.get_all_available_prediction_dates()

    if avail_dates:
        date_display_map = {}
        for d in avail_dates:
            if d == today_iso:
                date_display_map[f"Today ({d})"] = d
            elif d == tomorrow_iso:
                date_display_map[f"Tomorrow ({d})"] = d
            else:
                date_display_map[f"Matches for {d}"] = d
        date_display_map["All Matches (Consolidated)"] = "all"

        tab1_options = list(date_display_map.keys())

        # Find default index (Today if present)
        def_tab1_idx = 0
        for idx, opt in enumerate(tab1_options):
            if today_iso in opt:
                def_tab1_idx = idx
                break

        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        sel_date_label = st.radio(
            "Select Fixture Date:",
            options=tab1_options,
            index=def_tab1_idx,
            horizontal=True,
            help="Filter predictions by today's active schedule, tomorrow's upcoming fixtures, or all consolidated matches."
        )
        chosen_date_tab1 = date_display_map[sel_date_label]

        # Load appropriate predictions
        if chosen_date_tab1 == "all":
            dfs = []
            for pf in sorted(live_dir.glob("predictions_*.csv")):
                try:
                    dfs.append(pd.read_csv(pf))
                except Exception:
                    pass
            df_today = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["match_date", "player1", "player2"]).reset_index(drop=True) if dfs else pd.DataFrame()
            st.caption(f"Loaded all combined predictions ({len(df_today)} matches evaluated)")
        else:
            date_clean = chosen_date_tab1.replace("-", "")
            specific_file = live_dir / f"predictions_{date_clean}.csv"
            if specific_file.exists():
                df_today = pd.read_csv(specific_file)
                st.caption(f"Loaded predictions for `{chosen_date_tab1}` from `{specific_file.name}` ({len(df_today)} matches evaluated)")
            else:
                dfs = []
                for pf in sorted(live_dir.glob("predictions_*.csv")):
                    try:
                        df_tmp = pd.read_csv(pf)
                        if "match_date" in df_tmp.columns:
                            matched = df_tmp[df_tmp["match_date"] == chosen_date_tab1]
                            if not matched.empty:
                                dfs.append(matched)
                    except Exception:
                        pass
                if dfs:
                    df_today = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["match_date", "player1", "player2"]).reset_index(drop=True)
                    st.caption(f"Loaded predictions for `{chosen_date_tab1}` ({len(df_today)} matches evaluated)")
                else:
                    df_today = pd.DataFrame()

        if not df_today.empty:
            # Filter bar
            flt_c1, flt_c2, flt_c3 = st.columns(3)
            with flt_c1:
                edge_threshold = st.slider("Min Edge (%)", 0.0, 15.0, 0.0, 0.5)
            with flt_c2:
                surfaces_avail = df_today["surface"].dropna().unique().tolist()
                sel_surfaces = st.multiselect("Court Surface", options=surfaces_avail, default=surfaces_avail)
            with flt_c3:
                only_value_bets = st.checkbox("Show Only Actionable +EV Bets", value=False)

            # Apply Filters
            filtered_df = df_today.copy()
            if sel_surfaces:
                filtered_df = filtered_df[filtered_df["surface"].isin(sel_surfaces)]
            if only_value_bets and "recommended_bet" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["recommended_bet"] != "Pass (No Value)"]
            if "edge_pct" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["edge_pct"] >= edge_threshold]

            # Apply best-of-analysis to choose the single best pick for every match
            match_candidates = []
            for m_idx, row in filtered_df.iterrows():
                best_pick = TennisPredictionPipeline.extract_best_pick(row)
                conf_score = float(best_pick.get("win_prob_pct", 50.0))
                match_candidates.append({
                    "orig_idx": m_idx,
                    "row": row,
                    "best_pick": best_pick,
                    "score": conf_score,
                })

            # Rank matches by best-of-analysis confidence descending
            match_candidates.sort(key=lambda x: x["score"], reverse=True)
            total_matches_avail = len(match_candidates)

            # Cap at the top 10 best picks match predictions
            top_10_matches = match_candidates[:10]

            # Prominent Indicator & Navigation Callout for Total Day Slate
            st.markdown(f"""
            <div class="quant-card" style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 5px solid #0f172a; padding: 18px 22px; margin-bottom: 16px; border-radius: 10px; box-shadow: 0 2px 6px rgba(15,23,42,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                    <div style="flex: 1; min-width: 280px;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <span class="badge-pill badge-dark" style="font-weight: 700;">
                                {get_icon('shield', 13, '#ffffff')} TOP 10 BANKER PICKS
                            </span>
                            <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">
                                Showing Top {len(top_10_matches)} of {total_matches_avail} Evaluated Fixtures for {sel_date_label}
                            </span>
                        </div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a;">
                            Today's Top 10 Best Predictions (Single Optimal Pick per Match)
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; margin-top: 4px; line-height: 1.45;">
                            Each fixture below presents its single highest-conviction derivative pick derived from best-of market analysis. To review the <b>complete list of all {total_matches_avail} matches</b> for {sel_date_label} in the total day spreadsheet with live score settlement, access the <b>Results Tracker</b> page.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c_cta1, c_cta2 = st.columns([7, 3])
            with c_cta2:
                if st.button("📊 Open Total Day Spreadsheet & Results Tracker →", key="btn_jump_to_tracker", type="primary", use_container_width=True):
                    st.session_state["nav_tab_default"] = "Results Tracker"
                    st.rerun()

            st.markdown(f"#### Displaying Top **{len(top_10_matches)}** High-Confidence Match Predictions")

            # Interactive Cards View for Matches (Only 1 Single Best Pick per Match)
            for item in top_10_matches:
                idx = item["orig_idx"]
                row = item["row"]
                best_pick = item["best_pick"]
                p1 = str(row["player1"])
                p2 = str(row["player2"])
                tourney = str(row.get("tournament", "Tour Event"))
                surf = str(row.get("surface", "Hard"))
                rnd = str(row.get("round", "R32"))
                m_date_val = str(row.get("match_date", chosen_date_tab1 if chosen_date_tab1 != "all" else today_iso))
                m_time_val = str(row.get("match_time", ""))
                p1_prob = float(row.get("p1_win_prob", 0.5))
                p2_prob = 1.0 - p1_prob
                pred_winner = p1 if p1_prob >= 0.50 else p2
                win_prob_pct = max(p1_prob, p2_prob) * 100.0

                p1_odds = float(row.get("p1_odds", 1.90)) if not pd.isna(row.get("p1_odds")) else float(row.get("p1_fair_odds", 1.90))
                p2_odds = float(row.get("p2_odds", 1.90)) if not pd.isna(row.get("p2_odds")) else float(row.get("p2_fair_odds", 1.90))

                # Match with live ESPN results with exact date priority
                m_res = tracker.match_prediction_with_live_result(p1, p2, live_espn_results, match_date=m_date_val)
                if m_res:
                    st_name = m_res.get("status", "SCHEDULED")
                    sc_str = m_res.get("score", "")
                    wn_str = m_res.get("winner", "")
                    if not m_time_val or m_time_val == "nan":
                        m_time_val = m_res.get("match_time", "")
                    if st_name == "FINAL":
                        card_status_badge = f'<span class="badge-pill badge-emerald">{get_icon("check", 12, "#047857")} FINAL • {sc_str} ({wn_str} won)</span>'
                    elif st_name == "WALKOVER":
                        card_status_badge = f'<span class="badge-pill badge-slate">{get_icon("void", 12, "#475569")} WALKOVER (W/O) • {wn_str} adv</span>'
                    elif st_name == "RETIRED":
                        card_status_badge = f'<span class="badge-pill badge-amber">{get_icon("clock", 12, "#b45309")} RETIRED ({sc_str}) • {wn_str} won</span>'
                    elif st_name == "IN_PROGRESS":
                        card_status_badge = f'<span class="badge-pill badge-amber">{get_icon("pulse", 12, "#b45309")} LIVE ON COURT • {sc_str}</span>'
                    else:
                        card_status_badge = f'<span class="badge-pill badge-slate">{get_icon("clock", 12, "#475569")} SCHEDULED • Upcoming</span>'
                else:
                    card_status_badge = f'<span class="badge-pill badge-slate">{get_icon("clock", 12, "#475569")} SCHEDULED • Upcoming</span>'

                if not m_time_val or m_time_val == "nan":
                    m_time_val = "Scheduled"

                p1_pct = p1_prob * 100.0
                p2_pct = p2_prob * 100.0

                # Match Card Container
                st.markdown(f"""
                <div class="quant-card" style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <span style="display: inline-flex; align-items: center; gap: 6px; font-weight: 700; color: #0f172a; font-size: 1.05rem;">
                                {get_icon('trophy', 16, '#475569')} {tourney}
                            </span>
                            <span style="color: #64748b; font-size: 0.85rem;">• {surf} Court • {rnd}</span>
                            <span style="display: inline-flex; align-items: center; gap: 5px; color: #334155; font-size: 0.82rem; font-weight: 600; background: #f8fafc; padding: 3px 9px; border-radius: 6px; border: 1px solid #e2e8f0;">
                                {get_icon('calendar', 13, '#64748b')} {m_date_val} • {m_time_val}
                            </span>
                        </div>
                        <div>{card_status_badge}</div>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px;">
                        <div style="flex: 1; min-width: 260px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span style="font-size: 1.05rem; font-weight: {'800' if p1_prob >= 0.5 else '600'}; color: {'#0f172a' if p1_prob >= 0.5 else '#475569'};">
                                    {p1}
                                </span>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="odds-chip">@{p1_odds:.2f}</span>
                                    <span class="mono-num" style="font-size: 0.88rem; font-weight: 700; color: {'#0f172a' if p1_prob >= 0.5 else '#64748b'}; width: 46px; text-align: right;">{p1_pct:.1f}%</span>
                                </div>
                            </div>
                            <div class="prob-meter-container">
                                <div class="prob-meter-fill" style="width: {p1_pct:.1f}%;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                                <span style="font-size: 1.05rem; font-weight: {'800' if p2_prob > 0.5 else '600'}; color: {'#0f172a' if p2_prob > 0.5 else '#475569'};">
                                    {p2}
                                </span>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="odds-chip">@{p2_odds:.2f}</span>
                                    <span class="mono-num" style="font-size: 0.88rem; font-weight: 700; color: {'#0f172a' if p2_prob > 0.5 else '#64748b'}; width: 46px; text-align: right;">{p2_pct:.1f}%</span>
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right; min-width: 170px; padding-left: 16px; border-left: 1px solid #f1f5f9;">
                            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Projected Winner</div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #059669; margin: 2px 0;">{pred_winner}</div>
                            <div style="font-size: 0.82rem; color: #475569; font-weight: 600;">{win_prob_pct:.1f}% probability</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Grade Single Best Pick against live ESPN results
                p_grade = tracker.grade_prediction_pick(best_pick, p1, p2, m_res)
                settle = p_grade["settlement"]
                badge_txt = p_grade["badge"]

                if settle == "WON":
                    box_style = "background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;"
                    badge_markup = f'<span class="badge-pill badge-emerald">{get_icon("check", 12, "#047857")} {badge_txt}</span>'
                    action_markup = f'<div style="display: flex; align-items: center; justify-content: center; gap: 6px; background: #dcfce7; color: #15803d; font-weight: 800; font-size: 0.85rem; text-align: center; padding: 8px; border-radius: 6px; border: 1px solid #86efac; margin-bottom: 16px;">{get_icon("check", 14, "#15803d")} WON • CASHED</div>'
                elif settle == "FAILED":
                    box_style = "background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;"
                    badge_markup = f'<span class="badge-pill badge-rose">{get_icon("cross", 12, "#be123c")} {badge_txt}</span>'
                    action_markup = f'<div style="display: flex; align-items: center; justify-content: center; gap: 6px; background: #ffe4e6; color: #be123c; font-weight: 800; font-size: 0.85rem; text-align: center; padding: 8px; border-radius: 6px; border: 1px solid #fda4af; margin-bottom: 16px;">{get_icon("cross", 14, "#be123c")} LOST</div>'
                elif settle == "VOID":
                    box_style = "background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;"
                    badge_markup = f'<span class="badge-pill badge-slate">{get_icon("void", 12, "#475569")} {badge_txt}</span>'
                    action_markup = f'<div style="display: flex; align-items: center; justify-content: center; gap: 6px; background: #f1f5f9; color: #475569; font-weight: 800; font-size: 0.85rem; text-align: center; padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; margin-bottom: 16px;">{get_icon("void", 14, "#475569")} VOIDED (Walkover / Refund)</div>'
                elif settle == "IN_PLAY":
                    box_style = "background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px;"
                    badge_markup = f'<span class="badge-pill badge-amber">{get_icon("pulse", 12, "#b45309")} {badge_txt}</span>'
                    action_markup = f'<div style="display: flex; align-items: center; justify-content: center; gap: 6px; background: #fef3c7; color: #b45309; font-weight: 800; font-size: 0.85rem; text-align: center; padding: 8px; border-radius: 6px; border: 1px solid #fcd34d; margin-bottom: 16px;">{get_icon("pulse", 14, "#b45309")} IN PLAY ON COURT</div>'
                else:
                    box_style = "background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 18px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);"
                    badge_markup = f'<span class="badge-pill badge-dark">{get_icon("shield", 12, "#0f172a")} BEST SELECTION • {best_pick["market"]}</span>'
                    action_markup = None

                st.markdown(f"""
                <div style="{box_style}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 8px;">
                        <div>{badge_markup}</div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 0.82rem; color: #64748b; font-weight: 600;">Model Prob: <b style="color: #0f172a;">{best_pick['win_prob_pct']}%</b></span>
                            <span class="odds-chip" style="font-size: 0.95rem; font-weight: 800; color: #059669;">@{best_pick['fair_odds']:.2f}</span>
                        </div>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">
                        {best_pick['title']}
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; line-height: 1.45;">
                        {best_pick['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if action_markup:
                    st.markdown(action_markup, unsafe_allow_html=True)
                else:
                    if st.button(f"+ Add to Slip ({best_pick['title']})", key=f"btn_add_slip_top_{idx}", use_container_width=True):
                        add_to_betslip(
                            match=f"{p1} vs {p2}",
                            market=best_pick["market"],
                            pick=best_pick["title"],
                            odds=best_pick["fair_odds"],
                            prob=best_pick["win_prob_pct"] / 100.0,
                        )
                        st.rerun()

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    else:
        st.info("No active predictions file found. Click 'Re-Scrape Daily Fixtures from ESPN' above.")


# =============================================================================
# TAB 2: BANKER PICKS RESULTS & SETTLEMENT TRACKER (DEDICATED SECTION)
# =============================================================================

with tab_banker_tracker:
    st.markdown(f"### {get_icon('layers', 20, '#0f172a')} Total Day Predictions Spreadsheet & Settlement Tracker", unsafe_allow_html=True)
    st.markdown("Complete master ledger containing **all evaluated matches for the day** with the single optimal Best Pick per fixture, decimal odds, win probabilities, real-time scorelines, and automated sportsbook settlement.")

    # Top Control Bar: Date Selector & Sync Button
    dates_avail = tracker.get_all_available_prediction_dates()
    today_iso = datetime.now().strftime("%Y-%m-%d")
    tomorrow_iso = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    date_options = []
    # Make sure Today is listed first so the dashboard immediately opens to Today's genuine results
    if today_iso in dates_avail:
        date_options.append(f"{today_iso} (Today)")
    for d in dates_avail:
        if d == today_iso:
            continue
        elif d == tomorrow_iso:
            date_options.append(f"{d} (Tomorrow)")
        else:
            date_options.append(f"{d}")
    if not date_options and dates_avail:
        date_options = [f"{d}" for d in dates_avail]
    date_options.append("All Dates (Consolidated)")

    ctl_c1, ctl_c2, ctl_c3 = st.columns([4, 3, 3])
    with ctl_c1:
        chosen_date_label = st.selectbox(
            "Select Settlement Date",
            options=date_options,
            index=0,
            help="Filter banker pick settlement ledger by specific date or view all dates consolidated from system start."
        )
    with ctl_c2:
        if st.button("Sync Live Scores from ESPN", type="primary", use_container_width=True):
            with st.spinner("Connecting to ESPN live scoreboards..."):
                st.rerun()
    with ctl_c3:
        if st.session_state["demo_settlements"]:
            if st.button("Reset Simulation & Restore Feed", use_container_width=True):
                st.session_state["demo_settlements"] = {}
                st.toast("Restored live ESPN feed")
                st.rerun()

    # Load Banker Picks Ledger for selected date
    if "All Dates" in chosen_date_label:
        selected_date_val = "all"
    else:
        m_dt = re.search(r"\d{4}-\d{2}-\d{2}", chosen_date_label)
        selected_date_val = m_dt.group(0) if m_dt else "all"

    df_banker_ledger, summary_banker = tracker.get_banker_picks_ledger(
        selected_date=selected_date_val,
        live_results=live_espn_results,
    )

    if not df_banker_ledger.empty:
        total_p = summary_banker.get("total_picks", len(df_banker_ledger))
        won_p = summary_banker.get("won_picks", 0)
        failed_p = summary_banker.get("failed_picks", 0)
        void_p = summary_banker.get("void_picks", 0)
        pending_p = summary_banker.get("pending_picks", 0)
        strike_rate = summary_banker.get("strike_rate_pct", 0.0)
        net_units = summary_banker.get("net_units", 0.0)

        # Performance KPI Banner (6 tiles including Voided)
        st.markdown("---")
        k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)

        with k_col1:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('layers', 14, '#64748b')} Total Banker Picks
                </div>
                <div class="quant-val">{total_p}</div>
                <div class="quant-sub">{chosen_date_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col2:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('check', 14, '#059669')} Won (Cashed)
                </div>
                <div class="quant-val" style="color: #059669;">{won_p}</div>
                <div class="quant-sub">Successful Picks</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col3:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('cross', 14, '#e11d48')} Failed (Lost)
                </div>
                <div class="quant-val" style="color: #e11d48;">{failed_p}</div>
                <div class="quant-sub">Failed Picks</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col4:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('void', 14, '#64748b')} Voided
                </div>
                <div class="quant-val" style="color: #64748b;">{void_p}</div>
                <div class="quant-sub">Walkovers & Ret.</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col5:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('clock', 14, '#334155')} Scheduled / In-Play
                </div>
                <div class="quant-val" style="color: #334155;">{pending_p}</div>
                <div class="quant-sub">Awaiting Outcomes</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col6:
            net_color = "#059669" if net_units >= 0 else "#e11d48"
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                    {get_icon('trending_up', 14, net_color)} Strike Rate & P&L
                </div>
                <div class="quant-val" style="color: {net_color};">{strike_rate:.1f}%</div>
                <div class="quant-sub">Net: {net_units:+.2f} Units</div>
            </div>
            """, unsafe_allow_html=True)

        # Settlement Simulator Expander
        with st.expander("Test Settlement Simulator (Verify Won, Failed & Walkover Voided)"):
            st.caption("Test how any match result settlements are graded by simulating final scores, retirements, or walkovers.")
            match_names = df_banker_ledger["match"].tolist()
            chosen_test_match = st.selectbox("Select Match to Test", options=match_names, index=0)
            test_row = df_banker_ledger[df_banker_ledger["match"] == chosen_test_match].iloc[0]
            st.info(f"System's Selected Banker Pick: **{test_row['banker_pick']}** (@{test_row['fair_odds']:.2f})")

            ts_c1, ts_c2, ts_c3, ts_c4 = st.columns(4)
            with ts_c1:
                if st.button(f"Simulate {test_row['player1']} Wins (6-4 6-3)", key="btn_test_win", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "FINAL",
                        "winner": test_row['player1'],
                        "score": "6-4 6-3",
                        "p1_sets": 2,
                        "p2_sets": 0,
                        "p1_games": 12,
                        "p2_games": 7,
                        "total_games": 19,
                    }
                    st.toast(f"Simulated {test_row['player1']} win 6-4 6-3")
                    st.rerun()

            with ts_c2:
                if st.button(f"Simulate Walkover (W/O)", key="btn_test_wo", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "WALKOVER",
                        "winner": test_row['player1'],
                        "score": "W/O (Walkover)",
                        "p1_sets": 0,
                        "p2_sets": 0,
                        "p1_games": 0,
                        "p2_games": 0,
                        "total_games": 0,
                    }
                    st.toast("Simulated Walkover -> Selection VOIDED")
                    st.rerun()

            with ts_c3:
                if st.button(f"Simulate Retirement (6-2 2-0 ret)", key="btn_test_ret", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "RETIRED",
                        "winner": test_row['player2'],
                        "score": "2-6 0-2 ret",
                        "p1_sets": 0,
                        "p2_sets": 1,
                        "p1_games": 2,
                        "p2_games": 8,
                        "total_games": 10,
                    }
                    st.toast("Simulated Retirement 2-6 0-2 ret")
                    st.rerun()

            with ts_c4:
                if st.button(f"Simulate {test_row['player2']} Wins (6-2 6-2)", key="btn_test_loss", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "FINAL",
                        "winner": test_row['player2'],
                        "score": "2-6 2-6",
                        "p1_sets": 0,
                        "p2_sets": 2,
                        "p1_games": 4,
                        "p2_games": 12,
                        "total_games": 16,
                    }
                    st.toast(f"Simulated {test_row['player2']} win 6-2 6-2")
                    st.rerun()

        # Decluttering, Search & Filtering Controls
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        search_filter = st.text_input("🔍 Search Matchup, Player, or Tournament Name", "", placeholder="e.g. Sinner, Alcaraz, Paris, R32...")

        flt_col1, flt_col2, flt_col3, flt_col4 = st.columns([3, 3, 3, 3])
        with flt_col1:
            banker_status_filter = st.selectbox(
                "Filter by Settlement Outcome",
                options=["All Banker Picks", "Won Picks Only", "Failed Picks Only", "Voided Picks Only", "Scheduled / In-Play Only"]
            )
        with flt_col2:
            all_tourneys = ["All Tournaments"] + sorted(df_banker_ledger["tournament"].dropna().unique().tolist())
            sel_tourney = st.selectbox("Filter Tournament", options=all_tourneys)
        with flt_col3:
            b_surfaces = df_banker_ledger["surface"].dropna().unique().tolist()
            sel_b_surf = st.multiselect("Filter Court Surface", options=b_surfaces, default=b_surfaces, key="sel_b_surf_tracker")
        with flt_col4:
            view_mode = st.radio("Display View", options=["Total Day Spreadsheet (Master Table)", "Grouped Cards by Tournament"], horizontal=True)

        # Apply Filters
        filtered_bankers = df_banker_ledger.copy()
        if search_filter:
            sq = search_filter.strip().lower()
            filtered_bankers = filtered_bankers[
                filtered_bankers["match"].str.lower().str.contains(sq, na=False) |
                filtered_bankers["tournament"].str.lower().str.contains(sq, na=False) |
                filtered_bankers["banker_pick"].str.lower().str.contains(sq, na=False)
            ]
        if sel_b_surf:
            filtered_bankers = filtered_bankers[filtered_bankers["surface"].isin(sel_b_surf)]
        if sel_tourney != "All Tournaments":
            filtered_bankers = filtered_bankers[filtered_bankers["tournament"] == sel_tourney]

        if banker_status_filter == "Won Picks Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"] == "WON"]
        elif banker_status_filter == "Failed Picks Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"] == "FAILED"]
        elif banker_status_filter == "Voided Picks Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"] == "VOID"]
        elif banker_status_filter == "Scheduled / In-Play Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"].isin(["PENDING", "IN_PLAY"])]

        c_hdr1, c_hdr2 = st.columns([7, 3])
        with c_hdr1:
            st.markdown(f"#### Complete Slate: **{len(filtered_bankers)}** Matches with Optimal Best Picks")
        with c_hdr2:
            csv_export_df = filtered_bankers[[
                "date", "tournament", "surface", "round", "match", "banker_pick", "market", "fair_odds", "win_prob_pct", "settlement", "actual_score", "profit_units"
            ]].copy()
            csv_export_df.columns = [
                "Date", "Tournament", "Court", "Round", "Matchup", "Single Best Pick", "Market", "Odds", "Win Prob %", "Settlement Status", "Actual Score", "P&L (Units)"
            ]
            csv_bytes = csv_export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Full Slate Spreadsheet (CSV)",
                data=csv_bytes,
                file_name=f"tenpred_slate_{selected_date_val}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # 1. Total Day Spreadsheet View (Master Table)
        if view_mode == "Total Day Spreadsheet (Master Table)":
            table_display = filtered_bankers[[
                "date", "tournament", "surface", "round", "match", "banker_pick", "fair_odds", "win_prob_pct", "status_badge", "actual_score", "profit_units"
            ]].copy()
            table_display.columns = [
                "Date", "Tournament", "Court", "Round", "Matchup", "Single Best Pick", "Odds", "Win Prob %", "Outcome Status", "Actual Score", "P&L (Units)"
            ]
            st.dataframe(
                table_display,
                use_container_width=True,
                hide_index=True,
                height=min(680, 45 * len(table_display) + 40),
            )

        # 2. Grouped Cards by Tournament View
        else:
            for tourney_name, grp in filtered_bankers.groupby("tournament"):
                with st.expander(f"{tourney_name} ({len(grp)} Matches)", expanded=True):
                    for b_idx, b_row in grp.iterrows():
                        m_title = b_row["match"]
                        surf = b_row["surface"]
                        rnd = b_row["round"]
                        b_pick = b_row["banker_pick"]
                        b_odds = float(b_row["fair_odds"])
                        b_prob = float(b_row["win_prob_pct"])
                        settle = b_row["settlement"]
                        badge = b_row["status_badge"]
                        detail = b_row["outcome_detail"]
                        b_date = b_row["date"]

                        if settle == "WON":
                            border_style = "border: 1px solid #bbf7d0; background: #f0fdf4;"
                            badge_html = f'<span class="badge-pill badge-emerald">{get_icon("check", 12, "#047857")} {badge}</span>'
                        elif settle == "FAILED":
                            border_style = "border: 1px solid #fecdd3; background: #fff1f2;"
                            badge_html = f'<span class="badge-pill badge-rose">{get_icon("cross", 12, "#be123c")} {badge}</span>'
                        elif settle == "VOID":
                            border_style = "border: 1px solid #e2e8f0; background: #f8fafc;"
                            badge_html = f'<span class="badge-pill badge-slate">{get_icon("void", 12, "#475569")} {badge}</span>'
                        elif settle == "IN_PLAY":
                            border_style = "border: 1px solid #fef3c7; background: #fffbeb;"
                            badge_html = f'<span class="badge-pill badge-amber">{get_icon("pulse", 12, "#b45309")} {badge}</span>'
                        else:
                            border_style = "border: 1px solid #e2e8f0; background: #ffffff;"
                            badge_html = f'<span class="badge-pill badge-slate">{get_icon("clock", 12, "#475569")} Scheduled</span>'

                        st.markdown(f"""
                        <div class="quant-card" style="{border_style} margin-bottom: 10px; padding: 14px 18px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px;">
                                <div>
                                    <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{m_title}</span>
                                    <span style="color: #64748b; font-size: 0.82rem; margin-left: 8px;">• {surf} Court • {rnd} • Date: {b_date}</span>
                                </div>
                                <div>{badge_html}</div>
                            </div>
                            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px;">
                                <div style="flex: 1; min-width: 220px;">
                                    <div style="display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; color: #b45309; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">{get_icon('shield', 12, '#b45309')} Best Match Pick</div>
                                    <div style="font-size: 1.1rem; font-weight: 800; color: #059669;">{b_pick}</div>
                                    <div class="mono-num" style="font-size: 0.82rem; color: #475569;">
                                        Model: <b>{b_prob:.1f}%</b> | Odds: <b>@{b_odds:.2f}</b>
                                    </div>
                                </div>
                                <div style="text-align: right; min-width: 200px;">
                                    <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Settlement Status</div>
                                    <div style="font-size: 0.9rem; font-weight: 600; color: #0f172a;">{detail}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    else:
        st.info("No banker pick records found for the selected date. Click 'Sync Live Scores from ESPN' above.")


# =============================================================================
# TAB 3: HEAD-TO-HEAD MATCH SIMULATOR & DERIVATIVE PRICER
# =============================================================================

with tab_sim:
    st.subheader("Interactive Match Simulator & Multi-Market Pricer")
    st.markdown("Select any two professional players, tournament context, court speed, and clean decimal odds to generate full derivative forecasts.")

    # Optional Quick Match Pre-fill from Today's Schedule
    upcoming_path = Path("data/live/upcoming_fixtures.csv")
    df_sched = pd.read_csv(upcoming_path) if upcoming_path.exists() else pd.DataFrame()

    prefill_match = None
    if not df_sched.empty:
        sched_options = ["-- Custom Matchup --"] + [
            f"{r['player1']} vs {r['player2']} [{r.get('tournament', r.get('surface', 'Tour'))}]"
            for _, r in df_sched.iterrows()
        ]
        chosen_sched = st.selectbox(
            "Quick-Fill from Scheduled Matches (Optional)",
            options=sched_options,
            index=0,
            help="Choose an upcoming tournament match to auto-fill players, surface, court conditions, and market odds instantly."
        )
        if chosen_sched != "-- Custom Matchup --":
            match_idx = sched_options.index(chosen_sched) - 1
            prefill_match = df_sched.iloc[match_idx].to_dict()

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if prefill_match and prefill_match.get("player1") in all_players:
            default_p1 = prefill_match["player1"]
        elif "Jannik Sinner" in all_players:
            default_p1 = "Jannik Sinner"
        else:
            default_p1 = all_players[0]

        p1_idx = all_players.index(default_p1) if default_p1 in all_players else 0
        player1 = st.selectbox("Player 1", options=all_players, index=p1_idx)

        default_p1_odds = float(prefill_match.get("p1_odds", 1.65)) if prefill_match else 1.65
        p1_odds = st.number_input("Player 1 Decimal Odds", min_value=1.01, max_value=25.0, value=default_p1_odds, step=0.05)

    with col_p2:
        if prefill_match and prefill_match.get("player2") in all_players:
            default_p2 = prefill_match["player2"]
        elif "Carlos Alcaraz" in all_players:
            default_p2 = "Carlos Alcaraz"
        else:
            default_p2 = all_players[1]

        p2_idx = all_players.index(default_p2) if default_p2 in all_players else 1
        player2 = st.selectbox("Player 2", options=all_players, index=p2_idx)

        default_p2_odds = float(prefill_match.get("p2_odds", 2.35)) if prefill_match else 2.35
        p2_odds = st.number_input("Player 2 Decimal Odds", min_value=1.01, max_value=25.0, value=default_p2_odds, step=0.05)

    col_ctx1, col_ctx2, col_ctx3, col_ctx4 = st.columns(4)
    with col_ctx1:
        default_surf = prefill_match.get("surface", "Hard") if prefill_match else "Hard"
        surf_options = ["Hard", "Clay", "Grass"]
        surf_idx = surf_options.index(default_surf) if default_surf in surf_options else 0
        surface = st.selectbox("Surface", options=surf_options, index=surf_idx)

    with col_ctx2:
        is_indoor_default = False
        if prefill_match:
            t_name = str(prefill_match.get("tournament", "")).lower()
            if any(k in t_name for k in ["indoor", "vienna", "basel", "paris", "atp finals"]):
                is_indoor_default = True
        is_indoor = st.checkbox("Indoor Tournament (Fast CPI)", value=is_indoor_default)

    with col_ctx3:
        default_rnd = prefill_match.get("round", "R32") if prefill_match else "R32"
        rnd_opts = ["F (Final)", "SF (Semifinal)", "QF (Quarterfinal)", "R16", "R32", "R64", "R128"]
        rnd_idx = 0
        for i, r_o in enumerate(rnd_opts):
            if r_o.startswith(str(default_rnd)):
                rnd_idx = i
                break
        round_name = st.selectbox("Round", options=rnd_opts, index=rnd_idx)
        round_code = round_name.split(" ")[0]

    with col_ctx4:
        best_of = st.selectbox("Format", options=[3, 5], index=0)

    tournament_input = prefill_match.get("tournament", "ATP/WTA Tour") if prefill_match else "ATP/WTA Tour"

    btn_simulate = st.button("Run Match Simulation", type="primary", use_container_width=True)

    if btn_simulate:
        if player1 == player2:
            st.error("Please choose two distinct players for the matchup.")
        elif pipeline is None:
            st.error("Prediction pipeline is not loaded.")
        else:
            with st.spinner("Calculating matchup probabilities and derivative pricing..."):
                pred = pipeline.predict_match(
                    player1=player1,
                    player2=player2,
                    surface=surface,
                    tournament_level="M",
                    round_name=round_code,
                    best_of=best_of,
                    p1_odds=p1_odds,
                    p2_odds=p2_odds,
                    tournament=tournament_input,
                    is_indoor=is_indoor,
                )

            # Top KPI Summary Cards
            p1_prob = float(pred["p1_win_prob"])
            p2_prob = float(pred["p2_win_prob"])
            p1_fair = float(pred["p1_fair_odds"])
            p2_fair = float(pred["p2_fair_odds"])

            st.markdown("---")
            kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

            with kpi_c1:
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                        {get_icon('trophy', 14, '#0f172a')} Predicted Winner
                    </div>
                    <div class="quant-val" style="color: #0f172a;">{pred['predicted_winner']}</div>
                    <div class="quant-sub">Win Probability: {max(p1_prob, p2_prob):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c2:
                edge_val = pred.get("edge_pct", 0.0)
                edge_color = "#059669" if edge_val > 0 else "#64748b"
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                        {get_icon('arrow_up_right', 14, edge_color)} Market Edge
                    </div>
                    <div class="quant-val" style="color: {edge_color};">{edge_val:+.1f}%</div>
                    <div class="quant-sub">Target: {pred.get('recommended_bet', 'Pass')}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c3:
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                        {get_icon('wallet', 14, '#b45309')} Kelly Allocation
                    </div>
                    <div class="quant-val" style="color: #b45309;">{pred.get('kelly_stake_pct', '0.0%')}</div>
                    <div class="quant-sub">Bankroll: ${user_bankroll:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c4:
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label" style="display: flex; align-items: center; gap: 6px;">
                        {get_icon('target', 14, '#0f172a')} Expected Total Games
                    </div>
                    <div class="quant-val" style="color: #0f172a;">{pred['pred_total_games']:.1f}</div>
                    <div class="quant-sub">Speed Index: {'Fast (Indoor)' if is_indoor else 'Standard'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Executive Summary: Top 3 Simple Pick Recommendations
            st.markdown("---")
            st.markdown(f"""
            <div style="background: #ffffff; padding: 18px 22px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #0f172a; font-size: 1.15rem; display: inline-flex; align-items: center; gap: 8px;">
                        {get_icon('target', 18, '#0f172a')} Top 3 Match Selections
                    </span>
                    <span class="badge-pill badge-emerald">Ranked by Confidence</span>
                </div>
                <p style="margin: 6px 0 0 0; color: #64748b; font-size: 0.85rem;">
                    Derived directly from calibrated probability distributions across match winner, set handicap, and game lines.
                </p>
            </div>
            """, unsafe_allow_html=True)

            top_sim_picks = pipeline.extract_top_three_picks(pred)
            if top_sim_picks:
                sim_p_cols = st.columns(3)
                badges_sim = ["Banker Pick", "Spread Pick", "Value Edge"]
                badge_sim_styles = ["badge-emerald", "badge-dark", "badge-amber"]
                sim_icons = ["shield", "layers", "arrow_up_right"]
                sim_icon_colors = ["#047857", "#0f172a", "#b45309"]

                for s_i, sp in enumerate(top_sim_picks):
                    with sim_p_cols[s_i]:
                        st.markdown(f"""
                        <div class="quant-card" style="height: 100%;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span class="badge-pill {badge_sim_styles[s_i]}">
                                    {get_icon(sim_icons[s_i], 12, sim_icon_colors[s_i])} {badges_sim[s_i]}
                                </span>
                                <span class="mono-num" style="font-weight: 700; color: #059669;">@{sp['fair_odds']:.2f}</span>
                            </div>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">{sp['title']}</div>
                            <div class="mono-num" style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">
                                {sp['win_prob_pct']}%
                            </div>
                            <div style="font-size: 0.85rem; color: #475569; line-height: 1.45; margin-bottom: 12px;">
                                {sp['explanation']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("+ Add to Betslip", key=f"btn_add_sim_{s_i}", use_container_width=True):
                            add_to_betslip(
                                match=f"{player1} vs {player2}",
                                market=sp["market"],
                                pick=sp["title"],
                                odds=sp["fair_odds"],
                                prob=sp["win_prob_pct"] / 100.0,
                            )
                            st.rerun()

            # Multi-Market Output Grid
            st.markdown("---")
            st.markdown("### Specialized Market Breakdowns")

            m_col1, m_col2 = st.columns(2)

            with m_col1:
                # 1. Player to win a set
                st.markdown("#### 1. Player to Win At Least One Set (Yes / No)")
                p1_to_win_set = float(pred["p1_win_at_least_one_set_prob"])
                p2_to_win_set = float(pred["p2_win_at_least_one_set_prob"])
                df_set_win = pd.DataFrame([
                    {"Player": player1, "Win >= 1 Set Prob": f"{p1_to_win_set:.1%}", "Fair Decimal Odds": f"{pred['p1_win_at_least_one_set_fair_odds']:.2f}"},
                    {"Player": player2, "Win >= 1 Set Prob": f"{p2_to_win_set:.1%}", "Fair Decimal Odds": f"{pred['p2_win_at_least_one_set_fair_odds']:.2f}"},
                ])
                st.dataframe(df_set_win, use_container_width=True, hide_index=True)

                # 2. Set Handicap Spreads
                st.markdown("#### 2. Set Handicap Markets (-1.5 / +1.5 Sets)")
                p1_m15 = float(pred["prob_p1_set_handicap_minus_1_5"])
                p2_p15 = float(pred["prob_p2_set_handicap_plus_1_5"])
                df_set_hc = pd.DataFrame([
                    {"Selection": f"{player1} -1.5 Sets (Straight 2-0)", "Probability": f"{p1_m15:.1%}", "Fair Decimal Odds": f"{1.0/max(1e-4, p1_m15):.2f}"},
                    {"Selection": f"{player2} +1.5 Sets (Wins >=1 Set)", "Probability": f"{p2_p15:.1%}", "Fair Decimal Odds": f"{1.0/max(1e-4, p2_p15):.2f}"},
                ])
                st.dataframe(df_set_hc, use_container_width=True, hide_index=True)

                # 3. First Set Winner & Tiebreak
                st.markdown("#### 3. First Set & Tiebreak Frequency")
                p1_s1 = float(pred["p1_first_set_prob"])
                p2_s1 = float(pred["p2_first_set_prob"])
                df_s1 = pd.DataFrame([
                    {"Market": "1st Set Winner", f"{player1}": f"{p1_s1:.1%} (@{pred['p1_first_set_fair_odds']:.2f})", f"{player2}": f"{p2_s1:.1%} (@{pred['p2_first_set_fair_odds']:.2f})"},
                    {"Market": "Tiebreak Frequency", f"{player1}": f"Set 1 TB: {pred.get('prob_first_set_tiebreak', 0.2):.1%}", f"{player2}": f"Any TB: {pred.get('prob_any_tiebreak', 0.4):.1%}"},
                ])
                st.dataframe(df_s1, use_container_width=True, hide_index=True)

            with m_col2:
                # 4. Correct Set Score Betting Distribution
                st.markdown("#### 4. Exact Set Score Betting (Correct Score)")
                score_data = {
                    f"{player1} 2-0": float(pred.get("prob_set_score_2_0", 0.4)),
                    f"{player1} 2-1": float(pred.get("prob_set_score_2_1", 0.25)),
                    f"{player2} 2-1": float(pred.get("prob_set_score_1_2", 0.20)),
                    f"{player2} 2-0": float(pred.get("prob_set_score_0_2", 0.15)),
                }
                fig_scores = px.bar(
                    x=list(score_data.keys()),
                    y=[v * 100.0 for v in score_data.values()],
                    color=list(score_data.keys()),
                    color_discrete_sequence=["#0f172a", "#334155", "#475569", "#64748b"],
                    labels={"x": "Scoreline", "y": "Probability (%)"},
                    title="Exact Set Score Distribution",
                )
                fig_scores.update_layout(height=220, showlegend=False, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a"))
                st.plotly_chart(fig_scores, use_container_width=True)

                # 5. Total Match Games & Over Lines
                st.markdown("#### 5. Total Match Games Over/Under Lines")
                tg_lines = {
                    "Over 20.5": pred.get("prob_over_20_5", 0.5),
                    "Over 21.5": pred.get("prob_over_21_5", 0.5),
                    "Over 22.5": pred.get("prob_over_22_5", 0.5),
                    "Over 23.5": pred.get("prob_over_23_5", 0.5),
                }
                df_tg = pd.DataFrame([
                    {"Line": k, "Over Prob": f"{v:.1%}", "Under Prob": f"{1.0 - v:.1%}", "Fair Over Decimal Odds": f"{1.0/max(1e-4, v):.2f}"}
                    for k, v in tg_lines.items()
                ])
                st.dataframe(df_tg, use_container_width=True, hide_index=True)

                # 6. Game Handicap Spreads
                st.markdown("#### 6. Game Handicap Spread Lines")
                hc_lines = {
                    f"{player1} -2.5 Games": pred.get("prob_p1_cover_minus_2_5", 0.5),
                    f"{player1} -3.5 Games": pred.get("prob_p1_cover_minus_3_5", 0.5),
                    f"{player1} -4.5 Games": pred.get("prob_p1_cover_minus_4_5", 0.5),
                    f"{player1} +1.5 Games": pred.get("prob_p1_cover_plus_1_5", 0.5),
                }
                df_hc = pd.DataFrame([
                    {"Handicap Spread": k, "Cover Prob": f"{v:.1%}", "Fair Decimal Odds": f"{1.0/max(1e-4, v):.2f}"}
                    for k, v in hc_lines.items()
                ])
                st.dataframe(df_hc, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4: PLAYER RANKINGS & SURFACE ELO
# =============================================================================

with tab_elo:
    st.subheader("Player Surface Rankings & Elo Ratings")
    st.markdown("Dynamic surface-specific ratings tracking professional players across Hard, Clay, and Grass courts.")

    if not df_elo.empty:
        c_surf, c_search, c_limit = st.columns([2, 3, 2])
        with c_surf:
            selected_surface = st.selectbox("Surface Leaderboard", ["Overall", "Hard", "Clay", "Grass"])
        with c_search:
            search_query = st.text_input("Search Player Name", "")
        with c_limit:
            top_limit = st.slider("Display Top N", 10, 100, 25, 5)

        target_col = f"elo_{selected_surface.lower()}"
        df_rank = df_elo.sort_values(by=target_col, ascending=False).copy()

        if search_query:
            df_rank = df_rank[df_rank["player"].str.contains(search_query, case=False, na=False)]

        df_rank["Rank"] = range(1, len(df_rank) + 1)
        if selected_surface == "Overall":
            display_rank = df_rank[["Rank", "player", "elo_overall", "matches_played", "last_active"]].head(top_limit).copy()
            display_rank.columns = ["Rank", "Player Name", "Overall Elo", "Matches Logged", "Last Active"]
            chart_col = "Overall Elo"
        else:
            display_rank = df_rank[["Rank", "player", target_col, "elo_overall", "matches_played", "last_active"]].head(top_limit).copy()
            display_rank.columns = ["Rank", "Player Name", f"{selected_surface} Elo", "Overall Elo", "Matches Logged", "Last Active"]
            chart_col = f"{selected_surface} Elo"

        # Top 15 visual bar chart
        st.markdown(f"#### Top 15 Players on {selected_surface} Courts")
        top15 = display_rank.head(15)
        fig_elo = px.bar(
            top15,
            x=chart_col,
            y="Player Name",
            orientation="h",
            color=chart_col,
            color_continuous_scale=[[0, "#94a3b8"], [0.5, "#475569"], [1, "#0f172a"]],
        )
        fig_elo.update_layout(
            yaxis=dict(autorange="reversed"),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(fig_elo, use_container_width=True)

        st.markdown(f"#### Full Leaderboard Table ({selected_surface})")
        st.dataframe(display_rank, use_container_width=True, hide_index=True)
    else:
        st.warning("Elo database not loaded.")


# =============================================================================
# TAB 5: BANKROLL GROWTH & DRAWDOWN CALCULATOR
# =============================================================================

with tab_risk:
    st.subheader("Bankroll Growth & Drawdown Calculator")
    st.markdown("Project long-term portfolio growth, expected drawdown risk, and variance across sequential matches under fractional Kelly staking.")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        sim_bets = st.number_input("Number of Sequential Bets", min_value=50, max_value=2000, value=300, step=50)
    with r_col2:
        sim_win_rate = st.slider("Expected Win Rate (%)", 45.0, 75.0, 58.0, 0.5) / 100.0
    with r_col3:
        sim_odds = st.number_input("Average Decimal Odds", min_value=1.20, max_value=5.0, value=1.85, step=0.05)
    with r_col4:
        sim_stake_pct = st.slider("Fixed Stake Size (% of Bankroll)", 0.5, 4.0, 1.5, 0.25)

    if st.button("Run Portfolio Simulation", type="primary"):
        with st.spinner("Simulating portfolio trajectories..."):
            sim_res = risk_manager.simulate_bankroll(
                n_bets=sim_bets,
                avg_win_prob=sim_win_rate,
                avg_decimal_odds=sim_odds,
                stake_pct=sim_stake_pct,
                n_simulations=500,
            )

        # Risk Metrics Grid
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Expected Final Bankroll", f"${sim_res['expected_final_bankroll']:,.0f}", delta=f"{((sim_res['expected_final_bankroll']/user_bankroll) - 1.0):+.1%}")
        with m_c2:
            st.metric("Probability of Doubling", f"{sim_res['prob_doubling_bankroll_pct']:.1f}%")
        with m_c3:
            st.metric("Median Max Drawdown", f"{sim_res['median_max_drawdown_pct']:.1f}%")
        with m_c4:
            st.metric("95% Value-at-Risk (Worst DD)", f"{sim_res['worst_case_drawdown_95pct_var']:.1f}%")

        # Plot sample bankroll paths
        np.random.seed(42)
        steps = sim_bets
        n_paths = 30
        trajectories = np.zeros((n_paths, steps + 1))
        trajectories[:, 0] = user_bankroll

        for p in range(n_paths):
            b_curr = user_bankroll
            outcomes = np.random.binomial(1, sim_win_rate, size=steps)
            for s_idx, won in enumerate(outcomes):
                stk = b_curr * (sim_stake_pct / 100.0)
                if won == 1:
                    b_curr += stk * (sim_odds - 1.0)
                else:
                    b_curr -= stk
                trajectories[p, s_idx + 1] = max(0.0, b_curr)

        df_paths = pd.DataFrame(trajectories.T)
        fig_mc = go.Figure()
        for col_name in df_paths.columns:
            fig_mc.add_trace(go.Scatter(
                y=df_paths[col_name],
                mode='lines',
                line=dict(width=1.2, color="rgba(71, 85, 105, 0.18)"),
                showlegend=False,
            ))

        # Add median path
        fig_mc.add_trace(go.Scatter(
            y=df_paths.median(axis=1),
            mode='lines',
            line=dict(width=3.2, color="#0f172a"),
            name="Median Trajectory",
        ))

        fig_mc.update_layout(
            title="500 Simulated Portfolio Trajectories",
            xaxis_title="Sequential Matches Bet",
            yaxis_title="Portfolio Equity ($)",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(fig_mc, use_container_width=True)
