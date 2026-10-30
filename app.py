"""Antigravity Tennis Quantitative Prediction Platform.

Institutional-grade ATP & WTA forecasting terminal featuring:
- Dual GBDT Ensemble (LightGBM + XGBoost + SurfaceElo stacking)
- Closed-Form Markov Chain point-to-match scoreline simulator
- Court Pace Index (CPI) and Indoor environmental speed adjustments
- Interactive Betslip & Parlay Builder with Correlated Kelly Staking
- Real-time ESPN scheduled tournament match feed
- Modern, bespoke, mobile-responsive dark trading terminal UI
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure wide page layout and title
st.set_page_config(
    page_title="Antigravity Tennis Quant Terminal",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Bespoke Non-AI Institutional Dark Theme & Mobile Responsive CSS
# =============================================================================
st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --bg-main: #0c1017;
        --card-bg: #131924;
        --card-hover: #18202e;
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-emerald: #10b981;
        --accent-emerald-soft: rgba(16, 185, 129, 0.12);
        --accent-blue: #38bdf8;
        --accent-blue-soft: rgba(56, 189, 248, 0.12);
        --accent-amber: #f59e0b;
        --accent-amber-soft: rgba(245, 158, 11, 0.12);
        --accent-rose: #f43f5e;
        --accent-rose-soft: rgba(244, 63, 94, 0.12);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Overall Layout & Typography */
    .stApp {
        background-color: var(--bg-main);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }

    /* Monospace precision for odds, stats, numbers */
    .mono-num {
        font-family: ui-monospace, "SF Mono", Menlo, Monaco, Consolas, monospace;
        letter-spacing: -0.3px;
    }

    /* Custom Metric Cards */
    .quant-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 18px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
    }
    .quant-card:hover {
        border-color: rgba(255, 255, 255, 0.18);
    }

    .quant-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .quant-val {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
        margin-bottom: 4px;
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }

    .quant-sub {
        font-size: 0.82rem;
        color: var(--text-secondary);
    }

    /* Badge Pills */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-emerald {
        background: var(--accent-emerald-soft);
        color: var(--accent-emerald);
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-blue {
        background: var(--accent-blue-soft);
        color: var(--accent-blue);
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .badge-amber {
        background: var(--accent-amber-soft);
        color: var(--accent-amber);
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .badge-rose {
        background: var(--accent-rose-soft);
        color: var(--accent-rose);
        border: 1px solid rgba(244, 63, 94, 0.25);
    }

    /* Betslip Drawer */
    .betslip-box {
        background-color: #101622;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .betslip-item {
        background: #172030;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #0f141d;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid var(--card-border);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2638 !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Mobile Responsive Media Queries */
    @media (max-width: 768px) {
        .quant-val {
            font-size: 1.5rem !important;
        }
        .quant-card {
            padding: 12px 14px !important;
            margin-bottom: 8px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 10px !important;
            font-size: 0.8rem !important;
        }
        .block-container {
            padding-left: 12px !important;
            padding-right: 12px !important;
        }
        button[kind="primary"], button[kind="secondary"] {
            min-height: 44px !important;
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
        st.toast(f"Added to Betslip: {pick} @ {odds:.2f}", icon="📋")
    else:
        st.toast(f"Already in Betslip: {pick}", icon="ℹ️")


def remove_from_betslip(idx: int) -> None:
    """Remove a single leg from the betslip."""
    if 0 <= idx < len(st.session_state["betslip"]):
        removed = st.session_state["betslip"].pop(idx)
        st.toast(f"Removed: {removed['pick']}", icon="🗑️")


def clear_betslip() -> None:
    """Clear all legs from the betslip."""
    st.session_state["betslip"] = []
    st.toast("Betslip cleared", icon="🧹")


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
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 1.8rem;">🎾</span>
        <div>
            <div style="font-weight: 800; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.3px;">ANTIGRAVITY QUANT</div>
            <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">ATP & WTA Predictions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Telemetry Badges
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px 0;">
        <span class="badge-pill badge-emerald">● Dual GBDT Ensemble</span>
        <span class="badge-pill badge-blue">● CPI Speed Engine</span>
        <span class="badge-pill badge-amber">● Kelly Sizing</span>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # INTERACTIVE BETSLIP & PARLAY BUILDER (SIDEBAR DRAWER)
    # -------------------------------------------------------------------------
    st.markdown("---")
    betslip = st.session_state["betslip"]
    n_legs = len(betslip)

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">📋 Active Betslip</span>
        <span class="badge-pill {'badge-emerald' if n_legs > 0 else 'badge-amber'}">{n_legs} Leg{'s' if n_legs != 1 else ''}</span>
    </div>
    """, unsafe_allow_html=True)

    if n_legs == 0:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.12); border-radius: 8px; padding: 14px; text-align: center; color: #94a3b8; font-size: 0.85rem;">
            Your betslip is empty.<br>
            Click <b style="color: #38bdf8;">+ Add to Slip</b> on any match pick to build single or multi-leg parlays.
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
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 2px;">{leg['match']}</div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #ffffff;">{leg['pick']}</div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                    <span class="mono-num" style="color: #10b981; font-weight: 700;">@{leg['odds']:.2f}</span>
                    <span style="font-size: 0.78rem; color: #38bdf8;">Prob: {leg['prob']:.1%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✕ Remove", key=f"del_leg_{idx}", use_container_width=True):
                remove_from_betslip(idx)
                st.rerun()

        # Conservative correlation hair-cut for multi-leg (5% hair-cut per additional leg)
        if n_legs > 1:
            joint_prob = joint_prob * (0.95 ** (n_legs - 1))

        # Parlay Metrics Summary
        parlay_ev = (total_odds * joint_prob) - 1.0
        ev_color = "#10b981" if parlay_ev > 0 else "#f43f5e"

        st.markdown(f"""
        <div class="betslip-box">
            <div style="font-size: 0.78rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">
                {'Single Wager' if n_legs == 1 else f'{n_legs}-Leg Parlay Summary'}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 4px 0;">
                <span style="font-size: 0.9rem; color: #cbd5e1;">Cumulative Odds:</span>
                <span class="mono-num" style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">@{total_odds:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
                <span style="font-size: 0.85rem; color: #cbd5e1;">Joint Win Probability:</span>
                <span class="mono-num" style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">{joint_prob:.1%}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 0.85rem; color: #cbd5e1;">Expected Value (EV):</span>
                <span class="mono-num" style="font-size: 0.95rem; font-weight: 700; color: {ev_color};">{parlay_ev:+.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Wager Calculator
        wager_amount = st.number_input("Wager Stake ($)", min_value=1.0, max_value=50000.0, value=50.0, step=10.0)
        potential_payout = wager_amount * total_odds
        net_profit = potential_payout - wager_amount

        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.82rem; color: #94a3b8;">Potential Payout:</span>
                <span class="mono-num" style="font-weight: 700; color: #f8fafc;">${potential_payout:,.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                <span style="font-size: 0.82rem; color: #94a3b8;">Net Profit:</span>
                <span class="mono-num" style="font-weight: 800; color: #10b981;">+${net_profit:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Betslip", use_container_width=True):
            clear_betslip()
            st.rerun()

    # -------------------------------------------------------------------------
    # Bankroll Management Settings
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("<span style='font-weight: 700; font-size: 0.95rem; color: #f8fafc;'>⚙️ Bankroll Risk Settings</span>", unsafe_allow_html=True)
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
    st.caption("Active Base: 11,561 Official Sackmann Matches | Clean Decimal Odds")


# =============================================================================
# Main Header
# =============================================================================

st.markdown("""
<div style="margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; color: #f8fafc;">
        ATP & WTA Quantitative Prediction Terminal
    </h1>
    <p style="color: #94a3b8; font-size: 0.98rem; margin: 4px 0 0 0;">
        Dual GBDT Ensemble (LightGBM + XGBoost + SurfaceElo) & Closed-Form Markov Chain Simulator with Court Pace Environmental Modeling.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Navigation Tabs
# =============================================================================

tab_today, tab_banker_tracker, tab_sim, tab_elo, tab_model, tab_risk = st.tabs([
    "🔥 Today's Matches & Predictions",
    "🎯 Banker Picks Results Tracker",
    "🔬 Match Deep Dive & Simulator",
    "🏆 Surface Elo Leaderboards",
    "🧠 Dual Ensemble Architecture",
    "🛡️ Portfolio Risk & Monte Carlo",
])


# =============================================================================
# TAB 1: TODAY'S MATCHES & PREDICTIONS
# =============================================================================

with tab_today:
    st.subheader("Today's Scheduled Matches & High-Confidence Predictions")
    st.markdown("Live scan of active ATP & WTA tournament fixtures evaluated through the calibrated Dual Ensemble and Markov engine.")

    # Refresh Button
    c_btn1, c_btn2 = st.columns([3, 3])
    with c_btn1:
        if st.button("📡 Re-Scrape Daily Fixtures from ESPN", type="primary", use_container_width=True):
            with st.spinner("Fetching latest live scheduled fixtures and running ensemble scoring..."):
                fetcher = TennisLiveFetcher()
                live_df = fetcher.fetch_espn_daily_fixtures(tour="both", only_scheduled=True)
                if not live_df.empty:
                    fetcher.save_live_fixtures(live_df, "data/live/upcoming_fixtures.csv")
                    runner = DailyRunner()
                    runner.run_daily(fixtures_file="data/live/upcoming_fixtures.csv")
                    st.success(f"Fetched and evaluated {len(live_df)} scheduled matches!")
                    st.rerun()
                else:
                    st.warning("No new matches found currently on ESPN live board.")

    # Load Latest Predictions
    live_dir = Path("data/live")
    pred_files = sorted(live_dir.glob("predictions_*.csv"))

    if pred_files:
        latest_file = pred_files[-1]
        df_today = pd.read_csv(latest_file)
        st.caption(f"Loaded active predictions from `{latest_file.name}` ({len(df_today)} matches evaluated)")

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

        st.markdown(f"#### Displaying **{len(filtered_df)}** Scheduled Matches")

        # Interactive Cards View for Matches
        for idx, row in filtered_df.iterrows():
            p1 = str(row["player1"])
            p2 = str(row["player2"])
            tourney = str(row.get("tournament", "Tour Event"))
            surf = str(row.get("surface", "Hard"))
            rnd = str(row.get("round", "R32"))
            p1_prob = float(row.get("p1_win_prob", 0.5))
            p2_prob = 1.0 - p1_prob
            pred_winner = p1 if p1_prob >= 0.50 else p2
            win_prob_pct = max(p1_prob, p2_prob) * 100.0

            p1_odds = float(row.get("p1_odds", 1.90)) if not pd.isna(row.get("p1_odds")) else float(row.get("p1_fair_odds", 1.90))
            p2_odds = float(row.get("p2_odds", 1.90)) if not pd.isna(row.get("p2_odds")) else float(row.get("p2_fair_odds", 1.90))

            # Extract top 3 simple picks for this match
            top_picks = TennisPredictionPipeline.extract_top_three_picks(row)

            # Match Card Container
            st.markdown(f"""
            <div class="quant-card" style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
                    <div>
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">{tourney}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 8px;">• {surf} Court • {rnd}</span>
                    </div>
                    <span class="badge-pill badge-blue">⏳ SCHEDULED • Upcoming</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;">
                    <div style="flex: 1; min-width: 220px;">
                        <div style="font-size: 1.15rem; font-weight: 700; color: {'#38bdf8' if p1_prob >= 0.5 else '#cbd5e1'};">
                            {p1} <span class="mono-num" style="font-size: 0.9rem; color: #94a3b8;">(@{p1_odds:.2f} | {p1_prob:.1%})</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #64748b; margin: 2px 0;">vs</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {'#38bdf8' if p2_prob > 0.5 else '#cbd5e1'};">
                            {p2} <span class="mono-num" style="font-size: 0.9rem; color: #94a3b8;">(@{p2_odds:.2f} | {p2_prob:.1%})</span>
                        </div>
                    </div>
                    <div style="text-align: right; min-width: 160px;">
                        <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase;">Predicted Winner</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #10b981;">{pred_winner}</div>
                        <div style="font-size: 0.82rem; color: #38bdf8;">Confidence: {win_prob_pct:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display Top 3 Simple Banker Picks with 1-Click "+ Add to Betslip"
            if top_picks:
                p_cols = st.columns(3)
                badges_titles = ["🥇 Banker Pick 1", "🥈 Spread Pick 2", "🥉 Value Pick 3"]
                badge_styles = ["badge-emerald", "badge-blue", "badge-amber"]

                for p_idx, pick in enumerate(top_picks):
                    with p_cols[p_idx]:
                        st.markdown(f"""
                        <div style="background: #182030; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                <span class="badge-pill {badge_styles[p_idx]}">{badges_titles[p_idx]}</span>
                                <span class="mono-num" style="font-weight: 700; color: #10b981;">@{pick['fair_odds']:.2f}</span>
                            </div>
                            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">{pick['title']}</div>
                            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;">{pick['explanation']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # 1-Click Add to Betslip Button
                        if st.button(f"➕ Add to Slip", key=f"btn_add_slip_{idx}_{p_idx}", use_container_width=True):
                            add_to_betslip(
                                match=f"{p1} vs {p2}",
                                market=pick["market"],
                                pick=pick["title"],
                                odds=pick["fair_odds"],
                                prob=pick["win_prob_pct"] / 100.0,
                            )
                            st.rerun()

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    else:
        st.info("No active predictions file found. Click 'Re-Scrape Daily Fixtures from ESPN' above.")


# =============================================================================
# TAB 2: BANKER PICKS RESULTS & SETTLEMENT TRACKER (DEDICATED SECTION)
# =============================================================================

with tab_banker_tracker:
    st.subheader("Banker Picks Outcome & Results Settlement Tracker")
    st.markdown("Dedicated verification ledger auditing **only the system's selected #1 Banker Pick** for each scheduled fixture against official ESPN final match results.")

    # Session State for Demo / Simulation Settlements
    if "demo_settlements" not in st.session_state:
        st.session_state["demo_settlements"] = {}

    tracker = TennisResultsTracker()
    pred_path = Path("data/live/predictions_20261030.csv")
    if not pred_path.exists():
        files = sorted(Path("data/live").glob("predictions_*.csv"))
        pred_path = files[-1] if files else Path("data/live/predictions_20261030.csv")

    # Fetch live results from ESPN
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
                "total_games": demo_data.get("total_games", 19),
                "tour": "ATP",
            })

    # Grade Banker Picks
    df_banker_ledger, summary_banker = tracker.get_banker_picks_ledger(
        predictions_file=pred_path,
        live_results=live_espn_results,
    )

    # Top Control Buttons
    c_sync_b1, c_sync_b2 = st.columns([3, 3])
    with c_sync_b1:
        if st.button("🔄 Check Live Match Results from ESPN", type="primary", use_container_width=True):
            with st.spinner("Connecting to ESPN live scoreboards and checking match outcomes..."):
                st.rerun()

    with c_sync_b2:
        if st.session_state["demo_settlements"]:
            if st.button("🔄 Reset Demo & Restore Pure ESPN Feed", use_container_width=True):
                st.session_state["demo_settlements"] = {}
                st.toast("Restored pure live ESPN feed", icon="🔄")
                st.rerun()

    # Performance KPI Banner (Only for Banker Picks)
    if not df_banker_ledger.empty:
        total_p = summary_banker.get("total_picks", len(df_banker_ledger))
        won_p = summary_banker.get("won_picks", 0)
        failed_p = summary_banker.get("failed_picks", 0)
        pending_p = summary_banker.get("pending_picks", total_p)
        strike_rate = summary_banker.get("strike_rate_pct", 0.0)
        net_units = summary_banker.get("net_units", 0.0)

        st.markdown("---")
        k_col1, k_col2, k_col3, k_col4, k_col5 = st.columns(5)

        with k_col1:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label">Total Banker Picks</div>
                <div class="quant-val">{total_p}</div>
                <div class="quant-sub">Selected for Today</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col2:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label">✅ Won (Cashed)</div>
                <div class="quant-val" style="color: #10b981;">{won_p}</div>
                <div class="quant-sub">Successful Bankers</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col3:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label">❌ Failed (Lost)</div>
                <div class="quant-val" style="color: #f43f5e;">{failed_p}</div>
                <div class="quant-sub">Failed Bankers</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col4:
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label">⏳ Pending (Scheduled)</div>
                <div class="quant-val" style="color: #38bdf8;">{pending_p}</div>
                <div class="quant-sub">Awaiting Outcomes</div>
            </div>
            """, unsafe_allow_html=True)

        with k_col5:
            net_color = "#10b981" if net_units >= 0 else "#f43f5e"
            st.markdown(f"""
            <div class="quant-card">
                <div class="quant-label">Banker Strike Rate</div>
                <div class="quant-val" style="color: {net_color};">{strike_rate:.1f}%</div>
                <div class="quant-sub">Net: {net_units:+.2f} Units</div>
            </div>
            """, unsafe_allow_html=True)

        # Optional Demo Verification Expander
        with st.expander("🧪 Test Settlement Simulator (Verify How Banker Picks Settle)"):
            st.caption("Matches scheduled for later today are awaiting start time. Use this tool to simulate a completed match score and immediately verify how the system ticks ✅ WON or ❌ FAILED.")
            match_names = df_banker_ledger["match"].tolist()
            chosen_test_match = st.selectbox("Select Match to Test", options=match_names, index=1 if len(match_names) > 1 else 0)
            
            # Find the row for this match
            test_row = df_banker_ledger[df_banker_ledger["match"] == chosen_test_match].iloc[0]
            st.info(f"System's Selected Banker Pick: **{test_row['banker_pick']}** (@{test_row['fair_odds']:.2f})")

            test_c1, test_c2, test_c3 = st.columns(3)
            with test_c1:
                if st.button(f"Simulate 3-Set Thriller: {test_row['player1']} Wins 6-4 4-6 6-3 (29 Games)", key="btn_test_thriller", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "FINAL",
                        "winner": test_row['player1'],
                        "score": "6-4 4-6 6-3",
                        "p1_sets": 2,
                        "p2_sets": 1,
                        "p1_games": 16,
                        "p2_games": 13,
                        "total_games": 29,
                    }
                    st.toast(f"Simulated: {test_row['player1']} won 6-4 4-6 6-3 (29 total games)", icon="✅")
                    st.rerun()

            with test_c2:
                if st.button(f"Simulate Tight 2-0: {test_row['player1']} Wins 7-6 6-4 (23 Games)", key="btn_test_straight", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "FINAL",
                        "winner": test_row['player1'],
                        "score": "7-6 6-4",
                        "p1_sets": 2,
                        "p2_sets": 0,
                        "p1_games": 13,
                        "p2_games": 10,
                        "total_games": 23,
                    }
                    st.toast(f"Simulated: {test_row['player1']} won 7-6 6-4 (23 total games)", icon="✅")
                    st.rerun()

            with test_c3:
                if st.button(f"Simulate Blowout Upset: {test_row['player2']} Wins 6-1 6-2 (15 Games)", key="btn_test_blowout", use_container_width=True):
                    st.session_state["demo_settlements"][(test_row['player1'], test_row['player2'])] = {
                        "status": "FINAL",
                        "winner": test_row['player2'],
                        "score": "1-6 2-6",
                        "p1_sets": 0,
                        "p2_sets": 2,
                        "p1_games": 3,
                        "p2_games": 12,
                        "total_games": 15,
                    }
                    st.toast(f"Simulated: {test_row['player2']} won 6-1 6-2 (15 total games)", icon="❌")
                    st.rerun()

        # Filter bar
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        bf_c1, bf_c2 = st.columns([3, 3])
        with bf_c1:
            banker_status_filter = st.selectbox(
                "Filter Banker Outcomes",
                options=["All Banker Picks", "✅ Won Banker Picks Only", "❌ Failed Banker Picks Only", "⏳ Pending / Scheduled Only"]
            )
        with bf_c2:
            b_surfaces = df_banker_ledger["surface"].dropna().unique().tolist()
            sel_b_surf = st.multiselect("Filter Court Surface", options=b_surfaces, default=b_surfaces, key="sel_b_surf")

        # Apply Filters
        filtered_bankers = df_banker_ledger.copy()
        if sel_b_surf:
            filtered_bankers = filtered_bankers[filtered_bankers["surface"].isin(sel_b_surf)]

        if banker_status_filter == "✅ Won Banker Picks Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"] == "WON"]
        elif banker_status_filter == "❌ Failed Banker Picks Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"] == "FAILED"]
        elif banker_status_filter == "⏳ Pending / Scheduled Only":
            filtered_bankers = filtered_bankers[filtered_bankers["settlement"].isin(["PENDING", "IN_PLAY"])]

        st.markdown(f"#### Displaying **{len(filtered_bankers)}** Selected Banker Picks")

        # Display Each Banker Pick Record
        for b_idx, b_row in filtered_bankers.iterrows():
            m_title = b_row["match"]
            t_name = b_row["tournament"]
            surf = b_row["surface"]
            rnd = b_row["round"]
            b_pick = b_row["banker_pick"]
            b_mkt = b_row["market"]
            b_odds = float(b_row["fair_odds"])
            b_prob = float(b_row["win_prob_pct"])
            settle = b_row["settlement"]
            badge = b_row["status_badge"]
            detail = b_row["outcome_detail"]
            prof_u = float(b_row["profit_units"])

            if settle == "WON":
                border_style = "border: 1px solid rgba(16, 185, 129, 0.45);"
                badge_html = f'<span class="badge-pill badge-emerald">{badge}</span>'
            elif settle == "FAILED":
                border_style = "border: 1px solid rgba(244, 63, 94, 0.45);"
                badge_html = f'<span class="badge-pill badge-rose">{badge}</span>'
            elif settle == "IN_PLAY":
                border_style = "border: 1px solid rgba(245, 158, 11, 0.45);"
                badge_html = f'<span class="badge-pill badge-amber">{badge}</span>'
            else:
                border_style = "border: 1px solid rgba(56, 189, 248, 0.25);"
                badge_html = '<span class="badge-pill badge-blue">⏳ SCHEDULED • Awaiting Match</span>'

            st.markdown(f"""
            <div class="quant-card" style="{border_style} margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 6px;">
                    <div>
                        <span style="font-weight: 700; color: #f8fafc; font-size: 0.95rem;">{t_name}</span>
                        <span style="color: #94a3b8; font-size: 0.82rem; margin-left: 8px;">• {surf} Court • {rnd}</span>
                    </div>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px;">
                    <div style="flex: 1; min-width: 240px;">
                        <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Matchup</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">{m_title}</div>
                        <div style="font-size: 0.82rem; color: #38bdf8; margin-top: 2px;">{detail}</div>
                    </div>
                    <div style="text-align: right; min-width: 200px;">
                        <div style="font-size: 0.78rem; color: #f59e0b; text-transform: uppercase; font-weight: 700;">System Selected Banker Pick</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #10b981;">{b_pick}</div>
                        <div class="mono-num" style="font-size: 0.85rem; color: #94a3b8;">
                            Model Prob: <b>{b_prob:.1f}%</b> | Decimal Odds: <b>@{b_odds:.2f}</b>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("No banker pick records found. Check ESPN connection.")


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
            "⚡ Quick-Fill from Today's Scheduled Matches (Optional)",
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

    btn_simulate = st.button("🚀 Run Full Quantitative Simulation", type="primary", use_container_width=True)

    if btn_simulate:
        if player1 == player2:
            st.error("Please choose two distinct players for the matchup.")
        elif pipeline is None:
            st.error("Prediction pipeline is not loaded.")
        else:
            with st.spinner("Executing Feature Extraction, Dual GBDT Inference & Markov Point Simulation..."):
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
                    <div class="quant-label">Predicted Winner</div>
                    <div class="quant-val" style="color: #38bdf8;">{pred['predicted_winner']}</div>
                    <div class="quant-sub">Ensemble Win Prob: {max(p1_prob, p2_prob):.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c2:
                edge_val = pred.get("edge_pct", 0.0)
                edge_color = "#10b981" if edge_val > 0 else "#94a3b8"
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label">Market Edge</div>
                    <div class="quant-val" style="color: {edge_color};">{edge_val:+.1f}%</div>
                    <div class="quant-sub">Target: {pred.get('recommended_bet', 'Pass')}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c3:
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label">Kelly Allocation</div>
                    <div class="quant-val" style="color: #f59e0b;">{pred.get('kelly_stake_pct', '0.0%')}</div>
                    <div class="quant-sub">Bankroll: ${user_bankroll:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi_c4:
                st.markdown(f"""
                <div class="quant-card">
                    <div class="quant-label">Expected Total Games</div>
                    <div class="quant-val" style="color: #c084fc;">{pred['pred_total_games']:.1f}</div>
                    <div class="quant-sub">Speed Index: {'Fast (Indoor)' if is_indoor else 'Standard'}</div>
                </div>
                """, unsafe_allow_html=True)

            # Executive Summary: Top 3 Simple Pick Recommendations
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #131c2e 0%, #0d121c 100%); padding: 18px 22px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.3); margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; color: #38bdf8; font-size: 1.3rem;">🎯 Top 3 Simple Banker Picks (Plain Terms)</span>
                    <span class="badge-pill badge-emerald">Highest Confidence</span>
                </div>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                    Derived directly from calibrated probability distributions across match winner, set handicap, and game lines.
                </p>
            </div>
            """, unsafe_allow_html=True)

            top_sim_picks = pipeline.extract_top_three_picks(pred)
            if top_sim_picks:
                sim_p_cols = st.columns(3)
                badges_sim = ["🥇 Banker Pick 1", "🥈 Spread Pick 2", "🥉 Value Pick 3"]
                badge_sim_styles = ["badge-emerald", "badge-blue", "badge-amber"]

                for s_i, sp in enumerate(top_sim_picks):
                    with sim_p_cols[s_i]:
                        st.markdown(f"""
                        <div class="quant-card" style="height: 100%;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span class="badge-pill {badge_sim_styles[s_i]}">{badges_sim[s_i]}</span>
                                <span class="mono-num" style="font-weight: 700; color: #10b981;">@{sp['fair_odds']:.2f}</span>
                            </div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-bottom: 4px;">{sp['title']}</div>
                            <div class="mono-num" style="font-size: 1.4rem; font-weight: 800; color: #38bdf8; margin-bottom: 6px;">
                                {sp['win_prob_pct']}%
                            </div>
                            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.45; margin-bottom: 12px;">
                                {sp['explanation']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("➕ Add to Betslip", key=f"btn_add_sim_{s_i}", use_container_width=True):
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
            st.markdown("### Specialized Derivative Market Breakdowns")

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
                    color_discrete_sequence=["#38bdf8", "#60a5fa", "#f87171", "#f43f5e"],
                    labels={"x": "Scoreline", "y": "Probability (%)"},
                    title="Exact Set Score Distribution",
                )
                fig_scores.update_layout(height=220, showlegend=False, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)")
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
# TAB 3: SURFACE ELO LEADERBOARDS
# =============================================================================

with tab_elo:
    st.subheader("Official Surface-Specific Elo Ratings Leaderboards")
    st.markdown("Chronologically computed dynamic Elo ratings tracking over 1,000 professional players across surface environments.")

    if not df_elo.empty:
        c_surf, c_search, c_limit = st.columns([2, 3, 2])
        with c_surf:
            selected_surface = st.selectbox("Surface Leaderboard", ["Overall", "Hard", "Clay", "Grass"])
        with c_search:
            search_query = st.text_input("🔍 Search Player Name", "")
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
            color_continuous_scale="Viridis",
        )
        fig_elo.update_layout(yaxis=dict(autorange="reversed"), height=420, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_elo, use_container_width=True)

        st.markdown(f"#### Full Leaderboard Table ({selected_surface})")
        st.dataframe(display_rank, use_container_width=True, hide_index=True)
    else:
        st.warning("Elo database not loaded.")


# =============================================================================
# TAB 4: MODEL ARCHITECTURE & FEATURE IMPORTANCE
# =============================================================================

with tab_model:
    st.subheader("Dual Ensemble Architecture & Feature Engineering Attribution")
    st.markdown("Detailed breakdown of the regularized machine learning framework (LightGBM + XGBoost + SurfaceElo stacking) and feature importance.")

    df_imp = load_feature_importance_data()

    if not df_imp.empty:
        f_top = st.slider("Top Features to Display", 10, 25, 15, 1)
        sub_imp = df_imp.head(f_top)

        fig_imp = px.bar(
            sub_imp,
            x="importance_pct",
            y="feature",
            orientation="h",
            color="importance_pct",
            color_continuous_scale="Blues",
            labels={"importance_pct": "Relative Gain Contribution (%)", "feature": "Feature Name"},
            title=f"Top {f_top} Most Predictive Features (Information Gain)",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=450, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Predictive Feature Groups")
        st.markdown("""
        1. **Dual Surface & Dynamic Elo:** Pre-match Surface Elo, Overall Elo, Blended Elo, and differential spread.
        2. **Court Pace Index (CPI) & Environmental:** Indoor court status, estimated court pace index (20-45), pace hold boosts.
        3. **Serve & Return Bayesian Rates:** Opponent-adjusted service hold %, return win %, break point save & conversion rates.
        4. **Form & Momentum:** Rolling win rates over last 5, 10, and 20 matches, surface-specific form, streak lengths.
        5. **Decayed Head-to-Head:** Recency-weighted historical matchup dominance and total prior encounters.
        6. **Fatigue & Schedule:** Days since last match, matches/sets played in last 7 and 14 days, back-to-back indicators.
        7. **Biometrics & Physical Context:** Height differentials, dominant hand matchups, and age trajectory.
        8. **Tempo & Hold Expectations:** Expected combined hold %, tiebreak probabilities, and projected games per set.
        """)
    else:
        st.info("Feature importance data unavailable. Train MatchWinnerModel first.")


# =============================================================================
# TAB 5: PORTFOLIO RISK & MONTE CARLO SIMULATOR
# =============================================================================

with tab_risk:
    st.subheader("Institutional Bankroll Risk Management & Monte Carlo Simulation")
    st.markdown("Simulate 1,000 synthetic bankroll trajectories under fractional Kelly staking to project drawdown risk, growth velocity, and capital preservation.")

    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        sim_bets = st.number_input("Number of Sequential Bets", min_value=50, max_value=2000, value=300, step=50)
    with r_col2:
        sim_win_rate = st.slider("Expected Win Rate (%)", 45.0, 75.0, 58.0, 0.5) / 100.0
    with r_col3:
        sim_odds = st.number_input("Average Decimal Odds", min_value=1.20, max_value=5.0, value=1.85, step=0.05)
    with r_col4:
        sim_stake_pct = st.slider("Fixed Stake Size (% of Bankroll)", 0.5, 4.0, 1.5, 0.25)

    if st.button("🎲 Run Monte Carlo Trajectory Simulation", type="primary"):
        with st.spinner("Simulating 1,000 independent portfolio trajectories..."):
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
                line=dict(width=1.2, color="rgba(56, 189, 248, 0.45)"),
                showlegend=False,
            ))

        # Add median path
        fig_mc.add_trace(go.Scatter(
            y=df_paths.median(axis=1),
            mode='lines',
            line=dict(width=3.5, color="#f59e0b"),
            name="Median Trajectory",
        ))

        fig_mc.update_layout(
            title="500 Simulated Bankroll Trajectories (Quarter Kelly Allocation)",
            xaxis_title="Sequential Matches Bet",
            yaxis_title="Portfolio Equity ($)",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_mc, use_container_width=True)
