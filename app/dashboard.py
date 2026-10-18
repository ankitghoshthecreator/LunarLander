"""
LunarLander RL — Full-Screen Dark Sci-Fi Dashboard
===================================================
A real-time reinforcement learning simulation showcase with:
- Main Game Viewport (75-80% width) featuring live PyGame Box2D frame rendering
- Live Telemetry HUD Overlay (X/Y position, velocity, orientation, leg contacts)
- Top Control Bar (Gravity slider, Run/Pause/Reset, live status badge)
- Right-Side Training Log (PUBG / Among Us style scrollable feed with color-coded outcomes)
- Bottom Telemetry & Performance HUD (Rewards, steps, fuel, recent success rate)

Launch with:
    .\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ── Ensure project root and src/ are importable ──────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from rocketrl.frontend.adapter import SimulationEnvironmentAdapter
from rocketrl.frontend.config import (
    DEFAULT_GRAVITY,
    MAX_GRAVITY,
    MIN_GRAVITY,
    SimulationConfig,
)
from rocketrl.frontend.controller import SimulationController
from rocketrl.frontend.metrics import OUTCOME_COLORS, SessionLogManager

# ── Set Streamlit Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="LunarLander RL",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Sci-Fi Dark Glassmorphic Theme ────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global Reset & Dark Sci-Fi Palette */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 15%, #0f172a 0%, #070b14 100%) !important;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, monospace;
        overflow-x: hidden;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container {
        padding: 0.6rem 1.4rem 1.2rem 1.4rem !important;
        max-width: 100% !important;
    }
    #MainMenu, footer { visibility: hidden !important; }

    /* Top Control Bar */
    .top-bar-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 195, 255, 0.22);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.55), inset 0 0 15px rgba(0, 195, 255, 0.04);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 2.5px;
        color: #00d2ff;
        text-shadow: 0 0 12px rgba(0, 210, 255, 0.6);
        text-transform: uppercase;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .brand-subtitle {
        font-size: 0.68rem;
        color: #64748b;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: -2px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .status-ready   { background: rgba(0, 180, 216, 0.15); border: 1px solid #00b4d8; color: #00b4d8; }
    .status-running { background: rgba(0, 255, 136, 0.15); border: 1px solid #00ff88; color: #00ff88; box-shadow: 0 0 10px rgba(0,255,136,0.3); }
    .status-paused  { background: rgba(255, 170, 0, 0.15); border: 1px solid #ffaa00; color: #ffaa00; }
    .status-landed  { background: rgba(0, 255, 136, 0.25); border: 1px solid #00ff88; color: #00ff88; box-shadow: 0 0 12px rgba(0,255,136,0.4); }
    .status-crashed { background: rgba(255, 68, 85, 0.25); border: 1px solid #ff4455; color: #ff4455; box-shadow: 0 0 12px rgba(255,68,85,0.4); }
    .status-timeout { background: rgba(255, 170, 0, 0.25); border: 1px solid #ffaa00; color: #ffaa00; }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-glow 1.5s infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
    }

    /* Simulation Viewport Container */
    .viewport-box {
        position: relative;
        background: #02040a;
        border: 1px solid rgba(0, 195, 255, 0.3);
        border-radius: 10px;
        box-shadow: 0 0 25px rgba(0, 195, 255, 0.12), inset 0 0 15px rgba(0, 0, 0, 0.9);
        overflow: hidden;
    }

    /* Telemetry HUD Panel (Left Glass Card) */
    .telem-overlay {
        background: rgba(10, 16, 28, 0.82);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 195, 255, 0.22);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.45);
    }
    .telem-header {
        font-size: 0.68rem;
        color: #00d2ff;
        font-family: 'Consolas', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid rgba(0, 195, 255, 0.2);
        padding-bottom: 0.2rem;
        display: flex;
        justify-content: space-between;
    }
    .telem-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 6px 12px;
        font-size: 0.72rem;
        font-family: 'Consolas', monospace;
    }
    .telem-item {
        display: flex;
        flex-direction: column;
    }
    .telem-k { color: #64748b; font-size: 0.64rem; text-transform: uppercase; }
    .telem-v { color: #e2e8f0; font-weight: 700; }
    .telem-good { color: #00ff88; font-weight: 700; }
    .telem-warn { color: #ffaa00; font-weight: 700; }
    .telem-alert { color: #ff4455; font-weight: 700; }

    /* Bottom HUD bar */
    .bottom-hud {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 195, 255, 0.2);
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        margin-top: 0.5rem;
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        text-align: center;
        font-family: 'Consolas', monospace;
    }
    .hud-cell {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hud-lbl { font-size: 0.62rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .hud-val { font-size: 1.05rem; font-weight: 800; color: #f1f5f9; margin-top: 2px; }

    /* Right-Side Training Log (PUBG / Among Us Style) */
    .log-panel-card {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 195, 255, 0.22);
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.55);
        padding: 0.75rem 1rem;
        display: flex;
        flex-direction: column;
    }
    .log-panel-header {
        font-family: 'Consolas', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #00d2ff;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(0, 195, 255, 0.2);
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .feed-container {
        height: 480px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding-right: 4px;
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 195, 255, 0.3) transparent;
    }
    .feed-container::-webkit-scrollbar { width: 4px; }
    .feed-container::-webkit-scrollbar-thumb {
        background: rgba(0, 195, 255, 0.35);
        border-radius: 3px;
    }

    /* Feed item (PUBG/Among Us Kill-feed style) */
    .feed-item {
        background: rgba(10, 16, 28, 0.85);
        border-left: 3px solid;
        border-radius: 6px;
        padding: 0.45rem 0.65rem;
        font-family: 'Consolas', monospace;
        font-size: 0.74rem;
        display: flex;
        flex-direction: column;
        gap: 2px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        transition: transform 0.15s ease;
    }
    .feed-item:hover {
        transform: translateX(2px);
    }
    .feed-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .feed-ep-badge {
        font-weight: 700;
        color: #94a3b8;
    }
    .feed-outcome-tag {
        font-weight: 800;
        font-size: 0.7rem;
        letter-spacing: 0.8px;
    }
    .feed-details-row {
        display: flex;
        justify-content: space-between;
        color: #cbd5e1;
        font-size: 0.68rem;
    }
    .feed-rew { font-weight: 700; }
    .feed-time { color: #64748b; font-size: 0.62rem; }

    /* Streamlit Button Tweaks */
    div.stButton > button {
        border-radius: 6px !important;
        font-family: 'Consolas', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Model Checkpoint Loader (Cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_trained_model() -> Any:
    """Load default trained PPO model if available, else None (fallback)."""
    model_path = ROOT_DIR / "models" / "ppo_shaped_s0" / "best_model.zip"
    if not model_path.exists():
        # Fallback to default model or any best_model.zip
        candidates = list((ROOT_DIR / "models").glob("*/best_model.zip"))
        if candidates:
            model_path = candidates[0]
        else:
            return None

    try:
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), device="cpu")
    except Exception as exc:
        print(f"[Warning] Could not load model {model_path}: {exc}")
        return None


# ── Session State Management ─────────────────────────────────────────────────
def init_session() -> None:
    """Initialize stateful objects safely inside Streamlit session."""
    if "controller" not in st.session_state:
        config = SimulationConfig(gravity=DEFAULT_GRAVITY)
        model = get_trained_model()
        adapter = SimulationEnvironmentAdapter(
            config=config, reward_mode="shaped", model=model
        )
        log_manager = SessionLogManager(max_history=150)
        controller = SimulationController(adapter=adapter, log_manager=log_manager)

        st.session_state.controller = controller
        st.session_state.config = config
        st.session_state.slider_gravity = DEFAULT_GRAVITY
        st.session_state.model_loaded = model is not None


init_session()
ctrl: SimulationController = st.session_state.controller

# ── TOP CONTROL BAR ──────────────────────────────────────────────────────────
# Compact top control bar with title, gravity slider, status badge, and Run/Pause/Reset
with st.container():
    top_col_brand, top_col_grav, top_col_status, top_col_actions = st.columns(
        [2.3, 3.4, 1.8, 2.5], gap="small"
    )

    with top_col_brand:
        st.markdown(
            """
            <div>
                <div class="brand-title">🚀 LUNARLANDER RL</div>
                <div class="brand-subtitle">Box2D · Gymnasium-v3 · PPO Agent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col_grav:
        # Gravity slider (0.5 to 11.9 m/s², default 10.0, step 0.1)
        current_g = ctrl.selected_gravity
        selected_g = st.slider(
            label=f"Gravity: {current_g:.1f} m/s²",
            min_value=float(MIN_GRAVITY),
            max_value=float(MAX_GRAVITY),
            value=float(current_g),
            step=0.1,
            key="gravity_slider",
            help="Adjust downward gravitational acceleration in Box2D simulator. Applied live to environment.",
        )
        if abs(selected_g - current_g) >= 0.05:
            ctrl.set_gravity(selected_g)
            st.session_state.slider_gravity = selected_g

    with top_col_status:
        st.write("")  # vertical alignment
        status = ctrl.status
        if status == "RUNNING":
            badge_cls = "status-running"
            dot_color = "#00ff88"
        elif status == "PAUSED":
            badge_cls = "status-paused"
            dot_color = "#ffaa00"
        elif status == "LANDED":
            badge_cls = "status-landed"
            dot_color = "#00ff88"
        elif status == "CRASHED":
            badge_cls = "status-crashed"
            dot_color = "#ff4455"
        elif status == "TIMEOUT":
            badge_cls = "status-timeout"
            dot_color = "#ffaa00"
        else:
            badge_cls = "status-ready"
            dot_color = "#00b4d8"

        st.markdown(
            f"""
            <div style="margin-top: 10px;">
                <span class="status-badge {badge_cls}">
                    <span class="pulse-dot" style="background:{dot_color};"></span>
                    {status}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col_actions:
        st.write("")  # vertical alignment
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("▶ Run", use_container_width=True, key="btn_run"):
                ctrl.run()
        with btn_c2:
            if st.button("⏸ Pause", use_container_width=True, key="btn_pause"):
                ctrl.pause()
        with btn_c3:
            if st.button("🔄 Reset", use_container_width=True, key="btn_reset"):
                ctrl.reset(apply_new_gravity=True)

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)


# ── MAIN BODY: 78% VIEWPORT (LEFT) & 22% TRAINING LOG (RIGHT) ────────────────
col_viewport, col_feed = st.columns([78, 22], gap="medium")


# ── FRAGMENT: Live Simulation & HUD Update ───────────────────────────────────
# We use @st.fragment so frames update fluidly at ~25 FPS without rerunning
# the top slider, preventing UI input latency or flickering.
@st.fragment(run_every="40ms")
def render_live_simulation() -> None:
    # Advance simulation if currently in RUNNING state
    if ctrl.status == "RUNNING":
        ctrl.step(num_steps=1)

    frame = ctrl.last_frame
    telemetry = ctrl.adapter.get_telemetry()
    summary = ctrl.adapter.get_episode_summary()

    # Left Viewport & Telemetry
    with col_viewport:
        # 1. Telemetry HUD Overlay
        angle_deg = telemetry["angle"] * (180.0 / np.pi)
        ang_cls = "telem-alert" if telemetry["angle_warning"] else "telem-good"
        speed_cls = "telem-alert" if telemetry["speed_warning"] else "telem-good"
        leg_l_str = "CONTACT" if telemetry["left_leg"] else "CLEAR"
        leg_r_str = "CONTACT" if telemetry["right_leg"] else "CLEAR"
        leg_l_cls = "telem-good" if telemetry["left_leg"] else "telem-k"
        leg_r_cls = "telem-good" if telemetry["right_leg"] else "telem-k"

        st.markdown(
            f"""
            <div class="telem-overlay">
                <div class="telem-header">
                    <span>📡 Live Telemetry</span>
                    <span>Gravity: {ctrl.selected_gravity:.1f} m/s²</span>
                </div>
                <div class="telem-grid">
                    <div class="telem-item">
                        <span class="telem-k">X Position</span>
                        <span class="telem-v">{telemetry['x_pos']:+.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Y Altitude</span>
                        <span class="telem-v">{telemetry['y_pos']:.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Horizontal Vx</span>
                        <span class="telem-v">{telemetry['vx']:+.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Vertical Vy</span>
                        <span class="telem-v {speed_cls}">{telemetry['vy']:+.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Speed √(vx²+vy²)</span>
                        <span class="telem-v {speed_cls}">{telemetry['speed']:.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Tilt Angle</span>
                        <span class="telem-v {ang_cls}">{angle_deg:+.1f}° ({telemetry['angle']:+.3f} rad)</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Angular Vel</span>
                        <span class="telem-v">{telemetry['angular_velocity']:+.3f}</span>
                    </div>
                    <div class="telem-item">
                        <span class="telem-k">Landing Legs</span>
                        <span class="telem-v">
                            L:<span class="{leg_l_cls}">{leg_l_str}</span> | R:<span class="{leg_r_cls}">{leg_r_str}</span>
                        </span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Main Game Screen
        if frame is not None:
            st.image(frame, use_container_width=True)
        else:
            st.info("Simulation initialized. Click 'Run' to begin.")

        # 3. Bottom HUD Bar
        raw_rew = summary["raw_reward"]
        recent_sr = ctrl.log_manager.recent_success_rate(window=20)
        st.markdown(
            f"""
            <div class="bottom-hud">
                <div class="hud-cell">
                    <span class="hud-lbl">Current Episode</span>
                    <span class="hud-val">#{ctrl.current_episode}</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Total Reward (Raw)</span>
                    <span class="hud-val" style="color: {'#00ff88' if raw_rew >= 0 else '#ff5555'};">
                        {raw_rew:+.1f}
                    </span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Steps</span>
                    <span class="hud-val">{summary['steps']:,}</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Duration</span>
                    <span class="hud-val">{summary['duration_seconds']:.1f}s</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Success Rate (20ep)</span>
                    <span class="hud-val" style="color: #00d2ff;">{recent_sr:.0f}%</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Fuel Consumed</span>
                    <span class="hud-val" style="color: #cbd5e1;">{summary['fuel_used']:.0f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right Training Log Panel (PUBG / Among Us Style)
    with col_feed:
        records = ctrl.log_manager.recent_records
        st.markdown(
            f"""
            <div class="log-panel-card">
                <div class="log-panel-header">
                    <span>📋 Training Log</span>
                    <span style="font-size: 0.65rem; color: #94a3b8;">Ep #{ctrl.current_episode}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        if not records:
            st.markdown(
                """
                <div class="feed-container">
                    <div style="color: #64748b; font-size: 0.75rem; text-align: center; padding: 2rem 0; font-family: monospace;">
                        Simulation standing by.<br>
                        Click '▶ Run' to start rollouts.
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            feed_items_html = []
            for r in records:
                style = OUTCOME_COLORS.get(
                    r.outcome, OUTCOME_COLORS["OTHER TERMINATION"]
                )
                raw_str = f"{r.raw_reward:+.1f}"
                feed_items_html.append(
                    f"""
                    <div class="feed-item" style="border-left-color: {style['border']};">
                        <div class="feed-top-row">
                            <span class="feed-ep-badge">#{r.episode_id}</span>
                            <span class="feed-outcome-tag" style="color: {style['color']};">
                                {style['icon']} {r.outcome}
                            </span>
                        </div>
                        <div class="feed-details-row">
                            <span class="feed-rew">Reward: {raw_str}</span>
                            <span>{r.steps:,} steps</span>
                            <span class="feed-time">{r.timestamp}</span>
                        </div>
                    </div>
                    """
                )

            joined_items = "".join(feed_items_html)
            st.markdown(
                f"""
                <div class="feed-container">
                    {joined_items}
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Render the interactive fragment
render_live_simulation()
