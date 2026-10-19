"""
LunarLander RL — Full-Screen Dark Sci-Fi Dashboard
===================================================
A real-time reinforcement learning simulation showcase with:
- Main Game Viewport (76% width) featuring live high-resolution Box2D frame rendering
- Floating Telemetry Glass Card (X/Y position, velocities, angle, leg contacts, fuel)
- Top Control Bar (Gravity slider, Mode, Model Checkpoint, Run/Pause/Reset)
- Right-Side Training Log (Glassmorphism scrollable action feed with color-coded outcomes)
- Floating Bottom Performance HUD Dock (Episode, Total Reward, Steps, Fuel Used, Episode Time, Success Rate)

Launch with:
    .\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import base64
import html
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
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
from rocketrl.frontend.metrics import OUTCOME_COLORS, EpisodeRecord, SessionLogManager

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
    /* Global Reset & Sci-Fi Dark Palette */
    html, body, [data-testid="stAppViewContainer"] {
        background: #040814 !important;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, monospace;
        overflow-x: hidden;
    }
    [data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    .block-container {
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        max-width: 100% !important;
    }
    #MainMenu, footer { visibility: hidden !important; }

    /* Top Control Bar */
    .brand-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
        line-height: 2.2;
    }

    /* Streamlit Input / Dropdown Tweaks */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label {
        font-size: 0.72rem !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        margin-bottom: -4px !important;
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        color: #f1f5f9 !important;
        font-size: 0.78rem !important;
    }

    /* Slider Styling */
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top: -6px !important;
    }

    /* Action Buttons */
    div[data-testid="stButton"] button {
        border-radius: 6px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.4rem 0.8rem !important;
        transition: all 0.2s ease !important;
    }

    /* Vibrant Green Run Button */
    button[kind="primary"] {
        background: #00c853 !important;
        border: 1px solid #00e676 !important;
        color: #ffffff !important;
        box-shadow: 0 0 14px rgba(0, 200, 83, 0.45) !important;
    }
    button[kind="primary"]:hover {
        background: #00e676 !important;
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.65) !important;
    }

    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #e2e8f0 !important;
    }
    button[kind="secondary"]:hover {
        border-color: #00d2ff !important;
        color: #00d2ff !important;
        box-shadow: 0 0 10px rgba(0, 210, 255, 0.3) !important;
    }

    /* Viewport Card & Canvas (Edge-to-Edge Full Bleed Left Column) */
    .viewport-card {
        position: relative !important;
        width: 100% !important;
        aspect-ratio: 16 / 9 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        border: 1px solid rgba(0, 195, 255, 0.28) !important;
        box-shadow: 0 0 30px rgba(0, 195, 255, 0.12), 0 8px 32px rgba(0, 0, 0, 0.85) !important;
        background: #000000 !important;
    }

    .viewport-canvas {
        width: 100% !important;
        height: 100% !important;
        display: block !important;
        object-fit: cover !important;
    }

    /* Floating Telemetry Glass Card (Top-Left of Viewport) */
    .telem-glass-card {
        position: absolute !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 10 !important;
        width: 190px !important;
        background: rgba(8, 14, 26, 0.72) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(0, 195, 255, 0.22) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55) !important;
        pointer-events: none !important;
        font-family: 'Consolas', monospace !important;
    }
    .telem-row {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        font-size: 0.68rem !important;
        padding: 1.5px 0 !important;
    }
    .telem-k {
        color: #94a3b8 !important;
    }
    .telem-v {
        font-weight: 700 !important;
        color: #00ff88 !important;
    }
    .telem-alert {
        color: #ff4455 !important;
    }

    /* Floating Bottom Performance HUD Dock (Bottom of Viewport) */
    .bottom-hud-dock {
        position: absolute !important;
        bottom: 12px !important;
        left: 14px !important;
        right: 14px !important;
        z-index: 10 !important;
        display: grid !important;
        grid-template-columns: repeat(6, 1fr) !important;
        gap: 8px !important;
        pointer-events: none !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    .hud-pill {
        background: rgba(10, 16, 28, 0.78) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(0, 195, 255, 0.22) !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5) !important;
    }
    .hud-icon {
        font-size: 1.15rem !important;
    }
    .hud-text {
        display: flex !important;
        flex-direction: column !important;
    }
    .hud-lbl {
        font-size: 0.58rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        white-space: nowrap !important;
    }
    .hud-val {
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
        line-height: 1.1 !important;
        font-family: 'Consolas', monospace !important;
    }

    /* Right-Side Training Log Panel */
    .log-panel-card {
        height: calc(100vh - 78px) !important;
        background: rgba(12, 20, 36, 0.78) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(0, 195, 255, 0.22) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.55) !important;
        padding: 0.65rem 0.85rem !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .log-panel-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
        border-bottom: 1px solid rgba(0, 195, 255, 0.2) !important;
        padding-bottom: 0.4rem !important;
        margin-bottom: 0.5rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .training-badge {
        background: rgba(0, 255, 136, 0.15) !important;
        border: 1px solid #00ff88 !important;
        color: #00ff88 !important;
        padding: 1px 7px !important;
        border-radius: 12px !important;
        font-size: 0.62rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }
    .feed-container {
        flex: 1 !important;
        overflow-y: auto !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        padding-right: 4px !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(0, 195, 255, 0.35) transparent !important;
    }
    .feed-container::-webkit-scrollbar { width: 4px; }
    .feed-container::-webkit-scrollbar-thumb {
        background: rgba(0, 195, 255, 0.35);
        border-radius: 3px;
    }

    /* Feed item (Action / Kill-feed style) */
    .feed-item {
        background: rgba(10, 16, 28, 0.85) !important;
        border-left: 3px solid #00b4d8 !important;
        border-radius: 5px !important;
        padding: 0.4rem 0.6rem !important;
        font-family: 'Consolas', monospace !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 2px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35) !important;
        transition: transform 0.12s ease !important;
    }
    .feed-item:hover {
        transform: translateX(2px) !important;
    }
    .feed-title-line {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .feed-ep-label {
        color: #94a3b8 !important;
    }
    .feed-tag {
        font-weight: 800 !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.5px !important;
    }
    .feed-details-line {
        font-size: 0.66rem !important;
        color: #cbd5e1 !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        border-top: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding-top: 2px !important;
        margin-top: 1px !important;
    }
    .feed-empty {
        color: #64748b !important;
        font-size: 0.72rem !important;
        text-align: center !important;
        padding: 4rem 1rem !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        line-height: 1.5 !important;
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
# Compact top control bar matching reference mockup:
# Title | Gravity slider | Mode | Model Checkpoint | Run | Pause | Reset
top_c_title, top_c_grav, top_c_mode, top_c_model, top_c_btns = st.columns(
    [2.0, 2.7, 2.0, 2.4, 2.3], gap="small"
)

with top_c_title:
    st.markdown(
        """
        <div class="brand-title">🚀 LunarLander RL</div>
        """,
        unsafe_allow_html=True,
    )

with top_c_grav:
    # Gravity slider: range [0.5, 11.9], step 0.1, default 10.0 (displayed as -10.0 m/s²)
    current_g = ctrl.selected_gravity
    selected_g = st.slider(
        label=f"Gravity: -{current_g:.1f} m/s²",
        min_value=float(MIN_GRAVITY),
        max_value=float(MAX_GRAVITY),
        value=float(current_g),
        step=0.1,
        key="gravity_slider",
        help="Physical Box2D downward gravity [0.5 to 11.9 m/s²]. Applied live to environment.",
    )
    if abs(selected_g - current_g) >= 0.05:
        ctrl.set_gravity(selected_g)
        st.session_state.slider_gravity = selected_g

with top_c_mode:
    st.selectbox(
        "Mode",
        ["Inference (Trained Model)"],
        index=0,
        disabled=True,
    )

with top_c_model:
    st.selectbox(
        "Model Checkpoint",
        ["ppo_shaped_s0/best_model.zip"],
        index=0,
        disabled=True,
    )

with top_c_btns:
    st.write("")  # vertical alignment
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.button("▶ Run", type="primary", use_container_width=True, key="btn_run"):
            ctrl.run()
    with btn_c2:
        if st.button(
            "⏸ Pause", type="secondary", use_container_width=True, key="btn_pause"
        ):
            ctrl.pause()
    with btn_c3:
        if st.button(
            "🔄 Reset", type="secondary", use_container_width=True, key="btn_reset"
        ):
            ctrl.reset(apply_new_gravity=True)

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)


# ── LOG FEED HTML GENERATOR (Strictly unindented for CommonMark safety) ───────
def generate_log_feed_html(records: list[EpisodeRecord], current_ep: int) -> str:
    """Generate self-contained, valid HTML string for the training log panel."""
    lines: list[str] = [
        '<div class="log-panel-card">',
        '<div class="log-panel-header">',
        '<div style="display:flex;align-items:center;gap:6px;">',
        '<span style="font-size:1rem;">📊</span>',
        '<span>Training Log</span>',
        '</div>',
        '<div style="display:flex;align-items:center;gap:6px;">',
        f'<span style="font-size:0.68rem;color:#94a3b8;">Episode {current_ep}</span>',
        '<span class="training-badge">TRAINING</span>',
        '</div>',
        '</div>',
        '<div class="feed-container">',
    ]

    if not records:
        lines.append('<div class="feed-empty">')
        lines.append(
            'Awaiting completed episode rollouts...<br>Click <b>▶ Run</b> to start.'
        )
        lines.append('</div>')
    else:
        for r in records:
            style = OUTCOME_COLORS.get(r.outcome, OUTCOME_COLORS["OTHER TERMINATION"])
            rew_str = f"{r.raw_reward:+.1f}"
            rew_color = "#00ff88" if r.raw_reward >= 0 else "#ff5555"
            esc_outcome = html.escape(r.outcome)
            esc_time = html.escape(r.timestamp)

            lines.append(
                f'<div class="feed-item" style="border-left-color: {style["border"]};">'
            )
            # Line 1: ● 12:14:23 Episode 73: LANDED (SUCCESS)
            lines.append('<div class="feed-title-line">')
            lines.append('<div>')
            lines.append(
                f'<span style="color:{style["color"]};margin-right:4px;">●</span>'
            )
            lines.append(
                f'<span style="color:#64748b;font-size:0.64rem;margin-right:4px;">{esc_time}</span>'
            )
            lines.append(f'<span class="feed-ep-label">Episode {r.episode_id}:</span>')
            lines.append('</div>')
            lines.append(
                f'<span class="feed-tag" style="color: {style["color"]};">{esc_outcome}</span>'
            )
            lines.append('</div>')
            # Line 2: Total Reward: +247.9 | Steps: 321
            lines.append('<div class="feed-details-line">')
            lines.append(
                f'<span>Total Reward: <b style="color: {rew_color};">{rew_str}</b></span>'
            )
            lines.append('<span>|</span>')
            lines.append(f'<span>Steps: {r.steps:,}</span>')
            lines.append('</div>')
            lines.append('</div>')

    lines.append('</div>')  # close .feed-container
    lines.append('</div>')  # close .log-panel-card
    return "".join(lines)


# ── MAIN BODY: 76% VIEWPORT (LEFT) & 24% TRAINING LOG (RIGHT) ────────────────
col_viewport, col_feed = st.columns([76, 24], gap="small")


# ── FRAGMENT: Live Simulation & Full Bleed Overlays ──────────────────────────
# Fluid ~25 FPS loop updating frame, telemetry overlay, HUD, and feed
@st.fragment(run_every="40ms")
def render_live_simulation() -> None:
    # Advance simulation strictly when RUNNING
    if ctrl.status == "RUNNING":
        ctrl.step(num_steps=1)

    frame = ctrl.last_frame
    telemetry = ctrl.adapter.get_telemetry()
    summary = ctrl.adapter.get_episode_summary()

    # Left Viewport & Unified Glass Overlays
    with col_viewport:
        if frame is not None:
            # Convert frame to Base64 JPEG for ultra-crisp responsive rendering
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=85)
            b64_frame = base64.b64encode(buf.getvalue()).decode("ascii")
        else:
            b64_frame = ""

        # Telemetry & HUD State
        vy_cls = "telem-alert" if telemetry["vy"] < -0.8 else "telem-v"
        ang_cls = "telem-alert" if telemetry["angle_warning"] else "telem-v"
        leg_l_str = "Yes" if telemetry["left_leg"] else "No"
        leg_r_str = "Yes" if telemetry["right_leg"] else "No"
        leg_l_cls = "telem-v" if telemetry["left_leg"] else "telem-k"
        leg_r_cls = "telem-v" if telemetry["right_leg"] else "telem-k"

        raw_rew = summary["raw_reward"]
        recent_sr = ctrl.log_manager.recent_success_rate(window=50)
        fuel_used = summary["fuel_used"]
        fuel_used_pct = min(100.0, fuel_used)

        # Single self-contained HTML component: Canvas + Floating Top-Left Telemetry + Floating Bottom HUD
        viewport_html = f"""
        <div class="viewport-card">
            <img src="data:image/jpeg;base64,{b64_frame}" class="viewport-canvas" />

            <!-- Floating Telemetry Glass Card (TOP-LEFT) -->
            <div class="telem-glass-card">
                <div class="telem-row"><span class="telem-k">X Position</span><span class="telem-v">{telemetry['x_pos']:+.2f}</span></div>
                <div class="telem-row"><span class="telem-k">Y Position</span><span class="telem-v">{telemetry['y_pos']:.2f}</span></div>
                <div class="telem-row"><span class="telem-k">X Velocity</span><span class="telem-v">{telemetry['vx']:+.2f}</span></div>
                <div class="telem-row"><span class="telem-k">Y Velocity</span><span class="{vy_cls}">{telemetry['vy']:+.2f}</span></div>
                <div class="telem-row"><span class="telem-k">Angle (rad)</span><span class="{ang_cls}">{telemetry['angle']:+.2f}</span></div>
                <div class="telem-row"><span class="telem-k">Angular Vel.</span><span class="telem-v">{telemetry['angular_velocity']:+.2f}</span></div>
                <div class="telem-row"><span class="telem-k">Left Leg Contact</span><span class="{leg_l_cls}">{leg_l_str}</span></div>
                <div class="telem-row"><span class="telem-k">Right Leg Contact</span><span class="{leg_r_cls}">{leg_r_str}</span></div>
                <div class="telem-row"><span class="telem-k">Fuel Remaining</span><span class="telem-v">{telemetry['fuel_remaining']:.1f}%</span></div>
            </div>

            <!-- Floating Bottom Performance HUD Dock (BOTTOM) -->
            <div class="bottom-hud-dock">
                <div class="hud-pill">
                    <span class="hud-icon">🏆</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Episode</span>
                        <span class="hud-val">#{ctrl.current_episode}</span>
                    </div>
                </div>
                <div class="hud-pill">
                    <span class="hud-icon">🎯</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Total Reward</span>
                        <span class="hud-val" style="color: {'#00ff88' if raw_rew >= 0 else '#ff5555'};">{raw_rew:+.1f}</span>
                    </div>
                </div>
                <div class="hud-pill">
                    <span class="hud-icon">⁝⁝</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Steps</span>
                        <span class="hud-val">{summary['steps']:,}</span>
                    </div>
                </div>
                <div class="hud-pill">
                    <span class="hud-icon">⛽</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Fuel Used</span>
                        <span class="hud-val" style="color: #00d2ff;">{fuel_used_pct:.1f}%</span>
                    </div>
                </div>
                <div class="hud-pill">
                    <span class="hud-icon">⏱</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Episode Time</span>
                        <span class="hud-val">{summary['duration_seconds']:.1f}s</span>
                    </div>
                </div>
                <div class="hud-pill">
                    <span class="hud-icon">📈</span>
                    <div class="hud-text">
                        <span class="hud-lbl">Recent Success Rate</span>
                        <span class="hud-val" style="color: #00ff88;">{recent_sr:.0f}% (last 50)</span>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(viewport_html, unsafe_allow_html=True)

    # Right-Side Training Log Panel
    with col_feed:
        records = ctrl.log_manager.recent_records
        feed_html = generate_log_feed_html(records, ctrl.current_episode)
        st.markdown(feed_html, unsafe_allow_html=True)


# Render the interactive fragment
render_live_simulation()
