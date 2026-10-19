"""
LunarLander RL — Full-Screen Dark Sci-Fi Dashboard
===================================================
A real-time reinforcement learning simulation showcase with:
- Main Game Viewport (76% width) featuring live PyGame Box2D frame rendering
- Live Telemetry HUD Strip (X/Y position, velocities, speed, tilt angle, leg contacts, gravity)
- Top Control Bar (Gravity slider [0.5 - 11.9 m/s²], Run/Pause/Reset, live status badge)
- Right-Side Training Log (PUBG / Among Us style scrollable feed with color-coded outcomes)
- Bottom Performance HUD (Episode, Total Reward, Steps, Duration, 20-Ep Success Rate, Fuel)

Launch with:
    .\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
"""

from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path
from typing import Any

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
        background: radial-gradient(circle at 50% 10%, #0d1527 0%, #050811 100%) !important;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, monospace;
        overflow-x: hidden;
    }
    [data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
    .block-container {
        padding-top: 0.4rem !important;
        padding-bottom: 0.4rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    #MainMenu, footer { visibility: hidden !important; }

    /* Top Control Bar */
    .top-bar-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 195, 255, 0.22);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55), inset 0 0 15px rgba(0, 195, 255, 0.04);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .brand-title {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #00d2ff;
        text-shadow: 0 0 12px rgba(0, 210, 255, 0.6);
        text-transform: uppercase;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.65rem;
        color: #64748b;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 1px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.7rem;
        border-radius: 16px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'Consolas', monospace;
        letter-spacing: 1.2px;
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
        50% { opacity: 0.3; transform: scale(0.85); }
    }

    /* Telemetry HUD Strip (Above Canvas) */
    .telem-bar {
        background: rgba(10, 16, 28, 0.88);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 195, 255, 0.25);
        border-radius: 6px;
        padding: 0.35rem 0.65rem;
        margin-bottom: 0.4rem;
        display: flex;
        flex-wrap: nowrap;
        justify-content: space-between;
        align-items: center;
        gap: 6px;
        font-family: 'Consolas', monospace;
        font-size: 0.7rem;
        overflow-x: auto;
    }
    .telem-col {
        display: flex;
        align-items: center;
        gap: 4px;
        white-space: nowrap;
    }
    .telem-k { color: #64748b; font-size: 0.62rem; text-transform: uppercase; font-weight: 600; }
    .telem-v { color: #f1f5f9; font-weight: 700; }
    .telem-good { color: #00ff88; font-weight: 700; }
    .telem-warn { color: #ffaa00; font-weight: 700; }
    .telem-alert { color: #ff4455; font-weight: 700; }
    .telem-sep { color: rgba(0, 195, 255, 0.25); margin: 0 2px; }

    /* Simulation Viewport & Frame Styling */
    [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: #000000 !important;
        border: 1px solid rgba(0, 195, 255, 0.32) !important;
        border-radius: 8px !important;
        padding: 0 !important;
        margin: 0 auto !important;
        overflow: hidden !important;
        box-shadow: 0 0 24px rgba(0, 195, 255, 0.12), inset 0 0 25px rgba(0, 0, 0, 0.95) !important;
    }
    [data-testid="stImage"] img {
        max-height: 53vh !important;
        width: auto !important;
        max-width: 100% !important;
        aspect-ratio: 600 / 400 !important;
        object-fit: contain !important;
        display: block !important;
        margin: 0 auto !important;
    }

    /* Bottom HUD bar */
    .bottom-hud {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 195, 255, 0.2);
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin-top: 0.4rem;
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 6px;
        text-align: center;
        font-family: 'Consolas', monospace;
    }
    .hud-cell {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hud-lbl { font-size: 0.6rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }
    .hud-val { font-size: 0.98rem; font-weight: 800; color: #f1f5f9; margin-top: 1px; }

    /* Right-Side Training Log Panel (PUBG / Among Us Style) */
    .log-panel-card {
        background: rgba(15, 23, 42, 0.78);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 195, 255, 0.22);
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.55);
        padding: 0.6rem 0.8rem;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .log-panel-header {
        font-family: 'Consolas', monospace;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #00d2ff;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(0, 195, 255, 0.2);
        padding-bottom: 0.35rem;
        margin-bottom: 0.45rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .feed-container {
        height: calc(53vh + 88px);
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding-right: 4px;
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 195, 255, 0.35) transparent;
    }
    .feed-container::-webkit-scrollbar { width: 4px; }
    .feed-container::-webkit-scrollbar-thumb {
        background: rgba(0, 195, 255, 0.35);
        border-radius: 3px;
    }

    /* PUBG / Among Us Kill-feed style Item */
    .feed-item {
        background: rgba(10, 16, 28, 0.88);
        border-left: 3px solid #00b4d8;
        border-radius: 5px;
        padding: 0.4rem 0.6rem;
        font-family: 'Consolas', monospace;
        display: flex;
        flex-direction: column;
        gap: 2px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        transition: transform 0.12s ease;
    }
    .feed-item:hover {
        transform: translateX(2px);
    }
    .feed-title-line {
        font-size: 0.73rem;
        font-weight: 700;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .feed-ep-label {
        color: #94a3b8;
    }
    .feed-tag {
        font-weight: 800;
        font-size: 0.68rem;
        letter-spacing: 0.6px;
    }
    .feed-details-line {
        font-size: 0.67rem;
        color: #cbd5e1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 2px;
        margin-top: 1px;
    }
    .feed-details-line b {
        font-weight: 700;
    }
    .feed-empty {
        color: #64748b;
        font-size: 0.72rem;
        text-align: center;
        padding: 3rem 1rem;
        font-family: 'Consolas', monospace;
        line-height: 1.5;
    }

    /* Streamlit Button Styling */
    div.stButton > button {
        border-radius: 6px !important;
        font-family: 'Consolas', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px !important;
        padding: 0.35rem 0.6rem !important;
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
        [2.2, 3.5, 1.8, 2.5], gap="small"
    )

    with top_col_brand:
        st.markdown(
            """
            <div>
                <div class="brand-title">🚀 LUNARLANDER RL</div>
                <div class="brand-subtitle">Box2D · Autonomous PPO Agent</div>
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
            help="Adjust physical downward gravity in Box2D simulator. Clamped to safe Box2D physics limits [0.5, 11.9].",
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
                    {html.escape(status)}
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

st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)


# ── LOG FEED HTML GENERATOR (Strictly unindented for CommonMark safety) ───────
def generate_log_feed_html(records: list[EpisodeRecord], current_ep: int) -> str:
    """Generate self-contained, valid HTML string for the training log panel.

    All lines have zero indentation to prevent CommonMark from treating HTML
    as indented code blocks (<pre><code>).
    """
    lines: list[str] = [
        '<div class="log-panel-card">',
        '<div class="log-panel-header">',
        '<span>📋 TRAINING LOG</span>',
        f'<span style="font-size:0.65rem;color:#00d2ff;">EP #{current_ep}</span>',
        '</div>',
        '<div class="feed-container">',
    ]

    if not records:
        lines.append('<div class="feed-empty">')
        lines.append('Simulation standing by.<br>Click <b>▶ Run</b> to start rollouts.')
        lines.append('</div>')
    else:
        for r in records:
            style = OUTCOME_COLORS.get(r.outcome, OUTCOME_COLORS["OTHER TERMINATION"])
            rew_str = f"{r.raw_reward:+.1f}"
            rew_color = "#00ff88" if r.raw_reward >= 0 else "#ff5555"
            esc_outcome = html.escape(r.outcome)
            esc_icon = html.escape(style["icon"])
            esc_time = html.escape(r.timestamp)

            lines.append(
                f'<div class="feed-item" style="border-left-color: {style["border"]};">'
            )
            # Top line: Episode 73: LANDED (SUCCESS)
            lines.append('<div class="feed-title-line">')
            lines.append(f'<span class="feed-ep-label">Episode {r.episode_id}:</span>')
            lines.append(
                f'<span class="feed-tag" style="color: {style["color"]};">{esc_icon} {esc_outcome}</span>'
            )
            lines.append('</div>')
            # Bottom line: Total Reward: +247.9 | Steps: 321
            lines.append('<div class="feed-details-line">')
            lines.append(
                f'<span>Total Reward: <b style="color: {rew_color};">{rew_str}</b></span>'
            )
            lines.append(f'<span>Steps: {r.steps:,}</span>')
            lines.append(f'<span style="color:#64748b;">{esc_time}</span>')
            lines.append('</div>')
            lines.append('</div>')

    lines.append('</div>')  # close .feed-container
    lines.append('</div>')  # close .log-panel-card
    return "".join(lines)


# ── MAIN BODY: 76% VIEWPORT (LEFT) & 24% TRAINING LOG (RIGHT) ────────────────
col_viewport, col_feed = st.columns([76, 24], gap="medium")


# ── FRAGMENT: Live Simulation & Telemetry Update ─────────────────────────────
# We use @st.fragment so frames update fluidly at ~25 FPS without rerunning
# the whole script or slider, preventing UI input latency or flickering.
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
        # 1. Compact Cockpit Telemetry Bar
        angle_deg = telemetry["angle"] * (180.0 / np.pi)
        ang_cls = "telem-alert" if telemetry["angle_warning"] else "telem-good"
        speed_cls = "telem-alert" if telemetry["speed_warning"] else "telem-good"
        vy_cls = "telem-alert" if telemetry["vy"] < -0.8 else "telem-good"
        leg_l_str = "DOWN" if telemetry["left_leg"] else "OFF"
        leg_r_str = "DOWN" if telemetry["right_leg"] else "OFF"
        leg_l_cls = "telem-good" if telemetry["left_leg"] else "telem-k"
        leg_r_cls = "telem-good" if telemetry["right_leg"] else "telem-k"

        st.markdown(
            f"""
            <div class="telem-bar">
                <div class="telem-col"><span class="telem-k">X:</span><span class="telem-v">{telemetry['x_pos']:+.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">Y:</span><span class="telem-v">{telemetry['y_pos']:.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">Vx:</span><span class="telem-v">{telemetry['vx']:+.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">Vy:</span><span class="telem-v {vy_cls}">{telemetry['vy']:+.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">SPD:</span><span class="telem-v {speed_cls}">{telemetry['speed']:.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">TILT:</span><span class="telem-v {ang_cls}">{angle_deg:+.1f}°</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">ANG VEL:</span><span class="telem-v">{telemetry['angular_velocity']:+.2f}</span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">LEGS:</span><span class="telem-v">L:<span class="{leg_l_cls}">{leg_l_str}</span> R:<span class="{leg_r_cls}">{leg_r_str}</span></span></div>
                <span class="telem-sep">|</span>
                <div class="telem-col"><span class="telem-k">G:</span><span class="telem-v" style="color:#00d2ff;">{ctrl.selected_gravity:.1f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Main Game Viewport (Real PyGame / Box2D frame)
        if frame is not None:
            st.image(frame, use_container_width=True)
        else:
            st.info("Simulation standing by. Click '▶ Run' to start.")

        # 3. Bottom Performance HUD
        raw_rew = summary["raw_reward"]
        recent_sr = ctrl.log_manager.recent_success_rate(window=20)
        st.markdown(
            f"""
            <div class="bottom-hud">
                <div class="hud-cell">
                    <span class="hud-lbl">Episode</span>
                    <span class="hud-val">#{ctrl.current_episode}</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Reward (Raw)</span>
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
                    <span class="hud-lbl">Success (20ep)</span>
                    <span class="hud-val" style="color: #00d2ff;">{recent_sr:.0f}%</span>
                </div>
                <div class="hud-cell">
                    <span class="hud-lbl">Fuel Used</span>
                    <span class="hud-val" style="color: #cbd5e1;">{summary['fuel_used']:.0f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Right-Side Training Log Panel (PUBG / Among Us Style)
    with col_feed:
        records = ctrl.log_manager.recent_records
        feed_html = generate_log_feed_html(records, ctrl.current_episode)
        st.markdown(feed_html, unsafe_allow_html=True)


# Render the interactive fragment
render_live_simulation()
