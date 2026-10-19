"""Unit tests for the LunarLander Streamlit frontend components."""

from __future__ import annotations

import numpy as np
import pytest

from rocketrl.frontend.adapter import SimulationEnvironmentAdapter
from rocketrl.frontend.config import (
    DEFAULT_GRAVITY,
    MAX_GRAVITY,
    MIN_GRAVITY,
    SimulationConfig,
)
from rocketrl.frontend.controller import SimulationController
from rocketrl.frontend.metrics import (
    EpisodeRecord,
    SessionLogManager,
    classify_outcome,
)


def test_gravity_default_and_range_validation():
    """Verify default gravity value, bounds, and clamping."""
    cfg = SimulationConfig()
    assert cfg.gravity == pytest.approx(DEFAULT_GRAVITY)
    assert cfg.gym_gravity == pytest.approx(-DEFAULT_GRAVITY)

    # Test clamping
    clamped_low = cfg.validate_and_clamp_gravity(-5.0)
    assert clamped_low == pytest.approx(MIN_GRAVITY)

    clamped_high = cfg.validate_and_clamp_gravity(99.0)
    assert clamped_high == pytest.approx(MAX_GRAVITY)

    # Update gravity
    cfg.update_gravity(3.71)
    assert cfg.gravity == pytest.approx(3.71)
    assert cfg.gym_gravity == pytest.approx(-3.71)

    # Invalid string type raises ValueError
    with pytest.raises(ValueError, match="Gravity must be a valid float"):
        cfg.validate_and_clamp_gravity("not_a_number")


def test_gravity_application_to_environment():
    """Verify gravity is applied to Gymnasium environment and live Box2D world."""
    cfg = SimulationConfig(gravity=3.71)
    adapter = SimulationEnvironmentAdapter(config=cfg)
    try:
        obs, info = adapter.reset(seed=42)
        assert obs.shape == (8,)

        # Check unwrapped environment gravity
        unwrapped = adapter.env.unwrapped
        assert unwrapped.gravity == pytest.approx(-3.71)
        if unwrapped.world is not None:
            assert unwrapped.world.gravity.y == pytest.approx(-3.71)

        # Test live gravity update
        adapter.apply_gravity(9.81)
        assert unwrapped.gravity == pytest.approx(-9.81)
        if unwrapped.world is not None:
            assert unwrapped.world.gravity.y == pytest.approx(-9.81)
    finally:
        adapter.close()


def test_reset_applies_selected_gravity():
    """Verify reset creates/maintains the newly selected gravity."""
    adapter = SimulationEnvironmentAdapter(config=SimulationConfig(gravity=10.0))
    ctrl = SimulationController(adapter)
    try:
        ctrl.set_gravity(1.62)
        ctrl.reset(apply_new_gravity=True)
        assert adapter.env.unwrapped.gravity == pytest.approx(-1.62)
        assert adapter.env.unwrapped.world.gravity.y == pytest.approx(-1.62)
    finally:
        ctrl.cleanup()


def test_pause_preserves_simulation_state():
    """Verify pause does not reset the environment or discard current progress."""
    adapter = SimulationEnvironmentAdapter()
    ctrl = SimulationController(adapter)
    try:
        ctrl.run()
        ctrl.step(num_steps=5)
        step_before = adapter._step_count
        obs_before = np.copy(adapter._current_obs)

        # Pause
        ctrl.pause()
        assert ctrl.status == "PAUSED"

        # Calling step while paused must not advance simulation
        ctrl.step(num_steps=5)
        assert adapter._step_count == step_before
        np.testing.assert_array_equal(adapter._current_obs, obs_before)

        # Resume
        ctrl.run()
        assert ctrl.status == "RUNNING"
        ctrl.step(num_steps=1)
        assert adapter._step_count == step_before + 1
    finally:
        ctrl.cleanup()


def test_episode_reward_and_step_accumulation():
    """Verify reward and steps accumulate properly during rollouts."""
    adapter = SimulationEnvironmentAdapter()
    try:
        adapter.reset(seed=123)
        initial_steps = adapter._step_count
        assert initial_steps == 0
        assert adapter._raw_reward_acc == 0.0

        for _ in range(3):
            adapter.step(0)  # no-op action

        assert adapter._step_count == 3
        summary = adapter.get_episode_summary()
        assert summary["steps"] == 3
        assert isinstance(summary["raw_reward"], float)
        assert isinstance(summary["shaped_reward"], float)
    finally:
        adapter.close()


def test_outcome_classification():
    """Verify correct classification of landing, crash, and timeout."""
    # 1. Timeout takes precedence
    out = classify_outcome(terminated=False, truncated=True, info={}, observation=np.zeros(8))
    assert out == "TIMEOUT"

    # 2. Crash
    out = classify_outcome(terminated=True, truncated=False, info={"crashed": True}, observation=np.zeros(8))
    assert out == "CRASHED"

    # 3. Successful landing (both legs down, |x| <= 0.2)
    obs_success = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    out = classify_outcome(terminated=True, truncated=False, info={"landed": True}, observation=obs_success)
    assert out == "LANDED (SUCCESS)"

    # 4. Unsuccessful landing: landed, but only one leg touching
    obs_one_leg = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    out = classify_outcome(terminated=True, truncated=False, info={"landed": True}, observation=obs_one_leg)
    assert out == "UNSUCCESSFUL LANDING"

    # 5. Unsuccessful landing: landed, but outside center tolerance (|x| > 0.2)
    obs_off_pad = np.array([0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    out = classify_outcome(terminated=True, truncated=False, info={"landed": True}, observation=obs_off_pad)
    assert out == "UNSUCCESSFUL LANDING"


def test_session_log_limit_and_ordering():
    """Verify log manager enforces max history and orders newest first."""
    mgr = SessionLogManager(max_history=5)
    for i in range(1, 8):
        rec = EpisodeRecord(
            episode_id=i,
            outcome="LANDED (SUCCESS)" if i % 2 == 0 else "CRASHED",
            raw_reward=100.0 * i,
            shaped_reward=150.0 * i,
            steps=50 * i,
            duration_seconds=5.0,
            gravity=10.0,
        )
        mgr.add_record(rec)

    assert mgr.total_episodes == 7
    # Should only retain last 5
    assert len(mgr.records) == 5
    assert mgr.records[0].episode_id == 3
    assert mgr.records[-1].episode_id == 7

    # Newest first
    recent = mgr.recent_records
    assert recent[0].episode_id == 7
    assert recent[-1].episode_id == 3

    # Formatted message format verification
    assert recent[0].formatted_message.startswith("Episode 7:")


def test_telemetry_extraction():
    """Verify telemetry dict contains all expected fields and units."""
    adapter = SimulationEnvironmentAdapter()
    try:
        adapter.reset(seed=999)
        adapter.step(0)
        telem = adapter.get_telemetry()
        expected_keys = {
            "x_pos",
            "y_pos",
            "vx",
            "vy",
            "speed",
            "angle",
            "angular_velocity",
            "left_leg",
            "right_leg",
            "fuel_used",
            "angle_warning",
            "speed_warning",
        }
        assert expected_keys.issubset(set(telem.keys()))
        assert isinstance(telem["x_pos"], float)
        assert isinstance(telem["speed"], float)
    finally:
        adapter.close()


def test_render_frame_smoke():
    """Verify captured frame has expected shape (400, 600, 3), dtype uint8, and non-trivial content."""
    adapter = SimulationEnvironmentAdapter()
    try:
        adapter.reset(seed=42)
        frame = adapter.render_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (400, 600, 3)
        assert frame.dtype == np.uint8
        # Ensure frame contains rendered pixels (terrain, flags, stars/lander)
        assert frame.max() > 0
        assert frame.std() > 10.0
    finally:
        adapter.close()

