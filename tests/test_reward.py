"""Tests for ShapedLander reward correctness and telemetry accuracy.

Covers:
- Sparse / default / shaped / overshaped reward modes
- Potential-based shaping (PBRS) mathematical properties
- Correct separation of touchdown_* (true contact) vs episode_final_* (any terminal/truncated)
- Truncation handling: reward and info correctness when an episode times out
- Fuel-use proxy: engine-action counts, not physics fuel, are the metric
- Shaped-reward component sum consistency
"""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rocketrl.envs.shaped_lander import ShapedLander


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

class FakeLander(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, next_obs=None, reward=0.0, terminated=False, truncated=False):
        self.next_obs = np.zeros(8, dtype=np.float32) if next_obs is None else np.asarray(next_obs, dtype=np.float32)
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        return self.next_obs.copy(), self.reward, self.terminated, self.truncated, {}


# ---------------------------------------------------------------------------
# Pre-existing tests (preserved exactly)
# ---------------------------------------------------------------------------

def test_sparse_is_zero_on_nonterminal_steps():
    env = ShapedLander(FakeLander(reward=0.25), mode="sparse")
    env.reset()
    assert env.step(0)[1] == 0.0


def test_potential_shaping_is_zero_for_unchanged_state_when_gamma_one():
    obs = np.array([0.1, 0.2, 0.1, -0.1, 0.05, 0, 0, 0], dtype=np.float32)
    env = ShapedLander(FakeLander(next_obs=obs), mode="shaped", weights={"gamma": 1.0, "c_time": 0.0})
    env.reset()
    # Reset state differs from obs here; set a fake env whose reset matches step state.
    env._prev_phi = env._phi(obs)
    reward = env.step(0)[1]
    assert reward == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("base,landed,crashed", [(100.0, True, False), (-100.0, False, True), (0.0, False, False)])
def test_terminal_flags_follow_terminal_base_reward(base, landed, crashed):
    env = ShapedLander(FakeLander(reward=base, terminated=True), mode="default")
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["landed"] is landed
    assert info["crashed"] is crashed


def test_wrapper_preserves_observation_values_and_default_reward():
    observation = np.array([0.1, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    original = observation.copy()
    env = ShapedLander(FakeLander(next_obs=observation, reward=3.5), mode="default")
    env.reset()
    returned, reward, _, _, _ = env.step(0)
    np.testing.assert_array_equal(returned, original)
    assert reward == 3.5


def test_reward_components_sum_to_shaped_reward():
    env = ShapedLander(FakeLander(), mode="shaped")
    env.reset()
    _, reward, _, _, info = env.step(2)
    components = info["reward_components"]
    assert reward == pytest.approx(
        components["shaping"] + components["time"] + components["fuel"] + components["terminal"] + components["center"]
    )


# ---------------------------------------------------------------------------
# New tests: touchdown_* vs episode_final_* telemetry separation
# ---------------------------------------------------------------------------

class TestTouchdownTelemetrySeparation:
    """Verify that touchdown_* keys are only populated on true contact events,
    while episode_final_* keys are always populated at episode end."""

    def test_non_terminal_step_has_no_episode_final_keys(self):
        """Mid-episode steps must not set episode_final_* or touchdown_* keys."""
        env = ShapedLander(FakeLander(reward=0.5, terminated=False, truncated=False), mode="default")
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info.get("episode_final_speed") is None
        assert info.get("episode_final_angle") is None
        assert info.get("episode_final_x_offset") is None
        assert info.get("touchdown_speed") is None
        assert info.get("touchdown_angle") is None
        assert info.get("touchdown_x_offset") is None

    def test_true_landing_populates_both_touchdown_and_episode_final(self):
        """A landing (terminated=True, base_reward>=100) must populate both key families."""
        obs = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=obs, reward=100.0, terminated=True), mode="default")
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["landed"] is True
        # True touchdown: both families populated
        assert info.get("touchdown_speed") is not None
        assert info.get("touchdown_angle") is not None
        assert info.get("touchdown_x_offset") is not None
        assert info.get("episode_final_speed") is not None
        assert info.get("episode_final_angle") is not None
        assert info.get("episode_final_x_offset") is not None

    def test_timeout_has_episode_final_but_no_touchdown(self):
        """A timeout (truncated=True, terminated=False) must NOT populate touchdown_* keys
        but must populate episode_final_* keys.

        This is the core fix: previously timeouts incorrectly set touchdown_* = current state,
        which conflated hover/falling final states with genuine touchdowns.
        """
        obs = np.array([0.3, 0.5, 0.1, -0.05, 0.2, 0.01, 0.0, 0.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=obs, reward=0.0, terminated=False, truncated=True), mode="default")
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info.get("landed") is False
        assert info.get("crashed") is False
        # Timeout must NOT be treated as a touchdown
        assert info.get("touchdown_speed") is None, (
            "touchdown_speed must be None for timeouts; was previously set incorrectly"
        )
        assert info.get("touchdown_angle") is None
        assert info.get("touchdown_x_offset") is None
        # But episode_final_* MUST be available for logging
        assert info.get("episode_final_speed") is not None
        assert info.get("episode_final_angle") is not None
        assert info.get("episode_final_x_offset") is not None

    def test_crash_populates_both_key_families(self):
        """A crash (terminated=True, base_reward<=-100) is a true contact event."""
        obs = np.array([0.0, 0.0, 2.0, -3.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=obs, reward=-100.0, terminated=True), mode="default")
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["crashed"] is True
        assert info.get("touchdown_speed") is not None
        assert info.get("episode_final_speed") is not None


# ---------------------------------------------------------------------------
# New tests: truncation reward handling in shaped/sparse modes
# ---------------------------------------------------------------------------

class TestTruncationRewardHandling:
    """Ensure rewards on truncated (timeout) episodes are computed correctly."""

    def test_sparse_reward_is_zero_on_truncation(self):
        """Sparse mode: truncated episode must yield 0 reward (not landed, not crashed)."""
        env = ShapedLander(FakeLander(reward=0.0, terminated=False, truncated=True), mode="sparse")
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["landed"] is False
        assert info["crashed"] is False

    def test_shaped_truncation_has_no_terminal_bonus(self):
        """Shaped mode: truncated episode must not apply landing or crash bonus."""
        env = ShapedLander(FakeLander(reward=0.0, terminated=False, truncated=True), mode="shaped")
        env.reset()
        _, _, _, _, info = env.step(0)
        components = info["reward_components"]
        assert components["terminal"] == 0.0, "No terminal bonus expected on truncation"
        assert components["center"] == 0.0, "No center bonus expected on truncation"

    def test_shaped_truncation_reward_components_still_sum_correctly(self):
        """Component sum invariant must hold even on truncated episodes."""
        env = ShapedLander(FakeLander(reward=0.0, terminated=False, truncated=True), mode="shaped")
        env.reset()
        _, reward, _, _, info = env.step(2)
        components = info["reward_components"]
        assert reward == pytest.approx(
            components["shaping"] + components["time"] + components["fuel"]
            + components["terminal"] + components["center"],
            abs=1e-5,
        )

    def test_default_mode_passes_base_reward_through_on_truncation(self):
        """Default mode must pass the base env reward unchanged, even on truncation."""
        env = ShapedLander(FakeLander(reward=7.3, terminated=False, truncated=True), mode="default")
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(7.3)


# ---------------------------------------------------------------------------
# New tests: potential-based shaping (PBRS) properties
# ---------------------------------------------------------------------------

class TestPotentialBasedShaping:
    """Mathematical properties of the PBRS term F = gamma * phi(s') - phi(s)."""

    def test_shaping_is_negative_when_state_gets_worse(self):
        """Moving away from the pad should produce negative shaping."""
        near_obs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        far_obs = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=far_obs), mode="shaped", weights={"gamma": 1.0, "c_time": 0.0})
        env.reset()
        env._prev_phi = env._phi(near_obs)
        env._prev_components = env._potential_components(near_obs)
        _, reward, _, _, info = env.step(0)
        assert info["reward_components"]["shaping"] < 0.0

    def test_shaping_is_positive_when_approaching_pad(self):
        """Moving toward the pad should produce positive shaping."""
        far_obs = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        near_obs = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=near_obs), mode="shaped", weights={"gamma": 1.0, "c_time": 0.0})
        env.reset()
        env._prev_phi = env._phi(far_obs)
        env._prev_components = env._potential_components(far_obs)
        _, reward, _, _, info = env.step(0)
        assert info["reward_components"]["shaping"] > 0.0

    def test_shaping_prev_phi_updated_after_step(self):
        """_prev_phi must be updated to phi(s') after each shaped step."""
        obs = np.array([0.5, 0.5, 0.1, -0.1, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        env = ShapedLander(FakeLander(next_obs=obs), mode="shaped")
        env.reset()
        env.step(0)
        expected_phi = env._phi(obs)
        assert env._prev_phi == pytest.approx(expected_phi)

    def test_overshaped_shaping_is_scaled_version_of_shaped(self):
        """Overshaped shaping component must equal shaped shaping * overshaped_scale."""
        obs = np.array([0.3, 0.4, 0.1, -0.05, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        scale = 5.0

        env_s = ShapedLander(FakeLander(next_obs=obs), mode="shaped", weights={"gamma": 1.0, "c_time": 0.0})
        env_s.reset()
        env_s._prev_phi = env_s._phi(obs)
        env_s._prev_components = env_s._potential_components(obs)

        env_o = ShapedLander(
            FakeLander(next_obs=obs), mode="overshaped",
            weights={"gamma": 1.0, "overshaped_scale": scale}
        )
        env_o.reset()
        env_o._prev_phi = env_o._phi(obs)
        env_o._prev_components = env_o._potential_components(obs)

        # Use a state that is different from obs so shaping != 0
        next_obs = np.array([0.2, 0.3, 0.05, -0.02, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        env_s.env.next_obs = next_obs
        env_o.env.next_obs = next_obs

        _, _, _, _, info_s = env_s.step(0)
        _, _, _, _, info_o = env_o.step(0)

        shaped_val = info_s["reward_components"]["shaping"]
        overshaped_val = info_o["reward_components"]["shaping"]
        assert overshaped_val == pytest.approx(shaped_val * scale, rel=1e-5)


# ---------------------------------------------------------------------------
# New tests: fuel-use proxy labelling
# ---------------------------------------------------------------------------

class TestFuelUseProxy:
    """fuel_used counts engine-fire actions, not physics fuel consumption.
    The docstring and comments should make this clear; these tests verify
    the counting logic is correct.
    """

    def test_no_op_action_does_not_increment_fuel(self):
        """Action 0 (no-op) must not increment the engine-action counter."""
        env = ShapedLander(FakeLander(), mode="shaped")
        env.reset()
        env.step(0)  # no-op
        _, _, _, _, info = env.step(0)
        assert info["fuel_used"] == 0.0

    def test_main_engine_action_increments_fuel_by_one(self):
        """Action 2 (main engine) must increment fuel_used by 1 per step."""
        env = ShapedLander(FakeLander(), mode="shaped")
        env.reset()
        env.step(2)
        _, _, _, _, info = env.step(2)
        assert info["fuel_used"] == pytest.approx(2.0)

    def test_side_engine_action_increments_fuel_by_one(self):
        """Actions 1 and 3 (side engines) must each increment fuel_used by 1."""
        env = ShapedLander(FakeLander(), mode="shaped")
        env.reset()
        env.step(1)
        _, _, _, _, info = env.step(3)
        assert info["fuel_used"] == pytest.approx(2.0)

    def test_fuel_resets_on_new_episode(self):
        """fuel_used must reset to 0 on env.reset()."""
        env = ShapedLander(FakeLander(), mode="shaped")
        env.reset()
        env.step(2)
        env.step(2)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["fuel_used"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# New tests: sparse reward credit assignment
# ---------------------------------------------------------------------------

class TestSparseRewardCreditAssignment:
    """Document the known limitation: sparse reward is hard for PPO with gamma=0.999
    over 1000-step episodes. These tests encode the known behaviour as regression tests.
    """

    def test_sparse_landed_reward_is_positive(self):
        env = ShapedLander(FakeLander(reward=100.0, terminated=True), mode="sparse")
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward > 0.0
        assert info["landed"] is True

    def test_sparse_crashed_reward_is_negative(self):
        env = ShapedLander(FakeLander(reward=-100.0, terminated=True), mode="sparse")
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward < 0.0
        assert info["crashed"] is True

    def test_sparse_timeout_reward_is_zero(self):
        """The timeout zero-reward creates a hover equilibrium incentive.

        With gamma=0.999 and max_episode_steps=1000:
          discounted_landing_reward = 1.0 * 0.999^1000 ≈ 0.368
          discounted_crash_reward   = -1.0
        A risk-averse policy learns: never fire main engine = no crash = 0 reward > -0.368 expected.
        This is the documented failure mode for sparse reward training.
        """
        env = ShapedLander(FakeLander(reward=0.0, terminated=False, truncated=True), mode="sparse")
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["landed"] is False
        assert info["crashed"] is False


class TestSparseStepPenalty:
    """Sparse step penalty is a non-negative magnitude subtracted per transition."""

    def test_zero_penalty_preserves_sparse_rewards(self):
        ordinary = ShapedLander(FakeLander(reward=12.0), mode="sparse")
        ordinary.reset()
        assert ordinary.step(0)[1] == 0.0

        landing = ShapedLander(FakeLander(reward=100.0, terminated=True), mode="sparse")
        landing.reset()
        _, landing_reward, _, _, landing_info = landing.step(0)
        assert landing_reward == 1.0
        assert landing_info["reward_components"]["terminal"] == 1.0

        crash = ShapedLander(FakeLander(reward=-100.0, terminated=True), mode="sparse")
        crash.reset()
        _, crash_reward, _, _, crash_info = crash.step(0)
        assert crash_reward == -1.0
        assert crash_info["reward_components"]["terminal"] == -1.0

        timeout = ShapedLander(
            FakeLander(reward=0.0, terminated=False, truncated=True), mode="sparse"
        )
        timeout.reset()
        assert timeout.step(0)[1] == 0.0

    def test_nonzero_penalty_applies_on_ordinary_transition(self):
        penalty = 0.0001
        env = ShapedLander(
            FakeLander(reward=12.0), mode="sparse",
            weights={"sparse_step_penalty": penalty},
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)

        assert reward == pytest.approx(-penalty)
        assert not terminated and not truncated
        assert info["reward_components"]["time"] == pytest.approx(-penalty)
        assert info["reward_components"]["terminal"] == 0.0

    @pytest.mark.parametrize(
        ("base_reward", "expected_terminal", "expected_reward"),
        [(100.0, 1.0, 0.9999), (-100.0, -1.0, -1.0001)],
    )
    def test_terminal_rewards_are_preserved_with_penalty(
        self, base_reward, expected_terminal, expected_reward
    ):
        env = ShapedLander(
            FakeLander(reward=base_reward, terminated=True), mode="sparse",
            weights={"sparse_step_penalty": 0.0001},
        )
        env.reset()
        _, reward, _, _, info = env.step(0)

        assert info["reward_components"]["terminal"] == expected_terminal
        assert reward == pytest.approx(expected_reward)

    def test_timeout_gets_step_cost_without_terminal_reward(self):
        penalty = 0.0001
        env = ShapedLander(
            FakeLander(reward=0.0, terminated=False, truncated=True), mode="sparse",
            weights={"sparse_step_penalty": penalty},
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)

        assert not terminated and truncated
        assert reward == pytest.approx(-penalty)
        assert info["landed"] is False
        assert info["crashed"] is False
        assert info["reward_components"]["terminal"] == 0.0

    @pytest.mark.parametrize("invalid", [-0.1, float("nan"), float("inf"), "invalid"])
    def test_step_penalty_requires_finite_nonnegative_magnitude(self, invalid):
        with pytest.raises(ValueError, match="finite, non-negative penalty magnitude"):
            ShapedLander(FakeLander(), mode="sparse", weights={"sparse_step_penalty": invalid})


class TestSparseTimeoutPenalty:
    """Verify configurable b_timeout terminal penalty on genuine time-limit truncations."""

    def test_b_timeout_defaults_to_zero(self):
        env = ShapedLander(FakeLander(reward=0.0, terminated=False, truncated=True), mode="sparse")
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["reward_components"]["terminal"] == 0.0

    def test_b_timeout_applies_only_on_truncation_without_termination(self):
        timeout_penalty = 1.5
        env = ShapedLander(
            FakeLander(reward=0.0, terminated=False, truncated=True),
            mode="sparse",
            weights={"b_timeout": timeout_penalty},
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)
        assert not terminated and truncated
        assert reward == pytest.approx(-timeout_penalty)
        assert info["reward_components"]["terminal"] == pytest.approx(-timeout_penalty)

    def test_b_timeout_does_not_apply_when_landed(self):
        env = ShapedLander(
            FakeLander(reward=100.0, terminated=True, truncated=False),
            mode="sparse",
            weights={"b_timeout": 2.0},
        )
        env.reset()
        _, reward, terminated, _, info = env.step(0)
        assert terminated
        assert reward == 1.0
        assert info["reward_components"]["terminal"] == 1.0

    def test_b_timeout_does_not_apply_when_crashed(self):
        env = ShapedLander(
            FakeLander(reward=-100.0, terminated=True, truncated=False),
            mode="sparse",
            weights={"b_timeout": 2.0},
        )
        env.reset()
        _, reward, terminated, _, info = env.step(0)
        assert terminated
        assert reward == -1.0
        assert info["reward_components"]["terminal"] == -1.0

    def test_b_timeout_combines_with_step_penalty_on_truncation(self):
        env = ShapedLander(
            FakeLander(reward=0.0, terminated=False, truncated=True),
            mode="sparse",
            weights={"sparse_step_penalty": 0.01, "b_timeout": 1.0},
        )
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == pytest.approx(-1.01)
        assert info["reward_components"]["terminal"] == pytest.approx(-1.0)
        assert info["reward_components"]["time"] == pytest.approx(-0.01)

    @pytest.mark.parametrize("invalid", [-0.5, float("nan"), float("inf"), "bad"])
    def test_b_timeout_requires_finite_nonnegative_magnitude(self, invalid):
        with pytest.raises(ValueError, match="finite, non-negative penalty magnitude"):
            ShapedLander(FakeLander(), mode="sparse", weights={"b_timeout": invalid})
