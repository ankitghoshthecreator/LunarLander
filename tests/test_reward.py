import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from rocketrl.envs.shaped_lander import ShapedLander


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
