"""Deterministic tests for evaluation outcome and telemetry metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rocketrl.evaluate import evaluate


class FakeEvaluationEnv:
    def __init__(self, episodes):
        self.episodes = episodes
        self.episode_index = -1

    def reset(self, *, seed):
        self.episode_index += 1
        self.seed = seed
        return np.zeros(8, dtype=np.float32), {}

    def step(self, _action):
        observation, reward, terminated, truncated, info = self.episodes[self.episode_index]
        return observation.copy(), reward, terminated, truncated, dict(info)


class FakeEvaluationModel:
    def __init__(self):
        self.deterministic_flags = []

    def predict(self, _observation, *, deterministic):
        self.deterministic_flags.append(deterministic)
        return 0, None


class FakeLengthEvaluationEnv:
    def __init__(self, lengths):
        self.lengths = lengths
        self.episode_index = -1
        self.step_count = 0

    def reset(self, *, seed):
        self.episode_index += 1
        self.step_count = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, _action):
        self.step_count += 1
        ended = self.step_count == self.lengths[self.episode_index]
        observation = np.zeros(8, dtype=np.float32)
        info = {"episode_final_speed": 0.0} if ended else {}
        return observation, 1.0, ended, False, info


def _episode(*, x=0.0, left=0.0, right=0.0, reward=0.0,
             terminated=False, truncated=False, info=None):
    observation = np.array([x, 0.0, 3.0, 4.0, 0.25, 0.0, left, right], dtype=np.float32)
    return observation, reward, terminated, truncated, info or {}


def test_evaluate_classifies_outcomes_and_aggregates_only_touchdown_metrics():
    episodes = [
        _episode(x=0.1, left=1, right=1, reward=100, terminated=True, info={
            "landed": True, "episode_final_speed": 5, "episode_final_angle": 0.1,
            "episode_final_x_offset": 0.1, "touchdown_speed": 5,
            "touchdown_angle": 0.1, "touchdown_x_offset": 0.1,
        }),
        _episode(reward=-100, terminated=True, info={
            "crashed": True, "episode_final_speed": 2, "episode_final_angle": 0.2,
            "episode_final_x_offset": 0.2, "touchdown_speed": 2,
            "touchdown_angle": 0.2, "touchdown_x_offset": 0.2,
        }),
        _episode(truncated=True, info={
            "episode_final_speed": 3, "episode_final_angle": 0.3,
            "episode_final_x_offset": 0.3,
            # Defensive check: timeout classification excludes these even if a
            # malformed wrapper accidentally supplies touchdown values.
            "touchdown_speed": 99, "touchdown_angle": 99,
            "touchdown_x_offset": 99,
        }),
        # A touchdown outside the center tolerance is unsuccessful, but not a crash.
        _episode(x=0.4, left=1, right=1, reward=100, terminated=True, info={
            "landed": True, "episode_final_speed": 4, "episode_final_angle": 0.4,
            "episode_final_x_offset": 0.4, "touchdown_speed": 4,
            "touchdown_angle": 0.4, "touchdown_x_offset": 0.4,
        }),
        _episode(terminated=True, info={"episode_final_speed": 6}),
    ]
    model = FakeEvaluationModel()

    metrics = evaluate(model, FakeEvaluationEnv(episodes), episodes=5, seed=123, center_tolerance=0.2)

    assert metrics["success_rate"] == pytest.approx(0.2)
    assert metrics["crash_rate"] == pytest.approx(0.2)
    assert metrics["timeout_rate"] == pytest.approx(0.2)
    assert metrics["unsuccessful_landing_rate"] == pytest.approx(0.2)
    assert metrics["other_termination_rate"] == pytest.approx(0.2)
    assert sum(metrics[key] for key in (
        "success_rate", "crash_rate", "timeout_rate",
        "unsuccessful_landing_rate", "other_termination_rate",
    )) == pytest.approx(1.0)

    # Three contact outcomes have complete touchdown telemetry; the timeout is excluded.
    assert metrics["touchdown_count"] == 3
    assert metrics["mean_touchdown_speed"] == pytest.approx((5 + 2 + 4) / 3)
    assert metrics["mean_touchdown_angle"] == pytest.approx((0.1 + 0.2 + 0.4) / 3)
    assert metrics["mean_touchdown_offset"] == pytest.approx((0.1 + 0.2 + 0.4) / 3)

    # Episode-final metrics include timeout states, including its explicit speed of 3.
    assert metrics["mean_episode_final_speed"] == pytest.approx((5 + 2 + 3 + 4 + 6) / 5)
    assert model.deterministic_flags == [True] * 5


def test_evaluate_returns_unavailable_touchdown_metrics_when_no_contact_occurs():
    episodes = [
        _episode(truncated=True, info={
            "episode_final_speed": 3, "episode_final_angle": 0.3,
            "episode_final_x_offset": 0.4,
        }),
        _episode(truncated=True, info={
            "episode_final_speed": 5, "episode_final_angle": 0.5,
            "episode_final_x_offset": 0.6,
        }),
    ]

    metrics = evaluate(FakeEvaluationModel(), FakeEvaluationEnv(episodes), episodes=2, seed=9, center_tolerance=0.2)

    assert metrics["touchdown_count"] == 0
    assert metrics["mean_touchdown_speed"] is None
    assert metrics["mean_touchdown_angle"] is None
    assert metrics["mean_touchdown_offset"] is None
    assert metrics["mean_episode_final_speed"] == pytest.approx(4.0)
    assert metrics["mean_episode_final_angle"] == pytest.approx(0.4)
    assert metrics["mean_episode_final_offset"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("lengths", "expected_mean", "expected_median"),
    [([1, 3, 5], 3.0, 3), ([1, 2, 7, 10], 5.0, 4.5)],
)
def test_evaluate_reports_mean_and_median_episode_length(lengths, expected_mean, expected_median):
    metrics = evaluate(
        FakeEvaluationModel(), FakeLengthEvaluationEnv(lengths),
        episodes=len(lengths), seed=50, center_tolerance=0.2,
    )

    assert metrics["mean_episode_length"] == pytest.approx(expected_mean)
    assert metrics["median_episode_length"] == pytest.approx(expected_median)
