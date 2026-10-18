"""Adapter for Gymnasium LunarLander environment and PyGame rendering."""

from __future__ import annotations

import os
import time
from typing import Any

import gymnasium as gym
import numpy as np

from rocketrl.envs.shaped_lander import ShapedLander
from rocketrl.frontend.config import SimulationConfig
from rocketrl.frontend.metrics import OutcomeType, classify_outcome


class SimulationEnvironmentAdapter:
    """Manages the lifecycle of LunarLander environment with frame capture and telemetry."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        reward_mode: str = "shaped",
        model: Any = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.reward_mode = reward_mode
        self.model = model

        # Ensure headless SDL for servers/Streamlit
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        self.env: gym.Env | None = None
        self._current_obs: np.ndarray | None = None
        self._current_info: dict = {}
        self._raw_reward_acc: float = 0.0
        self._shaped_reward_acc: float = 0.0
        self._step_count: int = 0
        self._episode_start_time: float = time.time()
        self._is_terminated: bool = False
        self._is_truncated: bool = False

        self._build_env()

    def _build_env(self) -> None:
        """Construct the underlying Gymnasium environment with configured gravity."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        base_env = gym.make(
            "LunarLander-v3",
            continuous=self.config.continuous,
            gravity=self.config.gym_gravity,
            max_episode_steps=self.config.max_episode_steps,
            render_mode="rgb_array",
        )
        self.env = ShapedLander(base_env, mode=self.reward_mode)

    def apply_gravity(self, new_gravity: float) -> float:
        """Update gravity in config and apply live to the active Box2D world."""
        g = self.config.update_gravity(new_gravity)
        if self.env is not None and hasattr(self.env, "unwrapped"):
            unwrapped = self.env.unwrapped
            unwrapped.gravity = -abs(g)
            if hasattr(unwrapped, "world") and unwrapped.world is not None:
                unwrapped.world.gravity = (0.0, -abs(g))
        return g

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the simulation episode with the active configuration."""
        if self.env is None:
            self._build_env()

        # Ensure the configured gravity is present before reset
        if hasattr(self.env.unwrapped, "gravity"):
            self.env.unwrapped.gravity = self.config.gym_gravity

        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed

        obs, info = self.env.reset(**reset_kwargs)
        self._current_obs = np.asarray(obs, dtype=np.float32)
        self._current_info = dict(info)
        self._raw_reward_acc = 0.0
        self._shaped_reward_acc = 0.0
        self._step_count = 0
        self._episode_start_time = time.time()
        self._is_terminated = False
        self._is_truncated = False

        # Verify Box2D world has correct gravity
        if hasattr(self.env.unwrapped, "world") and self.env.unwrapped.world is not None:
            self.env.unwrapped.world.gravity = (0.0, self.config.gym_gravity)

        return self._current_obs, self._current_info

    def step(self, action: int | np.ndarray | None = None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one step using the model or provided action."""
        if self.env is None or self._current_obs is None:
            self.reset()

        if action is None:
            if self.model is not None:
                action, _ = self.model.predict(self._current_obs, deterministic=True)
                action = int(action) if np.asarray(action).ndim == 0 else action
            else:
                action = self.env.action_space.sample()

        obs, shaped_reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = np.asarray(obs, dtype=np.float32)
        self._current_info = dict(info)
        self._is_terminated = bool(terminated)
        self._is_truncated = bool(truncated)

        # Base Gymnasium reward is passed via ShapedLander's info
        base_reward = float(info.get("base_reward", shaped_reward))
        self._raw_reward_acc += base_reward
        self._shaped_reward_acc += float(shaped_reward)
        self._step_count += 1

        return self._current_obs, float(shaped_reward), self._is_terminated, self._is_truncated, self._current_info

    def render_frame(self) -> np.ndarray | None:
        """Capture RGB frame from the environment's PyGame renderer."""
        if self.env is None:
            return None
        return self.env.render()

    def get_telemetry(self) -> dict[str, Any]:
        """Extract structured, physical telemetry from current observation and info."""
        if self._current_obs is None:
            return {
                "x_pos": 0.0,
                "y_pos": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "angle": 0.0,
                "angular_velocity": 0.0,
                "left_leg": False,
                "right_leg": False,
                "fuel_used": 0.0,
                "angle_warning": False,
                "speed_warning": False,
            }

        o = self._current_obs
        x = float(o[0])
        y = float(o[1])
        vx = float(o[2])
        vy = float(o[3])
        angle = float(o[4])
        ang_vel = float(o[5])
        left_leg = bool(o[6] > 0.5)
        right_leg = bool(o[7] > 0.5)
        speed = float(np.hypot(vx, vy))
        fuel_used = float(self._current_info.get("fuel_used", 0.0))

        # Warnings for excessive tilt or descent rate
        angle_warning = abs(angle) > 0.35  # ~20 degrees
        speed_warning = speed > 1.2 or vy < -1.0

        return {
            "x_pos": x,
            "y_pos": y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "angle": angle,
            "angular_velocity": ang_vel,
            "left_leg": left_leg,
            "right_leg": right_leg,
            "fuel_used": fuel_used,
            "angle_warning": angle_warning,
            "speed_warning": speed_warning,
        }

    def get_episode_summary(self) -> dict[str, Any]:
        """Return cumulative metrics for current episode."""
        outcome = "RUNNING"
        if self._is_terminated or self._is_truncated:
            outcome = classify_outcome(
                self._is_terminated,
                self._is_truncated,
                self._current_info,
                self._current_obs if self._current_obs is not None else np.zeros(8),
            )

        duration = max(0.01, time.time() - self._episode_start_time)
        return {
            "raw_reward": self._raw_reward_acc,
            "shaped_reward": self._shaped_reward_acc,
            "steps": self._step_count,
            "duration_seconds": duration,
            "outcome": outcome,
            "is_done": self._is_terminated or self._is_truncated,
            "fuel_used": float(self._current_info.get("fuel_used", 0.0)),
        }

    def close(self) -> None:
        """Release Gymnasium and Box2D resources."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        self._current_obs = None
