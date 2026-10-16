"""Reward variants and episode telemetry for LunarLander-v3."""

from __future__ import annotations

from typing import Literal, Mapping

import gymnasium as gym
import numpy as np

RewardMode = Literal["sparse", "default", "shaped", "overshaped"]

DEFAULT_WEIGHTS: dict[str, float] = {
    "w_dist": 100.0,
    "w_vel": 100.0,
    "w_ang": 100.0,
    "w_leg": 10.0,
    "c_main": 0.3,
    "c_side": 0.03,
    "c_time": 0.05,
    "b_land": 100.0,
    "b_crash": 100.0,
    "b_center": 20.0,
    "gamma": 0.999,
    # Overshaped deliberately rewards proximity without an anti-hover cost.
    "overshaped_scale": 5.0,
}


class ShapedLander(gym.Wrapper):
    """Wrap a LunarLander environment with selectable reward design.

    ``default`` passes through Gymnasium's reward. ``sparse`` returns only
    terminal labels. ``shaped`` adds potential-based shaping, fuel/time costs,
    and terminal bonuses. ``overshaped`` amplifies proximity shaping and omits
    the time cost to make reward hacking observable.
    """

    MODES = frozenset(("sparse", "default", "shaped", "overshaped"))

    def __init__(
        self,
        env: gym.Env,
        mode: RewardMode = "shaped",
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(env)
        if mode not in self.MODES:
            raise ValueError(f"Unknown reward mode {mode!r}; choose from {sorted(self.MODES)}")
        self.mode: RewardMode = mode
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        unknown = set(weights or ()) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"Unknown reward weight(s): {', '.join(sorted(unknown))}")
        if not 0.0 <= self.weights["gamma"] <= 1.0:
            raise ValueError("weights.gamma must be in [0, 1]")
        self._prev_phi = 0.0
        self._prev_components = (0.0, 0.0, 0.0, 0.0)
        self._episode_fuel = 0.0

    def _phi(self, observation: np.ndarray) -> float:
        x, y, vx, vy, angle, _angular_velocity, left, right = np.asarray(observation)[:8]
        w = self.weights
        return float(
            -(w["w_dist"] * np.hypot(x, y)
              + w["w_vel"] * np.hypot(vx, vy)
              + w["w_ang"] * abs(angle))
            + w["w_leg"] * (left + right)
        )

    def _potential_components(self, observation: np.ndarray) -> tuple[float, float, float, float]:
        x, y, vx, vy, angle, _angular_velocity, left, right = np.asarray(observation)[:8]
        w = self.weights
        return (
            float(-w["w_dist"] * np.hypot(x, y)),
            float(-w["w_vel"] * np.hypot(vx, vy)),
            float(-w["w_ang"] * abs(angle)),
            float(w["w_leg"] * (left + right)),
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(observation)
        self._prev_components = self._potential_components(observation)
        self._episode_fuel = 0.0
        return observation, info

    def step(self, action):
        observation, base_reward, terminated, truncated, base_info = self.env.step(action)
        info = dict(base_info)
        obs = np.asarray(observation)
        action_id = int(np.asarray(action).item()) if np.asarray(action).ndim == 0 else None
        main_fired = action_id == 2
        side_fired = action_id in (1, 3)
        fuel_component = -self.weights["c_main"] * main_fired - self.weights["c_side"] * side_fired
        self._episode_fuel += float(main_fired) + float(side_fired)

        landed = bool(terminated and base_reward >= 100.0)
        crashed = bool(terminated and base_reward <= -100.0)
        x, y, vx, vy, angle = (float(v) for v in obs[:5])
        speed = float(np.hypot(vx, vy))

        distance_component, velocity_component, angle_component, leg_component = (0.0, 0.0, 0.0, 0.0)
        shaping_component = 0.0
        time_component = 0.0
        terminal_component = 0.0
        center_component = 0.0

        if self.mode == "default":
            reward = float(base_reward)
        elif self.mode == "sparse":
            reward = 1.0 if landed else (-1.0 if crashed else 0.0)
            terminal_component = reward
        else:
            w = self.weights
            if self.mode == "overshaped":
                fuel_component = 0.0
            phi = self._phi(obs)
            shaping_component = float(w["gamma"] * phi - self._prev_phi)
            current_components = self._potential_components(obs)
            distance_component, velocity_component, angle_component, leg_component = tuple(
                float(w["gamma"] * current - previous)
                for current, previous in zip(current_components, self._prev_components)
            )
            time_component = -w["c_time"] if self.mode == "shaped" else 0.0
            if self.mode == "overshaped":
                shaping_component *= w["overshaped_scale"]
                distance_component *= w["overshaped_scale"]
                velocity_component *= w["overshaped_scale"]
                angle_component *= w["overshaped_scale"]
                leg_component *= w["overshaped_scale"]
            terminal_component = w["b_land"] if landed else (-w["b_crash"] if crashed else 0.0)
            if landed:
                center_component = w["b_center"] * max(0.0, 1.0 - abs(x))
            reward = shaping_component + time_component + fuel_component + terminal_component + center_component
            self._prev_phi = phi
            self._prev_components = current_components

        info.update(
            landed=landed,
            crashed=crashed,
            base_reward=float(base_reward),
            x=x,
            y=y,
            speed=speed,
            touchdown_speed=speed if (terminated or truncated) else None,
            angle=abs(angle),
            touchdown_angle=abs(angle) if (terminated or truncated) else None,
            x_offset=abs(x),
            touchdown_x_offset=abs(x) if (terminated or truncated) else None,
            fuel_used=self._episode_fuel,
            reward_components={
                "distance": distance_component,
                "velocity": velocity_component,
                "angle": angle_component,
                "leg": leg_component,
                "shaping": shaping_component,
                "fuel": fuel_component if self.mode in ("shaped", "overshaped") else 0.0,
                "time": time_component,
                "terminal": terminal_component,
                "center": center_component,
            },
        )
        return observation, reward, terminated, truncated, info
