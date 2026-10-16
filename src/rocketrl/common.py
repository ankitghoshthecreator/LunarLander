"""Shared configuration and environment helpers for experiment commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

from rocketrl.envs import ShapedLander


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def make_lander(config: dict[str, Any], mode: str, render_mode: str | None = None) -> ShapedLander:
    env_kwargs = dict(config.get("env", {}))
    env_kwargs.setdefault("continuous", False)
    env_kwargs.setdefault("max_episode_steps", 1000)
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    env = gym.make("LunarLander-v3", **env_kwargs)
    return ShapedLander(env, mode=mode, weights=config.get("weights") if mode != "default" else None)
