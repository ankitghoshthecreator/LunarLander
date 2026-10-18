"""Simulation configuration and parameter management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MIN_GRAVITY: float = 0.5
MAX_GRAVITY: float = 11.9
DEFAULT_GRAVITY: float = 10.0


@dataclass
class SimulationConfig:
    """Central configuration for simulation parameters.

    Designed for extensibility: future physical parameters (wind, turbulence,
    engine power) can be added here without altering the rendering or simulation
    lifecycle.
    """

    gravity: float = DEFAULT_GRAVITY
    continuous: bool = False
    max_episode_steps: int = 1000
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gravity = self.validate_and_clamp_gravity(self.gravity)

    @staticmethod
    def validate_and_clamp_gravity(value: float) -> float:
        """Validate and clamp gravity to Gymnasium LunarLander supported range."""
        try:
            val = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gravity must be a valid float, got {value!r}") from exc
        return max(MIN_GRAVITY, min(MAX_GRAVITY, val))

    def update_gravity(self, new_gravity: float) -> float:
        """Update gravity to a clamped valid value."""
        self.gravity = self.validate_and_clamp_gravity(new_gravity)
        return self.gravity

    @property
    def gym_gravity(self) -> float:
        """Gymnasium expects downward gravity as a negative float."""
        return -abs(self.gravity)
