"""Simulation controller managing playback state, gravity updates, and episode transitions."""

from __future__ import annotations

from typing import Literal

import numpy as np

from rocketrl.frontend.adapter import SimulationEnvironmentAdapter
from rocketrl.frontend.config import SimulationConfig
from rocketrl.frontend.metrics import EpisodeRecord, SessionLogManager, classify_outcome


SimulationStatus = Literal["READY", "RUNNING", "PAUSED", "LANDED", "CRASHED", "TIMEOUT"]


class SimulationController:
    """Coordinates simulation stepping, user controls (Run/Pause/Reset), and episode logging."""

    def __init__(
        self,
        adapter: SimulationEnvironmentAdapter,
        log_manager: SessionLogManager | None = None,
    ) -> None:
        self.adapter = adapter
        self.log_manager = log_manager or SessionLogManager()
        self.status: SimulationStatus = "READY"
        self.last_outcome: str | None = None
        self.current_episode: int = 1
        self.selected_gravity: float = adapter.config.gravity
        self.last_frame: np.ndarray | None = None

        # Initial reset
        self.reset(apply_new_gravity=True)

    def set_gravity(self, new_gravity: float) -> float:
        """Update selected gravity and apply to environment."""
        self.selected_gravity = self.adapter.apply_gravity(new_gravity)
        return self.selected_gravity

    def run(self) -> None:
        """Start or resume simulation playback."""
        self.status = "RUNNING"

    def pause(self) -> None:
        """Pause simulation without resetting episode state."""
        self.status = "PAUSED"

    def reset(self, apply_new_gravity: bool = True) -> None:
        """Reset the current episode and apply the selected gravity."""
        if apply_new_gravity:
            self.adapter.apply_gravity(self.selected_gravity)

        self.adapter.reset()
        self.last_frame = self.adapter.render_frame()
        self.status = "READY"

    def step(self, num_steps: int = 1) -> tuple[np.ndarray | None, bool]:
        """Execute one or more simulation steps.

        In RUNNING mode, completed episodes are logged and the next episode is
        automatically started so rollouts continue seamlessly without freezing.

        Returns:
            (last_frame, episode_completed)
        """
        if self.status != "RUNNING":
            return self.last_frame, False

        episode_done = False
        for _ in range(num_steps):
            obs, reward, terminated, truncated, info = self.adapter.step()
            if terminated or truncated:
                episode_done = True
                summary = self.adapter.get_episode_summary()
                outcome = summary["outcome"]
                self.last_outcome = outcome

                # Record in session log manager
                record = EpisodeRecord(
                    episode_id=self.current_episode,
                    outcome=outcome,
                    raw_reward=summary["raw_reward"],
                    shaped_reward=summary["shaped_reward"],
                    steps=summary["steps"],
                    duration_seconds=summary["duration_seconds"],
                    gravity=self.selected_gravity,
                    fuel_used=summary["fuel_used"],
                )
                self.log_manager.add_record(record)
                self.current_episode += 1

                # Automatically prepare next episode and continue running
                self.adapter.reset()
                break

        self.last_frame = self.adapter.render_frame()
        return self.last_frame, episode_done

    def cleanup(self) -> None:
        """Release underlying environment and resources."""
        self.adapter.close()
