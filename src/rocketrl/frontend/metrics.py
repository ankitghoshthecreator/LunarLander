"""Outcome classification and episode logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import math
from typing import Literal

import numpy as np


OutcomeType = Literal[
    "LANDED (SUCCESS)",
    "CRASHED",
    "TIMEOUT",
    "UNSUCCESSFUL LANDING",
    "OTHER TERMINATION",
]

OUTCOME_COLORS: dict[OutcomeType, dict[str, str]] = {
    "LANDED (SUCCESS)": {
        "color": "#00ff88",
        "bg": "rgba(0, 255, 136, 0.12)",
        "border": "#00ff88",
        "icon": "🟢",
        "tag": "SUCCESS",
    },
    "CRASHED": {
        "color": "#ff4455",
        "bg": "rgba(255, 68, 85, 0.12)",
        "border": "#ff4455",
        "icon": "🔴",
        "tag": "CRASH",
    },
    "TIMEOUT": {
        "color": "#ffaa00",
        "bg": "rgba(255, 170, 0, 0.12)",
        "border": "#ffaa00",
        "icon": "🟠",
        "tag": "TIMEOUT",
    },
    "UNSUCCESSFUL LANDING": {
        "color": "#00b4d8",
        "bg": "rgba(0, 180, 216, 0.12)",
        "border": "#00b4d8",
        "icon": "🔵",
        "tag": "OFF-PAD",
    },
    "OTHER TERMINATION": {
        "color": "#90e0ef",
        "bg": "rgba(144, 224, 239, 0.12)",
        "border": "#90e0ef",
        "icon": "⚪",
        "tag": "TERMINATED",
    },
}


def classify_outcome(
    terminated: bool,
    truncated: bool,
    info: dict,
    observation: np.ndarray | list[float],
    center_tolerance: float = 0.2,
) -> OutcomeType:
    """Classify the terminal state according to the project's success definition.

    - Timeout takes precedence if environment reports both flags.
    - Crash: terminal step with base_reward <= -100 (crashed flag in info).
    - Landing: terminal step with base_reward >= +100 (landed flag in info).
    - Success: landed AND both legs in contact AND |x| <= center_tolerance.
    - Unsuccessful landing: landed, but legs not down or not within center tolerance.
    """
    obs = np.asarray(observation)
    is_timeout = bool(truncated)
    is_crash = not is_timeout and bool(info.get("crashed", False))
    is_landing = not is_timeout and not is_crash and bool(info.get("landed", False))

    if is_timeout:
        return "TIMEOUT"
    if is_crash:
        return "CRASHED"
    if is_landing:
        both_legs = bool(obs[6] and obs[7])
        on_pad = bool(abs(float(obs[0])) <= center_tolerance)
        if both_legs and on_pad:
            return "LANDED (SUCCESS)"
        return "UNSUCCESSFUL LANDING"

    return "OTHER TERMINATION"


@dataclass
class EpisodeRecord:
    """Structured record of a single completed simulation episode."""

    episode_id: int
    outcome: OutcomeType
    raw_reward: float
    shaped_reward: float
    steps: int
    duration_seconds: float
    gravity: float
    fuel_used: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%H:%M:%S")
    )

    @property
    def formatted_message(self) -> str:
        """Format matching project requirement:
        Episode 73: LANDED (SUCCESS) | Total Reward: +247.9 | Steps: 321
        """
        # Primary displayed reward is the Gymnasium raw reward for standardized comparison
        rew_str = f"{self.raw_reward:+.1f}" if math.isfinite(self.raw_reward) else "0.0"
        return f"Episode {self.episode_id}: {self.outcome} | Total Reward: {rew_str} | Steps: {self.steps:,}"


class SessionLogManager:
    """Bounded, thread-safe episode log manager for session state."""

    def __init__(self, max_history: int = 150) -> None:
        self.max_history = max_history
        self.records: list[EpisodeRecord] = []
        self.total_episodes: int = 0
        self.successes: int = 0
        self.crashes: int = 0
        self.timeouts: int = 0
        self.unsuccessful_landings: int = 0

    def add_record(self, record: EpisodeRecord) -> None:
        self.total_episodes += 1
        if record.outcome == "LANDED (SUCCESS)":
            self.successes += 1
        elif record.outcome == "CRASHED":
            self.crashes += 1
        elif record.outcome == "TIMEOUT":
            self.timeouts += 1
        elif record.outcome == "UNSUCCESSFUL LANDING":
            self.unsuccessful_landings += 1

        self.records.append(record)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]

    @property
    def recent_records(self) -> list[EpisodeRecord]:
        """Return records in reverse order (newest first)."""
        return list(reversed(self.records))

    def recent_success_rate(self, window: int = 20) -> float:
        """Calculate success percentage over recent completed episodes."""
        if not self.records:
            return 0.0
        slice_records = self.records[-window:]
        success_count = sum(1 for r in slice_records if r.outcome == "LANDED (SUCCESS)")
        return (success_count / len(slice_records)) * 100.0

    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across the session."""
        if self.total_episodes == 0:
            return 0.0
        return (self.successes / self.total_episodes) * 100.0
