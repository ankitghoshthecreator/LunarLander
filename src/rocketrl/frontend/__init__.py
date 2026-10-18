"""Frontend simulation adapter, controller, and metrics package."""

from rocketrl.frontend.adapter import SimulationEnvironmentAdapter
from rocketrl.frontend.config import (
    DEFAULT_GRAVITY,
    MAX_GRAVITY,
    MIN_GRAVITY,
    SimulationConfig,
)
from rocketrl.frontend.controller import SimulationController, SimulationStatus
from rocketrl.frontend.metrics import (
    EpisodeRecord,
    OutcomeType,
    SessionLogManager,
    classify_outcome,
)

__all__ = [
    "DEFAULT_GRAVITY",
    "MAX_GRAVITY",
    "MIN_GRAVITY",
    "EpisodeRecord",
    "OutcomeType",
    "SessionLogManager",
    "SimulationConfig",
    "SimulationController",
    "SimulationEnvironmentAdapter",
    "SimulationStatus",
    "classify_outcome",
]
