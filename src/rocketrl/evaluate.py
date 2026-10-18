"""Evaluate a saved agent using the common Gymnasium reward."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

import numpy as np
from stable_baselines3 import DQN, PPO

from rocketrl.common import load_config, make_lander


def _finite_metric(value: object) -> float | None:
    """Return a finite numeric metric, or None when it is absent or invalid."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_finite_metric(*values: object) -> float:
    """Return the first finite value, preserving valid zeros."""
    for value in values:
        result = _finite_metric(value)
        if result is not None:
            return result
    raise ValueError("At least one finite fallback metric is required")


def evaluate(model, env, episodes: int, seed: int, center_tolerance: float) -> dict[str, float | int | str | None]:
    rewards: list[float] = []
    lengths: list[int] = []
    episode_final_speeds: list[float] = []
    episode_final_angles: list[float] = []
    episode_final_offsets: list[float] = []
    touchdown_speeds: list[float] = []
    touchdown_angles: list[float] = []
    touchdown_offsets: list[float] = []
    fuel_used: list[float] = []
    main_engine_steps: list[int] = []
    side_engine_steps: list[int] = []
    successes = crashes = timeouts = unsuccessful_landings = other_terminations = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        length = 0
        main_steps = side_steps = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            action_id = int(np.asarray(action).item()) if np.asarray(action).ndim == 0 else None
            main_steps += int(action_id == 2)
            side_steps += int(action_id in (1, 3))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            if terminated or truncated:
                # A timeout takes precedence if an environment reports both flags.
                is_timeout = bool(truncated)
                is_crash = not is_timeout and bool(info.get("crashed", False))
                is_landing = not is_timeout and not is_crash and bool(info.get("landed", False))
                both_legs = bool(observation[6] and observation[7])
                success = bool(
                    is_landing
                    and both_legs
                    and abs(float(observation[0])) <= center_tolerance
                )
                successes += int(success)
                crashes += int(is_crash)
                timeouts += int(is_timeout)
                unsuccessful_landings += int(is_landing and not success)
                other_terminations += int(bool(terminated) and not is_timeout and not is_crash and not is_landing)
                rewards.append(total_reward)
                lengths.append(length)
                # Use episode_final_* keys (set on any terminal/truncated step) for
                # per-episode kinematics. Fall back to raw observation fields so this
                # function still works with unwrapped LunarLander environments.
                episode_final_speeds.append(_first_finite_metric(
                    info.get("episode_final_speed"), info.get("speed"),
                    np.hypot(observation[2], observation[3]),
                ))
                episode_final_angles.append(_first_finite_metric(
                    info.get("episode_final_angle"), info.get("angle"), abs(float(observation[4])),
                ))
                episode_final_offsets.append(_first_finite_metric(
                    info.get("episode_final_x_offset"), info.get("x_offset"), abs(float(observation[0])),
                ))

                # ShapedLander supplies these fields only for landings or crashes;
                # use one shared complete-case sample so touchdown_count applies to
                # each touchdown mean. Missing, nonnumeric, NaN, and infinite values
                # are excluded. Timeouts therefore contribute only to episode_final_*.
                touchdown_metrics = (
                    _finite_metric(info.get("touchdown_speed")),
                    _finite_metric(info.get("touchdown_angle")),
                    _finite_metric(info.get("touchdown_x_offset")),
                )
                if (is_landing or is_crash) and all(value is not None for value in touchdown_metrics):
                    touchdown_speeds.append(touchdown_metrics[0])
                    touchdown_angles.append(touchdown_metrics[1])
                    touchdown_offsets.append(touchdown_metrics[2])
                fuel_used.append(float(info.get("fuel_used", 0.0)))
                main_engine_steps.append(main_steps)
                side_engine_steps.append(side_steps)
                break

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_reward": statistics.fmean(rewards),
        "median_reward": statistics.median(rewards),
        # episode_final_* metrics include the final kinematic state regardless of outcome
        # (touchdown, crash, or timeout). They are NOT restricted to true touchdown episodes.
        "mean_episode_final_speed": statistics.fmean(episode_final_speeds),
        "mean_episode_final_angle": statistics.fmean(episode_final_angles),
        "mean_episode_final_offset": statistics.fmean(episode_final_offsets),
        "touchdown_count": len(touchdown_speeds),
        "mean_touchdown_speed": statistics.fmean(touchdown_speeds) if touchdown_speeds else None,
        "mean_touchdown_angle": statistics.fmean(touchdown_angles) if touchdown_angles else None,
        "mean_touchdown_offset": statistics.fmean(touchdown_offsets) if touchdown_offsets else None,
        "mean_fuel_used": statistics.fmean(fuel_used),
        "mean_main_engine_steps": statistics.fmean(main_engine_steps),
        "mean_side_engine_steps": statistics.fmean(side_engine_steps),
        "mean_episode_length": statistics.fmean(lengths),
        "median_episode_length": statistics.median(lengths),
        "crash_rate": crashes / episodes,
        "timeout_rate": timeouts / episodes,
        "unsuccessful_landing_rate": unsuccessful_landings / episodes,
        "other_termination_rate": other_terminations / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Model zip path (PPO or DQN)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--center-tolerance", type=float, default=0.2)
    parser.add_argument("--output", default="results/metrics.csv")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    config = load_config(args.config)
    algorithm = str(config.get("algo", "PPO")).upper()
    model_class = {"PPO": PPO, "DQN": DQN}.get(algorithm)
    if model_class is None:
        parser.error(f"Unsupported algorithm {algorithm!r}; choose PPO or DQN")
    model = model_class.load(args.model, device="cpu")
    env = make_lander(config, "default")
    try:
        metrics = evaluate(model, env, args.episodes, args.seed, args.center_tolerance)
    finally:
        env.close()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "name": config.get("name", Path(args.config).stem),
        "algorithm": algorithm,
        "model": args.model,
        "seed": args.seed,
        "center_tolerance": args.center_tolerance,
        **metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    exists = output.exists() and output.stat().st_size > 0
    if exists:
        with output.open("r", newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != fieldnames:
            raise ValueError(
                f"CSV schema at {output} does not match this evaluator; "
                "choose a new output path to preserve the existing results"
            )
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Appended metrics to {output}")


if __name__ == "__main__":
    main()
