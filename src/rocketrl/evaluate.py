"""Evaluate a saved agent using the common Gymnasium reward."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics

import numpy as np
from stable_baselines3 import DQN, PPO

from rocketrl.common import load_config, make_lander


def evaluate(model, env, episodes: int, seed: int, center_tolerance: float) -> dict[str, float | int | str]:
    rewards: list[float] = []
    lengths: list[int] = []
    touchdown_speeds: list[float] = []
    touchdown_angles: list[float] = []
    touchdown_offsets: list[float] = []
    fuel_used: list[float] = []
    successes = crashes = timeouts = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        length = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            length += 1
            if terminated or truncated:
                # Success is deliberately stricter than the environment's +100 label.
                both_legs = bool(observation[6] and observation[7])
                landed = bool(info.get("landed", False) and both_legs and abs(float(observation[0])) <= center_tolerance)
                successes += int(landed)
                crashes += int(bool(info.get("crashed", False)) or (bool(info.get("landed", False)) and not landed))
                timeouts += int(bool(truncated))
                rewards.append(total_reward)
                lengths.append(length)
                touchdown_speeds.append(float(info.get("speed", np.hypot(observation[2], observation[3]))))
                touchdown_angles.append(float(info.get("angle", abs(observation[4]))))
                touchdown_offsets.append(float(info.get("x_offset", abs(observation[0]))))
                fuel_used.append(float(info.get("fuel_used", 0.0)))
                break

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_reward": statistics.fmean(rewards),
        "median_reward": statistics.median(rewards),
        "mean_touchdown_speed": statistics.fmean(touchdown_speeds),
        "mean_touchdown_angle": statistics.fmean(touchdown_angles),
        "mean_touchdown_offset": statistics.fmean(touchdown_offsets),
        "mean_fuel_used": statistics.fmean(fuel_used),
        "mean_episode_length": statistics.fmean(lengths),
        "crash_rate": crashes / episodes,
        "timeout_rate": timeouts / episodes,
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
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Appended metrics to {output}")


if __name__ == "__main__":
    main()
