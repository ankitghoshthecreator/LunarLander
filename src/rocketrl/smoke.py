"""Inspect LunarLander spaces and run a short random-policy smoke test."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from rocketrl.envs import ShapedLander


OBSERVATION_FIELDS = (
    "x position", "y position", "x velocity", "y velocity",
    "angle", "angular velocity", "left leg contact", "right leg contact",
)
ACTION_FIELDS = {0: "no-op", 1: "left engine", 2: "main engine", 3: "right engine"}


def heuristic_action(observation) -> int:
    """Simple hand-built stabilizer, useful for checking successful terminal rewards."""
    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = observation
    target_angle = float(np.clip(x * 0.5 + vx, -0.4, 0.4))
    target_hover = 0.55 * abs(x)
    angle_error = (target_angle - angle) * 0.5 - angular_velocity
    hover_error = (target_hover - y) * 0.5 - vy * 0.5
    if left_leg or right_leg:
        angle_error = 0.0
        hover_error = -vy * 0.5
    if hover_error > abs(angle_error) and hover_error > 0.05:
        return 2
    if angle_error < -0.05:
        return 3
    if angle_error > 0.05:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", choices=("human", "rgb_array"), default=None)
    parser.add_argument("--policy", choices=("random", "heuristic"), default="random")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    env = ShapedLander(gym.make("LunarLander-v3", render_mode=args.render), mode="default")
    try:
        print("Observation space:", env.observation_space)
        print("Observation layout:", list(enumerate(OBSERVATION_FIELDS)))
        print("Action space:", env.action_space, ACTION_FIELDS)
        print("TimeLimit max_episode_steps:", env.spec.max_episode_steps)
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            print(f"Episode {episode + 1} initial observation:", observation)
            print(f"Episode {episode + 1} reset info fields:", sorted(reset_info))
            steps = 0
            while True:
                action = env.action_space.sample() if args.policy == "random" else heuristic_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    print(
                        f"Episode {episode + 1} terminal: steps={steps}, "
                        f"base_reward={info['base_reward']}, landed={info['landed']}, "
                        f"crashed={info['crashed']}, terminated={terminated}, truncated={truncated}"
                    )
                    print("Terminal info fields:", sorted(info))
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
