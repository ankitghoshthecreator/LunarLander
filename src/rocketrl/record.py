"""Record episodes from a trained PPO or DQN policy as video."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordVideo

from rocketrl.common import load_config, make_lander


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML config")
    parser.add_argument("--model", required=True, help="Trained PPO or DQN checkpoint")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", default="videos")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    config = load_config(args.config)
    algorithm = str(config.get("algo", "PPO")).upper()
    model_class = {"PPO": PPO, "DQN": DQN}.get(algorithm)
    if model_class is None:
        parser.error(f"Unsupported algorithm {algorithm!r}; choose PPO or DQN")
    model = model_class.load(args.model, device="cpu")
    base_env = make_lander(config, "default", render_mode="rgb_array")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = RecordVideo(
        base_env,
        video_folder=str(output),
        episode_trigger=lambda episode_id: episode_id < args.episodes,
        name_prefix=Path(args.model).stem,
    )
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    print(f"Recorded {args.episodes} episode(s) in {output}")


if __name__ == "__main__":
    main()
