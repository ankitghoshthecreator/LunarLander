"""Train PPO or DQN from a YAML experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from rocketrl.common import load_config, make_lander


class LatestCheckpointCallback(BaseCallback):
    """Keep an atomically replaceable latest checkpoint during long runs."""

    def __init__(self, path: Path, save_every: int, verbose: int = 0):
        super().__init__(verbose)
        self.path = path
        self.save_every = save_every
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= self.save_every:
            self.model.save(str(self.path))
            self._last_saved = self.num_timesteps
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, help="Override configured training steps")
    args = parser.parse_args()

    config = load_config(args.config)
    name = config["name"]
    n_envs = int(config.get("n_envs", 1))
    if n_envs < 1:
        parser.error("n_envs must be at least 1")
    run_name = f"{name}_s{args.seed}"
    model_dir = Path("models") / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    def build_training_env():
        return make_lander(config, config["reward_mode"])

    # make_vec_env adds a Monitor to each worker by default; that provides the
    # episode statistics used by EvalCallback without an extra VecMonitor.
    train_env = make_vec_env(build_training_env, n_envs=n_envs, seed=args.seed)
    eval_env = make_vec_env(lambda: make_lander(config, "default"), n_envs=1, seed=args.seed + 10_000)
    algorithm = str(config.get("algo", "PPO")).upper()
    model_class = {"PPO": PPO, "DQN": DQN}.get(algorithm)
    if model_class is None:
        raise ValueError(f"Unsupported algorithm {algorithm!r}; choose PPO or DQN")

    model = model_class(
        config.get("policy", "MlpPolicy"), train_env, seed=args.seed, device="cpu",
        tensorboard_log="runs/", verbose=1, **config.get("hparams", {}),
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval"),
        eval_freq=max(int(config.get("eval_freq", 10_000)) // n_envs, 1),
        n_eval_episodes=int(config.get("eval_episodes", 20)),
        deterministic=True,
    )
    latest_callback = LatestCheckpointCallback(
        model_dir / "latest", max(int(config.get("checkpoint_freq", 50_000)) // n_envs, 1)
    )
    try:
        model.learn(
            total_timesteps=args.timesteps or int(config["timesteps"]),
            callback=CallbackList([eval_callback, latest_callback]),
            tb_log_name=run_name,
            progress_bar=True,
        )
        model.save(str(model_dir / "latest"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
