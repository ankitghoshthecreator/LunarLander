"""Train PPO or DQN from a YAML experiment config."""

from __future__ import annotations

import argparse
import csv
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


class SuccessRateCallback(BaseCallback):
    """Periodically evaluate 100 fixed-seed episodes and record sample efficiency."""

    def __init__(
        self,
        config: dict,
        output_dir: Path,
        eval_freq: int = 50_000,
        episodes: int = 100,
        seed: int = 20_000,
        center_tolerance: float = 0.2,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.config = config
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed = seed
        self.center_tolerance = center_tolerance
        self.next_eval = eval_freq
        self.recorded_target = False

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True
        from rocketrl.common import make_lander

        env = make_lander(self.config, "default")
        successes = timeouts = 0
        lengths: list[int] = []
        fuels: list[float] = []
        try:
            for episode in range(self.episodes):
                observation, _ = env.reset(seed=self.seed + episode)
                length = 0
                while True:
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, _reward, terminated, truncated, info = env.step(action)
                    length += 1
                    if terminated or truncated:
                        both_legs = bool(observation[6] and observation[7])
                        success = bool(
                            info.get("landed", False)
                            and both_legs
                            and abs(float(observation[0])) <= self.center_tolerance
                        )
                        successes += int(success)
                        timeouts += int(bool(truncated))
                        lengths.append(length)
                        fuels.append(float(info.get("fuel_used", 0.0)))
                        break
        finally:
            env.close()

        rate = successes / self.episodes
        mean_length = sum(lengths) / len(lengths)
        mean_fuel = sum(fuels) / len(fuels)
        self.logger.record("eval/success_rate_100", rate)
        self.logger.record("eval/mean_episode_length_100", mean_length)
        self.logger.record("eval/mean_fuel_used_100", mean_fuel)
        self.logger.record("eval/timeout_rate_100", timeouts / self.episodes)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.output_dir / "success_evaluations.csv"
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("timesteps", "episodes", "success_rate", "mean_episode_length", "mean_fuel_used", "timeout_rate"),
            )
            if metrics_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow({
                "timesteps": self.num_timesteps,
                "episodes": self.episodes,
                "success_rate": rate,
                "mean_episode_length": mean_length,
                "mean_fuel_used": mean_fuel,
                "timeout_rate": timeouts / self.episodes,
            })
        if rate >= 0.9 and not self.recorded_target:
            (self.output_dir / "steps_to_90_success.txt").write_text(
                f"{self.num_timesteps}\n", encoding="utf-8"
            )
            self.recorded_target = True
        if self.verbose:
            print(
                f"100-episode success evaluation at {self.num_timesteps} steps: "
                f"success={rate:.1%}, mean length={mean_length:.1f}, mean fuel={mean_fuel:.1f}"
            )
        self.next_eval += self.eval_freq
        return True


def resolve_run_paths(
    config_name: str,
    seed: int,
    init_model: str | Path | None = None,
    name_override: str | None = None,
) -> tuple[str, Path]:
    """Resolve isolated run name and model directory for standard or curriculum runs."""
    if name_override:
        name = name_override
    elif init_model:
        init_p = Path(init_model)
        source_id = init_p.parent.name if init_p.stem in ("best_model", "latest") else init_p.stem
        name = f"{config_name}_from_{source_id}"
    else:
        name = config_name
    run_name = f"{name}_s{seed}"
    model_dir = Path("models") / run_name
    return run_name, model_dir


def create_or_load_model(
    model_class,
    config: dict,
    train_env,
    seed: int,
    init_model_path: str | Path | None = None,
    device: str = "cpu",
    tensorboard_log: str = "runs/",
    verbose: int = 1,
):
    """Instantiate a new model or initialize from an existing checkpoint with safety checks."""
    if init_model_path is not None:
        init_path = Path(init_model_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {init_path}")
        try:
            model = model_class.load(
                str(init_path),
                env=train_env,
                device=device,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
                **config.get("hparams", {}),
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load checkpoint {init_path}: {exc}") from exc

        if model.observation_space != train_env.observation_space:
            raise ValueError(
                f"Observation space mismatch: checkpoint has {model.observation_space}, "
                f"training env has {train_env.observation_space}"
            )
        if model.action_space != train_env.action_space:
            raise ValueError(
                f"Action space mismatch: checkpoint has {model.action_space}, "
                f"training env has {train_env.action_space}"
            )
        model.set_random_seed(seed)
        return model

    return model_class(
        config.get("policy", "MlpPolicy"),
        train_env,
        seed=seed,
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **config.get("hparams", {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, help="Override configured training steps")
    parser.add_argument("--init-model", type=str, default=None, help="Path to checkpoint (.zip) to warm-start policy")
    parser.add_argument("--name", type=str, default=None, help="Override experiment run name for output isolation")
    args = parser.parse_args()

    config = load_config(args.config)
    n_envs = int(config.get("n_envs", 1))
    if n_envs < 1:
        parser.error("n_envs must be at least 1")
    run_name, model_dir = resolve_run_paths(config["name"], args.seed, args.init_model, args.name)

    # Checkpoint output collision prevention
    if args.init_model:
        init_path = Path(args.init_model)
        if not init_path.is_file():
            parser.error(f"Supplied --init-model checkpoint does not exist: {init_path}")
        try:
            if init_path.resolve() == (model_dir / "best_model.zip").resolve() or init_path.resolve() == (model_dir / "latest.zip").resolve():
                parser.error(
                    f"Destination model directory {model_dir} would overwrite source checkpoint {init_path}. "
                    "Use a different --name or --seed to isolate output paths."
                )
        except OSError:
            pass

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

    model = create_or_load_model(
        model_class,
        config=config,
        train_env=train_env,
        seed=args.seed,
        init_model_path=args.init_model,
        device="cpu",
        tensorboard_log="runs/",
        verbose=1,
    )
    # Increase eval_episodes default from 20 to 50 to reduce best-model selection noise.
    # LunarLander landing has high episode-to-episode variance; 20 episodes is insufficient
    # to distinguish a 95%-success policy from an 80%-success policy reliably.
    n_eval_episodes = int(config.get("eval_episodes", 50))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval"),
        eval_freq=max(int(config.get("eval_freq", 10_000)) // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    latest_callback = LatestCheckpointCallback(
        model_dir / "latest", max(int(config.get("checkpoint_freq", 50_000)) // n_envs, 1)
    )
    success_callback = SuccessRateCallback(
        config,
        model_dir / "eval",
        eval_freq=int(config.get("success_eval_freq", 100_000)),
        episodes=int(config.get("success_eval_episodes", 100)),
        seed=args.seed + 20_000,
        center_tolerance=float(config.get("center_tolerance", 0.2)),
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps or int(config["timesteps"]),
            callback=CallbackList([eval_callback, latest_callback, success_callback]),
            tb_log_name=run_name,
            progress_bar=True,
        )
        model.save(str(model_dir / "latest"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
