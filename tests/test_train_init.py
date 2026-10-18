"""Unit tests for training initialization, curriculum transfer, and output isolation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from rocketrl.common import load_config, make_lander
from rocketrl.train import create_or_load_model, resolve_run_paths


class DummySpace:
    def __init__(self, shape=(8,)):
        self.shape = shape

    def __eq__(self, other):
        return hasattr(other, "shape") and self.shape == other.shape


class DummyEnv:
    def __init__(self, obs_shape=(8,), n_actions=4):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_actions)


class DummyModel:
    def __init__(self, obs_space=None, action_space=None):
        self.observation_space = obs_space or gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = action_space or gym.spaces.Discrete(4)
        self.seed = None

    def set_random_seed(self, seed: int):
        self.seed = seed


def test_resolve_run_paths_standard_run():
    run_name, model_dir = resolve_run_paths("ppo_sparse", seed=0)
    assert run_name == "ppo_sparse_s0"
    assert model_dir == Path("models/ppo_sparse_s0")


def test_resolve_run_paths_name_override():
    run_name, model_dir = resolve_run_paths(
        "ppo_sparse", seed=1, name_override="ppo_sparse_custom"
    )
    assert run_name == "ppo_sparse_custom_s1"
    assert model_dir == Path("models/ppo_sparse_custom_s1")


def test_resolve_run_paths_curriculum_auto_isolation():
    init_path = Path("models/ppo_shaped_s0/best_model.zip")
    run_name, model_dir = resolve_run_paths("ppo_sparse", seed=0, init_model=init_path)
    assert run_name == "ppo_sparse_from_ppo_shaped_s0_s0"
    assert model_dir == Path("models/ppo_sparse_from_ppo_shaped_s0_s0")
    # Output path is isolated and does not collide with the source checkpoint
    assert model_dir != init_path.parent


def test_create_or_load_model_standard_instantiation():
    mock_class = MagicMock(return_value="new_model_instance")
    env = DummyEnv()
    config = {"policy": "MlpPolicy", "hparams": {"n_steps": 512, "learning_rate": 1e-4}}

    model = create_or_load_model(
        mock_class,
        config=config,
        train_env=env,
        seed=42,
        init_model_path=None,
    )
    assert model == "new_model_instance"
    mock_class.assert_called_once_with(
        "MlpPolicy",
        env,
        seed=42,
        device="cpu",
        tensorboard_log="runs/",
        verbose=1,
        n_steps=512,
        learning_rate=1e-4,
    )


def test_create_or_load_model_missing_file_raises_error():
    mock_class = MagicMock()
    env = DummyEnv()
    config = {"hparams": {}}

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        create_or_load_model(
            mock_class,
            config=config,
            train_env=env,
            seed=0,
            init_model_path="nonexistent_checkpoint_file.zip",
        )


def test_create_or_load_model_observation_space_mismatch(tmp_path):
    checkpoint_file = tmp_path / "model.zip"
    checkpoint_file.write_text("fake_content")

    incompatible_model = DummyModel(
        obs_space=gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
    )
    mock_class = MagicMock()
    mock_class.load.return_value = incompatible_model

    env = DummyEnv(obs_shape=(8,))
    config = {"hparams": {}}

    with pytest.raises(ValueError, match="Observation space mismatch"):
        create_or_load_model(
            mock_class,
            config=config,
            train_env=env,
            seed=0,
            init_model_path=checkpoint_file,
        )


def test_create_or_load_model_action_space_mismatch(tmp_path):
    checkpoint_file = tmp_path / "model.zip"
    checkpoint_file.write_text("fake_content")

    incompatible_model = DummyModel(
        action_space=gym.spaces.Discrete(2)
    )
    mock_class = MagicMock()
    mock_class.load.return_value = incompatible_model

    env = DummyEnv(n_actions=4)
    config = {"hparams": {}}

    with pytest.raises(ValueError, match="Action space mismatch"):
        create_or_load_model(
            mock_class,
            config=config,
            train_env=env,
            seed=0,
            init_model_path=checkpoint_file,
        )


def test_curriculum_environment_preserves_sparse_reward_mode():
    config = load_config("configs/ppo_sparse.yaml")
    assert config["reward_mode"] == "sparse"

    env = make_lander(config, config["reward_mode"])
    try:
        assert env.mode == "sparse"
    finally:
        env.close()


def test_optimizer_state_restored_from_existing_checkpoint():
    """Verify that SB3 restores optimizer state when initializing from the shaped checkpoint."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    checkpoint_path = Path("models/ppo_shaped_s0/best_model.zip")
    if not checkpoint_path.is_file():
        pytest.skip("models/ppo_shaped_s0/best_model.zip not present")

    config = load_config("configs/ppo_sparse.yaml")
    train_env = make_vec_env(lambda: make_lander(config, "sparse"), n_envs=1)
    try:
        model = create_or_load_model(
            PPO,
            config=config,
            train_env=train_env,
            seed=7,
            init_model_path=checkpoint_path,
        )
        assert model.observation_space == train_env.observation_space
        assert model.action_space == train_env.action_space
        # Optimizer state should be restored and populated with parameter states
        opt_state = model.policy.optimizer.state_dict()
        assert "state" in opt_state
        assert len(opt_state["state"]) > 0
    finally:
        train_env.close()
