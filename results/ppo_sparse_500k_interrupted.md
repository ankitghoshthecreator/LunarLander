# Interrupted preliminary training run

Exact command: `.venv/Scripts/python.exe -m rocketrl.train --config configs/ppo_sparse.yaml --seed 0 --timesteps 500000`

This run was stopped with Ctrl+C after the displayed PPO rollout reached `total_timesteps | 360448`; the last displayed default-reward evaluation was at `Eval num_timesteps=360000`. It was not a completed 500000-step run and is excluded from `metrics_all.csv` and all aggregates. This file preserves the console values observed during that command.
