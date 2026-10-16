# Phase 1 environment smoke results

Run date: 2026-10-16

## Runtime

- Python 3.11.15, 64-bit Windows
- Gymnasium 1.2.3; Stable-Baselines3 2.7.0; PyTorch 2.14.1+cpu
- Box2D, pygame, TensorBoard, pandas, matplotlib, imageio, Gradio, and pytest imports succeeded.
- `pip check`: no broken requirements.
- `pytest`: 7 tests passed.

## Environment

- Observation shape: `(8,)`; fields are x, y, x velocity, y velocity, angle, angular velocity, left-leg contact, right-leg contact.
- Action space: `Discrete(4)`; no-op, left engine, main engine, right engine.
- Default `TimeLimit`: 1,000 steps.
- Reset info was empty for this Gymnasium version. Terminal info from `ShapedLander` contains `landed`, `crashed`, `base_reward`, x/y, speed, angle, touchdown metrics, fuel, and reward components.
- A human-render random-agent episode opened and ran; it crashed at step 59 with base reward `-100`.
- Three additional random episodes crashed with base reward `-100`.
- The built-in LunarLander heuristic landed in all 10 fixed-seed episodes, each with base reward `+100`.

## PPO compatibility

`ppo_default` completed with `--timesteps 20000` without errors. The 16 parallel environments and 1,024-step rollout caused PPO to collect 32,768 transitions before stopping. Evaluation ran on the default Gymnasium reward; at the 20,000-step evaluation it reported mean reward `-5.69` over 20 episodes. This is a compatibility sanity check, not a performance result.
