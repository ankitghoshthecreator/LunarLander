# 🚀 Rocket Lander RL — Reward-Shaped Autonomous Landing

Train a reinforcement-learning agent to land a rocket on a pad using an open-source physics simulator (OpenAI Gym → **Gymnasium**, Box2D engine), compare reward-shaping strategies, and watch the agent learn and land **in real time**. 100% open source.

> **One-line pitch:** "Same rocket, same algorithm, four different reward functions — see why reward design decides whether the agent learns to land or learns to cheat."

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [Core concepts](#2-core-concepts)
3. [Tech stack (chosen)](#3-tech-stack-chosen)
4. [Alternatives and when to pick them](#4-alternatives-and-when-to-pick-them)
5. [Architecture](#5-architecture)
6. [Repository structure](#6-repository-structure)
7. [Environment details](#7-environment-details)
8. [Reward shaping design](#8-reward-shaping-design)
9. [Training details](#9-training-details)
10. [Evaluation and metrics](#10-evaluation-and-metrics)
11. [Real-time visualization](#11-real-time-visualization)
12. [Deployment](#12-deployment)
13. [Setup and quickstart](#13-setup-and-quickstart)
14. [Core code skeletons](#14-core-code-skeletons)
15. [Experiments to run](#15-experiments-to-run)
16. [Troubleshooting](#16-troubleshooting)
17. [Stretch goals](#17-stretch-goals)
18. [References](#18-references)
19. [License](#19-license)

---

## 1. What this project is

| Item | Value |
|---|---|
| Task | Land a rocket softly, upright, on a marked pad |
| Simulator | Gymnasium `LunarLander-v3` (Box2D physics) |
| Learning method | Deep reinforcement learning (PPO primary, DQN/SAC for comparison) |
| Key technique | **Reward shaping** (incl. potential-based shaping) |
| Output | Trained model, learning curves, ablation results, live viewer, deployed demo |
| Cost | Free — runs on a laptop CPU |

**Why this is a good project:** it is small enough to finish fast, visual enough to demo in 30 seconds, and it teaches the single most important practical RL lesson: *the agent optimizes exactly what you reward, not what you meant.*

---

## 2. Core concepts

### 2.1 Reinforcement learning loop
Agent observes state → picks an action (fire engines) → simulator advances physics → returns next state + reward → repeat until the episode ends (landed, crashed, or flew out of bounds).

### 2.2 Sparse vs. shaped reward
- **Sparse reward:** +1 only on a perfect landing, 0 otherwise. Random exploration almost never lands, so the agent gets no learning signal. Training stalls.
- **Shaped reward:** gives continuous feedback every step (closer to the pad = better, slower = better, upright = better, less fuel = better). Learning is much faster.

### 2.3 Potential-based reward shaping (the theory-safe version)
Ng, Harada and Russell (1999) showed that adding `F(s, s') = γ·Φ(s') − Φ(s)` to the reward, where `Φ` is any function of state, **does not change the optimal policy**. Naive shaping can. This is why you can design `Φ` freely (distance, speed, angle) and not break the task. Non-potential shaping (like a flat fuel penalty) is still useful but can shift the optimum, so treat it as a design choice and test it.

### 2.4 Reward hacking (what you'll observe)
Common failure modes your experiments should demonstrate:
- **Hovering:** agent learns to hover near the pad forever because crashing costs more than waiting. Fix: small per-step time penalty.
- **Crash-fast:** if shaping rewards are tiny and the crash penalty is small, the agent may end episodes quickly.
- **Fuel hoarding:** too large a fuel penalty makes the agent free-fall and crash.
- **Off-pad landing:** landing anywhere counts unless you reward pad centering.

### 2.5 On-policy vs. off-policy
- **PPO** (on-policy): stable, forgiving, great default.
- **DQN** (off-policy, discrete actions): sample-efficient on this task.
- **SAC** (off-policy, continuous actions): best for the continuous-thrust variant.

---

## 3. Tech stack (chosen)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10–3.12 | Standard for RL |
| Simulator API | **Gymnasium** (maintained successor to OpenAI Gym) | Original Gym is no longer maintained; Gymnasium is the drop-in replacement |
| Physics engine | **Box2D** (via `gymnasium[box2d]`) | Built into LunarLander, fast, CPU-only |
| RL library | **Stable-Baselines3** (PPO, DQN, SAC) | Clean, well-tested, TensorBoard built in |
| Deep learning | **PyTorch** | SB3 backend |
| Hyperparameter reference | **RL Baselines3 Zoo** (tuned configs) | Known-good starting hyperparameters for LunarLander |
| Rendering (desktop) | **pygame** (via Gymnasium `render_mode="human"`) | Real-time window, zero extra code |
| Training dashboards | **TensorBoard** | Live reward/loss curves |
| Analysis | pandas, numpy, matplotlib | Result tables and plots |
| Video capture | `gym.wrappers.RecordVideo` + imageio/ffmpeg | Shareable MP4/GIF |
| Web demo | **Gradio** (or Streamlit) | Python-only UI, free hosting |
| Hosting | **Hugging Face Spaces** | Free, public URL |
| Packaging | `venv` + `requirements.txt` (+ optional Dockerfile) | Reproducible |
| Config | YAML | One file per experiment |
| Testing | pytest | Reward-function unit tests |
| Version control | Git + GitHub | Portfolio-ready |

**Hardware note:** MLP policies on LunarLander are tiny. Train on **CPU** (`device="cpu"`); a small GPU (including a 4 GB card) gives little or no speedup for this workload. Use multiple CPU environments (`n_envs=8–16`) instead.

---

## 4. Alternatives and when to pick them

### 4.1 Simulators / environments

| Option | Pros | Cons | Pick when |
|---|---|---|---|
| **Gymnasium LunarLander (chosen)** | Fastest setup, 2D, huge community | 2D, "lunar" not "rocket" | You want it done ASAP |
| Community Falcon-style 2D landers (e.g., the `RocketLander` env from EmbersArc's Gym fork; check repo status first) | Looks like a SpaceX landing, has gimbal and fuel | Older code, may need porting to Gymnasium API | You want the rocket look and are OK porting |
| **Custom Box2D env (your own)** | Full control of physics and reward | More work | Stretch goal; best for portfolio |
| **PyBullet** | 3D physics, open source, Gym-style API | More setup, slower to learn in 3D | You want 3D |
| **MuJoCo** | Accurate, now open source (DeepMind) | Learning curve; rocket model must be built | Research-grade physics |
| **MuJoCo MJX / Brax (JAX)** | Thousands of parallel envs on GPU/TPU | JAX stack, overkill here | Massive-scale training |
| **Unity ML-Agents** | Beautiful 3D visuals | Unity/C# toolchain, heavy | Visual polish is priority |
| **Godot RL Agents** | Fully open-source game engine + RL | Smaller ecosystem | You want a game-engine front end |
| NVIDIA Isaac Gym/Sim | Industry GPU physics | Not fully open source, needs NVIDIA GPU | Not recommended here |
| Kerbal Space Program + kRPC | Real rocket game | Proprietary game | Not open source |

### 4.2 RL libraries

| Option | Pros | Cons | Pick when |
|---|---|---|---|
| **Stable-Baselines3 (chosen)** | Simplest, reliable, great docs | Less flexible internals | Default choice |
| CleanRL | Single-file, readable implementations | Less plug-and-play | You want to understand/modify the algorithm |
| Ray RLlib | Distributed, scalable | Heavy, complex config | Large-scale training |
| Tianshou | Fast, modular | Smaller community | Research |
| SBX (SB3 in JAX) | Faster SAC/TD3 | Newer | You want speed with SB3-like API |
| TorchRL | Official PyTorch RL | Steeper learning curve | You want PyTorch-native |
| Write PPO from scratch | Deepest learning | Slow, bug-prone | Learning exercise after MVP |

### 4.3 Algorithms

| Algorithm | Action space | Notes |
|---|---|---|
| **PPO (primary)** | Discrete or continuous | Stable default |
| DQN / Double DQN | Discrete | Very sample-efficient on LunarLander |
| A2C | Both | Faster but less stable |
| **SAC** | Continuous | Strong for continuous thrust |
| TD3 | Continuous | Alternative to SAC |
| Evolution strategies / CMA-ES | Both | Fun comparison, no gradients |
| Imitation + RL fine-tune | Both | Advanced stretch |

### 4.4 Visualization

| Option | Use |
|---|---|
| **pygame window (chosen, desktop)** | Real-time local viewing |
| TensorBoard (chosen) | Live training curves |
| Weights & Biases | Cloud experiment tracking (free tier, but SaaS rather than fully open source) |
| MLflow | Open-source self-hosted tracking |
| Streamlit / Gradio (chosen, web) | Interactive demo |
| matplotlib live plot | Lightweight in-process curves |
| Rerun | Modern open-source visual logging |
| Three.js / planck.js | Fully in-browser version |

### 4.5 Deployment

| Option | Pros | Cons |
|---|---|---|
| **Hugging Face Spaces + Gradio (chosen)** | Free, easy, public URL | CPU-only, may sleep when idle |
| Streamlit Community Cloud | Easy | Needs GitHub repo, resource limits |
| **ONNX + onnxruntime-web + planck.js** | Runs fully client-side, no server | Must reimplement the env in JS (stretch) |
| Docker + Render/Fly.io/Railway | Full control | More setup, may cost |
| GitHub Pages | Free static hosting | Needs the client-side approach above |

---

## 5. Architecture

```
┌──────────────┐   obs, reward   ┌────────────────────┐
│  Gymnasium   │ ───────────────▶│  RewardWrapper      │
│  LunarLander │◀─────────────── │  (shaping modes)    │
│  (Box2D)     │     action      └─────────┬──────────┘
└──────────────┘                           │
                                           ▼
                                  ┌────────────────────┐
                                  │  SB3 Agent (PPO)   │───▶ TensorBoard (live curves)
                                  │  n_envs parallel   │───▶ checkpoints/latest.zip
                                  └─────────┬──────────┘
                                            │
              ┌─────────────────────────────┼───────────────────────────┐
              ▼                             ▼                           ▼
     watch.py (pygame live)       evaluate.py (metrics)       app.py (Gradio demo)
```

Key idea: the **reward wrapper** is the only thing that changes between experiments. Environment, algorithm, seeds, and budget stay fixed so results are comparable.

---

## 6. Repository structure

```
rocket-lander-rl/
├── README.md
├── PRD.md
├── PHASES.md
├── LICENSE
├── requirements.txt
├── Dockerfile                  # optional
├── configs/
│   ├── ppo_default.yaml        # baseline reward
│   ├── ppo_sparse.yaml
│   ├── ppo_shaped.yaml         # your potential-based reward
│   ├── ppo_overshaped.yaml     # intentionally flawed (hover exploit)
│   └── dqn_shaped.yaml
├── src/rocketrl/
│   ├── __init__.py
│   ├── envs/
│   │   └── shaped_lander.py    # reward wrapper + success detection
│   ├── train.py                # CLI: --config configs/ppo_shaped.yaml --seed 0
│   ├── evaluate.py             # N-episode eval → metrics JSON/CSV
│   ├── watch.py                # live pygame viewer (follows latest checkpoint)
│   ├── record.py               # MP4/GIF recorder
│   ├── callbacks.py            # checkpoint + custom metric callbacks
│   └── utils.py
├── app/
│   └── app.py                  # Gradio web demo
├── notebooks/
│   └── analysis.ipynb          # plots and comparison tables
├── tests/
│   └── test_reward.py
├── runs/                       # TensorBoard logs (gitignored)
├── models/                     # saved models (small ones committed)
├── videos/
└── results/                    # metrics.csv, plots/*.png
```

### Implementation status

The repository features the complete RL pipeline:
- `ShapedLander` reward wrapper supporting `default`, `sparse`, `shaped`, and `overshaped` modes with step penalties and configurable timeout penalties.
- `train.py`: Full SB3 PPO training with TensorBoard logging, evaluation callbacks, and curriculum-transfer support (`--init-model`).
- `evaluate.py`: Multi-seed evaluation with Wilson score confidence intervals, touchdown velocity/angle telemetry, and CSV/JSON reporting.
- `watch.py`: Live Pygame visualizer with interactive top-left gravitation controls (Earth 9.8, Moon 1.62, Mars 3.71, etc.) and live Box2D physics manipulation.
- Test suite: 57 passing unit tests across `tests/test_reward.py`, `tests/test_evaluate.py`, and `tests/test_train_init.py`.

### Local setup

Use 64-bit Python 3.11 on Windows. Gymnasium 1.2.3 uses the `Box2D` distribution,
which provides a Windows wheel for Python 3.11; this avoids building the legacy
`box2d-py` extension with SWIG and a C++ compiler. From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,analysis,media,web,dev]"
python -m pip check
python -m pytest
```

Verify the installed stack and run one LunarLander step:

```powershell
python -c "import gymnasium, stable_baselines3; from Box2D import b2World; print('Gymnasium', gymnasium.__version__); print('SB3', stable_baselines3.__version__); print('Box2D import OK')"
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); obs, info = env.reset(seed=0); result = env.step(env.action_space.sample()); print('obs shape:', obs.shape, 'step tuple length:', len(result)); env.close()"
```

If the `.venv` already exists, activate it and run the install/check commands
above; do not recreate it unless its base Python installation is missing.

The wrapper can be used directly around `gym.make("LunarLander-v3")`:

```python
import gymnasium as gym
from rocketrl import ShapedLander

env = ShapedLander(gym.make("LunarLander-v3"), mode="shaped")
observation, info = env.reset(seed=0)
```

Reward configuration examples are in `configs/`. For shaped PPO runs, keep the
config's `hparams.gamma` equal to `weights.gamma` so the policy discount and
potential-shaping discount agree.

---

## 7. Environment details

**Gymnasium `LunarLander-v3`** (use `v3`; `v2` is deprecated in recent Gymnasium).

### Observation (8 floats)
| Index | Meaning |
|---|---|
| 0 | x position (pad center = 0) |
| 1 | y position (altitude, pad ≈ 0) |
| 2 | x velocity |
| 3 | y velocity |
| 4 | angle |
| 5 | angular velocity |
| 6 | left leg contact (0/1) |
| 7 | right leg contact (0/1) |

### Actions
- **Discrete (default, 4):** 0 = no-op, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine
- **Continuous (`continuous=True`):** 2 floats in [-1, 1] — main throttle and lateral throttle

### Useful constructor options
```python
gym.make("LunarLander-v3",
         continuous=False,
         gravity=-10.0,            # change for variants (e.g., -3.7 for Mars-like)
         enable_wind=False,
         wind_power=15.0,
         turbulence_power=1.5,
         render_mode=None)         # "human" | "rgb_array"
```

### Built-in reward (what you are replacing/compare against)
- Per-step shaping from distance to pad, speed, tilt, and leg contact (difference between consecutive steps)
- −0.3 per frame for main engine firing, −0.03 per frame for side engines
- **+100** when the lander comes to rest, **−100** when it crashes or leaves the screen
- +10 per leg touching the ground
- Episode counts as "solved" at roughly **200 average reward**

### Episode termination
Crash, lander at rest, or |x| ≥ 1 (flew off-screen). Add a `TimeLimit` (default 1000 steps) so hovering episodes end.

---

## 8. Reward shaping design

Implement all variants behind one wrapper with a `mode` flag so experiments are one-line changes.

### 8.1 Variants to implement

| Mode | Description | Purpose |
|---|---|---|
| `sparse` | 0 every step; +1 on successful landing, −1 on crash | Show why sparse fails |
| `default` | Gymnasium's built-in reward, unchanged | Baseline |
| `shaped` | Potential-based shaping + fuel penalty + time penalty + terminal bonuses | Your main contribution |
| `overshaped` | Large proximity reward, no time penalty | Demonstrate reward hacking (hovering) |
| `curriculum` (stretch) | Start with easy shaping, anneal toward terminal-only | Advanced |

### 8.2 The `shaped` reward (starting weights, tune these)

Potential function:
```
Φ(s) = −( w_dist·√(x² + y²) + w_vel·√(vx² + vy²) + w_ang·|angle| ) + w_leg·(left_leg + right_leg)
shaping_reward = γ·Φ(s') − Φ(s)
```
Additional terms:
```
fuel_penalty   = −c_main·[main engine fired] − c_side·[side engine fired]
time_penalty   = −c_time                      (per step)
terminal bonus = +B_land   if landed softly, upright, near pad center
terminal bonus = −B_crash  if crashed or out of bounds
center bonus   = +B_center·(1 − |x|)          (only on successful landing)
```
Suggested starting values:

| Param | Start | Notes |
|---|---|---|
| `w_dist` | 100 | Matches built-in scale |
| `w_vel` | 100 | Penalize speed near ground more (optionally scale by 1/(y+ε)) |
| `w_ang` | 100 | Keeps it upright |
| `w_leg` | 10 | Leg contact |
| `c_main` | 0.3 | Fuel |
| `c_side` | 0.03 | Fuel |
| `c_time` | 0.05 | Anti-hover |
| `B_land` | 100 | |
| `B_crash` | 100 | |
| `B_center` | 20 | |
| `γ` | match PPO gamma | Use the same discount in both places |

### 8.3 Detecting landing success
In `LunarLander`, the built-in step returns exactly `+100` when the lander comes to rest and `−100` on crash/out-of-bounds at the terminal step. The wrapper can read `terminated` plus that terminal base reward to label outcomes, then also check `|x|` and leg contacts for "landed on pad". Verify this against your installed Gymnasium version in a unit test.

### 8.4 Reward-shaping experiment rules
1. Change **one** reward component at a time.
2. Fix seeds and training budget across modes.
3. Log reward components separately (distance term, fuel term, etc.) so you can see what the agent is optimizing.
4. Always evaluate with the **same** metric (success rate), never with your shaped reward, since the shaped reward is not the true objective.

---

## 9. Training details

### 9.1 PPO starting hyperparameters (from RL Baselines3 Zoo for LunarLander; verify the current values in the Zoo repo)
```yaml
algo: PPO
policy: MlpPolicy
n_envs: 16
n_timesteps: 1_000_000
n_steps: 1024
batch_size: 64
n_epochs: 4
gamma: 0.999
gae_lambda: 0.98
ent_coef: 0.01
device: cpu
```

### 9.2 DQN starting hyperparameters (Zoo, discrete actions)
```yaml
algo: DQN
policy: MlpPolicy
n_timesteps: 100_000   # often needs more (300k–500k) for reliable landing
learning_rate: 6.3e-4
batch_size: 128
buffer_size: 50_000
learning_starts: 0
gamma: 0.99
target_update_interval: 250
train_freq: 4
gradient_steps: -1
exploration_fraction: 0.12
exploration_final_eps: 0.1
policy_kwargs: { net_arch: [256, 256] }
```

### 9.3 SAC (continuous variant)
Use the Zoo's `LunarLanderContinuous` config as a starting point; typically 300k–500k steps, `ent_coef: auto`, larger replay buffer.

### 9.4 Practical training tips
- **Seeds:** run at least **3 seeds** per variant; RL results are noisy.
- **Normalization:** observations are already roughly scaled; `VecNormalize` is optional.
- **Checkpoints:** save every ~50k steps to `models/<run>/` and update `latest.zip`.
- **EvalCallback:** evaluate every ~10k steps on 20 episodes, save the best model.
- **Wall-clock:** roughly 10–30 minutes per PPO run on a modern CPU with 16 envs (varies by machine).
- **Early stopping:** `StopTrainingOnRewardThreshold` (e.g., mean reward ≥ 250) saves time during iteration.

---

## 10. Evaluation and metrics

Evaluate each trained model over **100 episodes** with fixed evaluation seeds.

| Metric | Definition |
|---|---|
| **Success rate** | % episodes ending at rest, both legs down, near pad |
| Mean/median episodic reward | Using the built-in reward (common yardstick across variants) |
| Landing offset | Mean |x| at touchdown |
| Touchdown speed | Mean √(vx²+vy²) at contact |
| Touchdown angle | Mean |angle| at contact |
| Fuel used | Count/weighted sum of engine-on steps |
| Episode length | Detects hovering (very long episodes) |
| Crash rate / timeout rate | Failure breakdown |
| Sample efficiency | Steps to reach 90% success |

**Outputs:** `results/metrics.csv`, learning-curve plots (mean ± std over seeds), a comparison bar chart, and a table in this README.

**Target for "done":** ≥ 90% success rate and mean built-in reward ≥ 200 for the `shaped` PPO model; `sparse` should visibly fail; `overshaped` should show long episodes (hovering).

---

## 11. Real-time visualization

Three levels, build in this order:

1. **Live training curves:** `tensorboard --logdir runs/` → open `localhost:6006`, auto-refreshes while training.
2. **Interactive landing viewer (pygame):**
   ```powershell
   python -m rocketrl.watch --model models/ppo_shaped_s0/best_model.zip
   ```
   - **Interactive gravitation HUD (top-left):** Click the input box to type any gravitation value (e.g., `9.8` for Earth, `1.62` for Moon, `3.71` for Mars, `15.0` for heavy gravity) and press Enter to instantly update the live Box2D physics simulation.
   - **Preset buttons:** Instant toggles for `[Earth]`, `[Moon]`, `[Mars]`, and `[Default]`.
   - **Hotkeys:**
     - `Up` / `Down` arrows: Nudge gravitation by ±0.5 m/s² in real time.
     - `R`: Instantly restart episode from the top under current gravity.
     - `Space`: Pause / resume physics.
   - **Live training follow mode:** Add `--follow` to auto-reload the checkpoint when updated during training.
3. **Web demo (Gradio):** select a model (sparse / default / shaped / overshaped), click Run; frames stream from `rgb_array` rendering with a side panel showing altitude, speed, angle, fuel, and the running reward.

Overlay ideas on the viewer: reward breakdown text, trajectory trail, success/crash counter, side-by-side agents (e.g., sparse vs. shaped running simultaneously).

Headless note: for servers/CI set `SDL_VIDEODRIVER=dummy` and use `render_mode="rgb_array"`.

---

## 12. Deployment

### Option A (recommended): Hugging Face Spaces + Gradio
1. Put `app/app.py`, `requirements.txt`, and small model files (`.zip`, a few hundred KB each) in a Space repo.
2. `app.py` loads models with SB3, runs an episode in `rgb_array` mode, and **yields frames** to a Gradio `Image` component for a live feel (or returns an MP4).
3. Use a Python version with a published Box2D wheel (Python 3.11 is the project target); no local SWIG build is needed for this runtime.
4. Push; the Space builds and gives a public URL.

### Option B: Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SDL_VIDEODRIVER=dummy
CMD ["python", "app/app.py"]
```

### Option C (stretch): fully in-browser
Export the policy to **ONNX**, run it with `onnxruntime-web`, and reimplement the lander in JavaScript using **planck.js** (JS port of Box2D). Host on GitHub Pages, no server needed. Note the JS physics must match the Python env closely, or the policy will behave differently.

### Release checklist
- [ ] README has GIFs/MP4 of each variant
- [ ] Public demo link works
- [ ] Models and configs committed, seeds documented
- [ ] `pip install -r requirements.txt` works on a clean machine

---

## 13. Setup and quickstart

```powershell
# Run from D:\AI\Rocket with 64-bit Python 3.11 installed
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,analysis,media,web,dev]"
python -m pip check
python -m pytest

# Random-agent environment inspection (add --render human for the pygame window)
python -m rocketrl.smoke --episodes 3 --render human
```

Start a shaped PPO run and evaluate its best checkpoint with the common built-in
reward. Evaluation seeds default to `10000` through `10099`; override `--seed`
to reproduce another fixed set. A successful landing requires both legs down
and an absolute pad offset no greater than `0.2`.

```powershell
python -m rocketrl.train --config configs/ppo_shaped.yaml --seed 0
python -m rocketrl.evaluate --config configs/ppo_shaped.yaml --model models/ppo_shaped_s0/best_model.zip
```

Training writes periodic `latest.zip` checkpoints, the best evaluated policy,
TensorBoard data under `runs/`, and evaluation logs under the run's model
directory. Evaluation appends a summary row to `results/metrics.csv` by default.
Override the configured budget with `--timesteps 20000` for a short run.

---

## 14. Core code skeletons

### 14.1 `src/rocketrl/envs/shaped_lander.py`
```python
import numpy as np
import gymnasium as gym

DEFAULTS = dict(
    w_dist=100.0, w_vel=100.0, w_ang=100.0, w_leg=10.0,
    c_main=0.3, c_side=0.03, c_time=0.05,
    b_land=100.0, b_crash=100.0, b_center=20.0, gamma=0.999,
)

class ShapedLander(gym.Wrapper):
    """mode: 'sparse' | 'default' | 'shaped' | 'overshaped'"""

    def __init__(self, env, mode="shaped", weights=None):
        super().__init__(env)
        self.mode = mode
        self.w = {**DEFAULTS, **(weights or {})}
        self._prev_phi = 0.0

    def _phi(self, o):
        x, y, vx, vy, ang, angv, l, r = o[:8]
        w = self.w
        return (-(w["w_dist"] * np.hypot(x, y)
                  + w["w_vel"] * np.hypot(vx, vy)
                  + w["w_ang"] * abs(ang))
                + w["w_leg"] * (l + r))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, base_r, term, trunc, info = self.env.step(action)
        landed = term and base_r >= 100          # verify in tests for your version
        crashed = term and base_r <= -100
        info.update(landed=landed, crashed=crashed, base_reward=base_r,
                    x=float(obs[0]), speed=float(np.hypot(obs[2], obs[3])),
                    angle=float(abs(obs[4])))

        w = self.w
        if self.mode == "default":
            r = base_r
        elif self.mode == "sparse":
            r = 1.0 if landed else (-1.0 if crashed else 0.0)
        else:  # shaped / overshaped
            phi = self._phi(obs)
            r = w["gamma"] * phi - self._prev_phi
            self._prev_phi = phi
            if self.mode == "shaped":
                r -= w["c_time"]
                r -= w["c_main"] * (action == 2) + w["c_side"] * (action in (1, 3))
            if landed:
                r += w["b_land"] + w["b_center"] * max(0.0, 1.0 - abs(obs[0]))
            if crashed:
                r -= w["b_crash"]
        return obs, r, term, trunc, info
```

### 14.2 `src/rocketrl/train.py` (core)
```python
import argparse, yaml, gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from rocketrl.envs.shaped_lander import ShapedLander

def make_env(mode, weights, seed=0):
    def _f():
        env = gym.make("LunarLander-v3", max_episode_steps=1000)
        return ShapedLander(env, mode=mode, weights=weights)
    return _f

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    vec = make_vec_env(make_env(cfg["reward_mode"], cfg.get("weights")),
                       n_envs=cfg.get("n_envs", 16), seed=a.seed,
                       vec_env_cls=SubprocVecEnv)
    # Eval env uses the DEFAULT reward so all variants share one yardstick
    eval_env = make_vec_env(make_env("default", None), n_envs=1, seed=a.seed + 1000)

    name = f'{cfg["name"]}_s{a.seed}'
    algo = {"PPO": PPO, "DQN": DQN}[cfg["algo"]]
    model = algo(cfg.get("policy", "MlpPolicy"), vec, seed=a.seed, device="cpu",
                 tensorboard_log="runs/", verbose=1, **cfg.get("hparams", {}))
    cbs = [
        EvalCallback(eval_env, best_model_save_path=f"models/{name}",
                     eval_freq=max(10_000 // cfg.get("n_envs", 16), 1),
                     n_eval_episodes=20, deterministic=True),
        CheckpointCallback(save_freq=max(50_000 // cfg.get("n_envs", 16), 1),
                           save_path=f"models/{name}", name_prefix="ckpt"),
    ]
    model.learn(a.timesteps or cfg["timesteps"], callback=cbs,
                tb_log_name=name, progress_bar=True)
    model.save(f"models/{name}/latest")
```

### 14.3 `src/rocketrl/watch.py` (interactive viewer with gravity control)

Run with:
```powershell
python -m rocketrl.watch --model models/ppo_shaped_s0/best_model.zip [--gravity 9.8] [--follow] [--episodes 0]
```

Features:
- **Interactive Top-Left Gravitation HUD:** Text input box for on-the-fly gravity adjustments (`9.8` Earth, `1.62` Moon, etc.) affecting Box2D physics dynamically.
- **Preset Buttons:** Quick buttons for Earth, Moon, Mars, and Default gravity.
- **Keyboard Shortcuts:** `Up`/`Down` arrows adjust gravity by ±0.5, `R` restarts episode, `Space` toggles pause.
- **Checkpoint Following:** Optional `--follow` polls for model updates every ~5 seconds.

### 14.4 Example config `configs/ppo_shaped.yaml`
```yaml
name: ppo_shaped
algo: PPO
reward_mode: shaped
n_envs: 16
timesteps: 1000000
hparams:
  n_steps: 1024
  batch_size: 64
  n_epochs: 4
  gamma: 0.999
  gae_lambda: 0.98
  ent_coef: 0.01
weights:
  c_time: 0.05
  b_center: 20
```

### 14.5 `tests/test_reward.py` (what to test)
- Potential-based shaping is ~0 when the state doesn't change.
- Sparse mode returns exactly 0 on non-terminal steps.
- `landed`/`crashed` flags match the terminal ±100 base reward in the installed Gymnasium version.
- Wrapper does not mutate the underlying observation.

---

## 15. Experiments to run

| # | Experiment | Compare | Expected finding |
|---|---|---|---|
| 1 | Sparse vs default vs shaped | success rate, steps-to-solve | Sparse fails or is very slow; shaped is fastest/most reliable |
| 2 | Shaped vs overshaped | episode length, success | Overshaped hovers (long episodes, low success) |
| 3 | Time-penalty sweep (0, 0.01, 0.05, 0.2) | hover rate, fuel | Too low = hover, too high = rushed crashes |
| 4 | Fuel-penalty sweep | fuel used, success | Trade-off curve |
| 5 | PPO vs DQN (shaped) | sample efficiency | DQN may be more sample-efficient in discrete mode |
| 6 | Wind on (`enable_wind=True`), train vs test | robustness | Policies trained without wind degrade; train with wind to fix |
| 7 | Gravity change (Mars/Moon-like) | generalization | Shows overfitting to dynamics |
| 8 (stretch) | Curriculum shaping | speed + final success | Annealing helps remove shaping dependence |

Run **3 seeds each**. Report mean ± std.

---

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `box2d-py` builds from source or `swig.exe` fails on Windows | Confirm Gymnasium is 1.2.3 or later (`python -m pip show gymnasium`). Gymnasium 1.2.3 changed its Box2D extra to the `Box2D` distribution, which has prebuilt Windows wheels for Python 3.11. Use `python -m pip install -e ".[train,dev]"`; do not install the older Gymnasium 1.2.1 pin. |
| `LunarLander-v2 is deprecated` | Use `LunarLander-v3` |
| pygame window doesn't open on a server | Use `render_mode="rgb_array"` and `SDL_VIDEODRIVER=dummy` |
| `SubprocVecEnv` hangs on Windows | Wrap the entry point in `if __name__ == "__main__":` or use `DummyVecEnv` |
| Reward goes up but agent never lands | Reward hacking; check episode length and per-component logs |
| Training unstable | Lower learning rate, add more envs, check `gamma` consistency |
| Results differ run to run | Expected; use 3+ seeds and report mean ± std |
| Video recording fails | Install `imageio-ffmpeg` or `moviepy` |
| SB3/Gymnasium version conflict | Upgrade both together; check SB3 release notes for the supported Gymnasium range |

---

## 17. Stretch goals

- Custom Falcon-9-style environment (gimbal, fuel mass, drone-ship pad that moves)
- Continuous thrust + SAC
- Random pad position and starting conditions (domain randomization)
- Wind and sensor-noise robustness tests
- Curriculum learning on shaping weights
- Hyperparameter search with Optuna
- ONNX export + in-browser demo with planck.js
- Side-by-side live comparison of two agents
- Reward-component dashboard overlaid on the render
- Sim-to-real flavor: add action latency and observe the performance drop

---

## 18. References

- Gymnasium docs — Lunar Lander environment
- Stable-Baselines3 docs and RL Baselines3 Zoo (tuned hyperparameters)
- Ng, Harada, Russell (1999), *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Mnih et al. (2015), *Human-level control through deep reinforcement learning* (DQN)
- Haarnoja et al. (2018), *Soft Actor-Critic*
- Box2D and planck.js documentation
- Hugging Face Spaces and Gradio documentation

---

## 19. License

MIT (recommended). All dependencies listed here are open source; check each library's license before redistributing bundled assets.
#   L u n a r L a n d e r  
 