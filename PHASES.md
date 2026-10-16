# PHASES — Rocket Lander RL Execution Plan

Four phases: **1 Requirements → 2 Training → 3 Visuals → 4 Deploy.**
Built for speed: the MVP (a working trained lander plus live viewer) is reachable by the end of Phase 3, Day 3. Phase 4 makes it public.

| Phase | Name | Est. time | Outcome |
|---|---|---|---|
| 1 | Requirements & Setup | 3–4 hrs | Env running, repo scaffolded, reward wrapper and tests done |
| 2 | Training | 1–1.5 days | Shaped agent lands ≥ 90%; ablations run; results table |
| 3 | Visuals | 0.5–1 day | Live viewer, recordings, local Gradio demo |
| 4 | Deploy | 0.5 day | Public demo, polished README, release |

Times are rough estimates for one developer; training runs happen in the background while you work on other tasks.

---

## Fast-track order (if you want it ASAP)

1. Phase 1 smoke test with a random agent (15 min)
2. Reward wrapper plus `ppo_default` training (start it immediately; it trains while you code the rest)
3. Launch `ppo_shaped`, `ppo_sparse`, and `ppo_overshaped` (one seed first; add seeds later)
4. While those train: build `watch.py` and TensorBoard
5. Record videos, build Gradio, deploy
6. Add extra seeds, DQN, and sweeps in the background; update the README table last

---

# PHASE 1 — Requirements & Setup

**Goal:** Lock the scope, get the simulator running, and build the reward wrapper so training can start right away.
**Duration:** 3–4 hours

## 1.1 Tasks

### A. Define requirements (30 min)
- [x] Read through `PRD.md`; freeze MVP scope (section 7.1)
- [x] Decide: discrete actions only for MVP (recommended)
- [x] Decide success definition: at rest, both legs down, |x| near pad center
- [x] Decide targets: ≥ 90% success, mean built-in reward ≥ 200
- [ ] Pick project name and create the GitHub repo (public, MIT license)

### B. Environment setup (45 min)
- [x] Install Python 3.10–3.12 and create a venv
- [x] Install `gymnasium[box2d]` (Gymnasium 1.2.3+ uses the Box2D wheel on Python 3.11; no local SWIG build is needed)
- [x] Install `stable-baselines3[extra]`, `torch`, `tensorboard`, `pygame`, `pandas`, `matplotlib`, `pyyaml`, `imageio`, `imageio-ffmpeg`, `gradio`, `pytest`
- [x] Run the random-agent smoke test with `render_mode="human"` (window opens, lander falls)
- [x] Run a 20k-step PPO sanity check to confirm the SB3 and Gymnasium versions work together
- [x] `pip freeze > requirements.txt`

### C. Scaffold the repo (30 min)
- [x] Create the initial package/config/test structure from README section 6
- [x] Add `.gitignore` (`runs/`, `.venv/`, `__pycache__/`, large model files)
- [x] Keep `README.md`, `PRD.md`, `PHASES.md` at the repository root
- [x] Add package metadata for editable installation (`pyproject.toml`)

### D. Understand the environment (30 min)
- [x] Print an observation and action space; confirm the 8-dim observation layout
- [x] Run a heuristic or random policy and log the terminal base rewards (confirm +100 landed, −100 crash)
- [x] Note `info` fields available and the default `TimeLimit` behavior

### E. Build the reward wrapper (60–90 min)
- [x] Implement `ShapedLander` with modes `sparse`, `default`, `shaped`, `overshaped` (README section 14.1)
- [x] Add `landed`, `crashed`, touchdown speed, angle, and x-offset to `info`
- [x] Support configurable weights through a mapping compatible with YAML `weights:` blocks
- [x] Log per-component rewards in `info` (distance, velocity, angle, fuel, time, terminal)
- [x] Write `tests/test_reward.py` for sparse rewards, potential invariance, terminal flags, and observation preservation
- [x] Write the YAML configs: `ppo_default`, `ppo_sparse`, `ppo_shaped`, `ppo_overshaped`

## 1.2 Deliverables
- Working venv and pinned `requirements.txt`
- Reward wrapper with passing tests
- 4 YAML configs
- Repo on GitHub with scaffolding and docs

## 1.3 Exit criteria
- Random-agent window opens
- `pytest` passes
- A 20k-step training run completes without errors
- You can state the shaped-reward formula from memory

## 1.4 Risks
| Risk | Fix |
|---|---|
| Box2D fails to install | Use Gymnasium 1.2.3+ with 64-bit Python 3.11 on Windows to install its prebuilt Box2D wheel; fall back to Docker if needed |
| SB3/Gymnasium mismatch | Upgrade both together, then pin |
| Terminal reward check fails | Print raw rewards and adjust `landed`/`crashed` logic for your version |

---

# PHASE 2 — Training

**Goal:** Train agents for each reward variant, reach ≥ 90% landing success with the shaped reward, and produce comparable results.
**Duration:** 1–1.5 days (mostly background compute)

## 2.1 Tasks

### A. Baseline first (immediately)
- [ ] Train `ppo_default` seed 0 (1M steps, 16 envs, CPU) and confirm it reaches roughly 200 mean reward
- [ ] Open TensorBoard and confirm curves look sane (reward rising, entropy decaying slowly)

### B. Train all reward variants (1 seed first)
- [ ] `ppo_sparse`: expect failure or very slow learning
- [ ] `ppo_shaped`: expect reliable landing
- [ ] `ppo_overshaped`: expect hovering or poor success
- [ ] Use the **default** reward in `EvalCallback` for every run so results are comparable
- [ ] Save `best` and `latest` models per run

### C. Tune the shaped reward
- [ ] If success < 90%, adjust in this order: more timesteps, `c_time`, `w_vel` near the ground, `b_center`, PPO `ent_coef` and learning rate
- [ ] Change one thing at a time and record each change in a log (`results/tuning_log.md`)
- [ ] Add a velocity-near-ground scaling term if landings are too hard (optional)

### D. Evaluation pipeline
- [ ] Implement `evaluate.py`: 100 episodes, fixed seeds, deterministic policy
- [ ] Compute: success rate, mean/median reward, touchdown speed/angle/offset, fuel used, episode length, crash rate, timeout rate
- [ ] Save to `results/metrics.csv`
- [ ] Add the callback that records steps-to-90%-success

### E. Scale up experiments
- [ ] Run seeds 0, 1, 2 for each variant (run in parallel if CPU cores allow)
- [ ] Sweep `c_time` ∈ {0, 0.01, 0.05, 0.2} (shows hovering vs rushing)
- [ ] Sweep fuel penalty (`c_main`) at 2–3 values
- [ ] Train DQN with the shaped reward for the PPO-vs-DQN comparison
- [ ] (Optional) Wind-on evaluation of the no-wind policy, then train with wind

### F. Analysis
- [ ] `notebooks/analysis.ipynb`: learning curves (mean ± std over seeds), success-rate bar chart, fuel vs success scatter, episode-length histogram (shows hovering)
- [ ] Write the key takeaways (3–5 bullets) for the README

## 2.2 Config reference (starting point)
```yaml
# configs/ppo_shaped.yaml
name: ppo_shaped
algo: PPO
reward_mode: shaped
n_envs: 16
timesteps: 1000000
hparams: {n_steps: 1024, batch_size: 64, n_epochs: 4, gamma: 0.999, gae_lambda: 0.98, ent_coef: 0.01}
weights: {c_time: 0.05, b_center: 20}
```

## 2.3 Deliverables
- Trained models for all variants (≥ 1 seed MVP; 3 seeds target)
- `results/metrics.csv`
- Learning-curve and comparison plots
- `results/tuning_log.md`
- Documented sparse failure and overshaped hovering

## 2.4 Exit criteria
- `ppo_shaped` ≥ 90% success and ≥ 200 mean built-in reward over 100 episodes
- `sparse` and `overshaped` outcomes recorded
- Metrics CSV and at least one comparison plot generated

## 2.5 Risks
| Risk | Fix |
|---|---|
| Shaped agent plateaus below 90% | Longer training; raise `b_land`; adjust `c_time`; use Zoo hyperparameters exactly |
| Overshaped doesn't hover | Remove time penalty entirely and increase proximity weight |
| Noisy results | More seeds; report mean ± std |
| CPU too slow | Reduce `n_envs`, shorten to 500k steps for iteration, run full length only for final |

---

# PHASE 3 — Visuals

**Goal:** Make the learning and landing visible in real time, and produce shareable media.
**Duration:** 0.5–1 day (start while Phase 2 runs are training)

## 3.1 Tasks

### A. Live training curves (20 min)
- [ ] `tensorboard --logdir runs/`; confirm live refresh during training
- [ ] Log custom scalars: success rate, mean episode length, fuel per episode

### B. Live landing viewer, desktop (1–2 hrs)
- [ ] Implement `watch.py` with `render_mode="human"`
- [ ] Add `--follow` to reload `latest.zip` periodically so you can watch it improve while training runs in another terminal
- [ ] Add a CLI flag to pick the model, seed, and wind on/off
- [ ] Overlay text: episode count, landings vs crashes, running reward (use pygame font or `cv2.putText` on `rgb_array`)
- [ ] (Optional) trajectory trail drawn behind the lander

### C. Recording media (45 min)
- [ ] `record.py` using `RecordVideo` or manual frame capture with imageio
- [ ] Record one success clip per variant (sparse, default, shaped, overshaped)
- [ ] Record "before/after" clips: early checkpoint (crashes) vs final (lands)
- [ ] Convert to GIFs (small size) for the README; keep MP4s for the demo
- [ ] Capture a TensorBoard screenshot and learning-curve PNG

### D. Web UI, local (2–3 hrs)
- [ ] `app/app.py` with Gradio:
  - dropdown: choose agent (sparse / default / shaped / overshaped)
  - button: Run episode(s)
  - output: streamed frames (Python generator yielding `rgb_array` frames) plus a telemetry panel (altitude, speed, angle, fuel, reward)
  - results table tab showing `metrics.csv`
- [ ] Set `SDL_VIDEODRIVER=dummy` and `render_mode="rgb_array"` (no window needed)
- [ ] Cap frame rate (~20–30 FPS) and frame size to keep streaming smooth
- [ ] Fallback: render the whole episode to MP4 and display it if streaming is laggy

### E. Optional extras
- [ ] Side-by-side view of two agents (e.g., sparse vs shaped) running the same seed
- [ ] Reward-component bar chart updating live
- [ ] Wind toggle in the UI

## 3.2 Deliverables
- `watch.py` live viewer with `--follow`
- TensorBoard dashboard
- MP4/GIF per variant plus before/after clips
- Local Gradio app

## 3.3 Exit criteria
- You can watch an agent improve live while training runs
- Every variant has a recorded clip
- Local Gradio app runs an episode end to end and shows telemetry

## 3.4 Risks
| Risk | Fix |
|---|---|
| pygame window issues | Use `rgb_array` and render in Gradio/matplotlib instead |
| Gradio streaming laggy | Lower resolution/FPS; fall back to pre-rendered MP4 |
| Large GIF files | Reduce resolution, FPS, and duration; use MP4 where possible |

---

# PHASE 4 — Deploy

**Goal:** Put it online for free, polish the repo, and release.
**Duration:** 0.5 day

## 4.1 Tasks

### A. Prepare the deployable bundle (45 min)
- [ ] Pick the final model per variant (`best.zip`); keep them small (a few hundred KB each)
- [ ] Make `app/app.py` load models by relative path
- [ ] Trim `requirements.txt` to inference needs (gymnasium[box2d], stable-baselines3, torch CPU, gradio, numpy, imageio); consider the CPU-only torch build to cut build time
- [ ] Confirm the Space Python version has a published Box2D wheel (use Python 3.11; SWIG is only needed on Python 3.14+)
- [ ] (Optional) write the `Dockerfile` (README section 12) and test locally

### B. Deploy to Hugging Face Spaces (1 hr)
- [ ] Create a new Space (SDK: Gradio, free CPU hardware)
- [ ] Push `app.py`, `requirements.txt`, `packages.txt`, and the model files
- [ ] Watch the build logs; fix Box2D/swig or dependency issues
- [ ] Test: pick each agent, run an episode, check smoothness
- [ ] Add a README header for the Space with title, emoji, and a short description

### C. Polish the GitHub repo (1–1.5 hrs)
- [ ] Update `README.md`: demo link, hero GIF, results table, key findings, quickstart
- [ ] Add a short "Reward shaping lessons" section (3–5 bullets from your experiments)
- [ ] Verify the clean-install path: new venv, install, `pytest`, short training run, watch
- [ ] Add LICENSE (MIT), pin versions, tag `v1.0`
- [ ] Add repo topics: `reinforcement-learning`, `reward-shaping`, `gymnasium`, `stable-baselines3`, `box2d`, `ppo`

### D. Share (optional, 30 min)
- [ ] 30–60 second screen recording or GIF for LinkedIn/Instagram/X
- [ ] Short write-up: "Why my rocket learned to hover instead of land" (a strong hook)
- [ ] Post the demo link and repo

### E. Stretch deploy options (only after v1.0)
- [ ] Export to ONNX and verify the outputs match PyTorch
- [ ] Port the lander physics to planck.js and run the policy with `onnxruntime-web`
- [ ] Host on GitHub Pages (no backend)
- [ ] Add a Docker image for one-command local run

## 4.2 Deliverables
- Public demo URL
- Release-ready repo (`v1.0` tag)
- Final README with results and media
- (Optional) social post and write-up

## 4.3 Exit criteria
- Demo link loads and runs an episode in under ~30 s
- Clean-machine install works from the README alone
- Results table and clips are in the README

## 4.4 Risks
| Risk | Fix |
|---|---|
| Space build fails on Box2D | Pin Python 3.11 to use the published Box2D wheel; fall back to Docker SDK Space |
| Free tier sleeps | Note "may take ~30 s to wake" on the README |
| Demo too slow on shared CPU | Lower FPS/resolution; serve pre-rendered clips with a "run live" option |
| Model files too large | Keep only the best model per variant; avoid committing checkpoints |

---

## Cross-phase checklist

| Item | Phase |
|---|---|
| Pin dependency versions | 1 |
| Reward unit tests | 1 |
| Eval uses the default reward for every variant | 2 |
| ≥ 3 seeds before claiming results | 2 |
| Log reward components separately | 2 |
| Record before/after clips | 3 |
| Test clean install | 4 |
| README results table and demo link | 4 |

## Definition of done
- [ ] Shaped PPO ≥ 90% success over 100 episodes
- [ ] Sparse and overshaped failures documented with metrics and video
- [ ] Live viewer and TensorBoard working
- [ ] Public demo working
- [ ] Repo reproducible from scratch
- [ ] README complete
