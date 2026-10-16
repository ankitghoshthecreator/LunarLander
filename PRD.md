# PRD — Rocket Lander RL (Reward-Shaped Autonomous Landing)

| Field | Value |
|---|---|
| Product name | Rocket Lander RL |
| Type | Open-source ML/RL portfolio project with live demo |
| Owner | Ankit |
| Status | Draft v1 |
| Target timeline | MVP in 2–3 days, full release in ~5 days |
| Related docs | `README.md` (technical reference), `PHASES.md` (execution plan) |

---

## 1. Summary

Build an open-source system that trains a reinforcement-learning agent to land a rocket in a physics simulator (Gymnasium + Box2D), uses **reward shaping** to make learning feasible, compares several reward designs, and lets anyone **watch the agent learn and land in real time**, both locally and in a public web demo.

## 2. Problem statement

- Landing a rocket from a sparse "success/fail" signal is nearly impossible for an RL agent: random exploration rarely succeeds, so there is nothing to learn from.
- Hand-designed shaped rewards fix this but can introduce **reward hacking** (e.g., hovering forever).
- Tutorials show a trained agent but rarely show **why** reward design matters or let a viewer compare designs side by side.

## 3. Goals

| ID | Goal |
|---|---|
| G1 | Train an agent that lands reliably (≥ 90% success over 100 episodes) |
| G2 | Implement and compare ≥ 4 reward variants (sparse, default, shaped, overshaped) |
| G3 | Demonstrate at least one concrete reward-hacking failure and its fix |
| G4 | Provide real-time visualization of both training progress and landing behavior |
| G5 | Ship a public, free, shareable demo plus a reproducible repo |
| G6 | Use only open-source tools and free hosting |
| G7 | Finish fast: usable MVP in ≤ 3 days |

## 4. Non-goals (v1)

- Real rocket hardware or high-fidelity aerospace modeling
- 3D physics (candidate stretch only)
- Large-scale distributed or GPU training
- Multi-agent or competitive scenarios
- Mobile apps
- Paid services or proprietary simulators

## 5. Target users and use cases

| User | Need | Use case |
|---|---|---|
| **Developer (Ankit)** | A polished, explainable RL project for portfolio and interviews | Show end-to-end ownership: env design, reward design, training, eval, deploy |
| **Recruiter / interviewer** | Understand the project in under a minute | Open the demo link, watch a landing, read the comparison table |
| **RL learner / student** | Learn reward shaping with runnable code | Clone, run, modify weights, observe results |
| **Open-source contributor** | Extend with new envs/algorithms | Add a config and a reward mode |

## 6. User stories

1. As a viewer, I can open a link and see a rocket land itself without installing anything.
2. As a viewer, I can switch between agents trained with different rewards and see how their behavior differs.
3. As a learner, I can run one command to train an agent and another to watch it improve live.
4. As a developer, I can change reward weights in a YAML file and rerun without editing code.
5. As a developer, I can see a table and plots comparing success rate, fuel, and sample efficiency across variants.
6. As a developer, I can see why an "overshaped" agent hovers rather than lands.
7. As a reviewer, I can reproduce results from a clean machine using documented steps and fixed seeds.

## 7. Scope

### 7.1 MVP (must ship)
- Gymnasium `LunarLander-v3` environment with reward wrapper (4 modes)
- PPO training pipeline with YAML configs and seeds
- Evaluation script with success-rate metrics
- Live pygame viewer and TensorBoard curves
- Recorded videos/GIFs of each variant
- README with results table

### 7.2 Release scope (should ship)
- Gradio web demo hosted on Hugging Face Spaces
- DQN comparison
- Time-penalty and fuel-penalty sweeps
- Unit tests for the reward wrapper
- Analysis notebook with plots

### 7.3 Stretch (could ship)
- Continuous-action SAC variant
- Wind and gravity robustness tests
- Custom Falcon-style environment
- Curriculum reward shaping
- Optuna hyperparameter search
- ONNX + planck.js in-browser demo

## 8. Functional requirements

### 8.1 Environment and reward

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Use Gymnasium `LunarLander-v3` with Box2D physics | Must |
| FR-2 | Provide a reward wrapper with modes `sparse`, `default`, `shaped`, `overshaped` selectable by config | Must |
| FR-3 | `shaped` mode uses potential-based shaping plus fuel penalty, time penalty, and terminal bonuses | Must |
| FR-4 | All reward weights configurable via YAML | Must |
| FR-5 | Wrapper exposes `landed`, `crashed`, touchdown speed, angle, and x-offset in `info` | Must |
| FR-6 | Support optional wind and gravity parameters via config | Should |
| FR-7 | Log reward components individually for analysis | Should |

### 8.2 Training

| ID | Requirement | Priority |
|---|---|---|
| FR-8 | Train with Stable-Baselines3 PPO using parallel environments | Must |
| FR-9 | CLI: `python -m rocketrl.train --config <yaml> --seed <n>` | Must |
| FR-10 | Save checkpoints periodically and keep `latest` and `best` models | Must |
| FR-11 | Evaluate during training on the **default** reward so all variants share a common yardstick | Must |
| FR-12 | Log to TensorBoard (reward, episode length, losses) | Must |
| FR-13 | Support DQN as a second algorithm | Should |
| FR-14 | Support multiple seeds per variant (≥ 3) | Should |
| FR-15 | Support SAC with continuous actions | Could |

### 8.3 Evaluation

| ID | Requirement | Priority |
|---|---|---|
| FR-16 | Evaluate a model over N=100 episodes with fixed seeds | Must |
| FR-17 | Report success rate, mean reward, touchdown speed/angle/offset, fuel, episode length, crash and timeout rates | Must |
| FR-18 | Export results to CSV and generate comparison plots (mean ± std across seeds) | Must |
| FR-19 | Report steps-to-90%-success (sample efficiency) | Should |

### 8.4 Visualization

| ID | Requirement | Priority |
|---|---|---|
| FR-20 | Live local viewer that renders the Box2D scene in real time | Must |
| FR-21 | Viewer can follow the latest checkpoint and reload it while training runs elsewhere | Must |
| FR-22 | Record MP4/GIF per variant | Must |
| FR-23 | Web UI to pick a trained agent and watch it land with streamed frames | Should |
| FR-24 | On-screen telemetry: altitude, speed, angle, fuel used, running reward, landing counter | Should |
| FR-25 | Side-by-side comparison of two agents | Could |

### 8.5 Deployment

| ID | Requirement | Priority |
|---|---|---|
| FR-26 | Public web demo hosted on a free tier (Hugging Face Spaces) | Should |
| FR-27 | `requirements.txt` that installs on a clean machine; Dockerfile optional | Must / Could |
| FR-28 | Small model files committed or hosted so the demo works without retraining | Must |
| FR-29 | README includes demo link, GIFs, results table, and quickstart | Must |

## 9. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | **Cost** | $0 (open-source tools, free hosting) |
| NFR-2 | **Hardware** | Trains on a laptop CPU; no GPU required |
| NFR-3 | **Training time** | One PPO run ≤ ~30 minutes on a modern CPU (approximate; hardware dependent) |
| NFR-4 | **Real-time feel** | Local viewer ≥ 30 FPS; web demo smooth enough to follow (≥ ~15 FPS) |
| NFR-5 | **Reproducibility** | Fixed seeds, pinned dependencies, documented commands |
| NFR-6 | **Portability** | Works on Windows, macOS, and Linux with Python 3.10–3.12 |
| NFR-7 | **Maintainability** | Config-driven experiments; reward logic isolated in one module with tests |
| NFR-8 | **Licensing** | All dependencies open source; project under MIT |
| NFR-9 | **Documentation** | A new user can run the quickstart in under 15 minutes |

## 10. Success metrics

| Metric | Target |
|---|---|
| Landing success rate (shaped PPO, 100 eval episodes) | ≥ 90% |
| Mean built-in reward (shaped PPO) | ≥ 200 |
| Sparse-reward baseline | Clearly worse (documented failure) |
| Overshaped baseline | Demonstrates hovering or other hacking (documented) |
| Seeds per variant | ≥ 3 |
| Public demo | Live and loads in < 30 s |
| Quickstart success | Clean-machine install works in < 15 min |
| Time to MVP | ≤ 3 days |

## 11. Constraints and assumptions

**Constraints**
- Open-source only; free hosting only.
- Single developer; timeline is short, so scope is cut aggressively toward the MVP.
- CPU-only training.

**Assumptions**
- Gymnasium `LunarLander-v3` and Stable-Baselines3 versions installed together are compatible (verified at setup in Phase 1).
- On Windows with Python 3.11, Gymnasium 1.2.3+ installs Box2D from its published wheel rather than compiling `box2d-py` locally.
- Default PPO hyperparameters (from RL Zoo) are a good starting point and need only light tuning.
- Hugging Face Spaces free tier can run Box2D plus SB3 inference for a demo.

## 12. Dependencies

| Dependency | Purpose | Risk |
|---|---|---|
| Gymnasium (Box2D extra) | Simulator | Platform-specific Box2D wheel availability |
| Stable-Baselines3, PyTorch | Training | Version drift |
| TensorBoard | Live curves | Low |
| pygame | Local rendering | Headless setup needed on servers |
| Gradio | Web demo | Low |
| Hugging Face Spaces | Hosting | Free tier sleeps when idle |
| imageio / ffmpeg | Video | Install friction |

## 13. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Box2D install fails | Medium | High | Pin Gymnasium 1.2.3+ and use a supported Python/platform wheel; fall back to Docker |
| Shaped agent doesn't reach 90% | Medium | Medium | Start from Zoo hyperparameters; sweep weights; add more timesteps |
| Reward hacking is not reproducible on demand | Medium | Low | Remove time penalty and boost proximity weight to provoke hovering |
| Results too noisy | High | Medium | ≥ 3 seeds, report mean ± std |
| Real-time web demo laggy | Medium | Medium | Lower frame size/FPS; fallback to pre-rendered MP4 |
| Scope creep | High | High | Hard-freeze MVP scope; stretch only after release |
| Version conflicts (SB3 vs Gymnasium) | Medium | Medium | Pin versions after first working install |
| Hugging Face Space build fails | Low | Medium | Pin a Python version with a published Box2D wheel and test Docker locally |

## 14. Acceptance criteria

The project is **done** when:
1. A fresh clone plus documented commands reproduces training and evaluation.
2. `shaped` PPO meets the success targets in section 10.
3. `sparse` and `overshaped` failure modes are demonstrated with metrics and video.
4. A live viewer and TensorBoard show progress in real time.
5. The public demo link works and lets a viewer select an agent and watch it land.
6. README contains results table, GIFs, quickstart, and the demo link.
7. Reward unit tests pass.

## 15. Milestones

| Milestone | Deliverable | Target |
|---|---|---|
| M1 — Requirements done | Env running, repo scaffolded, docs ready | Day 1 (morning) |
| M2 — Training works | Shaped PPO lands reliably; ablation runs launched | Day 1–2 |
| M3 — Visuals done | Live viewer, recordings, Gradio demo working locally | Day 3 |
| M4 — Deployed | Public demo + polished README | Day 3–5 |

## 16. Open questions

1. Plain `LunarLander` only, or build a custom Falcon-style environment as a stretch goal?
2. Discrete actions only for the MVP, or add continuous (SAC) early?
3. Is the in-browser (ONNX + JS physics) version worth the extra effort?
4. What demo name and branding to use for the public Space?
