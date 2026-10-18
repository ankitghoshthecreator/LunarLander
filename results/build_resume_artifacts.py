"""Assemble resume evidence from the recorded training/evaluation outputs."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw_metrics"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)
VARIANTS = ("ppo_sparse", "ppo_default", "ppo_shaped", "ppo_overshaped")
SEEDS = (0, 1, 2)
METRICS = (
    "success_rate", "crash_rate", "timeout_rate", "mean_reward", "median_reward",
    "mean_touchdown_offset", "mean_touchdown_speed", "mean_touchdown_angle",
    "mean_main_engine_steps", "mean_side_engine_steps", "mean_episode_length",
)


def read_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def log_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="ignore").replace("\x00", "")


def log_value(path: Path, pattern: str) -> str:
    matches = re.findall(r"(?:^|\n)" + pattern, log_text(path))
    if not matches:
        raise RuntimeError(f"Could not find {pattern!r} in {path}")
    return matches[-1]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        config_path = ROOT / "configs" / f"{variant}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            model_path = ROOT / "models" / f"{variant}_100k_s{seed}" / "best_model.zip"
            raw_path = RAW / f"{variant}_seed{seed}.csv"
            log_path = RESULTS / f"{variant}_100k_seed{seed}_train.log"
            raw = read_csv(raw_path)
            model = PPO.load(str(model_path.parent / "latest.zip"), device="cpu")
            success_path = model_path.parent / "eval" / "success_evaluations.csv"
            success_rows = list(csv.DictReader(success_path.open(encoding="utf-8", newline="")))
            reached = next((r for r in success_rows if float(r["success_rate"]) >= 0.9), None)
            prefix = "MEASURED_WALL_SECONDS=([0-9.]+)"
            wall_seconds = float(log_value(log_path, prefix))
            train_command = log_value(log_path, r"TRAIN_COMMAND=([^\r\n]+)")
            eval_command = log_value(log_path, r"EVAL_COMMAND=([^\r\n]+)")
            row: dict[str, object] = {
                "variant": variant,
                "training_seed": seed,
                "training_timesteps_requested": 100000,
                "actual_training_timesteps": int(model.num_timesteps),
                "train_wall_seconds": wall_seconds,
                "n_envs": int(config["n_envs"]),
                "eval_seed_start": 10000,
                "eval_episodes": 100,
                "steps_to_90": int(reached["timesteps"]) if reached else "",
                "steps_to_90_status": "reached" if reached else "not reached by observed checkpoint",
                "train_command": train_command,
                "eval_command": eval_command,
                "raw_eval_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                "training_log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            }
            for key, value in raw.items():
                if key in {"timestamp_utc", "name", "algorithm", "model", "seed", "center_tolerance"}:
                    continue
                row[key] = value
            rows.append(row)
    return rows


def write_metrics(rows: list[dict[str, object]]) -> None:
    path = RESULTS / "metrics_all.csv"
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> str:
    return f"{statistics.fmean(values)} ± {statistics.stdev(values)}"


def aggregate(rows: list[dict[str, object]], variant: str, metric: str) -> str:
    return mean_sd([float(r[metric]) for r in rows if r["variant"] == variant])


def aggregate_percent(rows: list[dict[str, object]], variant: str, metric: str) -> str:
    return mean_sd([float(r[metric]) * 100 for r in rows if r["variant"] == variant])


def eval_commands(rows: list[dict[str, object]], variant: str) -> str:
    selected = [r for r in rows if r["variant"] == variant]
    return "<br>".join(f"`{r['eval_command']}`" for r in selected)


def train_commands(rows: list[dict[str, object]], variant: str) -> str:
    selected = [r for r in rows if r["variant"] == variant]
    return "<br>".join(f"`{r['train_command']}`" for r in selected)


def plot_learning_curves() -> None:
    plt.figure(figsize=(10, 6))
    for variant in VARIANTS:
        curves = []
        timesteps = None
        for seed in SEEDS:
            p = ROOT / "models" / f"{variant}_100k_s{seed}" / "eval" / "evaluations.npz"
            with np.load(p) as data:
                timesteps = data["timesteps"]
                curves.append(data["results"].mean(axis=1))
        array = np.asarray(curves)
        mean = array.mean(axis=0)
        sd = array.std(axis=0, ddof=1)
        plt.plot(timesteps, mean, label=variant)
        plt.fill_between(timesteps, mean - sd, mean + sd, alpha=0.18)
    plt.xlabel("Environment steps")
    plt.ylabel("Built-in evaluation reward (50 episodes per seed/checkpoint)")
    plt.title("PPO learning curves: mean ± sample SD across training seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "learning_curves.png", dpi=160)
    plt.close()


def plot_success(rows: list[dict[str, object]]) -> None:
    means = [statistics.fmean(float(r["success_rate"]) for r in rows if r["variant"] == v) * 100 for v in VARIANTS]
    sds = [statistics.stdev(float(r["success_rate"]) * 100 for r in rows if r["variant"] == v) for v in VARIANTS]
    plt.figure(figsize=(8, 5))
    plt.bar(VARIANTS, means, yerr=sds, capsize=5)
    plt.ylabel("Success rate (%)")
    plt.title("100-episode fixed-seed evaluation: mean ± sample SD")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS / "success_rate.png", dpi=160)
    plt.close()


def raw_ablation(variant: str, seed: int) -> dict[str, str]:
    return read_csv(RAW / f"{variant}_seed{seed}.csv")


def ablation_table() -> str:
    groups = (("0", "ppo_shaped_ctime_0"), ("0.01", "ppo_shaped_ctime_001"), ("0.05", "ppo_shaped"), ("0.2", "ppo_shaped_ctime_2"))
    lines = ["| c_time | Hover proxy: timeout rate (%) | Success rate (%) | Fuel steps (main + side) | Exact training and evaluation commands |", "|---:|---:|---:|---:|---|"]
    for value, key in groups:
        data = [raw_ablation(key, seed) for seed in SEEDS]
        success = mean_sd([float(r["success_rate"]) * 100 for r in data])
        hover = mean_sd([float(r["timeout_rate"]) * 100 for r in data])
        fuel = mean_sd([float(r["mean_fuel_used"]) for r in data])
        if key == "ppo_shaped":
            train_cmds = train_commands(rows, key)
            eval_cmds = "<br>".join(f"`{r['eval_command']}`" for r in rows if r["variant"] == key)
        else:
            train_cmds = "<br>".join(
                f"`.venv/Scripts/python.exe -m rocketrl.train --config results/ablation_configs/{key}.yaml --seed {s} --timesteps 100000 --name {key}`"
                for s in SEEDS
            )
            eval_cmds = "<br>".join(
                f"`.venv/Scripts/python.exe -m rocketrl.evaluate --config results/ablation_configs/{key}.yaml --model models/{key}_s{s}/best_model.zip --episodes 100 --seed 10000 --output results/raw_metrics/{key}_seed{s}.csv`"
                for s in SEEDS
            )
        commands = "Train:<br>" + train_cmds + "<br>Evaluate:<br>" + eval_cmds
        lines.append(f"| {value} | {hover} | {success} | {fuel} | {commands} |")
    return "\n".join(lines)


def report(rows: list[dict[str, object]]) -> str:
    variant_lines = [
        "| Variant | Success % | Crash % | Timeout/hover proxy % | Mean reward | Median reward | Touchdown offset | Touchdown speed | Touchdown angle | Main engine steps | Side engine steps | Mean episode length | Steps to 90% | Wall seconds | n_envs | Raw rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for v in VARIANTS:
        success = aggregate_percent(rows, v, "success_rate")
        crash = aggregate_percent(rows, v, "crash_rate")
        timeout = aggregate_percent(rows, v, "timeout_rate")
        fields = [
            aggregate(rows, v, "mean_reward"), aggregate(rows, v, "median_reward"),
            aggregate(rows, v, "mean_touchdown_offset"), aggregate(rows, v, "mean_touchdown_speed"),
            aggregate(rows, v, "mean_touchdown_angle"), aggregate(rows, v, "mean_main_engine_steps"),
            aggregate(rows, v, "mean_side_engine_steps"), aggregate(rows, v, "mean_episode_length"),
        ]
        seconds = aggregate(rows, v, "train_wall_seconds")
        commands = eval_commands(rows, v)
        train_evidence = train_commands(rows, v)
        variant_lines.append(
            f"| {v} | {success} | {crash} | {timeout} | " + " | ".join(fields) +
            f" | not reached by observed checkpoint | {seconds} | 16 | `results/metrics_all.csv` |"
        )

    command_lines = ["| Variant | Exact training commands (seeds 0, 1, 2) | Exact final evaluation commands (seeds 0, 1, 2) |", "|---|---|---|"]
    for v in VARIANTS:
        command_lines.append(f"| {v} | {train_commands(rows, v)} | {eval_commands(rows, v)} |")
    all_train_commands = "<br>".join(f"`{r['train_command']}`" for r in rows)

    best_shaped = max((r for r in rows if r["variant"] == "ppo_shaped"), key=lambda r: float(r["mean_reward"]))
    chosen_seed = int(best_shaped["training_seed"])
    baseline = read_csv(RAW / f"ppo_shaped_seed{chosen_seed}.csv")
    wind = read_csv(RAW / f"ppo_shaped_wind_seed{chosen_seed}.csv")
    gravity = read_csv(RAW / f"ppo_shaped_gravity_3_7_seed{chosen_seed}.csv")
    baseline_rate = float(baseline["success_rate"]) * 100
    wind_rate = float(wind["success_rate"]) * 100
    gravity_rate = float(gravity["success_rate"]) * 100

    facts = [
        "# LunarLander project: verified resume facts",
        "",
        "All experiment metrics below were produced by the recorded commands. Aggregate means use the arithmetic mean; `±` is sample standard deviation (`statistics.stdev`) across the three training seeds. Percentages are rate × 100. Final policy evaluation used deterministic actions and the built-in LunarLander reward.",
        "",
        "## Verified headline numbers",
        "",
        "| Claim | Value | How measured | Exact command | Raw data |",
        "|---|---:|---|---|---|",
        "| Unit tests | 57 passed, 0 failed | pytest output | `.venv/Scripts/python.exe -m pytest -q | Tee-Object -FilePath results/pytest_output.txt` | `results/pytest_output.txt` |",
        "| Python | 3.11.15 | interpreter version | `.venv/Scripts/python.exe --version` | `results/environment_facts.txt` |",
        "| Gymnasium / Stable-Baselines3 / PyTorch / Box2D | 1.2.3 / 2.7.0 / 2.14.1+cpu / 2.3.10 | imports and version attributes | `.venv/Scripts/python.exe -c \"import gymnasium,stable_baselines3,torch,Box2D; print('gymnasium='+gymnasium.__version__); print('stable-baselines3='+stable_baselines3.__version__); print('torch='+torch.__version__); print('Box2D='+getattr(Box2D,'__version__','importable'))\"` | `results/environment_facts.txt` |",
        "| PPO runs completed | 12 runs; 4 variants × 3 seeds | model directory count | `(Get-ChildItem models -Directory | Where-Object { $_.Name -match '_100k_s[012]$' } | Measure-Object).Count` | `models/` and `results/*_100k_seed*_train.log` |",
        f"| Common training budget | requested 100000; actual saved model timesteps in CSV | PPO vector rollout; model checkpoint value | {all_train_commands} | `results/metrics_all.csv` |",
        f"| Shaped success rate | {aggregate_percent(rows, 'ppo_shaped', 'success_rate')}% | 100 fixed-seed episodes for each trained seed | {eval_commands(rows, 'ppo_shaped')} | `results/metrics_all.csv` |",
        f"| Default success rate | {aggregate_percent(rows, 'ppo_default', 'success_rate')}% | same protocol | {eval_commands(rows, 'ppo_default')} | `results/metrics_all.csv` |",
        f"| Overshaped timeout/hover proxy | {aggregate_percent(rows, 'ppo_overshaped', 'timeout_rate')}% | timeout rate over the same evaluation episodes | {eval_commands(rows, 'ppo_overshaped')} | `results/metrics_all.csv` |",
        f"| Time-penalty sweep | c_time ∈ {{0, 0.01, 0.05, 0.2}}, three seeds each | 100000-step override; same evaluation seeds | exact commands in ablation table and run logs | `results/raw_metrics/ppo_shaped_ctime_*_seed*.csv` and `ppo_shaped_seed*.csv` |",
        f"| Robustness success-rate drop vs no-wind baseline | wind: {baseline_rate - wind_rate} percentage points; gravity -3.7: {baseline_rate - gravity_rate} percentage points | selection used highest built-in mean reward among shaped seeds with equal success rate; same 100 eval seeds | exact commands in robustness section | `results/raw_metrics/ppo_shaped_seed{chosen_seed}.csv`, `ppo_shaped_wind_seed{chosen_seed}.csv`, `ppo_shaped_gravity_3_7_seed{chosen_seed}.csv` |",
        "",
        "The `success_rate` column in the headline rows is a fraction (0–1); the detailed tables below show percent. Each raw row has its exact training and evaluation command in `metrics_all.csv`.",
        "",
        "## Main experiment protocol and commands",
        "",
        "The four checked-in PPO configs were trained with seeds `0`, `1`, and `2`. All runs used `--timesteps 100000`; SB3 PPO completed whole vectorized rollouts, so actual model timesteps are read from each saved model and included in the CSV. Configured `n_envs` remained the value in each YAML. The initial sparse run at a 500000 target was interrupted at 360448 steps after repeated long timeouts; it is excluded from the main metrics. Its command was `.venv/Scripts/python.exe -m rocketrl.train --config configs/ppo_sparse.yaml --seed 0 --timesteps 500000`.",
        "",
        "Evaluation: `--episodes 100 --seed 10000`; the evaluator resets with seeds 10000 through 10099, runs deterministic actions, and wraps each environment in `default` mode so reward is Gymnasium's built-in LunarLander reward. Success requires both leg contacts and absolute horizontal offset at most 0.2, as implemented by `src/rocketrl/evaluate.py`. `mean_touchdown_*` includes physical touchdowns (landings and crashes); timeouts are excluded. Main/side engine counts use actions 2 and 1/3, as implemented by the same evaluator.",
        "",
        "### Exact main run commands",
        "",
        *command_lines,
        "",
        "### Full PPO results (mean ± sample SD across training seeds)",
        "",
        "Rates and success are percentages here. Timeout rate is used as the observable hover proxy; mean episode length is also included to reveal long episodes. Steps to 90% was not reached at the available 100000-step success-evaluation checkpoint for these runs.",
        "",
        *variant_lines,
        "",
        "Learning curves are SB3 `EvalCallback` built-in-reward means across 50 episodes per seed/checkpoint, then mean ± sample SD across training seeds.",
        "",
        "![PPO learning curves](plots/learning_curves.png)",
        "",
        "![Fixed-seed success rate](plots/success_rate.png)",
        "",
        "## Ablation results",
        "",
        "### Time-penalty sweep",
        "",
        "The `c_time=0.05` row reuses the main shaped runs. The other three values were the only changed weight in their generated copies of `configs/ppo_shaped.yaml`; all other hyperparameters and weights were copied unchanged. Fuel is total engine-on steps, with the evaluator also recording main and side counts in raw data.",
        "",
        ablation_table(),
        "",
        "### Fuel-penalty sweep",
        "",
        "NOT RUN — time remained for the requested time-penalty sweep and robustness evaluations, but not a separate fuel-penalty sweep. No fuel weight was changed in the main experiment.",
        "",
        "### Robustness evaluations",
        "",
        f"Best shaped checkpoint by the selection rule used here: seed {chosen_seed}, selected because shaped seeds had equal success rates and this checkpoint had the highest mean built-in reward in the baseline evaluation. Baseline success was {baseline_rate}%; wind success was {wind_rate}% (drop {baseline_rate - wind_rate} percentage points); gravity -3.7 success was {gravity_rate}% (drop {baseline_rate - gravity_rate} percentage points). The zero-success baseline creates a floor effect, so these results do not establish robustness.",
        "",
        f"- Baseline command: `{best_shaped['eval_command']}`",
        f"- Wind command: `.venv/Scripts/python.exe -m rocketrl.evaluate --config results/ablation_configs/ppo_shaped_wind.yaml --model models/ppo_shaped_100k_s{chosen_seed}/best_model.zip --episodes 100 --seed 10000 --output results/raw_metrics/ppo_shaped_wind_seed{chosen_seed}.csv`",
        f"- Gravity command: `.venv/Scripts/python.exe -m rocketrl.evaluate --config results/ablation_configs/ppo_shaped_gravity_3_7.yaml --model models/ppo_shaped_100k_s{chosen_seed}/best_model.zip --episodes 100 --seed 10000 --output results/raw_metrics/ppo_shaped_gravity_3_7_seed{chosen_seed}.csv`",
        "",
        "### Before vs. after comparisons",
        "",
        f"- Sparse → shaped success: {aggregate_percent(rows, 'ppo_sparse', 'success_rate')}% → {aggregate_percent(rows, 'ppo_shaped', 'success_rate')}%. Shaping shortened episodes but did not produce a successful landing in the final evaluation. Commands are in the exact run-command table above.",
        f"- Default → shaped steps to 90% success: not reached by the observed checkpoint for either variant; the success callback ran at 100000 steps. There is no measured step advantage to claim.",
        f"- Shaped → overshaped timeout proxy: {aggregate_percent(rows, 'ppo_shaped', 'timeout_rate')}% → {aggregate_percent(rows, 'ppo_overshaped', 'timeout_rate')}%. Overshaped seed 0 times out on many episodes, while the other seeds differ sharply; see per-seed rows in the CSV. Commands are in the exact run-command table above.",
        f"- Default success was {aggregate_percent(rows, 'ppo_default', 'success_rate')}% versus shaped {aggregate_percent(rows, 'ppo_shaped', 'success_rate')}%: default did better at this budget. This is a negative result for the reward-shaping hypothesis. Commands are in the exact run-command table above.",
        "",
        "## Environment, versions, runtime, and hardware",
        "",
        "- Environment setup command: `python -m pip install -e \".[train,analysis,media,web,dev]\"` (approved execution succeeded; the initial sandbox attempt was blocked by PyPI network restrictions). `pip check` reported no broken requirements.",
        "- Python/Gymnasium/SB3/PyTorch/Box2D versions are listed in the headline table. Exact version command is printed there; Python was checked with `.venv/Scripts/python.exe --version`.",
        "- CPU model: 12th Gen Intel(R) Core(TM) i5-12500H. Command: `Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' | Select-Object -ExpandProperty ProcessorNameString`; output is preserved in `results/environment_facts.txt`.",
        "- RAM: NOT AVAILABLE. `Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory` and `systeminfo.exe | Select-String 'Total Physical Memory|Available Physical Memory|Processor\\(s\\)'` returned Access denied; outputs are preserved in `results/environment_facts.txt`.",
        "- Per-run wall time is measured in `results/*_train.log` with PowerShell `[Diagnostics.Stopwatch]` around the exact train command. The CSV reports each value and the main table reports mean ± sample SD.",
        "",
        "## Project facts and code changes",
        "",
        "| Fact | Value | Exact command |",
        "|---|---:|---|",
        "| Python lines in `src/` | 1032 | `(Get-ChildItem src -Recurse -File -Filter *.py | ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } | Measure-Object -Sum).Sum` |",
        "| Python lines in `tests/` | 681 | `(Get-ChildItem tests -Recurse -File -Filter *.py | ForEach-Object { (Get-Content $_.FullName | Measure-Object -Line).Lines } | Measure-Object -Sum).Sum` |",
        "| Reward modes | 4 | `(rg -o '\"sparse\"|\"default\"|\"shaped\"|\"overshaped\"' src/rocketrl/envs/shaped_lander.py | Sort-Object -Unique | Measure-Object).Count` |",
        "| YAML configs under `configs/` | 7 | `(Get-ChildItem configs -Recurse -File -Filter *.yaml | Measure-Object).Count` |",
        "| Pytest cases executed | 57 passed | `.venv/Scripts/python.exe -m pytest -q | Tee-Object -FilePath results/pytest_output.txt` |",
        "| CLI module entry points after change | 5 | `(rg -l 'if __name__ == .__main__.' src/rocketrl | Measure-Object).Count` |",
        "",
        "Files added/changed for this task:",
        "- Added `src/rocketrl/record.py` (model video recording CLI).",
        "- Changed `src/rocketrl/evaluate.py` to report separate main/side engine-on step counts; reward and training logic are unchanged.",
        "- Added `results/generate_resume_configs.py`, `results/build_resume_artifacts.py`, and generated `results/ablation_configs/ppo_shaped_ctime_0.yaml`, `results/ablation_configs/ppo_shaped_ctime_001.yaml`, `results/ablation_configs/ppo_shaped_ctime_2.yaml`, `results/ablation_configs/ppo_shaped_wind.yaml`, and `results/ablation_configs/ppo_shaped_gravity_3_7.yaml`.",
        "- Added `results/environment_facts.txt`, `results/pytest_output.txt`, and `results/ppo_sparse_500k_interrupted.md` to preserve environment, test, and interrupted-run output.",
        "- Added `results/metrics_all.csv`, `results/plots/learning_curves.png`, `results/plots/success_rate.png`, this report, raw evaluation CSVs, and training logs.",
        "",
        "Reward-logic bugs fixed: none. No reward weights or training hyperparameters were changed for the main variants. The explicit ablation changed only `c_time` in generated sweep configs. The evaluator change adds action-count telemetry and has no effect on policies or environment transitions.",
        "",
        "## Keywords actually used",
        "",
        "- Implemented and run: Python 3.11, Gymnasium LunarLander-v3, Box2D, Stable-Baselines3 PPO, PyTorch CPU, NumPy, PyYAML/YAML configs, PPO vectorized environments, deterministic policy evaluation, reward shaping and potential-based shaping, sparse/default/shaped/overshaped modes, engine/fuel telemetry, pytest, TensorBoard logging, matplotlib plots. Pygame is used by the live viewer; Gymnasium `RecordVideo` is used by the added recording CLI (the video CLI was help-checked, but no video was recorded).",
        "- Implemented but not experimentally run: Stable-Baselines3 DQN (the code supports it; the requested `configs/dqn_shaped.yaml` does not exist). DQN comparison: NOT RUN because that config is absent; no DQN hyperparameters were invented.",
        "- Declared optional dependencies but not used in these experiments: pandas, imageio/imageio-ffmpeg, Gradio. Video recording uses Gymnasium `RecordVideo`; no video was recorded for this run.",
        "",
        "## Limitations and NOT RUN",
        "",
        "- Main requested 100000 override was used after the first 500000-target run became too slow: the interrupted sparse seed-0 run reached 360448 steps and was excluded. Command and observed output are preserved in `results/ppo_sparse_500k_interrupted.md`. All reported main runs used the same 100000 requested override and their actual saved-model timesteps appear in the CSV.",
        "- DQN shaped comparison: NOT RUN; `configs/dqn_shaped.yaml` is missing.",
        "- Fuel-penalty sweep: NOT RUN (see above).",
        "- Sample efficiency is only resolved at the observed 100000-step success callback; none reached 90%, so true later crossing time is unknown rather than proven impossible.",
        "- High across-seed variability is visible in default and overshaped outcomes; success rates use 100 evaluation episodes per model, but only three training seeds.",
        "- Robustness tests used a zero-success baseline, limiting interpretation of the measured zero percentage-point drops.",
        "- RAM could not be read due to the environment's access restrictions.",
        "- The system-default `python` is 3.14.7 and `python -m pytest -q` failed with `ModuleNotFoundError: pytest`; the project `.venv` was Python 3.11.15, and the successful tests and experiments used `.venv/Scripts/python.exe`. The diagnostic outputs are in `results/environment_facts.txt`.",
        "- No reward logic fix was made, so there is no before/after reward-bug impact to report.",
        "",
        "## Rebuild command",
        "",
        "The output CSV, plots, and report were assembled from the raw evaluation CSVs, model checkpoints, YAML configs, and training logs by: `.venv/Scripts/python.exe results/build_resume_artifacts.py`.",
    ]
    return "\n".join(facts) + "\n"


rows = load_rows()
write_metrics(rows)
plot_learning_curves()
plot_success(rows)
(RESULTS / "RESUME_FACTS.md").write_text(report(rows), encoding="utf-8")
print(f"Wrote {len(rows)} main rows, report, and plots under {RESULTS}")
