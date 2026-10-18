"""Generate the explicitly requested c_time and robustness configs."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "ablation_configs"
base = yaml.safe_load((ROOT / "configs" / "ppo_shaped.yaml").read_text(encoding="utf-8"))
for value, tag in ((0.0, "0"), (0.01, "001"), (0.2, "2")):
    config = {**base, "name": f"ppo_shaped_ctime_{tag}", "weights": {**base["weights"], "c_time": value}}
    (OUTPUT / f"ppo_shaped_ctime_{tag}.yaml").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"ppo_shaped_ctime_{tag}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

for name, env in (("wind", {"enable_wind": True}), ("gravity_3_7", {"gravity": -3.7})):
    config = {**base, "name": f"ppo_shaped_{name}", "env": env}
    (OUTPUT / f"ppo_shaped_{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
