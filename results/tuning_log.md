# Phase 2 run and tuning log

All reported evaluations use the Gymnasium default reward and deterministic policies. Formal metrics use 100 episodes with reset seeds `10000` through `10099`; each training run uses seed 0.

## Baseline runs

| Run | Configured steps | Collected steps | Eval success | Mean reward | Notes |
|---|---:|---:|---:|---:|---|
| PPO default, seed 0 | 1,000,000 | 1,015,808 | 98% | 261.64 | Formal evaluation of `best_model.zip`; 1% crash and 1% timeout. First 100-episode callback evaluation above 90% was at 500k (96%). |
| PPO shaped, seed 0 | 1,000,000 | 1,015,808 | 99% | 259.13 | `c_time=0.05`, `b_center=20`; formal evaluation of `best_model.zip`; 1% crash, no timeouts. First callback above 90% was at 600k (94%). |

Mean episode length/fuel from formal evaluation:

- Default: `318.95` steps, `263.73` fuel units.
- Shaped: `334.40` steps, `273.94` fuel units.

## Observations

- Both default and shaped PPO met the 90% success and 200 mean-reward targets in their 100-episode evaluations.
- Shaped PPO had zero successes and 100% timeouts at its 500k callback evaluation, then reached 94% at 600k and 97% at 1M. This learning curve is noisy and shows the fixed budget matters.
- No reward weights have been tuned yet. Sparse and overshaped runs are pending.
