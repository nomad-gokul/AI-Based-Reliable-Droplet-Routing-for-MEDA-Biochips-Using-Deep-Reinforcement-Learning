# MEDA droplet routing: simulator and classical baselines

Preliminary work for the BTP "AI-based reliable droplet routing for MEDA biochips
using deep reinforcement learning" (NSUT, supervisor Dr. Ankur Gupta).

## Authors

* Kashika Puri
* Gokul Kumar
* Mohammed Eshaan

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

Supervisor: Dr. Ankur Gupta, Department of Computer Science and Engineering, NSUT.
The mid-semester report is in [`report/BTP_report_Final.pdf`](report/BTP_report_Final.pdf).

## What is here

```
medaroute/env.py        MEDA chip simulator (Gymnasium environment)
medaroute/routers.py    Plain A* and health-aware A*, plus an executor
experiments/exp1_visualize.py          one task, both routers' paths on a worn chip
experiments/exp2_degradation_levels.py success / routing time vs degradation level
experiments/exp3_lifetime.py           success rate as one chip wears out
experiments/exp4_ppo.py                PPO agents with vs without the health map
tests/                  pytest checks for the chip, its dynamics and the routers
run_experiments.ipynb   notebook that runs the experiments
report/                 BTP mid-semester report (PDF)
results/                figures and CSVs produced by the scripts (not committed)
```

## Setup and run

```bash
pip install -r requirements.txt
python experiments/exp1_visualize.py          # ~2 s
python experiments/exp2_degradation_levels.py # ~10 s
python experiments/exp3_lifetime.py           # ~1 min
```
Every script takes command-line options (`--help`), e.g. chip size, degradation
fraction, number of chips, deadline slack.

## Chip model

Based on Liang et al., ICML 2021 (`github.com/tcliang-tw/meda-env`), whose code
needs TensorFlow 1 and old `stable_baselines`, so it was reimplemented.

* Droplet = square block of microelectrodes (3x3 by default), 8 directions of motion.
* A move succeeds with probability = mean health of the electrodes under the droplet;
  a failed move leaves the droplet in place (the actuation is retried).
* A fraction of electrodes is degradable, each with a decay factor in [0.6, 1.0].
  Every 50 actuations its health is multiplied by that factor (charge trapping).
* Reward (for RL later): +1 at goal, -0.05 for a step closer, -0.1 otherwise.
* Observation: 4 x H x W — droplet, goal, blocked cells (empty for now), and the
  electrode health map (the Elfar et al., TCAD 2023 addition; `observe_health=False`
  removes it, for the ablation).

Differences from meda-env: one-microelectrode steps (finer MEDA control), exact
goal overlap, health updated every step, optional pre-aged chips (`pre_age_max`)
and fixed reservoir/mixer sites (`task_mode="ports"`).

## Baselines

* **A\***: every move costs 1; ignores electrode health.
* **Health-aware A\***: leaving a position costs 1/p (p = move-success probability),
  i.e. the expected number of actuation cycles, so it minimises expected routing time.

A task **succeeds** if the droplet reaches the goal within a deadline of
1.5 x its shortest-path length.

## Results (30 x 30 chip)

**Experiment 2** – 500 pre-aged chips per level, one random task each, both routers on the same chip/task:

| Degradable electrodes | A* success | Health-aware A* success |
|---|---|---|
| 0%  | 1.000 | 1.000 |
| 10% | 0.994 | 1.000 |
| 30% | 0.918 | 0.970 |
| 50% | 0.634 | 0.848 |
| 70% | 0.228 | 0.492 |

**Experiment 3** – 10 fresh chips (50% degradable), 3000 tasks each between 6 fixed sites:

| Router | Tasks before first failure | Overall success |
|---|---|---|
| A* | 241 ± 43 | 0.47 ± 0.05 |
| Health-aware A* | 397 ± 110 | 0.72 ± 0.04 |

Observations:
1. Using health information matters more as the chip degrades.
2. Plain A* wears out the same corridors; health-aware A* spreads the wear and lasts ~65% longer.
3. Even health-aware A* declines steadily: it reacts to current health but cannot
   anticipate the wear its own routes cause — the motivation for a learned (DRL) router.

## Next steps

* PPO + small CNN agent (Stable-Baselines3) on this environment.
* Ablation: agent with vs without the health channel (Liang vs Elfar formulation).
* Curriculum learning over chip size and degradation level.

## License

MIT, see [`LICENSE`](LICENSE).
