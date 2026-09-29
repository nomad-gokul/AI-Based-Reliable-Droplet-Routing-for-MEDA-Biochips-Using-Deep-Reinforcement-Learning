<div align="center">

# AI-Based Reliable Droplet Routing for MEDA Biochips

**Moving droplets across a lab-on-a-chip whose electrodes wear out with use,
using health-aware path planning and deep reinforcement learning.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Gymnasium](https://img.shields.io/badge/env-Gymnasium-0081a5)](medaroute/env.py)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3%20PPO-orange)](experiments/exp4_ppo.py)
[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

B.Tech Project · Department of Computer Science and Engineering ·
Netaji Subhas University of Technology (NSUT) · Supervisor: Dr. Ankur Gupta

</div>

---

## Key results at a glance

> [!IMPORTANT]
> **Reading electrode health pays off, and it matters more the more worn the chip is.**
>
> | | Plain A\* | Health-aware A\* | Gain |
> |---|:---:|:---:|:---:|
> | Success on a heavily worn chip (70% degradable electrodes) | 22.8% | **49.2%** | **2.2× more tasks completed** |
> | Tasks before the first failure on a fresh chip | 241 | **397** | **~65% longer chip life** |
> | Share of 3,000 back-to-back tasks completed | 47% | **72%** | **+25 points** |
>
> Even the health-aware router keeps declining as the chip wears out: it reacts to the
> current health but can't anticipate the wear its own routes cause. That gap is what the
> deep reinforcement learning router is meant to close.

> [!NOTE]
> **First deep RL agents already beat plain A\*.** Two PPO agents trained for 300k steps
> reach ~100% success during training and **68% / 66%** on 500 unseen test chips, vs **64%** for A\*.
> They don't yet beat health-aware A\* (**79%**); closing that gap is the next phase
> (see [Experiment 4](#experiment-4-first-ppo-agents)).

<p align="center">
  <img src="results/exp2_success.png" width="49%" alt="Success rate vs degradation level">
  <img src="results/exp3_lifetime.png" width="49%" alt="Chip lifetime">
</p>

---

## Contents

1. [The problem](#the-problem)
2. [Our approach](#our-approach)
3. [Repository layout](#repository-layout)
4. [Getting started](#getting-started)
5. [Results in detail](#results-in-detail)
6. [Simulator details](#simulator-details)
7. [Next steps](#next-steps)
8. [Authors](#authors)
9. [References and license](#references-and-license)

---

## The problem

**Microfluidic biochips** (labs-on-a-chip) run lab procedures such as diluting,
mixing and detecting, on a device the size of a coin. A **MEDA** (Micro-Electrode-Dot-Array)
biochip does this by moving tiny droplets across a grid of thousands of microelectrodes:
switching on the electrodes next to a droplet pulls it one step in that direction.

The catch is that **microelectrodes degrade the more they are used** (charge trapping).
A worn electrode may fail to move the droplet, so the move has to be retried. That wastes
time, and a bioassay has deadlines. A router that always takes the shortest path keeps
driving droplets over the same electrodes and wears out whole corridors of the chip.

**Goal:** route droplets so that tasks finish on time *and* the chip stays usable for as long as possible.

## Our approach

```mermaid
flowchart LR
    A["1. MEDA simulator<br/>Gymnasium env with<br/>use-dependent wear"] --> B["2. Classical baselines<br/>A* and health-aware A*"]
    B --> C["3. DRL router<br/>PPO + small CNN,<br/>with / without health map"]
    C --> D["4. Curriculum learning<br/>chip size and<br/>degradation level"]
```

| Router | What it optimises | Knows electrode health? |
|---|---|:---:|
| **A\*** | Fewest moves (every move costs 1) | ✗ |
| **Health-aware A\*** | Expected routing time: leaving a cell costs `1/p`, where `p` is the move-success probability, i.e. the expected number of actuation cycles | ✓ |
| **PPO (no health map)** | Learned policy; sees droplet, goal and blocked cells (Liang et al. style) | ✗ |
| **PPO (health map)** | Learned policy; also sees the electrode health map (Elfar et al. style) | ✓ |

A task **succeeds** if the droplet reaches its goal within a deadline of **1.5 × its shortest-path length**.

---

## Repository layout

```
.
├── medaroute/
│   ├── env.py                      MEDA chip simulator (Gymnasium environment)
│   └── routers.py                  A*, health-aware A* and an executor that runs a plan on the chip
├── experiments/
│   ├── exp1_visualize.py           one task: both routers' paths on a worn chip
│   ├── exp2_degradation_levels.py  success rate / routing time vs degradation level
│   ├── exp3_lifetime.py            success rate as one chip wears out over 3,000 tasks
│   └── exp4_ppo.py                 PPO agents with vs without the health map
├── results/                        figures and CSV files produced by the experiments
├── tests/                          pytest checks for the chip, its dynamics and the routers
├── run_experiments.ipynb           Google Colab notebook that runs everything
├── report/BTP_report_Final.pdf     mid-semester project report
├── CONTRIBUTORS.md
└── LICENSE
```

---

## Getting started

### Install

```bash
git clone https://github.com/nomad-gokul/AI-Based-Reliable-Droplet-Routing-for-MEDA-Biochips-Using-Deep-Reinforcement-Learning.git
cd AI-Based-Reliable-Droplet-Routing-for-MEDA-Biochips-Using-Deep-Reinforcement-Learning
pip install -r requirements.txt
```

### Check that everything works

```bash
python -m pytest        # 20 tests, under a second
```

### Reproduce the results

| Command | What it does | Time (4-core CPU) |
|---|---|---|
| `python experiments/exp1_visualize.py` | Draws both routers' paths on one worn chip | ~2 s |
| `python experiments/exp2_degradation_levels.py` | 500 chips × 5 degradation levels | ~15 s |
| `python experiments/exp3_lifetime.py` | 10 chips × 3,000 tasks each | ~1 min |
| `python experiments/exp4_ppo.py --timesteps 300000` | Trains and evaluates both PPO agents | ~12 min |

Figures and CSV files are written to `results/`. Every script takes command-line options
(chip size, degradation fraction, number of chips, deadline slack, …); run it with `--help` to list them.

**No local Python?** Open [`run_experiments.ipynb`](run_experiments.ipynb) in Google Colab
and run the cells top to bottom (select a T4 GPU runtime for Experiment 4).

### Use the simulator in your own code

```python
from medaroute.env import MEDARoutingEnv
from medaroute.routers import ROUTERS, run_router

env = MEDARoutingEnv(30, 30, frac_degradable=0.5, pre_age_max=15)
obs, info = env.reset(seed=0)          # obs: 4 x 30 x 30 (droplet, goal, blocked, health)
result = run_router(env, ROUTERS["Health-aware A*"])
print(result)                          # success, steps taken, planned path length
```

---

## Results in detail

All numbers below come from running the scripts in this repository with their default
settings. The raw numbers are in the CSV files in [`results/`](results/).

### Experiment 1: the two baselines on one task

<p align="center"><img src="results/exp1_paths.png" width="520" alt="Paths of A* and health-aware A*"></p>

A 30×30 chip where half the electrodes can degrade (red = worn, green = healthy).
Both routers need **23 moves**, but A\* (dashed) goes straight through weak electrodes
while health-aware A\* (solid) curves around them. Expected routing time drops from
**~33 to ~29 cycles (about 12% faster)** without a longer path.

### Experiment 2: success rate vs degradation level

500 pre-aged 30×30 chips per level, one random task each; both routers solve identical copies of each chip.

| Degradable electrodes | A\* success | Health-aware A\* success | Gain | A\* time (cycles) | Health-aware A\* time (cycles) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0%  | 1.000 | 1.000 | – | 14.6 | 14.6 |
| 10% | 0.994 | **1.000** | +0.6 pts | 15.4 | **14.8** |
| 30% | 0.918 | **0.970** | +5.2 pts | 17.5 | **16.3** |
| 50% | 0.634 | **0.848** | +21.4 pts | 19.2 | **18.6** |
| 70% | 0.228 | **0.492** | **+26.4 pts** | 18.2 | 19.8 |

<p align="center">
  <img src="results/exp2_success.png" width="49%" alt="Success rate vs degradation">
  <img src="results/exp2_steps.png" width="49%" alt="Routing time vs degradation">
</p>

- Both routers plan paths of almost the same length (14.2–14.6 moves), so the difference comes entirely from *where* the path goes.
- Plain A\* collapses as the chip degrades, from 91.8% success at 30% to 22.8% at 70%.
- At 70%, health-aware A\* looks slightly slower only because it also finishes the harder tasks that A\* fails.

### Experiment 3: chip lifetime

10 fresh 30×30 chips (50% degradable), each running 3,000 tasks between six fixed
reservoir/mixer sites, as a real bioassay would. Both routers get the same chips and task sequences.

| Router | Tasks before first failure | Overall success |
|---|:---:|:---:|
| A\* | 241 ± 43 | 0.47 ± 0.05 |
| **Health-aware A\*** | **397 ± 110** | **0.72 ± 0.04** |

<p align="center"><img src="results/exp3_lifetime.png" width="75%" alt="Rolling success rate over 3000 tasks"></p>
<p align="center"><img src="results/exp3_health.png" width="85%" alt="Electrode health after 3000 tasks"></p>

- A\* keeps using the same straight corridors between sites until they die; health-aware A\* spreads the wear over a larger area.
- Health-aware A\* still declines steadily: it reacts to current health but cannot plan for the wear its own routes will cause. **This is the main motivation for a learned (DRL) router.**
- "Tasks before first failure" varies a lot between chips (one unlucky failure ends the count), so overall success is the more stable metric.

### Experiment 4: first PPO agents

Two PPO agents (Stable-Baselines3, small 3-layer CNN) trained for 300k steps each on
12×12 chips with 50% degradable electrodes, a new random pre-aged chip every episode.
One agent sees the electrode health map; the other doesn't. All four methods are then tested
on the **same 500 unseen chips**.

| Method | Test success rate | Routing time of successful tasks (cycles) |
|---|:---:|:---:|
| A\* | 0.644 | 9.0 |
| **Health-aware A\*** | **0.788** | **8.8** |
| PPO without health map | 0.684 | 9.1 |
| PPO with health map | 0.662 | 9.1 |

<p align="center">
  <img src="results/exp4_learning_curve.png" width="49%" alt="PPO training curves">
  <img src="results/exp4_eval.png" width="49%" alt="Test success of all four methods">
</p>

- **Both agents learn to route:** training success climbs from ~0 to ~99–100% (left), and on unseen chips both beat plain A\*.
- **Neither beats health-aware A\* yet, and the health map hasn't helped so far.** The agent with the health channel
  learns more slowly (90% training success after ~160k steps vs ~100k), so it has less time left to refine its policy.
- The training step limit (3× shortest path) is looser than the test deadline (1.5×), and health affects the reward only
  indirectly through failed moves. These are the first things to change (see [Next steps](#next-steps)).
- With 500 test tasks the standard error is about ±2 percentage points and only one seed was trained, so the gaps
  between the PPO agents and A\* are not yet conclusive.

> Numbers are from a CPU run of `exp4_ppo.py --timesteps 300000`. RL training is not bit-for-bit reproducible across
> hardware, so a GPU run gives slightly different values (the report's Colab run got 0.690 and 0.650).

---

## Simulator details

The chip model follows Liang et al. (ICML 2021, [`tcliang-tw/meda-env`](https://github.com/tcliang-tw/meda-env)).
Their code needs TensorFlow 1 and the old `stable_baselines`, so it was reimplemented on Gymnasium.

| Aspect | Model |
|---|---|
| Droplet | Square block of microelectrodes (3×3 by default) that moves in 8 directions |
| Move success | Probability = mean health of the electrodes under the droplet; a failed move leaves it in place and is retried |
| Wear | A fraction of electrodes is degradable, each with a decay factor in [0.6, 1.0]; every 50 actuations its health is multiplied by that factor |
| Observation | 4 × H × W: droplet, goal, blocked cells, electrode health map (`observe_health=False` drops the health channel for the ablation) |
| Reward | +1 at the goal, −0.05 for a step that gets closer, −0.1 otherwise |

Differences from meda-env: one-microelectrode steps (finer MEDA control), exact goal overlap,
health updated after every step, optional pre-aged chips (`pre_age_max`) and fixed
reservoir/mixer sites (`task_mode="ports"`).

---

## Next steps

- [ ] Train PPO for 1–2 M steps with a tighter training deadline, over 3–5 seeds
- [ ] Add a health term to the reward (`degrade_penalty` is already supported by the simulator)
- [ ] Ablation: agent with vs without the health channel (Liang vs Elfar formulation)
- [ ] Curriculum learning over chip size and degradation level
- [ ] Evaluate the learned router on the chip-lifetime benchmark (Experiment 3)

---

## Authors

| Name | Roll number |
|---|---|
| Kashika Puri | 2023UCA1885 |
| Gokul Kumar | 2023UCA1919 |
| Mohammed Eshaan | 2023UCA1960 |

Supervised by **Dr. Ankur Gupta**, Department of Computer Science and Engineering, NSUT.
The full mid-semester report is in [`report/BTP_report_Final.pdf`](report/BTP_report_Final.pdf).
See also [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## References and license

- T.-C. Liang, J. Zhou, Y.-S. Chan, T.-Y. Ho, K. Chakrabarty and C.-Y. Lee, "Parallel droplet control in MEDA biochips using multi-agent reinforcement learning," *Proc. 38th ICML*, PMLR vol. 139, pp. 6588–6599, 2021.
- M. Elfar, Y.-C. Chang, H. H.-Y. Ku, T.-C. Liang, K. Chakrabarty and M. Pajic, "Deep reinforcement learning-based approach for efficient and reliable droplet routing on MEDA biochips," *IEEE Trans. CAD*, vol. 42, no. 4, pp. 1212–1222, 2023.
- The full reference list is in the [report](report/BTP_report_Final.pdf).

Released under the [MIT License](LICENSE).
