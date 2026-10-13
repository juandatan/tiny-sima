# Tiny-SIMA

Tiny-SIMA is a small research project about building an agent that can play
Craftax, a 2D open-world survival game.

The agent will use:

- A vision-language model (VLM) to decide what to do next.
- A lightweight reinforcement-learning policy to carry out that decision.
- Structured memory to remember progress, discoveries, and failures.

The project is designed to train and run on one accessible GPU, such as an
RTX 4090, L40, or A100. Training, testing, and evaluation should also be
runnable from Jupyter notebooks.

## Research question

Can a small open-weight VLM improve long-horizon survival by selecting useful
skills, when a learned low-level policy estimates which skills are currently
possible and executes them?

## System overview

Tiny-SIMA has two levels:

1. **High-level planner**
   - Observes the game at important moments.
   - Reads the current objective and memory.
   - Chooses a skill such as `collect wood` or `craft stone pickaxe`.
   - Uses an open-weight VLM running in PyTorch.

2. **Low-level executor**
   - Receives the selected skill.
   - Takes game actions until the skill succeeds, fails, or times out.
   - Predicts how likely each skill is to succeed from the current state.
   - Uses PPO implemented in JAX, Flax, and Optax.

The planner selects from a fixed set of skills instead of producing arbitrary
instructions. This keeps its decisions grounded in actions that the executor
has learned.

The current Phase 3 implementation tests the executor in isolation. There is
not yet a VLM choosing skills. The training environment assigns one requested
skill to each environment slot at reset, guarantees that its initial
preconditions hold, and supplies the corresponding skill ID to the shared
policy at every action step. A new episode for the same requested skill starts
after success, timeout, or environment termination.

```text
Long-term objective
        |
        v
VLM planner + memory
        |
        v
Candidate skill
        |
        v
Skill success estimate
        |
        v
Goal-conditioned RL executor
        |
        v
Craftax actions and observations
        |
        +---- outcome and state changes ----> planner
```

## Example skills

The first version will use a small skill registry:

- Collect wood, stone, coal, and other resources.
- Find food or water.
- Restore health, hunger, or thirst.
- Place a crafting table or furnace.
- Craft tools and weapons.
- Explore an area.
- Fight or retreat from an enemy.
- Move between dungeon floors.

A planner decision will have a typed format similar to:

```json
{
  "skill": "collect",
  "target": "wood",
  "quantity": 2,
  "horizon": 64
}
```

## Planner and executor loop

The planner will not run at a fixed frame interval. It will choose a new skill
when:

- The current skill succeeds.
- The current skill times out.
- The executor believes the skill is no longer possible.
- The agent enters danger or needs food, water, or health.
- The environment changes in an important way.

The planner will consider both:

- **Usefulness:** Does this skill help with the long-term objective?
- **Feasibility:** Can the executor complete this skill from the current state?

## Memory

The planner will receive a small structured memory rather than an unlimited
free-form transcript. It may contain:

- The current long-term objective.
- Completed skills and achievements.
- Important inventory changes.
- Recent failures and their causes.
- Episode-specific discoveries, such as potion effects.

This will allow us to test whether memory helps the agent adapt during long
episodes.

## Technology choices

The project will use a hybrid stack:

- **JAX:** Craftax environments, PPO training, rollouts, and evaluation.
- **Flax and Optax:** Neural networks and optimization.
- **PyTorch:** Open-weight VLM inference and optional fine-tuning.
- **Jupyter:** Reproducible training and evaluation entry points.

JAX is the best fit for the high-frequency environment and RL loop because
Craftax is JAX-native. PyTorch has better support for current open-weight VLMs.
The two systems only need to communicate when selecting a new skill, so this
boundary should not be a performance bottleneck.

## Quick start

Use Python 3.11 or 3.12. Python 3.10 is not supported because the pinned Flax
version requires Python 3.11 or newer.

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

On a Linux machine with an NVIDIA GPU, install the CUDA-enabled JAX extra:

```bash
pip install -e ".[dev,notebook,cuda12]"
```

Run the automated checks:

```bash
pytest
pyright
ruff check .
```

Measure environment throughput:

```bash
tiny-sima-benchmark --num-envs 64 --num-steps 256
```

Run the short baseline:

```bash
tiny-sima-train --config configs/ppo_debug.yaml
```

Evaluate its final checkpoint:

```bash
tiny-sima-evaluate runs/ppo-debug/checkpoints/final
```

Train and evaluate the categorical goal-conditioned executor:

```bash
tiny-sima-train-skills --config configs/skill_ppo_debug.yaml
tiny-sima-evaluate-trained-skills \
  runs/skill-ppo-debug/checkpoints/final \
  --episodes-per-skill 64
```

Evaluate a trained executor on randomized local-navigation scenarios:

```bash
tiny-sima-evaluate-trained-skills \
  runs/skill-ppo-1m/checkpoints/final \
  --scenario local_navigation \
  --horizon 32 \
  --episodes-per-skill 128
```

Validate the persistent skill chain with the scripted oracle, then substitute
the learned executor:

```bash
tiny-sima-evaluate-chain --executor scripted
tiny-sima-evaluate-chain \
  --executor learned \
  --checkpoint runs/skill-ppo-1m/checkpoints/final
```

The larger Phase 1 configuration is available at `configs/ppo_1m.yaml`.
The longer Phase 3 configuration is available at `configs/skill_ppo_1m.yaml`.
Generated runs and checkpoints are written under `runs/` and are not tracked
by Git.

### JupyterHub

After pulling the repository, open
`notebooks/00_environment_smoke_test.ipynb` and select the Python kernel you
want to use. The first cell:

- Finds the repository root.
- Installs Tiny-SIMA into the active kernel.
- Installs the CUDA 12 JAX dependencies when `nvidia-smi` reports an available
  GPU.

Run all cells to reset, step, render, and benchmark Craftax. No terminal
installation is required for this notebook.

To override automatic GPU detection, set `TINY_SIMA_USE_CUDA=1` or
`TINY_SIMA_USE_CUDA=0` in the JupyterHub environment before starting the
kernel.

Jupyter notebooks reserve 60% of GPU memory for JAX by default. Override this
before starting the kernel when needed:

```bash
export TINY_SIMA_JAX_MEMORY_FRACTION=0.75
```

Keep only one JAX notebook kernel running during training. Shutting a notebook
tab does not necessarily stop its kernel.

### One-million-step baseline

The training notebook supports both the debug and 1M configurations:

1. Open `notebooks/01_train_baseline.ipynb`.
2. Restart its kernel so the JAX memory setting is applied before import.
3. Shut down other GPU-backed notebook kernels.
4. Set `RUN_1M = True` in the configuration cell.
5. Run all cells.

The 1M configuration uses 64 environments, 64 rollout steps, and optimistic
resets. It writes intermediate and final checkpoints under `runs/ppo-1m/`.

Training logs report two throughput values:

- `sps`: overall steps per second, including compilation.
- `steady_sps`: steps per second after the first compiled update.

Each training run also writes:

- `gpu_metrics.csv`: periodic `nvidia-smi` utilization, memory, and power
  samples.
- `gpu_summary.json`: peak and mean hardware measurements.
- `summary.json`: training results plus GPU and JAX allocator summaries.

`nvidia-smi` memory includes memory reserved by JAX. For estimating the
policy's actual memory requirement, use
`jax_memory.peak_bytes_in_use_mib` in `summary.json`.

After training, open `notebooks/02_evaluate_baseline.ipynb`, change
`run_name` to `ppo-1m`, and run all cells.

### Recorded Phase 1 baseline

The first one-million-step run completed on a single NVIDIA L40S:

| Measurement | Result |
| --- | ---: |
| Training steps | 1,048,576 |
| Training seed | 0 |
| Updates | 256 |
| Total time, including compilation | 132.8 seconds |
| Compile and first update | 48.0 seconds |
| Post-compilation throughput | 12,321.7 steps/second |
| Completed training episodes | 7,090 |
| Mean completed training return | 4.260 |

The final checkpoint was evaluated deterministically for 64 episodes with
seed 1:

| Measurement | Result |
| --- | ---: |
| Mean return | 3.647 |
| Return standard deviation | 1.499 |
| Minimum return | 0.100 |
| Maximum return | 7.100 |
| Mean episode length | 176.95 |

This is a single training seed and establishes the engineering baseline. It
is not yet a statistically robust comparison.

The instrumented repeat run produced similar throughput and measured hardware
use:

| Measurement | Result |
| --- | ---: |
| Post-compilation throughput | 12,129.8 steps/second |
| JAX peak live allocation | 172.1 MiB |
| JAX preallocated pool | 27,276 MiB |
| Peak live allocation as a fraction of the pool | 0.63% |
| Mean GPU utilization, including compilation | 39.9% |
| Mean power draw | 88.5 W |
| Peak power draw | 103.7 W |

The 27.8 GiB reported by `nvidia-smi` is almost entirely JAX's configured 60%
preallocation. The executor is compute-underutilized and has substantial room
for a larger environment batch or model.

### Executor scaling sweep

The completed L40S sweep produced:

| Environments | Batch size | Tail steps/second | Tail GPU use | JAX peak memory | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4,096 | 16,006 | 25.9% | 172.1 MiB | 1.00x |
| 128 | 8,192 | 31,339 | 28.2% | 173.5 MiB | 1.96x |
| 256 | 16,384 | 59,256 | 28.4% | 253.2 MiB | 3.70x |
| 512 | 32,768 | 112,383 | 28.3% | 481.6 MiB | 7.02x |
| 1,024 | 65,536 | 197,427 | 96.8% | 945.3 MiB | 12.33x |
| 2,048 | 131,072 | 318,774 | 99.5% | 1,897.6 MiB | 19.92x |
| 4,096 | 262,144 | 410,578 | 99.6% | 3,754.4 MiB | 25.65x |

GPU saturation begins around 1,024 environments. The 2,048-environment
configuration reaches about 78% of the maximum measured throughput with half
the environment batch and memory of 4,096. Doubling from 2,048 to 4,096 adds
only 29% more throughput, so 2,048 is the practical hardware-efficiency point
and 4,096 is the maximum-throughput point.

Open `notebooks/01_profile_executor_scaling.ipynb` and run all cells to
reproduce the sweep. Completed candidates are reused automatically. Each
candidate:

- Runs for 32 PPO updates.
- Uses the same rollout length and optimizer settings as the baseline.
- Runs in a fresh subprocess.
- Records cumulative and tail-window throughput, utilization, power, and JAX
  peak memory.

Results are written to `runs/executor-scaling/scaling_summary.json` and
`scaling_summary.csv`. This sweep compares hardware throughput only; changing
the PPO batch size may also change learning behavior.

We will not change the learning default based on throughput alone. At one
million total steps, 64 environments produce 256 PPO updates, while 512,
1,024, 2,048, and 4,096 environments produce only 32, 16, 8, and 4 updates.
The next learning experiment will compare 64, 256, and 512 environments over
equal environment steps and multiple seeds.

### Final learning-scale comparison

The final comparison ran nine isolated training jobs:

- Environment counts: 64, 256, and 512.
- Training seeds: 0, 1, and 2.
- Training budget: 1,048,576 environment steps per run.
- Evaluation: 64 deterministic episodes using the same evaluation seed.
- PPO hyperparameters: unchanged from the baseline.

| Environments | PPO updates | Mean training time | Evaluation return | Mean episode length |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 132.2 seconds | 3.636 ± 0.299 | 159.74 |
| 256 | 64 | 87.5 seconds | 1.293 ± 0.240 | 148.05 |
| 512 | 32 | 84.2 seconds | 0.803 ± 0.051 | 151.75 |

Increasing to 256 environments saved 34% wall-clock time but reduced mean
return by 64%. Increasing to 512 saved 36% wall-clock time but reduced return
by 78%. Tiny-SIMA therefore retains 64 environments as the learning default.
Larger batches remain useful for throughput-oriented rollout generation and
evaluation.

The comparison can be reproduced with
`notebooks/02_compare_learning_scaling.ipynb`. Aggregate JSON and CSV files
are written under `runs/learning-scaling-comparison/`.

## Development phases

### Phase 1: Environment and baseline

Goal: establish a correct, reproducible Craftax training setup.

Tasks:

- Create the Python project structure.
- Install and pin the required dependencies.
- Add Craftax environment smoke tests.
- Add a notebook that resets, steps, and renders the environment.
- Measure environment throughput and GPU memory use.
- Train or reproduce a small PPO-RNN baseline.
- Save checkpoints and training metrics.
- Add a notebook that evaluates a saved checkpoint.

Phase 1 is complete when:

- A fresh environment can run the setup instructions.
- Automated environment tests pass.
- The smoke-test notebook runs from top to bottom.
- A short baseline training run completes without errors.
- A checkpoint can be loaded and evaluated.
- Throughput, memory use, and baseline results are recorded.

### Phase 2: Skill benchmark

Goal: turn short Craftax objectives into a testable skill suite.

Tasks:

- Define the initial skill registry.
- Implement skill preconditions and success predicates.
- Build short skill episodes from valid starting states.
- Add sparse skill rewards and time limits.
- Evaluate scripted and random baselines.

Initial skills will focus on gathering resources, restoring basic needs,
placing structures, and crafting simple tools.

The first benchmark stage uses nearby-target scenarios. Each scenario starts
from a generated Craftax world, injects the minimum valid target and inventory
state, disables hostile spawning, and continues to use the real Craftax
transition logic. Episodes terminate on skill success, environment failure,
or a skill-specific time limit and use a one-shot sparse completion reward.

The second evaluation stage uses `local_navigation` scenarios. Targets are
randomly placed two to four tiles away inside the initial observation. The
generator clears a guaranteed route, randomizes the player's direction, and
adds irrelevant resource and structure distractors outside the route. A
shortest-path scripted policy must retain 100% success; this distinguishes
malformed scenarios from learned-policy generalization failures.
`place_table` is excluded because it has no visible destination to navigate
toward; it remains covered by the nearby-target benchmark.

### Phase 3: Goal-conditioned executor

Goal: train one policy that can perform multiple skills.

PPO minibatches are stratified so every minibatch contains every requested
skill, and losses are averaged across skills because this phase optimizes equal
competence over an externally balanced skill suite. Per-skill advantage
normalization is available experimentally but disabled by default: short
rollouts without any successful examples can otherwise amplify critic noise.
A later planner experiment may instead weight skills by the planner's real
request distribution.

Tasks:

- Condition the PPO-RNN policy on a skill ID.
- Train over a curriculum of feasible skills.
- Add a skill-success prediction head.
- Add optional learned skill termination.
- Compare categorical skill IDs with text embeddings.
- Evaluate on held-out world seeds and goal paraphrases.

### Phase 4: Hierarchical planner

Goal: connect the executor to an open-weight VLM.

The first Phase 4 harness composes skills before introducing the VLM. It uses a
persistent controlled world and the chain:

```text
collect wood -> collect wood -> place table
-> collect wood -> craft wood pickaxe -> collect stone
```

The planner ranks every registered candidate using:

```text
combined score = utility * feasibility * failure penalty
```

Utility is currently supplied by the scripted objective. Feasibility is an
explicit symbolic precondition and will later be replaced or augmented by a
calibrated executor success estimate. The VLM will replace the utility scorer
without changing the executor or evaluation loop.

The ordered sequence is private to the scripted baseline. Planner-visible
context contains only the objective and prior execution outcomes, and chain
success is determined from the resulting Craftax state. This allows a future
planner to discover a different valid ordering without being shown or graded
against the reference answer.

Unlike an open-loop plan, the harness replans after every skill outcome. It
records success, timeout, lost precondition, infeasibility, environment
termination, native return, candidate factors, and repeated-failure penalties.
This provides a typed feedback channel for planner memory and avoids silently
attributing executor failures to the planner.

Tasks:

- Define the planner input and output schemas.
- Add a small quantized VLM.
- Rank candidate skills using usefulness and predicted feasibility.
- Replan after success, failure, danger, or timeout.
- Compare scripted, VLM-only, feasibility-only, and combined planners.

### Phase 5: Memory and visual grounding

Goal: study how observations and memory affect planning.

Tasks:

- Add structured episodic memory.
- Test memory on episode-specific discoveries.
- Compare symbolic state, image plus inventory, and pixel-only inputs.
- Compare a single screenshot with a short frame history.
- Test held-out language instructions and paraphrases.

### Phase 6: Full evaluation

Goal: produce a clear and reproducible research result.

Tasks:

- Run long-horizon survival and crafting objectives.
- Run all important ablations over multiple random seeds.
- Measure success, survival, efficiency, and compute use.
- Create evaluation notebooks, plots, and videos.
- Document limitations and failed approaches.

## Evaluation

The main measurements will be:

- Skill success rate.
- Long-horizon objective success rate.
- Achievement coverage and total return.
- Survival duration.
- Invalid planner decisions.
- Number of planner calls per episode.
- Accuracy and calibration of skill-success predictions.
- Environment steps per second.
- Training time and peak GPU memory.

Important comparisons will include:

- Flat PPO versus hierarchical control.
- Scripted planner versus VLM planner.
- VLM-only planning versus feasibility-grounded planning.
- No memory versus structured memory.
- Symbolic observations versus visual observations.
- Skill IDs versus language-conditioned skills.

## Compute scope

The project should remain usable on a single GPU.

To keep this practical:

- Begin with symbolic Craftax observations.
- Use pixels only after the basic system works.
- Start with a small VLM.
- Keep the VLM frozen during the first experiments.
- Run the VLM only at skill boundaries.
- Use parameter-efficient fine-tuning only if prompting is insufficient.
- Report wall-clock time and memory for every major experiment.

## Intended claims and limitations

Tiny-SIMA is primarily a project about hierarchical planning, language
grounding, memory, and compute-efficient reinforcement learning.

It will not initially reproduce the full scope of SIMA:

- Craftax is a turn-based 2D environment, not a real-time 3D game.
- Its action space is simpler than keyboard-and-mouse control.
- Early experiments may give the executor symbolic observations.
- The VLM will select skills rather than directly generate every game action.

These limitations will be kept explicit. The goal is a small, rigorous system
whose components and results can be understood and reproduced.

## Planned repository structure

```text
tiny-sima/
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── 00_environment_smoke_test.ipynb
│   ├── 01_train_baseline.ipynb
│   ├── 01_profile_executor_scaling.ipynb
│   ├── 02_evaluate_baseline.ipynb
│   ├── 02_compare_learning_scaling.ipynb
│   ├── 03_evaluate_skills.ipynb
│   ├── 04_train_goal_conditioned.ipynb
│   └── 05_hierarchical_evaluation.ipynb
├── src/
│   └── tiny_sima/
│       ├── envs/
│       ├── goals/
│       ├── executor/
│       ├── planner/
│       ├── memory/
│       └── evaluation/
├── tests/
├── configs/
└── scripts/
```

## Current status

**Phases 1 through 3 are complete. Phase 4 planner scaffolding and persistent
skill-chain evaluation are in progress.**

Implemented:

- Python package and dependency configuration.
- Craftax environment creation, vectorization, rendering, and benchmarking.
- A recurrent PPO baseline with checkpoint and metric output.
- Checkpoint evaluation.
- Automated configuration, environment, model, checkpoint, and PPO tests.
- Command-line scripts and Phase 1 notebooks.
- Automatic `nvidia-smi` sampling and JAX allocator peak-memory reporting.
- An initial six-skill registry with preconditions, progress, success, and
  sparse-reward predicates.
- Randomized local-navigation evaluation with controlled distractors and
  guaranteed paths.
- Persistent SayCan-style skill ranking and closed-loop chain evaluation.
- Scripted and checkpoint-backed chain executors with per-decision traces.

Verified locally on October 8, 2026:

- A clean Python 3.11 environment installs successfully.
- All 9 automated tests pass.
- The environment, training, and evaluation notebooks run from top to bottom.
- A 16-environment CPU benchmark reached about 3,205 environment steps/second.
- The 8,192-step PPO smoke run completed in 11.6 seconds, including compilation.
- The saved checkpoint loaded and completed an independent 8-episode evaluation.

The short-run returns are smoke-test results only and are not evidence of
meaningful policy learning.

Optimized locally on October 9, 2026:

- Added optimistic vectorized resets for training and benchmarking.
- Improved the 16-environment CPU benchmark from about 3,205 to 4,587
  environment steps/second.
- Added post-compilation throughput reporting.
- Added shared-GPU memory controls to the notebooks.

Verified on an NVIDIA L40S on October 11, 2026:

- The one-million-step baseline completed in 132.8 seconds.
- Post-compilation throughput reached about 12,322 environment steps/second.
- A deterministic 64-episode evaluation reached a mean return of 3.647.
- The instrumented repeat peaked at 172.1 MiB of live JAX allocations.
- Overall GPU utilization averaged 39.9%, confirming that the baseline leaves
  substantial accelerator capacity unused.

Started on October 11, 2026:

- All 48 automated tests pass locally.
- Defined `collect_wood`, `collect_stone`, `drink_water`, `place_table`,
  `craft_wood_pickaxe`, and `craft_wood_sword`.
- Added repeatable state-delta success predicates instead of relying on
  one-time achievement flags.
- Added feasibility preconditions, normalized progress, and one-shot sparse
  rewards.

Completed on October 12, 2026:

- The three-seed learning comparison selected 64 environments as the default.
- Added nearby-target episodes for all six initial skills.
- Added sparse completion rewards, eight-step time limits, and batched reset
  and step functions.
- Added scripted and random skill baselines.
- Converted `notebooks/03_evaluate_skills.ipynb` into an executable Phase 2
  benchmark.

Started on October 12, 2026:

- Added a balanced multi-skill environment with automatic short-episode
  resets.
- Added a recurrent actor-critic conditioned on learned categorical skill
  embeddings.
- Added a separate goal-conditioned PPO trainer and checkpoint evaluator.
- Added debug and one-million-step configurations.
- Added `notebooks/04_train_goal_conditioned.ipynb` for notebook-based
  training and per-skill evaluation.
- Added macro and per-skill success metrics so short skills cannot dominate
  the reported result through more frequent episode resets.
- Added explicit failure categories for feasible timeout, lost preconditions,
  and unrelated environment termination.
- Added skill-stratified minibatches and optional per-skill advantage
  normalization. The latter is disabled by default after a debug ablation
  showed that it collapsed unsolved interaction skills.
- Added side-by-side deterministic and stochastic checkpoint evaluation.
- Added controlled action diagnostics for initial target-facing accuracy,
  follow-up interaction accuracy after an oracle turn, direct-action accuracy,
  target-action probability, and per-stage policy entropy.

The next step is to rerun the goal-conditioned baseline with balanced metrics,
then add progressively less-structured starting states.
