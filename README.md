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

The goal-conditioned executor and persistent skill-chain harness are now
working. Phase 4 adds a frozen VLM that observes a rendered Craftax frame,
structured inventory and survival values, the objective, and recent skill
outcomes. It scores the fixed skill catalog once at each option boundary.
Symbolic preconditions independently filter infeasible choices before the
selected skill is passed to the JAX executor.

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
  This is three separate questions, answered separately: are the preconditions
  met, is a target actually in reach, and how reliably does this policy finish
  that skill?

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

To run the optional open-weight VLM planner:

```bash
pip install -e ".[dev,notebook,cuda12,vlm]"
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

Train from scratch on the static local-navigation distribution:

```bash
tiny-sima-train-skills --config configs/skill_ppo_local_debug.yaml
# After the debug run succeeds:
tiny-sima-train-skills --config configs/skill_ppo_local_1m.yaml
```

The local-navigation trainer uses randomized visible targets and controlled
distractors for the five spatial skills. `place_table` remains a balanced
nearby-target task because it has no destination to navigate toward.

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

Validate the complete rendered-state-to-grounded-decision VLM path before
running the longer planner evaluations:

```bash
pip install -e ".[vlm]"
tiny-sima-smoke-vlm --config configs/vlm_planner_qwen3_2b.yaml --seed 51
```

The same real-model check is available as an opt-in integration test:

```bash
TINY_SIMA_RUN_VLM_SMOKE=1 \
  pytest -q tests/test_vlm_backend_smoke.py
```

The larger Phase 1 configuration is available at `configs/ppo_1m.yaml`.
The longer Phase 3 configurations are available at
`configs/skill_ppo_1m.yaml` and `configs/skill_ppo_local_1m.yaml`.
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

The first Phase 4 harness composes skills in a persistent controlled world
using the chain:

```text
collect wood -> collect wood -> place table
-> collect wood -> craft wood pickaxe -> collect stone
```

The planner ranks every registered candidate using:

```text
combined score = utility * feasibility * failure penalty
```

Utility can be supplied by the scripted objective or by a frozen open-weight
VLM. The first VLM backend uses `Qwen/Qwen3-VL-2B-Instruct` through
Transformers. It emits one structured score for every candidate skill in a
single generation. The executor and chain loop are unchanged.

Feasibility is composed from two sources that answer different questions:

```text
feasibility = precondition * P(success | skill, reachable, steps to target)
```

Symbolic preconditions alone cannot ground the ranking. A resource
precondition asks whether a block exists anywhere on the 64x64 map, which is
true in essentially every state, so the Can term carried no information and
the planner ranked unreachable skills above reachable ones. Reachability
replaces that existence test with a bounded breadth-first search over walkable
tiles inside the observation window.

Reachability grades rather than vetoes. A target that is not in view is
unknown, not impossible: both the executor and the scripted oracle can walk
out of the observation window to find one. Hard-gating on view-limited
reachability dropped the four-episode scripted chain from 1.0 to 0.5 success
with the lost episodes terminating in `no_viable_skill`. Unreached targets are
therefore scored at `unknown_feasibility = 0.25`, which demotes an unreachable
favourite without refusing it when nothing nearer makes progress.

How reliably *this* policy finishes what is in reach is measured rather than
inferred. The first attempt tried to read it from the executor's own critic:
the skill environment pays one sparse reward on success and terminates there,
so

```text
V(s, skill) = E[gamma ** k | success] * P(success)
```

looks monotone in success probability with only its scale wrong, and isotonic
regression against observed outcomes should recover a probability. **That did
not work, and the measurement is reported here rather than buried.** Across 432
labelled probes the critic separates success at AUC 0.43 - below chance - with
per-skill AUCs near 0.5. The cause is the training distribution:
`local_navigation` always spawns a reachable target, so the critic never saw a
state where its skill was impossible and encodes *which* skill was requested
instead of whether the state admits it. Because the cross-skill relation
between value and success is decreasing while an isotonic fit must be
non-decreasing, the fit collapses onto the base rate and the Can term becomes a
multiplier varying by 1.15x - which cannot reorder candidates whose rank
utilities differ by 4x.

The same probes show where the signal is. Reachability splits success 0.70
against 0.03, and per-skill success runs from 0.17 (`place_table`) to 0.63
(`collect_wood`). `planner/competence.py` therefore tabulates success by
`(skill, reachable, distance band)` and shrinks each cell toward its parent by
pseudo-counts - distance toward the skill's reachable rate, the skill toward
the reachability rate, that toward the base rate - so a thin cell inherits an
estimate rather than reading 0 or 1. A floor keeps a measured 3% rate
selectable, for the same reason reachability grades rather than vetoes. The
critic path is kept as the ablation that produced the refutation, not as a
recommended Can term.

Both heads are fitted on one seed and scored on another. The in-sample ECE of
an isotonic fit is identically zero by construction, so a single-seed protocol
reports a perfect calibration number for a head carrying no information.

Probing is the expensive part of that measurement, so it runs batched on
device: one `jax.lax.while_loop` over the whole `(episode, skill)` batch per
boundary, with a fixed batch shape so the compiled probe is reused across
boundaries. The serial Python version spent 2187s on 432 rollouts, almost all
of it in per-step device synchronisation.

With the Can term grounded, the planner itself is the binding constraint: it
ranked an unreachable skill first at 74.5% of boundaries where a reachable one
existed, and no downstream factor can undo that. Output is schema-constrained,
so field order is a hard scaffold rather than a request - the default prompt
style hands the model `available_now` and `blocked_reason` per candidate and
makes it echo the availability list in a field declared *before*
`selected_skill`. The prompt that produced 74.5% is preserved byte for byte as
`PlannerPromptStyle.BASELINE` so the change is measured against it instead of
assumed to help. The echoed list is also recorded, which separates a perception
failure (wrong list) from an instruction-following failure (right list, then an
unavailable choice).

Measured on 48 prefix boundaries, the change took the ungrounded top choice
from 0.750 to 0.026, and with it the Can terms' rescue rate to zero - every
condition sharing the new prompt now scores the same. It also exposed two new
errors. The schema fixes the ranking's length but cannot make its entries
distinct, and the model filled the slots by cycling the skills it had just
reported as available: 10 of 48 responses failed as
`ranking_not_a_permutation`, every one with a correct `selected_skill`. Those
rankings are now repaired at parse time - each skill keeps its first position,
omitted skills follow in catalog order - and flagged, so the prefix evaluation
reports the strict-parse accuracy from the same generations. The other error
is at the last boundary, where the model picked `collect_wood` over an
available, claimed `collect_stone` in six of eight episodes. Catalog order
always lists `collect_wood` first, so a `vlm_shuffled_available` condition
lists the available skills in a seeded order and is otherwise identical; the
per-boundary table shows whether the error follows the position.

Measuring a prompt change is only useful if the loop is short, and generation
dominates it. The controlled-prefix evaluation therefore advances every episode
in lockstep, one boundary at a time, and decodes all of their decisions in one
constrained batch: `SayCanPlanner.select_many` collects the states, the scorer
renders and caches them per boundary, and `TransformersVLMBackend.generate_batch`
groups the requests by catalog and schema - a constrained generator owns one
logits processor, which allocates a guide per sequence - and decodes in chunks
of `max_batch_size`. Each episode keeps its own rollout key, so lockstep visits
exactly the states that sequential advancing would. Batching is optional at the
backend boundary: `generate_all` falls back to one call at a time for any
backend implementing only `generate`. The closed loop cannot batch this way,
because each decision depends on where the previous skill left that episode.
The oracle ignores the planner, so its rollout is collected once as a
`PrefixTrajectory` and every condition is scored on it, and the frame renderer
and the reachability search are compiled once rather than dispatched op by op
or recompiled per call - together roughly a second per boundary that every
condition paid, including those the response cache answered.

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

Fixed-plan learned-executor evaluations batch persistent worlds into one
compiled JAX state machine. Completed environments are masked while each
world tracks its current option, horizon, retries, and termination reason. The
generic Python evaluator remains available for arbitrary planners and detailed
per-decision traces.

Tasks:

- Define the planner input and output schemas. **Implemented.**
- Add a small frozen VLM backend. **Implemented; evaluation pending.**
- Rank candidate skills using usefulness and predicted feasibility.
  **Implemented: preconditions, reachability, and a measured per-skill
  competence prior. The critic-as-affordance variant was measured and
  retired.**
- Replan after success, failure, danger, or timeout.
- Compare scripted, VLM-only, feasibility-only, and combined planners.
  **Nine-condition ladder implemented in notebook 06, including the retired
  critic control, the baseline prompt, and a position-bias control; first GPU
  prefix run recorded, closed loop pending.**

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
│   ├── 05_hierarchical_evaluation.ipynb
│   └── 06_evaluate_vlm_planner.ipynb
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

**Phases 1 through 3 are complete. Phase 4 VLM integration and hierarchical
evaluation are in progress.**

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
- A model-independent VLM backend interface, Outlines-constrained skill
  rankings, strict validation, and one-call whole-catalog utility scoring.
- Planning-only controlled-prefix evaluation, separate from executor errors.
- Horizon-bounded reachability feasibility, graded so an out-of-view target is
  demoted rather than refused.
- Batched on-device skill probing that labels every candidate at a boundary
  with whether the executor actually finished it.
- A measured competence prior over `(skill, reachable, distance band)` with
  empirical-Bayes shrinkage, fitted on one seed and scored on another.
- Critic-as-affordance scoring with isotonic recalibration, retained as the
  ablation that measured its own failure at AUC 0.43.
- A schema-constrained planner prompt that reports skill availability before
  the decoder can name a choice, with the baseline prompt preserved for
  comparison.
- Batched constrained generation across episodes at a shared boundary, with a
  sequential fallback for backends that do not support it.
- Parse-time repair of rankings that repeat a skill, flagged per response so
  strict-parse accuracy is reported from the same generations.
- One oracle trajectory shared by every prefix-ablation condition.
- Static notebook checks for parse errors, stale imports, and undefined names.

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

Progress recorded on October 14, 2026:

- A one-million-step executor trained from scratch on local-navigation
  scenarios reached deterministic success rates of 0.953 for wood, 0.648 for
  stone, 0.555 for water, and 1.000 for both wood-tool crafting skills in the
  persistent-world evaluation.
- The matched chain ablation below used 64 episodes per condition, identical
  seeds and retry limits, and the same local-navigation checkpoint.

| Plan order | Policy | Chain success | Mean skills | Mean steps |
| --- | --- | ---: | ---: | ---: |
| Default | Deterministic | 0.094 | 3.39 | 130.6 |
| Default | Stochastic | 0.344 | 4.11 | 119.0 |
| Collect-first | Deterministic | 0.266 | 4.25 | 117.0 |
| Collect-first | Stochastic | **0.578** | **4.88** | **91.8** |

The result shows that both plan order and stochastic low-level execution
materially affect chain success. Remaining failures are concentrated in
`collect_wood` and `collect_stone`, so the next executor work should improve
recovery and navigation rather than expanding the skill catalog immediately.
The compiled batched evaluator also reduces cached matched conditions from
tens of seconds to roughly 0.3 seconds.

Phase 4 implementation started on October 14, 2026:

- Added a frozen VLM utility scorer that sees the screenshot, structured
  state, objective, candidate catalog, and recent outcomes without seeing the
  private reference plan.
- Split stable instructions into a true system message and state-dependent
  context into a user message.
- Replaced self-reported floating-point utilities with a constrained complete
  ranking and deterministic reciprocal-square rank utilities. The wider score
  separation prevents one executor timeout from immediately displacing the
  planner's top prerequisite.
- Added complete-catalog JSON validation and fail-closed handling for
  malformed, incomplete, or invented skills.
- Cached the compiled Outlines generator for each candidate catalog instead
  of rebuilding its structured-decoding index at every skill boundary.
- Added an end-to-end smoke runner and opt-in real-model integration test for
  rendering, multimodal generation, response validation, and grounded SayCan
  selection. Notebook 06 runs this smoke check before the longer ablations.
- Changed VLM utility ranking from global goal association to the best next
  skill given the updated state and recent outcomes, matching SayCan's
  closed-loop re-planning pattern more closely.
- Added per-skill current values and concise world-progress facts to the
  planner context, and limited progress assessments to 96 characters so
  structured generations finish cleanly.
- Ordered structured generation as progress assessment, explicit next-skill
  selection, then full ranking. Validation rejects responses where the
  selected skill is not ranked first.
- Added a typed mechanics contract for every skill and supply its requirements,
  resource costs, and successful effects as authoritative static planner
  knowledge. Dynamic inventory, world state, and recent outcomes remain in the
  per-decision user message, while executor-only action IDs and training
  horizons are excluded from the VLM context.
- Added final failed-skill and failure-outcome counts to closed-loop
  evaluation results and notebook 06.
- Kept VLM dependencies optional and isolated from JAX training.
- Added planner-prefix metrics for accepted objective-progressing choices,
  complete utility responses, and grounded choices. Closed-loop evaluation
  reports invalid planner output separately from a legitimate lack of viable
  skills, alongside objective success, skill failures, steps, and return.
- All 92 regular automated tests pass locally; the real-model VLM smoke test
  is opt-in. Ruff and Pyright report no issues.

Planner grounding fixed on October 16, 2026:

- Diagnosed the reported ranking failure. `skill_precondition` tests whether a
  block exists anywhere on the 64x64 map, so the Can term was satisfied at
  nearly every boundary and could not demote an out-of-range resource.
- Added horizon-bounded reachability in `goals/reachability.py`: a
  breadth-first search over walkable tiles, expressed as iterated min-plus
  relaxation so it stays inside `jax.lax.fori_loop`, confined by default to the
  7x9 observation window. It reports interaction distances per skill and
  distinguishes the 8-neighbourhood used for crafting from the faced tile used
  by `DO` and `PLACE_TABLE`.
- Made the gate graded rather than vetoing after a hard gate regressed the
  four-episode scripted chain from 1.0 to 0.5 success. "Not reachable in view"
  means unknown, not impossible, because the executor can walk out of the
  window; unreached targets score `unknown_feasibility = 0.25`.
- Added `planner/affordance.py`, which reads `V(s, skill)` for the whole
  catalog in one forward pass at option boundaries and multiplies the result by
  the reachability gate.
- Added `planner/calibration.py`: isotonic recalibration by pool-adjacent
  violators in plain numpy, with Brier score, expected calibration error, and
  rank-based AUC. No scikit-learn dependency is introduced for 60 lines of
  monotone regression.
- Added `evaluation/affordance.py`, which collects calibration data by probing
  every skill at every chain boundary from a copy of the boundary state and
  discarding the probe rollouts. It reports `unreachable_successes`, the count
  of skills a hard gate would have refused that the executor completed anyway.
- Fixed the planner metrics so they can detect this class of failure.
  `grounded_selection_rate` now means the chosen skill was usable from that
  exact state; its previous precondition-based definition was satisfied almost
  everywhere and so measured nothing. Added `ungrounded_top_choice_rate`, which
  isolates the VLM's grounding error independent of the Can term, and
  `feasibility_rescue_rate`, which shows how often the Can term recovered an
  accepted skill. Ground truth is now computed independently of whichever
  scorer the planner under test uses.
- Softened the prompt text to match graded semantics: an out-of-range resource
  is ranked below anything that makes progress now, not declared impossible.
- Rebuilt `notebooks/06_evaluate_vlm_planner.ipynb` around a six-condition
  ladder (`scripted_oracle`, `vlm_only`, `feasibility_only`,
  `vlm_plus_symbolic`, `vlm_plus_reachability`, `vlm_plus_critic_affordance`),
  a cheap precondition-versus-reachability table that needs no model weights,
  and a calibration cell that shares one checkpoint load with the closed-loop
  executor so the calibrated critic belongs to the policy under test.
- Added static notebook checks in `tests/test_notebooks.py`. The notebooks need
  a remote GPU, so what is verified here is what wastes a remote session: every
  cell parses, every `tiny_sima` symbol imported still exists, and no cell reads
  a name that no earlier cell binds.
- All 153 regular automated tests pass locally; the real-model VLM smoke test
  remains opt-in. Ruff and Pyright report no issues.

Critic affordances measured and retired on October 19, 2026:

- Ran the calibration harness on the GPU. The critic-as-affordance hypothesis
  failed: pooled AUC 0.428 over 432 labelled probes, per-skill AUCs near 0.5,
  and an isotonic fit that collapses to two values spanning `[0.240, 0.276]`.
  The cross-skill relation between value and success is *decreasing*, which a
  pool-adjacent-violators fit cannot represent, so the Can term degenerated to
  a near-constant multiplier. A factor varying by 1.15x cannot reorder
  candidates whose reciprocal-square rank utilities differ by 4x.
- Diagnosed the cause as the training distribution rather than the fit.
  `local_navigation` always places a reachable target, so `V(s, skill)` encodes
  which skill was requested, not whether the state admits it.
- Dropped the critic from the default feasibility composition. The code stays
  as the ablation that produced the measurement, with the result recorded in
  its module docstring so the next reader does not repeat it.
- Added `planner/competence.py`: a `(skill, reachable, distance band)` lookup
  table over the three features the same probes show carry signal -
  reachability splits success 0.70 against 0.03, per-skill success runs 0.17 to
  0.63. Cells are shrunk toward their parents by pseudo-counts, so an
  unobserved cell inherits its parent's estimate instead of reading 0 or 1, and
  a floor keeps a measured 3% rate selectable for the same reason reachability
  grades rather than vetoes. It is a SayCan competence head, just not one read
  out of a value function that was never trained to answer the question.
- Fixed the calibration protocol. Both heads are now fitted on one seed and
  scored on another; the previously reported ECE was in-sample and an isotonic
  fit's in-sample ECE is identically zero by construction. The returned report
  carries a `held_out` flag so the distinction cannot be lost again.
- Batched the probes. `probe_skills_batched` runs the whole `(episode, skill)`
  batch for a boundary in one `jax.lax.while_loop` with a fixed batch shape, so
  the compiled probe is reused across boundaries; `collect_affordance_samples`
  advances all episodes in lockstep over the boundary index to feed it. The
  serial version spent 2187s on 432 rollouts, nearly all of it in the six
  per-step device synchronisations inside `LearnedSkillExecutor.execute`.
- Attacked the remaining error, which is the planner: it ranked an unreachable
  skill first at 74.5% of boundaries where a reachable one existed. Because
  generation is schema-constrained left to right, field order is a hard
  scaffold - the new default prompt style supplies `available_now` and
  `blocked_reason` per candidate and requires the model to echo the
  availability list in a field declared before `selected_skill`. The system
  prompt no longer tells the planner that a separate affordance model handles
  executability, which was the instruction licensing the failure.
- Preserved the previous prompt and system prompt byte for byte as
  `PlannerPromptStyle.BASELINE`, and added a `vlm_baseline_prompt` condition
  that differs from `vlm_plus_reachability` in the prompt alone, so the change
  is measured rather than assumed. The echoed availability list is recorded as
  `claimed_available`, which separates a perception failure from an
  instruction-following failure.
- Extended notebook 06 to an eight-condition ladder with the held-out
  calibration seed, both competence heads scored side by side, the fitted
  competence table printed cell by cell with its support, and both prompt
  styles printed before any model weights load.
- Batched planner generation, so measuring a prompt costs a batch per boundary
  instead of a generation per episode per boundary.
  `TransformersVLMBackend.generate_batch` groups requests by catalog and schema
  and decodes in chunks of the new `max_batch_size`; `generate_all` falls back
  to sequential calls for a backend without it; `VLMUtilityScorer`
  `score_all_batched` renders and caches per item and batches only the misses;
  `SayCanPlanner.select_many` keeps feasibility per state, which is free next
  to generation. `evaluate_planner_prefixes` advances every episode in lockstep
  over the boundary index, each with its own rollout key, so the states visited
  are unchanged and one oracle failure no longer shortens the batch.
- Replaced notebook 06's single `QUICK_EVAL` switch with `smoke`, `prompt`, and
  `full` presets over independent probing, prefix, and closed-loop knobs, and
  enabled `autoreload`. The `prompt` preset runs eight full prefix episodes,
  loads both fits from disk, and skips the closed loop, so a prompt edit costs
  one cell run rather than a kernel restart - which here means reinstalling the
  package and reloading the VLM. 48 prefix boundaries give a standard error of
  about 0.063 on an `ungrounded_top` near 0.745, which is enough to see a real
  prompt effect without the ladder's confirmation conditions.
- Read the first `full` run. The available-first prompt took the ungrounded
  top choice from 0.750 to 0.026, and the competence prior separated held-out
  success at AUC 0.738 against the critic's 0.369. Accepted accuracy fell from
  0.938 to 0.646 even so, for two new reasons, both addressed below.
- Repaired rankings that repeat a skill. 10 of 48 available-first responses
  cycled the available skills to fill the ranking, every one with the right
  `selected_skill`; repairing them would have lifted accepted accuracy to about
  0.854. `VLMUtilityScorer.repair_ranking` defaults on, each repaired response
  is flagged, and `evaluate_planner_prefixes` reports
  `strict_accepted_next_skill_accuracy` and `repaired_utility_response_rate`
  beside it. The scorer cache now holds raw text, so the parse mode can change
  without regenerating.
- Added `vlm_shuffled_available`, the position-bias control for the last
  boundary, where `collect_wood` - always listed first - was chosen over an
  available `collect_stone` in six of eight episodes. `available_shuffle_seed`
  orders the available list per boundary as a pure function of the seed and
  the prompt, so the cache still hits.
- Raised the available-first assessment limit from 96 to 160 characters and
  asked for English, since assessments were cut off mid-sentence and drifted
  into Chinese near the cutoff; the baseline keeps 96. `max_new_tokens` rose
  from 256 to 320 so a longer assessment cannot truncate the JSON.
- Fixed notebook 06's per-skill table, which read the prior's predictions on
  precondition-passing probes against the critic's success rate over all
  probes, so `place_table` showed 0.98 predicted against 0.17 observed.
- Removed about a second per boundary from every prefix condition. The frame
  renderer is compiled once per texture size (253 ms to 0.6 ms a frame on CPU),
  `travel_distance_map` is compiled once per config instead of retracing its
  loop on every call, and `collect_prefix_trajectory` runs the scripted oracle
  once for the whole ladder rather than once per condition.
- All 191 regular automated tests pass locally; the real-model VLM smoke test
  remains opt-in. Ruff and Pyright report no issues.
