# Tiny-SIMA

Tiny-SIMA is a small research project about building an agent that can play
Craftax, a 2D open-world survival game.

The agent will use:

- A vision-language model (VLM) to decide what to do next.
- A lightweight reinforcement-learning policy to carry out that decision.
- Structured memory to remember progress, discoveries, and failures.

The project is designed to train and run on one accessible GPU, such as an
RTX 4090 or A100. Training, testing, and evaluation should also be runnable
from Jupyter notebooks.

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

Use Python 3.11 for the initial development environment.

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

The larger Phase 1 configuration is available at `configs/ppo_1m.yaml`.
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

After training, open `notebooks/02_evaluate_baseline.ipynb`, change
`run_name` to `ppo-1m`, and run all cells.

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

### Phase 3: Goal-conditioned executor

Goal: train one policy that can perform multiple skills.

Tasks:

- Condition the PPO-RNN policy on a skill ID.
- Train over a curriculum of feasible skills.
- Add a skill-success prediction head.
- Add optional learned skill termination.
- Compare categorical skill IDs with text embeddings.
- Evaluate on held-out world seeds and goal paraphrases.

### Phase 4: Hierarchical planner

Goal: connect the executor to an open-weight VLM.

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
│   ├── 02_evaluate_baseline.ipynb
│   ├── 03_train_skills.ipynb
│   └── 04_hierarchical_evaluation.ipynb
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

**Phase 1 is in progress.**

Implemented:

- Python package and dependency configuration.
- Craftax environment creation, vectorization, rendering, and benchmarking.
- A recurrent PPO baseline with checkpoint and metric output.
- Checkpoint evaluation.
- Automated configuration, environment, model, checkpoint, and PPO tests.
- Command-line scripts and Phase 1 notebooks.

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

Still required to complete Phase 1:

- Record remote GPU throughput and memory use.
- Run and record the one-million-step baseline.
