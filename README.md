# Tiny-SIMA

Tiny-SIMA is a small research project about building an agent that plays
Craftax, a 2D open-world survival game, with two levels of control:

- A frozen open-weight vision-language model (VLM) chooses the next skill.
- A goal-conditioned reinforcement-learning policy carries that skill out.

It trains and runs on one accessible GPU, such as an RTX 4090, L40, or A100,
and every experiment has a Jupyter notebook entry point.

Superseded results, the development history, and the dated changelog are in
[ARCHIVE.md](ARCHIVE.md).

## Research question

Can a small open-weight VLM improve long-horizon play by selecting useful
skills, when a learned low-level policy executes them and a measured model of
that policy says which skills are currently possible?

## Status

As of October 21, 2026:

- **Phases 1 to 3 are complete.** The environment, PPO baseline, skill
  benchmark, and goal-conditioned executor are done.
- **Phase 4, the hierarchical planner, is measured.** The grounded VLM picks
  an objective-progressing skill at 45 of 48 controlled boundaries, which is
  every boundary where one is reachable. No planner response was invalid in
  any condition.
- **The executor is now the binding constraint.** The same six skills in a
  different fixed order succeed in 0.250 or 0.609 of 64 closed-loop episodes
  (paired p < 0.001). `collect_wood` timeouts end most failed episodes under
  every planner. Until the executor improves, no planner difference is
  readable.
- **Current work is executor improvement.** The randomized training scenario,
  its diagnostics, and notebook 07 are implemented. The 10M-step training runs
  and the verification in notebook 06 are pending.
- **Phases 5 and 6, memory and full evaluation, have not started.**

## System overview

```text
Objective + rendered frame + inventory + recent outcomes
        |
        v
VLM planner: ranks the whole skill catalog         (Say: usefulness)
        |
        v
x feasibility: reachability or competence prior    (Can: possibility)
x failure penalty for repeated failures
        |
        v
Best-scoring skill
        |
        v
Goal-conditioned PPO-RNN executor (JAX)
        |
        v
Craftax actions until success, timeout, or lost precondition
        |
        +---- outcome and new state ----> planner
```

**Planner.** `Qwen/Qwen3-VL-2B-Instruct` runs frozen through Transformers. It
is called once per option boundary, not per frame. It sees:

- the rendered frame;
- structured inventory and survival values;
- the objective;
- recent skill outcomes.

Generation is schema-constrained with Outlines and greedy, so a response is a
function of the prompt. The schema's field order works as a scaffold. Under
the default `available_first` prompt style, the model must echo which skills
are available before it may name a choice. It then emits a full ranking, and
the rank sets the skill's utility by a reciprocal-square rule. Rankings that
repeat a skill, or that do not put the selected skill first, are repaired at
parse time and flagged.

**Feasibility.** Two Can terms are in use, and both grade rather than veto:

- **Reachability:** a breadth-first search over walkable tiles inside the 7x9
  observation window. A target out of view counts as unknown, not impossible,
  and scores `unknown_feasibility = 0.25`.
- **Competence prior:** success measured for this executor, tabulated by
  `(skill, reachable, distance band)`. Each cell is shrunk toward its parent,
  and a floor keeps rare successes selectable.

The final score is `utility * feasibility * failure_penalty`.

**Executor.** One recurrent PPO policy, conditioned on a learned embedding of
the skill ID, handles all six skills:

- `collect_wood`
- `collect_stone`
- `drink_water`
- `place_table`
- `craft_wood_pickaxe`
- `craft_wood_sword`

Episodes pay one sparse reward on success. The current baseline executor
trained for one million steps on `local_navigation` scenarios.

**Evaluation chain.** The closed loop runs one persistent world per episode,
with the objective `wood -> wood -> table -> wood -> wood pickaxe -> stone`.
The reference order is private to the scripted baseline. The planner sees only
the objective and its own outcomes, and success is read from the final Craftax
state. An episode ends at success, after 12 decisions, or when one skill has
failed twice.

**Memory.** The memory package is still a placeholder. Recent outcomes in the
planner context are the only memory so far.

## Latest findings

### Planner ablations

These results come from notebook 06 on October 21, 2026, with the
one-million-step `local_navigation` executor. Evaluation seed 51 was used for
both parts:

- **Prefix ladder:** 8 episodes x 6 boundaries, on states the scripted oracle
  reaches, so every condition is scored on identical states. At 3 of the 48
  boundaries the oracle's skill is out of reach, so a grounded planner can
  accept at most 45.
- **Closed loop:** 64 episodes with a stochastic executor. Every condition
  starts episode `i` in the same world, so conditions are compared with an
  exact McNemar test.

| Condition | Prompt | Can term | Prefix accepted (strict) | Ungrounded top choice | Closed-loop success |
| --- | --- | --- | ---: | ---: | ---: |
| `scripted_oracle` | none, default order | none | 1.000 | 0.000 | 0.250 |
| `scripted_collect_first` | none, wood first | none | n/a | n/a | **0.609** |
| `feasibility_only` | none, constant utility | reachability | 0.729 | 0.000 | 0.172 |
| `vlm_plus_reachability` | available-first, seeded order | reachability | 0.938 (0.917) | 0.042 | 0.297 |
| `vlm_fixed_order` | available-first, catalog order | reachability | 0.896 (0.750) | 0.042 | 0.203 |
| `vlm_only` | available-first, seeded order | none | 0.896 (0.875) | 0.042 | 0.266 |
| `baseline_prompt_only` | baseline | none | 0.312 | 0.729 | 0.000 |
| `baseline_plus_reachability` | baseline | reachability | 0.958 | 0.729 | 0.266 |
| `baseline_plus_competence_prior` | baseline | competence prior | 1.000 | 0.729 | 0.266 |

The column definitions:

- **Prefix accepted:** the fraction of boundaries where the chosen skill
  progresses the objective.
- **Strict:** the same fraction with no parse-time repair.
- **Ungrounded top choice:** how often the model's own first choice was
  unreachable while a reachable skill existed.

The paired closed-loop comparisons:

| Reference | Candidate | Both | Ref only | Cand only | Diff | p | Isolates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `scripted_oracle` | `scripted_collect_first` | 11 | 5 | 28 | +0.359 | <0.001 | plan order |
| `scripted_collect_first` | `vlm_plus_reachability` | 14 | 25 | 5 | -0.312 | <0.001 | VLM against the best fixed plan |
| `feasibility_only` | `vlm_plus_reachability` | 3 | 8 | 16 | +0.125 | 0.152 | VLM against no utility |
| `vlm_fixed_order` | `vlm_plus_reachability` | 6 | 7 | 13 | +0.094 | 0.263 | position bias |
| `vlm_only` | `vlm_plus_reachability` | 17 | 0 | 2 | +0.031 | 0.500 | Can under the grounded prompt |
| `baseline_prompt_only` | `baseline_plus_reachability` | 0 | 0 | 17 | +0.266 | <0.001 | Can as the grounding |
| `baseline_plus_reachability` | `baseline_plus_competence_prior` | 14 | 3 | 3 | +0.000 | 1.000 | which Can |
| `baseline_plus_reachability` | `vlm_plus_reachability` | 6 | 11 | 13 | +0.031 | 0.839 | prompt against Can grounding |

What the tables show:

1. **Grounding works in the prompt or in the Can term, and either is
   enough.**
   - The available-first prompt cuts the ungrounded top choice from 0.729 to
     0.042.
   - Under the baseline prompt, the Can term does the grounding. Without it,
     the planner succeeds in no episode, and 61 of 64 episodes end on an
     infeasible skill, mostly `craft_wood_pickaxe`. With reachability it
     succeeds in 17, and those 17 discordant pairs all favour reachability.
   - Under the available-first prompt, adding the Can term changes two
     episodes.
2. **Which Can term is used does not show in the closed loop.** Reachability
   and the competence prior each succeed in 17 of 64 episodes, with 3
   discordant pairs each way. On the prefix ladder the prior accepts 48 of 48
   and reachability 46. Both are above the achievable 45, because they are
   credited when they choose an unreachable reference skill.
3. **Seeded order scores at or above catalog order in every run, but never
   significantly.** In this run it won 19 against 13 (p = 0.26).
   - At the last boundary, catalog order misses in 3 of 8 episodes against 1
     for seeded order. In earlier runs, catalog order's misses there were
     `collect_wood`, which it always lists first, chosen over an available
     `collect_stone`.
   - Catalog order also needed a ranking repair in 0.146 of responses, against
     0.021 for seeded order.
4. **The VLM does not beat the best fixed plan.**
   - `vlm_plus_reachability` loses to `scripted_collect_first`, 19 against 39
     (p < 0.001).
   - It beats `feasibility_only` by 0.125, which is not significant
     (p = 0.15).
   - The available-first conditions complete more skills than the default
     scripted plan, 6.3 to 7.2 against 3.9, yet finish the chain about as
     often, 0.203 to 0.297 against 0.250. They hit the 12-decision limit in 11 or 12 episodes, repeating
     sub-goals after timeouts. The grounded baseline-prompt conditions hit it
     in 5 or 6, and `feasibility_only` in 33.
5. **The executor sets the ceiling.**
   - Plan order alone moves the scripted plan from 0.250 to 0.609, with 28
     discordant pairs against 5.
   - All 48 of the default plan's failures are timeouts, 41 of them in
     `collect_wood`.
   - `collect_wood` is the final failed skill in 36 to 41 of 64 episodes under
     every VLM condition except `baseline_prompt_only`.
   - Held out, `collect_wood` succeeds 0.941 of the time when the target is
     two steps away or closer, and 0.581 at three to six steps.
   - At these success rates, detecting a 0.10 difference takes about 300
     unpaired episodes an arm. Planner differences of that size stay
     unreadable until the executor's variance comes down.

### Feasibility measurements

The skills were probed at every chain boundary, 432 probes on the fit seed (7)
and 432 on the held-out seed (23):

- **Competence prior:** held-out AUC 0.738 and Brier score 0.208, over the 192
  probes whose precondition held.
- **Reachability:** splits success 0.739 against 0.041. Eleven skills the
  predicate called unreachable were completed anyway, by walking out of the
  observation window. Each of those is an episode a hard gate would have
  thrown away.
- **Critic:** held-out AUC 0.369, below chance. Isotonic calibration brings
  its error (ECE) to 0.047 only because the fit collapses onto the base rate.

### Refuted approaches, kept as ablations

| Approach | Measurement that refuted it | Status |
| --- | --- | --- |
| Executor critic `V(s, skill)` as the Can term | AUC 0.43 in the first run and 0.369 held out. `local_navigation` always spawns a reachable target, so the critic learned which skill was asked for, not whether it is possible. | Fitted and scored on every probing run, and retired from the ladder: identical closed loop at 2.8 times the cost. |
| Symbolic preconditions as the Can term | They test whether a block exists anywhere on the 64x64 map, which is true almost everywhere. `vlm_plus_symbolic` matched reachability exactly. | Kept inside every Can term; retired as a condition. |
| Hard reachability gate | The four-episode scripted chain fell from 1.0 to 0.5 success, with the lost episodes ending in `no_viable_skill`. | Replaced by grading, with unknown targets at 0.25. |
| Sampled decoding | `vlm_only` fell from 43 to 38 accepted of 48 on identical states. | Greedy decoding by default. |
| Per-skill advantage normalization | It collapsed the unsolved interaction skills in a debug ablation. | Available, off by default. |
| Larger PPO batches at equal steps | 256 environments lost 64% of return, and 512 lost 78%. | 64 environments remain the default. |

## Current work: executor improvement

Plan order changes nothing about the planner, so the gap between the two fixed
plans is in the states each order hands the executor. `local_navigation`
trains on none of those states. Every episode starts:

- at the map centre, in a cleared window;
- with one target and a reserved straight route;
- with the minimal inventory;
- at dawn.

A chain boundary differs in each of those ways.

The work is ordered so that each step is measured before the next is paid for:

1. **Benchmark the executor alone.** `benchmark_executor` runs both fixed
   plans over 512 worlds with a greedy executor. It records every
   first-attempt boundary, so success can be read by skill and by distance
   band. It uses seed 101, which is disjoint from the calibration and
   evaluation seeds.
2. **Attribute failures.** `probe_counterfactuals` edits the boundary states
   one training difference at a time:
   - remove tables;
   - clear the view;
   - reset the inventory, the vitals, or the clock.

   It then reruns the option from the same key. A replay-agreement check on
   the unedited state confirms that the comparison is exact.
   `failed_rollout_features` describes what a failed option did, such as
   reaching the target without facing it.
3. **Widen the training distribution.** The `randomized` scenario undoes each
   `local_navigation` control through one `ScenarioRandomization` knob:
   - obstacles on the route;
   - extra targets no closer than the sampled one;
   - a table in view;
   - a player offset from the centre;
   - natural rather than cleared terrain;
   - random holdings and vitals;
   - a start time up to step 300.

   With every knob at its `local_navigation` value it reproduces that setup,
   so each knob can be ablated on its own. Every world stays solvable, and all
   six skills, including `place_table`, train on it.
4. **Train and select.** `notebooks/07_improve_executor.ipynb` trains three
   randomized seeds and a step-matched `local_navigation` control. It compares
   each against the baseline on the benchmark worlds, greedy and stochastic,
   and checks that `local_navigation` success has not regressed. It then
   writes the best run to `runs/executor-improvement-full/selected_executor.json`.
5. **Verify in the full system.** Notebook 06 reads that file and measures the
   baseline and the candidate side by side:
   - It fits a competence prior for each executor.
   - It pairs their probes on identical oracle states, which compares the
     executors with the planner held fixed.
   - It runs every closed-loop condition under both executors on the same
     worlds.

   Its `executor` preset runs only this comparison.

**Success criterion.** The candidate closes most of the 0.250-to-0.609
plan-order gap, and its `scripted_oracle` beats the baseline's in the paired
executor table, without a `local_navigation` regression.

Tasks:

- Benchmark the executor on both fixed plans. **Implemented.**
- Attribute boundary failures with counterfactual edits. **Implemented.**
- Train on the randomized scenario, with every control switchable.
  **Implemented; the 10M-step runs are pending.**
- Compare executors in notebook 06, with a refitted competence prior.
  **Implemented; pending the selected run.**
- If the gap remains: mine boundary states from chain rollouts on disjoint
  seeds, then consider prioritized level replay or behaviour cloning from the
  scripted policy.

### Seeds

Each seed is reserved for one purpose, so that selecting on one never biases
a result reported on another:

| Seed | Use |
| ---: | --- |
| 7 | Fitting the critic calibration and the competence prior |
| 23 | Held-out scoring of both feasibility heads |
| 51 | Planner prefix ladder and closed-loop evaluation |
| 101 | Executor benchmark and run selection, in notebook 07 |
| 202 | Randomized-scenario evaluation |

Boundary states mined for training must come from seeds disjoint from all of
these.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| 1. Environment and baseline | A reproducible Craftax PPO-RNN setup with throughput and memory measured | Complete |
| 2. Skill benchmark | Six skills with preconditions, success predicates, sparse rewards, and scripted and random baselines | Complete |
| 3. Goal-conditioned executor | One policy for every skill, trained on `local_navigation` | Complete |
| 4. Hierarchical planner | A frozen VLM in a SayCan loop, with a grounded ablation ladder | Planner measured; executor improvement in progress |
| 5. Memory and visual grounding | Structured episodic memory; symbolic against pixel inputs; frame history; paraphrased instructions | Not started |
| 6. Full evaluation | Long-horizon objectives, multi-seed ablations, compute accounting, and documented failures | Not started |

The Phase 3 plan had more items than were built. Learned skill termination,
text-embedding skill conditioning, and paraphrase evaluation were deferred.
The skill-success head became the measured competence prior, after the critic
failed as one.

Phase 4 items still open:

- Replanning on danger or low vitals. The planner currently replans only at
  option boundaries.
- A planner that beats the best fixed plan once the executor allows the
  comparison.

## Quick start

Use Python 3.11 or 3.12. The pinned Flax version needs Python 3.11 or newer.

```bash
./scripts/setup_env.sh
source .venv/bin/activate
# On Linux with an NVIDIA GPU, plus the optional VLM planner:
pip install -e ".[dev,notebook,cuda12,vlm]"
```

Run the checks. The real-model VLM test is opt-in.

```bash
pytest
pyright
ruff check .
TINY_SIMA_RUN_VLM_SMOKE=1 pytest -q tests/test_vlm_backend_smoke.py
```

Train the executors:

```bash
# Current baseline executor: one million steps on local_navigation.
tiny-sima-train-skills --config configs/skill_ppo_local_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_local_1m.yaml

# Executor-improvement candidate: about 10M steps, 96 environments.
tiny-sima-train-skills --config configs/skill_ppo_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_randomized_10m.yaml
```

Evaluate them:

```bash
tiny-sima-evaluate-trained-skills runs/skill-ppo-local-1m/checkpoints/final \
  --scenario local_navigation --horizon 32 --episodes-per-skill 128
tiny-sima-evaluate-chain --executor scripted
tiny-sima-evaluate-chain --executor learned \
  --checkpoint runs/skill-ppo-local-1m/checkpoints/final
tiny-sima-smoke-vlm --config configs/vlm_planner_qwen3_2b.yaml --seed 51
```

Other entry points:

- `tiny-sima-train` and `tiny-sima-evaluate` cover the flat PPO baseline, with
  `configs/ppo_debug.yaml` and `configs/ppo_1m.yaml`.
- `tiny-sima-benchmark`, `tiny-sima-profile`, and
  `tiny-sima-compare-learning` measure throughput and learning scale.
- `tiny-sima-evaluate-skills` runs the scripted and random skill baselines.

Runs and checkpoints go under `runs/`, which Git does not track. Each
training run writes:

- `summary.json`, with training results and the GPU and JAX allocator
  summaries;
- `gpu_metrics.csv`;
- `gpu_summary.json`.

Use `jax_memory.peak_bytes_in_use_mib` for the policy's real memory use.
`nvidia-smi` also counts JAX's preallocated pool.

### Notebooks

| Notebook | Purpose |
| --- | --- |
| `00_environment_smoke_test` | Install into the kernel, then reset, step, render, and benchmark Craftax |
| `01_train_baseline`, `02_evaluate_baseline` | Flat PPO baseline |
| `01_profile_executor_scaling`, `02_compare_learning_scaling` | Throughput sweep and learning-scale comparison |
| `03_evaluate_skills` | Scripted and random skill baselines |
| `04_train_goal_conditioned` | Train and evaluate the goal-conditioned executor |
| `05_hierarchical_evaluation` | Local-navigation skill evaluation and fixed-plan chains with the learned executor |
| `06_evaluate_vlm_planner` | Planner ablation ladder, feasibility fits, and the executor comparison |
| `07_improve_executor` | Executor diagnostics, randomized training, and run selection |

Notebooks 06 and 07 have presets:

- Notebook 06: `smoke`, `prompt`, `executor`, and `full`.
- Notebook 07: `smoke` and `full`.

Both notebooks reuse completed work rather than repeating it.

### JupyterHub

The first cell of every notebook except `02_evaluate_baseline` does the
setup:

- It finds the repository root.
- It installs Tiny-SIMA into the active kernel.
- It adds the CUDA 12 JAX dependencies when `nvidia-smi` reports a GPU.

Two environment variables, set before the kernel starts, control it:

- `TINY_SIMA_USE_CUDA=1` or `TINY_SIMA_USE_CUDA=0` overrides GPU detection.
- `TINY_SIMA_JAX_MEMORY_FRACTION`, for example `0.75`, changes JAX's share of
  GPU memory from its default of 60%.

Keep only one JAX kernel running during training. Closing a notebook tab does
not always stop its kernel.

## Evaluation principles

- **Separate planner errors from executor errors.** The prefix ladder scores
  the planner on oracle states, and the closed loop scores the whole system.
  Invalid planner output, no viable skill, and skill failures are all reported
  separately.
- **Pair everything that can be paired.** Conditions share worlds, so
  comparisons use the discordant episodes rather than two success rates.
- **Fit and score on different seeds.** An isotonic fit's in-sample
  calibration error is zero by construction.
- **Keep refuted approaches runnable, with the number that refuted them.**
- **Report compute.** Every major run records wall-clock time and peak JAX
  memory.

## Compute scope and limitations

Everything runs on one GPU:

- The VLM is small and frozen, and runs only at skill boundaries.
- The executor uses symbolic Craftax observations.
- Parameter-efficient fine-tuning is reserved for the case where prompting
  proves insufficient.

Tiny-SIMA does not reproduce SIMA's scope:

- Craftax is turn-based and 2D.
- Its action space is simpler than keyboard and mouse.
- The VLM selects from a fixed skill catalog rather than issuing game
  actions.

The aim is a small system whose components and results can be understood and
reproduced.

## Repository structure

```text
tiny-sima/
├── README.md               current findings and status
├── ARCHIVE.md              superseded results, history, and changelog
├── pyproject.toml
├── configs/                PPO, skill-PPO, and VLM planner configurations
├── notebooks/              00-07, as listed above
├── scripts/                setup and standalone baseline scripts
├── src/tiny_sima/
│   ├── envs/               Craftax wrappers, skill scenarios, persistent chains
│   ├── goals/              skill registry, predicates, reachability
│   ├── executor/           PPO and goal-conditioned PPO-RNN, checkpoints
│   ├── planner/            SayCan planner, VLM scorer, feasibility terms
│   ├── evaluation/         skill, chain, planner, affordance, executor diagnostics
│   └── memory/             placeholder for Phase 5
└── tests/
```
