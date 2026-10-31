# Tiny-SIMA archive

Material moved out of [README.md](README.md) on October 21, 2026, when the
README was reduced to the current findings and status; on October 22, when the
executor-improvement phase finished; on October 28, when the README moved
to the Phase 4b sub-goal setup; and on October 31, when Phase 4b was
measured. Everything here was accurate when it
was written. Later runs supersede some of it: condition names changed on
October 20, the executor changed on October 22, and the README's tables hold
the latest numbers.

Contents:

- [Phase 1 measurements](#phase-1-measurements): the one-million-step PPO
  baseline, the executor scaling sweep, and the learning-scale comparison.
- [Phase 4 development history](#phase-4-development-history): how the planner
  ladder reached its current form, including the critic refutation, the prompt
  iterations, and the batching work.
- [Original design notes](#original-design-notes): the skill registry, decision
  format, replanning triggers, and memory as first planned.
- [Feature inventory](#feature-inventory-as-of-october-21-2026): the
  implemented-features list as of October 21, 2026.
- [Planner ablations with the baseline executor](#planner-ablations-with-the-baseline-executor):
  the October 21 ladder, closed loop, and feasibility fits with the
  one-million-step `local_navigation` executor, reproduced on October 22.
- [Executor improvement plan and diagnostics](#executor-improvement-plan-and-diagnostics):
  the plan as written before training, and notebook 07's benchmark,
  counterfactual, and evaluation tables.
- [Phase 4 per-turn planner results](#phase-4-per-turn-planner-results):
  the full stone-pickaxe tables, the six-skill ablations and feasibility
  probes with the randomized executor, the executor-improvement result, and
  the Phase 4 closing diagnosis and work items, as of October 26, 2026.
- [README material moved on October 31, 2026](#readme-material-moved-on-october-31-2026):
  the Phase 4 baselines table and the Phase 4b plan as of October 28.
- [Changelog](#changelog): dated status entries from October 8, 2026 onward.

## Phase 1 measurements

These runs fixed the PPO defaults that every later executor inherits: 64
environments and 64 rollout steps.

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

## Phase 4 development history

This is the Phase 4 narrative as of October 20, 2026. It uses the condition
names from before the ladder was restructured. In today's names:

| Name used below | Current name |
| --- | --- |
| `vlm_shuffled_available` | `vlm_plus_reachability` |
| `vlm_plus_reachability` (catalog order) | `vlm_fixed_order` |
| `vlm_baseline_prompt` | `baseline_plus_reachability` |
| `vlm_plus_symbolic`, `vlm_plus_critic_affordance` | retired from runs |

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

A second prefix run on identical code and identical oracle states moved
`vlm_only` from 43 to 38 accepted of 48. Decoding had never been set
explicitly, so `generate` fell back to the checkpoint's sampling defaults, and
a swing that size is as large as the gaps between conditions: shuffled order's
44 against fixed order's 38 in that run says nothing while fixed order alone
spans 38 to 43. Decoding is now greedy (`do_sample: false`). The same run had
one response that selected `place_table` but ranked `collect_wood` first; both
were acceptable, but a mismatch ended the closed-loop episode. Repair now
moves the selection to the top of the ranking, since the schema decodes it
first, and flags the response like a cycled ranking. Three of the 48
boundaries have an oracle skill that is out of reach, so a grounded planner
can accept at most 45.

Greedy decoding made one thing exact: under the available-first prompt,
preconditions, reachability, the critic, and the competence prior produced
byte-identical 64-episode closed loops. The prompt already carries what the Can
term knows - `available_now`, `blocked_reason`, and `steps_to_target` are
computed from the same reachability search - so the model's top choice is
reachable at 96% of boundaries and no Can term has anything left to rescue.
Grounding can live in the prompt or in the Can term, and which Can term is
better is only measurable under the prompt that leaves grounding to it. The
ladder is laid out on those two axes: the available-first prompt with and
without reachability, plus catalog order as the position-bias ablation, and
the baseline prompt with no Can term, reachability, or the competence prior.
Seeded order is now the available-first default. The closed loop adds the
collect-first scripted plan as the stronger fixed reference, and reports
paired comparisons, since every condition starts episode `i` from the same
world.

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
backend implementing only `generate`. The closed loop batches the same way:
a decision depends only on its own episode's state and history, so
`evaluate_chain_episodes` plans every running episode in one `select_many` call
per decision index and runs the chosen skills together in one compiled loop.
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
generic evaluator accepts arbitrary Python planners and keeps detailed
per-decision traces. It advances episodes in lockstep between decisions, and
`LearnedSkillExecutor.execute_many` runs each decision's options as one
compiled loop, padded to the episode count so one compilation serves every
decision. Each option splits its own key exactly as a lone option would, so
lockstep results equal running the episodes one by one.

Tasks:

- Define the planner input and output schemas. **Implemented.**
- Add a small frozen VLM backend. **Implemented; evaluation pending.**
- Rank candidate skills using usefulness and predicted feasibility.
  **Implemented: preconditions, reachability, and a measured per-skill
  competence prior. The critic-as-affordance variant was measured and
  retired.**
- Replan after success, failure, danger, or timeout.
- Compare scripted, VLM-only, feasibility-only, and combined planners.
  **Ladder implemented in notebook 06 on two axes, prompt grounding against
  Can-term grounding, with a position-bias control, a collect-first scripted
  plan, and paired closed-loop comparisons; three 64-episode closed loops
  recorded, the last with greedy decoding and both repairs. A run of the
  restructured ladder is pending.**

## Original design notes

These were the plans written before the skill registry and the planner
existed. The registry that was built has six skills. The planner returns a
ranking over the whole catalog rather than one typed decision, and it replans
at every option boundary. The memory package is still a placeholder.

### Example skills

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

### Planner and executor loop

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

### Memory

The planner will receive a small structured memory rather than an unlimited
free-form transcript. It may contain:

- The current long-term objective.
- Completed skills and achievements.
- Important inventory changes.
- Recent failures and their causes.
- Episode-specific discoveries, such as potion effects.

This will allow us to test whether memory helps the agent adapt during long
episodes.

## Feature inventory as of October 21, 2026

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
- Parse-time repair of rankings that repeat a skill or that disagree with the
  selected skill, flagged per response so strict-parse accuracy is reported
  from the same generations.
- Greedy decoding by default, so responses are a function of the prompt.
- One oracle trajectory shared by every prefix-ablation condition.
- Static notebook checks for parse errors, stale imports, and undefined names.

## Planner ablations with the baseline executor

These results come from notebook 06 on October 21, 2026, with the
one-million-step `local_navigation` executor, `runs/skill-ppo-local-1m`. The
October 22 run reproduced every closed-loop number below exactly. Its prefix
ladder matched too, except `baseline_plus_competence_prior`, which by then
used the candidate executor's prior. Evaluation seed 51 was used for both
parts:

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

What the tables showed:

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
     often, 0.203 to 0.297 against 0.250. They hit the 12-decision limit in 11
     or 12 episodes, repeating sub-goals after timeouts. The grounded
     baseline-prompt conditions hit it in 5 or 6, and `feasibility_only` in
     33.
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

The feasibility fits for this executor, probed at every chain boundary, 432
probes on the fit seed (7) and 432 on the held-out seed (23):

- **Competence prior:** held-out AUC 0.738 and Brier score 0.208, over the 192
  probes whose precondition held.
- **Reachability:** splits success 0.739 against 0.041. Eleven skills the
  predicate called unreachable were completed anyway, by walking out of the
  observation window. Each of those is an episode a hard gate would have
  thrown away.
- **Critic:** held-out AUC 0.369, below chance. Isotonic calibration brings
  its error (ECE) to 0.047 only because the fit collapses onto the base rate.

## Executor improvement plan and diagnostics

The plan below was the README's "Current work" section from October 21, 2026,
written before any 10M-step run. The measurements after it come from
notebook 07's `full` preset on seed 101, and from seed 202 for the randomized
evaluation.

### The plan as written

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

Tasks, with their state on October 21:

- Benchmark the executor on both fixed plans. **Implemented.**
- Attribute boundary failures with counterfactual edits. **Implemented.**
- Train on the randomized scenario, with every control switchable.
  **Implemented; the 10M-step runs are pending.**
- Compare executors in notebook 06, with a refitted competence prior.
  **Implemented; pending the selected run.**
- If the gap remains: mine boundary states from chain rollouts on disjoint
  seeds, then consider prioritized level replay or behaviour cloning from the
  scripted policy.

The gap closed, so the last task was not needed.

### Baseline benchmark and attribution

With a greedy executor, the baseline completed the default plan in 0.086 of
512 worlds and the collect-first plan in 0.248, a gap of +0.162 (33 against
116 discordant). The default plan's third `collect_wood`, the first one after
the table is placed, is where that gap lives:

| Skill | Occurrence | Table present | n | Success |
| --- | ---: | --- | ---: | ---: |
| `collect_wood` | 1 | no | 512 | 0.973 |
| `collect_wood` | 2 | no | 503 | 0.903 |
| `place_table` | 1 | no | 467 | 1.000 |
| `collect_wood` | 3 | yes | 467 | 0.158 |
| `craft_wood_pickaxe` | 1 | yes | 87 | 0.977 |
| `collect_stone` | 1 | yes | 85 | 0.447 |

Counterfactual edits over the 4,639 recorded boundaries, each paired against
the unedited replay:

| Edit | n | Ref only | Edit only | Diff | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| `remove_tables` | 4,207 | 16 | 511 | +0.118 | <0.001 |
| `clear_view` | 4,639 | 23 | 795 | +0.166 | <0.001 |
| `training_inventory` | 4,639 | 22 | 85 | +0.014 | <0.001 |
| `training_vitals` | 4,639 | 41 | 45 | +0.001 | 0.747 |
| `training_clock` | 4,639 | 5 | 8 | +0.001 | 0.581 |
| all | 4,639 | 28 | 793 | +0.165 | <0.001 |

- Clearing the view lifted `collect_wood` from 0.781 to 0.977 and
  `collect_stone` from 0.342 to 0.781. Removing tables lifted them to 0.893
  and 0.723.
- The unedited replay agreed with the chain at 0.998 of boundaries, not
  1.000, and the notebook flagged it. The edits are near-exact rather than
  exact, which does not affect effects of this size.
- Of 934 failed first attempts, 648 were `collect_wood` and 283
  `collect_stone`. A failed `collect_wood` came near its target in 0.202 of
  attempts and faced it in 0.080, visited 3.5 distinct positions, and spent
  0.946 of its steps revisiting them. The policy stayed within a few tiles
  rather than failing at the interaction.

### Training and selection

The three randomized seeds and the `local-matched` control each trained for
9,830,400 steps with 96 environments and a 64-step skill horizon, in 1,601 to
1,781 s. Training macro success, by step:

| Run | 1.5M | 2.15M | 3.1M | 4.9M | 9.8M |
| --- | ---: | ---: | ---: | ---: | ---: |
| `randomized-seed0` | 0.912 | 0.988 | 0.996 | 0.995 | 0.999 |
| `randomized-seed1` | 0.943 | 0.989 | 0.990 | 0.997 | 1.000 |
| `randomized-seed2` | 0.943 | 0.988 | 0.988 | 0.998 | 1.000 |
| `local-matched-seed0` | 0.956 | 0.972 | 0.993 | 0.999 | 1.000 |

The 512-world benchmark on seed 101:

| Executor | Greedy default | Greedy collect-first | Greedy gap | Stochastic default | Stochastic collect-first | Stochastic gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.086 | 0.248 | +0.162 | 0.309 | 0.627 | +0.318 |
| `randomized-seed0` | 0.996 | 0.998 | +0.002 | 0.996 | 0.994 | -0.002 |
| `randomized-seed1` | 0.988 | 0.988 | +0.000 | 0.996 | 0.994 | -0.002 |
| `randomized-seed2` | 0.982 | 0.986 | +0.004 | 0.994 | 0.996 | +0.002 |
| `local-matched-seed0` | 0.395 | 0.648 | +0.254 | 0.516 | 0.834 | +0.318 |

Held-out randomized worlds on seed 202, by skill:

| Skill | Baseline | `randomized-seed0` | `randomized-seed1` | `randomized-seed2` | `local-matched-seed0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `collect_wood` | 0.582 | 1.000 | 1.000 | 1.000 | 0.859 |
| `collect_stone` | 0.547 | 1.000 | 1.000 | 1.000 | 0.922 |
| `drink_water` | 0.480 | 0.996 | 0.996 | 0.996 | 0.906 |
| `place_table` | 0.902 | 1.000 | 1.000 | 1.000 | 0.898 |
| `craft_wood_pickaxe` | 0.676 | 1.000 | 1.000 | 1.000 | 0.953 |
| `craft_wood_sword` | 0.715 | 1.000 | 1.000 | 1.000 | 0.984 |

The `local_navigation` regression check put every new run, `local-matched`
included, at 1.000 on all five skills it covers. The baseline scored 0.922
for wood, 0.684 for stone, 0.594 for water, and 1.000 for both crafting
skills.

Notebook 07 selects the randomized run with the best stochastic default-plan
success. `randomized-seed0` and `randomized-seed1` tied at 0.996, and the
first in order was kept. The selection cell's printed output was not saved
with the notebook; notebook 06's candidate path confirms the choice.

## Phase 4 per-turn planner results

The README's Phase 4 sections as of October 26, 2026, moved here on
October 28 when the README switched to the Phase 4b sub-goal setup. In
Phase 4 the VLM chose every skill directly. The README's summary of the
stone-pickaxe results followed on October 31, to
[Phase 4 baselines as of October 28, 2026](#phase-4-baselines-as-of-october-28-2026).

### Stone-pickaxe chain, seed 51

These results come from two `full` runs of notebook 06 on seed 51 with the
seven-skill `randomized-seed2` executor. Run 1 came before the
`recipe_aware` prompt and the plan veto existed, and run 2 came after them.
The chain needs 8 skills, and episodes stop at 16 decisions.

- **Prefix ladder:** 8 episodes x 8 boundaries, on the oracle's states.
- **Derived states:** 27 search states, 48 surplus states and 172 detour
  states, all generated from those boundaries (see items 6, 8 and 10 under
  [Closing Phase 4](#closing-phase-4)).
- **Closed loop:** 64 paired episodes.

| Condition | Prefix accepted | Search (Say) | Surplus | Detour | Closed loop, run 1 | Closed loop, run 2 | Decision limit, run 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scripted_oracle` | 1.000 | n/a | n/a | n/a | 0.969 | 0.969 | 0 |
| `scripted_collect_first` | n/a | n/a | n/a | n/a | 1.000 | 1.000 | 0 |
| `feasibility_only` | 0.766 | 0.000 | 0.833 | 0.459 | 0.000 | 0.000 | 64 |
| `baseline_prompt_only` | 0.281 | n/a | n/a | n/a | 0.031 | 0.031 | 0 |
| `baseline_plus_reachability` (main) | 0.641 | 0.074 (0.074) | 0.146 | 0.674 | 0.469 | 0.453 | 34 |
| `baseline_plus_reachability_skip_held` | 0.641 | n/a | n/a | n/a | 0.516 | 0.484 | 32 |
| `baseline_plus_competence_prior` | 0.641 | 0.074 (0.074) | 0.125 | 0.674 | 0.531 | 0.484 | 33 |
| `objective_first_plus_reachability` | 0.766 | 0.926 (0.852) | 0.104 | 0.779 | 0.562 | 0.547 | 29 |
| `objective_first_plus_competence_prior` | 0.766 | 1.000 (0.852) | 0.021 | 0.820 | 0.453 | 0.500 | 31 |
| `recipe_aware_plus_reachability` | 0.750 | 1.000 (1.000) | 0.250 | 0.640 | n/a | 0.375 | 39 |
| `recipe_aware_plus_competence_prior` | 0.750 | 1.000 (1.000) | 0.229 | 0.698 | n/a | 0.672 | 20 |
| `plan_only_plus_reachability` | 1.000 | 1.000 | 1.000 | 1.000 | n/a | **1.000** | 0 |
| `baseline_plus_plan` | 1.000 | 1.000 | 1.000 | 1.000 | n/a | 0.984 | 0 |
| `recipe_aware_plus_plan` | 1.000 | 1.000 | 1.000 | 1.000 | n/a | **1.000** | 0 |
| `vlm_plus_reachability` | 0.859 | 0.000 (0.000) | 0.312 | 0.634 | 0.422 | 0.422 | 30 |
| `vlm_fixed_order` | 0.828 | n/a | n/a | n/a | 0.359 | 0.359 | 39 |
| `vlm_only` | 0.828 | n/a | n/a | n/a | 0.406 | 0.406 | 30 |

The column definitions:

- **Search (Say):** the accepted rate, with `say_accepted` in brackets.
  `say_accepted` is the rate at which the VLM's own first choice is accepted,
  before the Can term is applied.
- **Surplus and Detour:** accepted rates.
- **n/a:** the condition was not scored there, or had not been added yet.

Paired closed-loop comparisons from run 2:

| Reference | Candidate | Both | Ref only | Cand only | Diff | p | Isolates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `scripted_collect_first` | `baseline_plus_reachability` | 29 | 35 | 0 | -0.547 | <0.001 | VLM against the best fixed plan |
| `baseline_plus_reachability` | `baseline_plus_plan` | 29 | 0 | 34 | +0.531 | <0.001 | registry plan veto |
| `baseline_plus_reachability_skip_held` | `baseline_plus_plan` | 31 | 0 | 32 | +0.500 | <0.001 | plan veto against the one held-craft rule |
| `plan_only_plus_reachability` | `baseline_plus_plan` | 63 | 1 | 0 | -0.016 | 1.000 | VLM tie-breaking among on-plan skills |
| `recipe_aware_plus_reachability` | `recipe_aware_plus_plan` | 24 | 0 | 40 | +0.625 | <0.001 | plan veto, under recipe quantities |
| `objective_first_plus_reachability` | `recipe_aware_plus_reachability` | 15 | 20 | 9 | -0.172 | 0.061 | recipe quantities computed |
| `objective_first_plus_competence_prior` | `recipe_aware_plus_competence_prior` | 21 | 11 | 22 | +0.172 | 0.080 | the same, under the measured Can |
| `baseline_plus_reachability` | `objective_first_plus_reachability` | 18 | 11 | 17 | +0.094 | 0.345 | reachability discounted once |
| `baseline_plus_reachability` | `baseline_plus_reachability_skip_held` | 27 | 2 | 4 | +0.031 | 0.688 | held-craft veto |
| `objective_first_plus_reachability` | `objective_first_plus_competence_prior` | 29 | 6 | 3 | -0.047 | 0.508 | which Can (run 1: 8 against 1, p = 0.039) |
| `baseline_plus_reachability` | `vlm_plus_reachability` | 16 | 13 | 11 | -0.031 | 0.839 | prompt against Can grounding |

What the runs show:

1. **The plan veto closes the chain, as an upper bound.** It reads the
   chain's formal goal and the whole map, which the agent does not have, so
   it bounds what a planner could reach rather than being one.
   - `baseline_plus_plan` succeeds in 63 of 64 episodes and wins 34
     discordant pairs against the main condition, losing none. Its one lost
     episode is a `collect_stone` timeout.
   - `plan_only_plus_reachability` and `recipe_aware_plus_plan` succeed in 64
     of 64, which matches the collect-first bar.
   - None of the three makes an off-objective decision. They complete 7.94 to
     8.00 skills per episode, and the chain needs 8.
   - Before any VLM ran, the plan agreed with the hand-written accepted sets
     at all 311 derived states.
2. **With the veto in place, the VLM adds nothing on this chain.**
   - `plan_only` against `baseline_plus_plan` is 1 discordant pair against 0
     (p = 1.0).
   - `plan_only` takes 59 seconds for 64 episodes, against 291 for
     `baseline_plus_plan`.
   - Here the accepted set is nearly always one skill, so the VLM is left
     nothing to choose.
3. **No VLM condition without the veto reaches the bar, and none beats the
   main condition significantly.**
   - The best is `recipe_aware_plus_competence_prior`, at 43 of 64. Against
     `objective_first_plus_competence_prior` it wins 22 to 11 (p = 0.080).
   - Under reachability the same prompt does worse, 24 of 64 against 35
     (9 against 20, p = 0.061).
   - The two `recipe_aware` conditions have identical ladder, search and
     surplus choices, yet their closed loops differ by 19 episodes.
4. **`recipe_aware` fixed the search stall in the Say term, but not the
   counting.**
   - Search states: `say_accepted` rose from 0.852 to 1.000.
   - Surplus states: the VLM's first choice is accepted at 1 of 48, against 0
     for `objective_first`. It still collects stone at 36 to 37 of the 48,
     where wood is the only step.
   - Detour states: 0.640 and 0.698, against 0.779 and 0.820 for
     `objective_first`.
     - Worse after surplus resources: 15 and 21 of 32 after a second wood
       (against 26 and 27), and 2 of 12 after a second stone (against 6 and
       8).
     - Better after a held pickaxe or a sword: 5 and 7 of 8 (against 4), and
       8 of 12 (against 6).
   - Ladder: boundary 3 rose from 5/8 to 8/8, and boundary 7 fell from 3/8
     to 0/8.
   - Its commonest closed-loop waste is a new mistake: collecting stone when
     `craft_stone_pickaxe` is the only accepted step, in 104 decisions under
     reachability and 86 under the prior.
5. **Seed 202's lead for `objective_first` did not replicate.** On 202,
   `objective_first_plus_competence_prior` reached 0.641, against 0.469 for
   the main condition. On 51 it reached 0.453 and 0.500.
6. **Single runs are noisy at this size.**
   - Run 2 reproduced exactly every scripted number, every `available_first`
     number, and every ladder and search number.
   - The closed loops of the `baseline` and `objective_first` conditions
     moved by 1 to 3 episodes. The cause has not been found.
   - The one significant result from run 1, `objective_first` under
     reachability against the prior (8 against 1, p = 0.039), became 6
     against 3 (p = 0.508).
7. **The ladder still does not predict the closed loop.** The main condition
   accepts 0.641 on the ladder and `vlm_plus_reachability` 0.859, yet they
   succeed in 29 and 27 episodes.
8. **The competence prior is informative on this executor.** Held out, its
   AUC is 0.917 and its base rate 0.991. The critic's AUC is 0.457.

### Planner ablations, six-skill chain

These results come from notebook 06 on October 22, 2026, with the
randomized-scenario executor that notebook 07 selected. Evaluation seed 51 was
used for both parts:

- **Prefix ladder:** 8 episodes x 6 boundaries, on states the scripted oracle
  reaches, so every condition is scored on identical states. At 3 of the 48
  boundaries the oracle's skill is out of reach, so a grounded planner can
  accept at most 45.
- **Closed loop:** 64 episodes with a stochastic executor. Every condition
  starts episode `i` in the same world, so conditions are compared with an
  exact McNemar test.

| Condition | Prompt | Can term | Prefix accepted (strict) | Ungrounded top choice | Closed-loop success | Decision limit | Mean skills |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `scripted_oracle` | none, default order | none | 1.000 | 0.000 | **1.000** | 0 | 6.00 |
| `scripted_collect_first` | none, wood first | none | n/a | n/a | 0.984 | 0 | 5.98 |
| `feasibility_only` | none, constant utility | reachability | 0.729 | 0.000 | 0.422 | 37 | 10.75 |
| `vlm_plus_reachability` | available-first, seeded order | reachability | 0.938 (0.917) | 0.042 | 0.781 | 14 | 8.28 |
| `vlm_fixed_order` | available-first, catalog order | reachability | 0.896 (0.750) | 0.042 | 0.578 | 26 | 9.58 |
| `vlm_only` | available-first, seeded order | none | 0.896 (0.875) | 0.042 | 0.766 | 10 | 7.89 |
| `baseline_prompt_only` | baseline | none | 0.312 | 0.729 | 0.109 | 4 | 4.56 |
| `baseline_plus_reachability` | baseline | reachability | 0.958 | 0.729 | 0.969 | 2 | 7.25 |
| `baseline_plus_competence_prior` | baseline | competence prior | 0.979 | 0.729 | 0.969 | 2 | 7.12 |

The column definitions:

- **Prefix accepted:** the fraction of boundaries where the chosen skill
  progresses the objective.
- **Strict:** the same fraction with no parse-time repair.
- **Ungrounded top choice:** how often the model's own first choice was
  unreachable while a reachable skill existed.
- **Decision limit:** closed-loop episodes that ended after 12 decisions.
- **Mean skills:** skills completed per closed-loop episode. The chain needs
  six.

The prefix ladder is the same as on October 21 except for
`baseline_plus_competence_prior`, which fell from 1.000 to 0.979 because it now
uses the candidate executor's prior.

The paired closed-loop comparisons, with the same comparison under the
baseline executor for reference:

| Reference | Candidate | Both | Ref only | Cand only | Diff | p | Baseline executor | Isolates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scripted_oracle` | `scripted_collect_first` | 63 | 1 | 0 | -0.016 | 1.000 | +0.359 (<0.001) | plan order |
| `scripted_collect_first` | `vlm_plus_reachability` | 49 | 14 | 1 | -0.203 | 0.001 | -0.312 (<0.001) | VLM against the best fixed plan |
| `feasibility_only` | `vlm_plus_reachability` | 23 | 4 | 27 | +0.359 | <0.001 | +0.125 (0.152) | VLM against no utility |
| `vlm_fixed_order` | `vlm_plus_reachability` | 35 | 2 | 15 | +0.203 | 0.002 | +0.094 (0.263) | position bias |
| `vlm_only` | `vlm_plus_reachability` | 49 | 0 | 1 | +0.016 | 1.000 | +0.031 (0.500) | Can under the grounded prompt |
| `baseline_prompt_only` | `baseline_plus_reachability` | 7 | 0 | 55 | +0.859 | <0.001 | +0.266 (<0.001) | Can as the grounding |
| `baseline_plus_reachability` | `baseline_plus_competence_prior` | 62 | 0 | 0 | +0.000 | 1.000 | +0.000 (1.000) | which Can |
| `baseline_plus_reachability` | `vlm_plus_reachability` | 49 | 13 | 1 | -0.188 | 0.002 | +0.031 (0.839) | prompt against Can grounding |

Each condition under both executors, paired on the same worlds:

| Condition | Baseline executor | Candidate executor | Both | Base only | Cand only | Diff | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scripted_oracle` | 0.250 | 1.000 | 16 | 0 | 48 | +0.750 | <0.001 |
| `scripted_collect_first` | 0.609 | 0.984 | 38 | 1 | 25 | +0.375 | <0.001 |
| `feasibility_only` | 0.172 | 0.422 | 6 | 5 | 21 | +0.250 | 0.002 |
| `vlm_plus_reachability` | 0.297 | 0.781 | 16 | 3 | 34 | +0.484 | <0.001 |
| `vlm_fixed_order` | 0.203 | 0.578 | 10 | 3 | 27 | +0.375 | <0.001 |
| `vlm_only` | 0.266 | 0.766 | 14 | 3 | 35 | +0.500 | <0.001 |
| `baseline_prompt_only` | 0.000 | 0.109 | 0 | 0 | 7 | +0.109 | 0.016 |
| `baseline_plus_reachability` | 0.266 | 0.969 | 16 | 1 | 46 | +0.703 | <0.001 |
| `baseline_plus_competence_prior` | 0.266 | 0.969 | 16 | 1 | 46 | +0.703 | <0.001 |

The baseline executor reproduced the October 21 run exactly: every success
count, termination count, and paired comparison matches. Its full ladder and
the reading of it are in
[ARCHIVE.md](#planner-ablations-with-the-baseline-executor).

What the tables show:

1. **The executor is no longer the ceiling.**
   - Both scripted plans finish the chain, in 64 and 63 of 64 episodes. Plan
     order, which moved the baseline executor's scripted plan from 0.250 to
     0.609, now changes one episode.
   - Every condition improved, by 0.109 to 0.750. No condition has more than
     5 episodes that only the baseline executor solved.
   - No skill failed in any episode of `vlm_plus_reachability`,
     `baseline_plus_reachability`, or `baseline_plus_competence_prior`. Under
     the baseline executor, `collect_wood` was the final failed skill in 36 to
     41 episodes of each.
2. **The planner's sequencing sets the ceiling now.**
   - `vlm_plus_reachability` loses to `scripted_collect_first`, 1 against 14
     (p = 0.001).
   - All 14 of its failures end at the 12-decision limit. It completes 8.28
     skills an episode where the chain needs 6, so the lost episodes are spent
     on decisions that do not progress the objective, not on executor
     timeouts.
   - The other planned conditions fail the same way: `vlm_fixed_order` hits
     the limit in 26 episodes, `vlm_only` in 10, and `feasibility_only` in 37.
3. **The grounded prompt now underperforms the ungrounded prompt with
   reachability.**
   - `baseline_plus_reachability` succeeds in 62 of 64 against 50 for
     `vlm_plus_reachability`, with 13 discordant pairs against 1
     (p = 0.002). Under the baseline executor the same comparison favoured the
     grounded prompt by 0.031, not significantly (p = 0.84).
   - The prefix ladder does not show it: on oracle states the two accept
     0.958 and 0.938.
   - A hypothesis, not yet tested: echoing `available_now` biases the model
     toward skills that are almost always available, `collect_wood` and
     `drink_water`.
4. **Grounding still has to come from the prompt or the Can term.** Without
   either, the baseline prompt succeeds in 7 episodes, and 57 end on an
   infeasible skill, 50 of them `craft_wood_pickaxe`. Reachability wins 55
   discordant pairs and loses none. Under the grounded prompt, adding the Can
   term changes one episode.
5. **With the executor's variance gone, two planner effects are significant.**
   - VLM utility against constant utility: 27 against 4 (p < 0.001), from 16
     against 8 (p = 0.15) under the baseline executor.
   - Seeded against catalog order: 15 against 2 (p = 0.002), from 13 against
     7 (p = 0.26). Catalog order also needed a ranking repair in 0.146 of
     prefix responses, against 0.021 for seeded order.
6. **Which Can term is used still does not show.** The competence prior
   refitted on the new executor is flat (see below), so its closed loop
   matches reachability: 62 of 64 each, with no discordant pair. On the prefix
   ladder the prior accepts 47 of 48 and reachability 46. Both are above the
   achievable 45, because they are credited when they choose an unreachable
   reference skill.
7. **This objective is saturated** for the executor and for both fixed plans.
   Executor differences beyond this point need a longer chain.

### Feasibility measurements, six-skill chain

Each executor was probed at every chain boundary, 432 probes on the fit seed
(7) and 432 on the held-out seed (23). Held out:

| Measurement | Baseline executor | Candidate executor |
| --- | ---: | ---: |
| Success, all probes | 0.301 | 0.442 |
| Success, target reachable (161 probes) | 0.739 | 0.994 |
| Success, target unreachable (271 probes) | 0.041 | 0.114 |
| Unreachable targets completed anyway | 11 | 31 |
| Competence prior base rate, precondition held (192 probes) | 0.677 | 0.995 |
| Competence prior AUC | 0.738 | 0.500 |
| Critic AUC | 0.369 | 0.437 |

The paired probes start both executors from identical scripted-oracle states,
so they compare the executors with the planner held fixed:

| Probes | n | Baseline | Candidate | Base only | Cand only | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 432 | 0.301 | 0.442 | 1 | 62 | <0.001 |
| reachable | 161 | 0.739 | 0.994 | 1 | 42 | <0.001 |
| unreachable | 271 | 0.041 | 0.114 | 0 | 20 | <0.001 |
| `collect_wood` | 72 | 0.611 | 1.000 | 0 | 28 | <0.001 |
| `drink_water` | 72 | 0.583 | 1.000 | 0 | 30 | <0.001 |
| `collect_stone` | 72 | 0.111 | 0.153 | 1 | 4 | 0.375 |
| `place_table` and each crafting skill | 72 | 0.167 | 0.167 | 0 | 0 | 1.000 |

What the probes show:

- **The rates for `place_table` and the crafting skills only count how often
  the precondition holds,** at 12 of 72 probes. Where it holds, both
  executors succeed every time.
- **`collect_stone` needs a pickaxe, which is also held at 12 of 72 probes.**
  There the candidate succeeds in 11 and the baseline in 8.
- **The competence prior is degenerate on the candidate.** Every fitted cell
  is p = 1.000, so it cannot rank candidates. It carries information only
  while the executor fails within reach.
- **The critic is still below chance,** at 0.437. `collect_wood` and
  `drink_water` never fail on the candidate, so their per-skill AUCs are
  undefined.
- **Out-of-view targets matter more now.** The candidate completed 31 skills
  that the predicate called unreachable, against 11 for the baseline. Each of
  those is an episode a hard gate would have thrown away.

### Executor improvement result

With the baseline executor, plan order alone moved the scripted plan from
0.250 to 0.609, so the gap was in the states each order hands the executor.
`local_navigation` trains on none of those states. Every episode starts at
the map centre in a cleared window, with one target, a reserved straight
route, the minimal inventory, and the clock at dawn. Notebook 07 attributed
the gap and trained it away. The full plan and diagnostics are in
[ARCHIVE.md](#executor-improvement-plan-and-diagnostics).

- **Attribution.** Counterfactual edits on 4,639 greedy boundary states
  showed that clutter in view cost the most. Clearing the view rescued +0.166
  of first attempts and removing tables +0.118. Resetting the inventory gained
  +0.014, and the vitals and the clock +0.001 each.
- **Training.** The `randomized` scenario undoes each `local_navigation`
  control through its own knob (`configs/skill_ppo_randomized_10m.yaml`).
  Three seeds trained for 9.8M steps with 96 environments, about 29 minutes
  each.
- **Selection.** Each run was benchmarked on 512 worlds of seed 101 with the
  stochastic executor. `randomized-seed0` was selected on default-plan
  success, tied with seed 1 at 0.996:

| Executor | Default plan | Collect-first plan | Gap | Default only / collect-first only |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.309 | 0.627 | +0.318 | 41 / 204 |
| `randomized-seed0` (selected) | 0.996 | 0.994 | -0.002 | 2 / 1 |
| `randomized-seed1` | 0.996 | 0.994 | -0.002 | 3 / 2 |
| `randomized-seed2` | 0.994 | 0.996 | +0.002 | 2 / 3 |
| `local-matched-seed0` | 0.516 | 0.834 | +0.318 | 28 / 191 |

- **The scenario did the work, not the budget or the horizon.** The
  `local-matched` control trained at the same budget, horizon, and distance
  range on `local_navigation`. It raised success, but its plan-order gap
  equals the baseline's.
- **No regression.** Every randomized run succeeds 1.000 on all five skills
  of the `local_navigation` check, against 0.594 to 1.000 for the baseline. On
  randomized worlds of seed 202, they succeed 0.996 to 1.000 per skill,
  against 0.480 to 0.902 for the baseline.
- **Verified in the full system.** On seed 51 in notebook 06, the scripted
  oracle went from 16 to 64 of 64 (48 discordant pairs against 0), and the
  plan-order gap from +0.359 to -0.016. That meets the success criterion.

### Closing Phase 4

**The diagnosis.** Notebook 06 read the per-decision traces of the October 22
closed loop, candidate executor, seed 51:

- `vlm_plus_reachability` made 163 off-objective decisions, and 162 were
  `say_led`: the VLM ranked an unaccepted skill first and the planner took it.
  The Can term was not the cause.
- `baseline_plus_reachability` solved 13 of the 14 worlds the grounded prompt
  lost. Its own waste was a second `craft_wood_pickaxe`, 23 decisions, and
  wood beyond what the chain needs, 25.
- The executor finished nearly every skill asked of it, so the remaining gap
  is the planner's.

**What changed:**

1. **Main condition.** `baseline_plus_reachability` replaces
   `vlm_plus_reachability`, which is kept as an ablation and recorded as
   refuted: 50 against 62 of 64 (13 discordant against 1, p = 0.002).
2. **Held-craft veto.** `SkipHeldCraftFeasibility` scores zero any craft
   whose only effects are tools already held. It reads the skill registry's
   declared effects, never the chain's accepted set, so it cannot leak the
   objective. `baseline_plus_reachability_skip_held` pairs it against the main
   condition.
3. **A longer chain.** The `wood_to_stone_pickaxe` chain and a seventh skill,
   `craft_stone_pickaxe` (one wood and one stone at a table). The six-skill
   chain is saturated: the oracle solves 64 of 64 and the main condition 62.
   Training config: `configs/skill_ppo_stone_pickaxe_randomized_10m.yaml`, or
   notebook 07's `stone_pickaxe` preset.
4. **Confirmation seed.** Notebook 06's `CONFIRM` reruns on seed 202, which
   no planner choice was made on.

**The stall before stone.** The seed 51 run refuted the veto (59 against 61,
3 discordant to 1 against) and located the remaining failures. All three of
the main condition's lost episodes, and the veto's, end the same way: the
wood pickaxe is held, stone is out of view, and the planner collects wood until
the decision limit. The executor is not the cause. Every `collect_stone` the
main condition requested succeeded, and the oracle, which asks for stone
regardless, solved all 64 worlds. Three factors compound:

- The `baseline` prompt tells the model to rank an out-of-view target below
  any skill that can progress now, so it ranks stone fourth or fifth.
- The reachability Can term discounts the out-of-view target again, to 0.25.
- Rank utilities are `1/rank^2`. Stone ranked fifth scores 0.01, against
  0.25 for wood ranked second. Even ranked first, stone only ties a reachable
  wood ranked second, and ties keep catalog order, which puts wood first.

An explore skill was considered and not built: the collect skills already
search, and the planner would face the same choice between exploring and
collecting wood that it gets wrong now.

5. **`objective_first` prompt.** The baseline prompt without the instruction
   to demote an out-of-view target, and with surplus resources and held tools
   named as non-progress. The schema and the assessment limit are the
   baseline's, and the `baseline` and `available_first` prompts are
   byte-identical to before. It runs as `objective_first_plus_reachability`
   and `objective_first_plus_competence_prior`. The prior replaces the 0.25
   guess with the executor's measured success when searching, which this
   executor puts near 1.
6. **Search-state test.** The prefix ladder follows the oracle, which reaches
   stone while it is nearby, so it never showed the stall.
   `collect_search_trajectory` detours from each oracle boundary with
   `collect_wood` until the only accepted step collects a resource that is
   out of view. Seed 51 gives 14 such states from 16 episodes. It reports
   `say_accepted`, the VLM's own first choice, beside `accepted`, so the prompt
   and the Can term are measured apart.

**Miscounting on the stone-pickaxe chain.** The first stone-pickaxe run, on
seed 202, put `objective_first` ahead (0.641 under the competence prior,
against 0.469 for the main condition) and the search-state test at 1.000, so
the out-of-view stall is fixed. The waste left is counting. The model ignored
the catalog's recipe ("Stone: 0 (5/5 needed)" for a pickaxe that takes one),
collected stone again while wood was the only accepted step (the commonest
off-objective choice by far), and ranked `collect_wood` last right after
collecting one wood. That run was on the confirmation seed, so the fix is
selected on 51.

7. **`recipe_aware` prompt.** `objective_first` with three fields added to
   each candidate, all computed from the skill registry and the inventory,
   never the chain: `inventory_shortfall` against the skill's own declared
   requirements, `result_held` for a craft of a tool already held, and
   `fills_shortfall_of`, the candidates whose shortfall it reduces. Its
   blanket "demote redundant repetitions" is replaced with repetition
   conditioned on that shortfall. It runs as `recipe_aware_plus_reachability`
   and `recipe_aware_plus_competence_prior`, each paired against its
   `objective_first` counterpart.
8. **Surplus-state test.** `collect_surplus_trajectory` detours from each
   oracle boundary with `collect_stone` and keeps every state where stone is
   held and `collect_wood` is the only accepted step: the oracle never
   reaches it, and it is where the closed loop's commonest off-objective
   choice was made.

**From the registry rather than the diagnosis.** Items 2, 6, 7 and 8 each
answered one class of mistake found by reading a closed loop. Every one was a
mistake the skill registry could already rule out: each skill declares its
`requires` and `effects`, so the next useful step is a planning problem with
an exact answer. The next two items solve it once instead of one class at a
time.

9. **Plan-progress veto.** `PlanProgressFeasibility` runs a breadth-first
   search over the registry's declared mechanics, from the chain's formal
   `goal` (for example `inventory.stone_pickaxe >= 1`), and scores zero any
   skill that starts no shortest plan. World facts mean the target exists on
   the map; distance stays with the reachability term it wraps. When no plan
   exists, for example wood short with no tree left, or when every on-plan
   skill scores zero under that term, it returns the reachability scores
   unchanged. The plan reproduces both chains' hand-written accepted sets
   exactly at every ladder, search, surplus and detour state tested. It reads
   the goal, never the accepted set or the reference plan. It runs as:
   - `plan_only_plus_reachability`, with no VLM.
   - `baseline_plus_plan`, paired against the main condition, the held-craft
     veto and the plan alone.
   - `recipe_aware_plus_plan`, paired against `recipe_aware_plus_reachability`.

   With the scripted executor on seed 51, `plan_only` solves 16 of 16
   stone-pickaxe worlds, against 0 of 16 for reachability alone. A hard veto
   cannot route around a step the executor keeps failing, and `off_plan > 0`
   is the knob for that.
10. **Detour-state test.** `collect_detour_trajectory` generalizes the
    search and surplus tests. From each oracle boundary it requests every
    skill in the catalog repeatedly and keeps the state after each request
    the accepted set did not contain, up to two, labelled `<skill>x<k>`. The
    states are labelled by the chain's exact accepted set, so no hand-written
    predicate is needed. Seed 51 gives about 43 states per episode on the
    stone-pickaxe chain, covering surplus stone and wood, a second wood
    pickaxe, a sword, and drinking with nothing to drink for. Reachability
    alone accepts 0.46 of them and the plan veto 1.00. Notebook 06 checks
    plan agreement on every derived trajectory before running any VLM, then
    prints accuracy per label.

**To run, in order:**

1. Notebook 06 `full` on seed 51, then with `CONFIRM = True`. The veto run
   is done. The seed 202 run in progress confirms the main condition.
   Afterwards, run `full` on seed 51 again for the `objective_first` rows and
   the search-state test. An `objective_first` condition becomes the main
   condition only if it also holds on seed 202.
2. Notebook 07 `stone_pickaxe`, which trains three seven-skill seeds and
   records the selected one.
3. Notebook 06 with `CHAIN = WOOD_TO_STONE_PICKAXE_CHAIN` and the
   seven-skill executor selected. Read `scripted_oracle` first: the planner
   conditions cannot beat what the executor finishes. The seed 202 and
   seed 51 runs are done (see
   [Stone-pickaxe chain, seed 51](#stone-pickaxe-chain-seed-51)). Next,
   confirm the plan-veto conditions on 202 with `CONFIRM = True`.

4. Notebook 06's Phase 4b section on seed 51 with the same executor. It
   runs the sub-goal conditions on the stone-pickaxe, terse and
   stone-pickaxe-and-sword chains, and saves to
   `subgoal_loop_seed51_n64.json` beside the checkpoint.

**The bar** stays the collect-first scripted plan on each chain.

## README material moved on October 31, 2026

These are the README's Phase 4 baselines and its Phase 4b plan as of
October 28, 2026. They moved here once Phase 4b was measured on the GPU. The
README's [Phase 4b results](README.md#phase-4b-results) rerun the per-turn
arm, the plan veto and the scripted plans with provenance, and they
supersede the table below.

### Phase 4 baselines as of October 28, 2026

This is the per-turn arm and the upper bound that Phase 4b was to be
compared against. They come from notebook 06 on seed 51, with the
seven-skill `randomized-seed2` executor, on the stone-pickaxe chain: 8
skills, 16 decisions, 64 paired episodes.

| Condition | Closed-loop success | Decision limit |
| --- | ---: | ---: |
| `scripted_oracle` | 0.969 | 0 |
| `scripted_collect_first` (the bar) | 1.000 | 0 |
| `baseline_plus_reachability` | 0.453 | 34 |
| `objective_first_plus_reachability` | 0.547 | 29 |
| `recipe_aware_plus_competence_prior`, best per-turn | 0.672 | 20 |
| `baseline_plus_plan`, plan veto | 0.984 | 0 |
| `recipe_aware_plus_plan`, plan veto | 1.000 | 0 |
| `plan_only_plus_reachability`, plan veto without a VLM | 1.000 | 0 |

What the Phase 4 runs established:

- **Grounding has to come from somewhere.** The ungrounded prompt with no
  Can term solves 2 of 64.
- **The planner's sequencing, not the executor, sets the ceiling.** Every
  per-turn failure ends at the decision limit, on surplus resources, held
  tools, or a stall before an out-of-view target.
- **Each prompt fix moved the mistakes rather than removing them.**
  `objective_first` fixed the out-of-view stall. `recipe_aware` fixed the
  search states but not the counting.
- **The plan veto removes every off-objective decision.** With it in place,
  the VLM adds nothing on this chain: `plan_only` against `baseline_plus_plan`
  is 1 discordant pair against 0.
- **Single runs are noisy at this size.** Repeat runs moved closed loops by
  1 to 3 episodes.
- **The competence prior is informative on this executor.** Held out, its AUC
  is 0.917. The critic's is 0.457.

The notebook 06 cells that produced them were removed once Phase 4 closed.
They are at commit `282ef35`.

### Phase 4b experiments as planned

The two experiments that have since run, as the README described them:

1. **Phase 4b on the chains.** Notebook 06, section "Phase 4b", on seed 51
   with the seven-skill executor. It runs three conditions on the
   stone-pickaxe, terse and pickaxe-and-sword chains:
   - `subgoal_formal_goal`, the hierarchy's ceiling under the memory belief;
   - `subgoal_vlm_memory`;
   - `subgoal_vlm_map_scan`.

   Results go to `subgoal_loop_seed51_n64.json` beside the checkpoint. Read
   `subgoal_formal_goal` first. Then pair `subgoal_vlm_memory` against the
   per-turn arm and the plan veto.
2. **The open-ended task.** Notebook 08 runs the three designs with
   creatures off and on. It defaults to 4 episodes per design and setting,
   because a sub-goal design can ask the VLM up to three times per decision;
   raise `NUM_EPISODES` for the comparison. Results go to
   `runs/open-ended/`.

The next steps were, in order:

1. **Run the experiments**, then confirm the chain results on seed 202 with
   `CONFIRM = True`.
2. **Add the survival and hard-tier skills, to cover all of Craftax-Classic.**
   Survival first, since the open-ended runs with creatures will show what
   kills the agent:
   - eat a cow, sleep, and place and eat a plant;
   - then stone tools, furnace, coal, iron and diamond;
   - then combat.

   Each needs registry entries, scenario support and executor training.
   About 3M steps per added stage should suffice, since training success
   reached 0.988 in every randomized seed by 2.15M steps.
3. **The Qwen3-VL-8B scaling test**, once the 2B results are in.

The survival and stone-tier skills were then added in one step, so the
executor trains once: see the October 31 changelog entry.

## Changelog

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
- Read the first closed loop, 16 episodes and run before the ranking repair.
  Invalid planner output ended 13 of 16 episodes in every condition sharing the
  available-first prompt, so those five conditions scored identically at 0.062
  success after 20.8 environment steps. The scripted plan reached only 0.188,
  and 12 of its 13 failure-limit endings were repeated `collect_wood`
  timeouts, so the executor caps closed-loop success once the planner is
  fixed. The baseline prompt's 0.375 is not distinguishable from the oracle at
  this size (Fisher p = 0.43). The run took about 70 minutes, 45 seconds an
  episode even for the oracle, which never calls the VLM.
- Compiled the learned executor. `LearnedSkillExecutor.execute` stepped the
  environment from Python, dispatching each Craftax operation separately and
  reading four flags back per step; one 64-step option took 8.9 s on CPU and
  now takes 13 ms after a 2.3 s compile. The compiled loop matches the
  stepwise reference on outcome, step count, and final world state, and is
  cached on the environment's class rather than its instance, so a fresh but
  identical environment does not recompile.
- Ran closed-loop episodes in lockstep. `evaluate_chain_episodes` plans every
  running episode in one batched generation per decision index and executes
  their skills together, with results equal to running them one by one. A
  16-episode scripted closed loop takes 5.5 s on CPU and 64 episodes 10 s, so
  notebook 06's `full` preset now runs 64 closed-loop episodes, a standard
  error near 0.05 instead of 0.11. The closed-loop cell prints
  `planner_error_counts`, which says which validation ended an episode.
- Read the second prefix run. On identical states `vlm_only` accepted 38 of
  48 boundaries against 43 in the first, because decoding fell back to the
  checkpoint's sampling defaults. `VLMPlannerConfig.do_sample` now defaults to
  false and is passed to every generation. Every avoidable fixed-order miss
  in that run was `collect_wood` over `collect_stone` at the last boundary,
  each time the first entry of the model's own `available_now`. Shuffled
  order missed there twice in eight, once where stone was out of reach; the
  greedy rerun has to confirm the difference.
- Read the first 64-episode closed loop, run with sampled decoding and before
  the mismatch repair. Nine conditions took 31 minutes, against 70 for 16
  episodes before lockstep. Success ranged from 0.172 (`feasibility_only`) to
  0.328 (`vlm_baseline_prompt`), with the scripted plan at 0.250; no pair
  differs significantly (baseline against oracle Fisher p = 0.44, against
  `vlm_only` p = 0.16). The executor sets the ceiling: all 48 of the oracle's
  failures were timeouts, 41 of them in `collect_wood`, and `collect_wood` was
  the final failed skill in 29 to 34 episodes of every other condition. The
  mismatch ended 5 to 7 episodes in every available-first condition. Those
  conditions also hit the 12-decision limit in 14 or 15 episodes while
  completing 6.8 skills on average against the oracle's 3.9, and shuffled order
  hit it in 9. Held out, the critic's AUC was 0.369 and the competence prior's
  0.738.
- Repaired a `selected_skill` the ranking does not put first, which was the one
  invalid response left after the cycling repair and would have ended about
  one closed-loop episode in six. Strict parsing still rejects it.
- All 197 regular automated tests pass locally; the real-model VLM smoke test
  remains opt-in. Ruff and Pyright report no issues.

Planner ladder restructured on October 20, 2026:

- Read the first full run with greedy decoding and both repairs. No response
  was invalid in any condition, prefix or closed loop. On the prefix ladder the
  shuffled order accepted 45 of 48 - every achievable boundary - against 43
  for the fixed order with any Can term and 41 for `vlm_only`; both avoidable
  fixed-order misses were `collect_wood` over an available `collect_stone` at
  the last boundary, once with the assessment saying stone came next. The
  baseline prompt's 46 is above the achievable 45 because it was credited for
  choosing unreachable reference skills, with a 0.729 ungrounded top choice
  that reachability rescued at 0.646 of boundaries. Closed-loop success out of
  64: shuffled 19, baseline prompt 17, scripted plan 16, `vlm_only` 15, the
  four fixed-order Can terms 14 each with byte-identical episodes, and
  `feasibility_only` 11. No pair is significant (shuffled against fixed order
  Fisher p = 0.42, against the scripted plan p = 0.69), and detecting a 0.10
  difference at these rates takes about 300 episodes an arm unpaired. The
  scripted plan's 48 failures were all timeouts, 41 in `collect_wood`; the
  held-out competence table puts `collect_wood` at 0.94 success with a target
  two steps away or closer and 0.58 at three to six. Held out, the critic's AUC
  was 0.369 and the competence prior's 0.738; the critic condition took 294 s
  of closed loop against 106 s for reachability.
- Restructured notebook 06's ladder around where grounding lives. Under the
  available-first prompt: `vlm_plus_reachability` (seeded order, formerly
  `vlm_shuffled_available`), `vlm_fixed_order` (formerly
  `vlm_plus_reachability`), and `vlm_only`. Under the baseline prompt:
  `baseline_prompt_only`, `baseline_plus_reachability` (formerly
  `vlm_baseline_prompt`), and `baseline_plus_competence_prior`. The three
  baseline conditions share one scorer, so the prompt is generated once.
- Retired from runs, with code and numbers kept: `vlm_plus_symbolic`, which
  matched reachability exactly; `vlm_plus_critic_affordance`, AUC 0.369 and an
  identical closed loop at 2.8 times the cost; and the competence prior under
  the available-first prompt, identical to reachability, which moved to the
  baseline prompt. The critic is still fitted and scored whenever probing
  runs, so its AUC stays reproduced.
- Made seeded order notebook 06's available-first default, since it scored at
  or above catalog order in every run that measured both and catalog order's
  last-boundary error recurred in each. The library's `VLMUtilityScorer` still
  defaults to catalog order.
- Added `scripted_collect_first` to the closed loop, the October 14 plan that
  collects all three wood before placing the table (0.578 against 0.344 for
  the default order with this executor), as the stronger fixed plan for the
  VLM to be measured against.
- Added `compare_paired_successes`, an exact McNemar test over two
  `evaluate_skill_chain` results on the same seed and episode count; it
  refuses results that do not pair. Notebook 06 prints a paired table for the
  comparisons each pair of conditions isolates.
- All 201 regular automated tests pass locally; the real-model VLM smoke test
  remains opt-in. Ruff and Pyright report no issues.

Executor improvement started on October 21, 2026:

- Added `evaluation/executor_diagnostics.py`: the executor-only benchmark over
  both fixed plans, boundary success by skill and distance band,
  counterfactual boundary edits with a replay-agreement check, and
  failed-rollout features. A greedy executor in a mob-free world replays a
  boundary exactly, so an edit's effect is not sampling noise.
- Added the `randomized` skill scenario and its eight `skill_*` configuration
  knobs. Its reset draws randomness only in its own branch, so `nearby` and
  `local_navigation` worlds are bit-identical to before for every seed. The
  straight-route scripted policy refuses randomized worlds; the chain
  evaluator's breadth-first policy solves every one tested.
- Added `configs/skill_ppo_randomized_debug.yaml`, which trained end to end
  in 68 s on CPU, and `configs/skill_ppo_randomized_10m.yaml`.
- Added `notebooks/07_improve_executor.ipynb` with `smoke` and `full`
  presets. Runs that already have a summary are not retrained.
- Notebook 06 now saves fits beside the checkpoint they measure, so pointing
  `CHECKPOINT` at a new executor cannot overwrite the baseline's competence
  prior.
- All 250 regular automated tests pass locally; the real-model VLM smoke test
  remains opt-in. Ruff and Pyright report no issues.

Notebook 06 adapted and README reorganized on October 21, 2026:

- Read the first full run of the restructured ladder. Its prefix,
  closed-loop, and paired tables are in
  [Planner ablations with the baseline executor](#planner-ablations-with-the-baseline-executor).
- Notebook 06 now measures the baseline executor against the one notebook 07
  selects. It fits a competence prior for each executor and pairs their probes
  on identical oracle states. It also runs every closed-loop condition under
  both executors on the same worlds. A new `executor` preset runs only that
  comparison. Saved closed loops carry a planner fingerprint, so a stale run
  cannot be paired against a new one.
- Notebook 07 writes `selected_executor.json` with the run it selected and the
  seed it selected on, and notebook 06 reads it.
- Moved superseded README material and this changelog into `ARCHIVE.md`.

Executor improvement completed on October 22, 2026:

- Ran notebook 07's `full` preset. Three `randomized` seeds and a step-matched
  `local_navigation` control trained for 9.8M steps each, in about 27 to 30
  minutes apiece. On the stochastic benchmark, seed 101, every randomized seed
  completed both fixed plans in 0.994 to 0.996 of 512 worlds, against 0.309
  and 0.627 for the baseline. The plan-order gap fell from +0.318 to between
  -0.002 and +0.002. Nothing regressed on `local_navigation`. The tables are
  in [Executor improvement plan and diagnostics](#executor-improvement-plan-and-diagnostics).
- Recorded the `local-matched` control as a refuted approach. Ten million steps
  on `local_navigation` raised default-plan success from 0.309 to 0.516 but left
  the gap at +0.318, so the scenario closed the gap, not the budget or the
  horizon.
- Selected `randomized-seed0` on stochastic default-plan success, at 0.996.
- Ran notebook 06's `full` preset with both executors. The baseline executor
  reproduced the October 21 closed loop exactly. Under the candidate, the
  scripted oracle succeeded in 64 of 64 episodes against 16 (48 discordant
  pairs against 0). The plan-order gap went from +0.359 to -0.016, and
  `vlm_plus_reachability` from 0.297 to 0.781.
- Paired held-out probes from identical oracle states: the candidate
  succeeded in 0.442 of 432 against 0.301 (62 discordant against 1), 0.994
  against 0.739 where the target was reachable.
- The competence prior refitted on the candidate is flat: every cell is
  p = 1.000, with held-out base rate 0.995 and AUC 0.500. Its closed loop is
  identical to reachability's, 62 of 64 each with no discordant pair.
  Recorded as a refuted approach under a strong executor. The candidate's
  critic AUC is 0.437, still below chance.
- Read the new bottleneck. Every `vlm_plus_reachability` failure ends at the
  12-decision limit with no skill failure, at 8.28 skills an episode. The
  ungrounded `baseline` prompt with reachability now beats the grounded
  `available_first` prompt, 62 against 50 of 64 (13 discordant against 1,
  p = 0.002). That reverses the baseline executor's +0.031 (p = 0.84).
- Moved the baseline-executor ladder, its feasibility fits, and the
  executor-improvement plan from the README into this archive. The README's
  current work is now planner sequencing.

Planner sequencing diagnosed on October 25, 2026:

- Read the October 22 closed-loop traces in notebook 06, candidate executor,
  seed 51. 162 of `vlm_plus_reachability`'s 163 off-objective decisions were
  `say_led`. `baseline_plus_reachability` solved 13 of the 14 worlds it lost;
  its own waste was a second wood pickaxe (23 decisions) and extra wood (25).
- Made `baseline_plus_reachability` the main condition. Recorded
  `available_first` as a refuted approach for the main condition, 50 against
  62 of 64 (13 discordant against 1, p = 0.002), kept as an ablation.
  `DEFAULT_PROMPT_STYLE` is unchanged so that saved ablation runs keep their
  meaning; the notebook names the baseline style explicitly.
- Added `SkipHeldCraftFeasibility` and the
  `baseline_plus_reachability_skip_held` condition.
- Added `craft_stone_pickaxe` as an opt-in seventh skill (`num_skills: 7`),
  the `wood_to_stone_pickaxe` chain with four trees and a 16-decision limit,
  its collect-first plan, and seven-skill training configs and notebook 07
  presets. Six-skill checkpoints, competence priors, and system prompts are
  byte-identical to before. Executors now refuse skills they were not trained
  on, because JAX clamps an out-of-range embedding index silently.
- Executor benchmarks pair local-navigation results by skill, so a
  seven-skill run compares against the six-skill baseline.
- Added notebook 06's `CHAIN` and `CONFIRM` knobs.

Held-craft veto refuted on October 26, 2026:

- Notebook 06 `full`, candidate executor, seed 51. The main condition solved
  61 of 64, and `baseline_plus_reachability_skip_held` 59, with 3 discordant
  episodes to 1 against the veto (p = 0.625). Recorded as a refuted approach.
  It cut `craft_wood_pickaxe` waste from 22 decisions to 0, and surplus
  `collect_wood` rose from 39 to 74.
- All of the main condition's failures stall at the same state: the wood
  pickaxe is held and stone is out of view. The baseline prompt demotes an
  out-of-view target, the reachability term discounts it to 0.25, and
  `1/rank^2` utilities make stone ranked fifth worth 0.01 against 0.25 for
  wood. The oracle solved 64 of 64, so the executor finds stone when asked.
- Considered and not built: an explore skill. The collect skills already
  search, and the planner would face the same choice.
- Added the `objective_first` prompt style. It is the baseline without the
  instruction to demote an out-of-view target, with the baseline's schema and
  assessment limit. The `baseline` and `available_first` prompts are
  byte-identical to before. Added the `objective_first_plus_reachability`
  and `objective_first_plus_competence_prior` conditions.
- Added `collect_search_trajectory` and `is_search_boundary`, a prefix test on
  the out-of-view states the oracle's own prefixes never reach: 14 states from
  16 episodes on seed 51.

Notebook 06 trimmed to Phase 4b on October 28, 2026:

- Removed the per-turn ablation ladder: the prefix ladder and its inspection,
  the search, surplus and detour states, the per-turn closed loop with its
  paired table and plot, the sequencing diagnosis, the paired executor probes,
  and the precondition grounding check. The cells are at commit `282ef35`.
- Removed `MAIN_CONDITION`, the `prompt` and `executor` presets,
  `MEASURE_BASELINE` and `REUSE_SAVED_CLOSED_LOOPS`. The notebook measures
  the one selected executor.
- Kept the competence-prior probing, the VLM feasibility study, Phase 4b and
  model learning. Phase 4b rebuilds the per-turn arms it is paired against:
  the scripted plans, `recipe_aware_plus_competence_prior` and
  `recipe_aware_plus_plan`.

Phase 4b measured on the chains, October 29 to 31, 2026:

- The 2B sub-goal planner repeated sub-goals that already held until its
  rejection rounds ran out. Feedback now names held sub-goals and their
  values (`a86fe31`) and the ones re-proposed after being reported held
  (`e50ea85`). It also rejects `at_least` below 1 as always true
  (`95cce9a`).
- Saved results record their provenance and commit, and a baseline is
  reused only when its provenance matches (`d637e26`). Notebook 06 gained
  the `incremental` preset, and notebook 08 gained `REUSE_DESIGNS`.
- Notebook 06 `incremental` at `e50ea85`. On seed 51, `subgoal_vlm_memory`
  solved 64 of 64 on all three chains, against 45, 63 and 1 for
  `recipe_aware_plus_competence_prior`. Seed 202 confirmed it, against 49, 63
  and 2.

Open-ended task measured, October 31, 2026:

- Notebook 08 on seed 51 with Qwen3-VL-2B at `e50ea85`. The sub-goal VLM
  reached 4.88 and 4.94 achievements per episode, against 6.88 and 6.94 for
  the curriculum. It had spent its retries on achievements no skill
  produces.
- The open-ended sub-goals now offer only the achievements the skills
  unlock (`d8e4d70`). The sub-goal VLM rose to 6.38 and 6.44.
- Added `configs/vlm_planner_qwen3_8b.yaml` and ran notebook 08 with
  Qwen3-VL-8B on seeds 51 and 202 (`6ff3b5d`, `7853c8d`). The notebooks now
  keep the Hugging Face cache on the shared volume. The README's
  [Phase 4b results](README.md#phase-4b-results) have the tables.

Sixteen skills added on October 31, 2026 (`740103b`):

- Added nine skills after `craft_stone_pickaxe`, in one step so the executor
  trains once instead of once per stage:
  - `craft_stone_sword`
  - `collect_coal`
  - `place_stone`
  - `place_furnace`
  - `collect_sapling`
  - `place_plant`
  - `eat_cow`
  - `sleep`
  - `eat_plant`

  Each has registry entries, scenario support and planner rules. Sleep
  length follows fatigue, and the eat effects match Craftax: +6 food for a
  cow and +4 for a plant, capped at 9.
- Added the `sixteen_skill_smoke` and `sixteen_skill` presets to notebook
  07, with 256 environments and 26M steps.
- Per-skill tables across evaluation and planning now derive from the
  registry. The chain switches branch over every skill, because
  `lax.switch` clamps an out-of-range index to the last branch.
- Memory treats cows as obtainable. Model induction offers the survival
  fields to sixteen-skill models.
- Notebook 03 benchmarks the new skills at horizon 16. Notebook 08 offers
  every skill the selected executor knows.
