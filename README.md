# Tiny-SIMA

Tiny-SIMA is a small research project about building an agent that plays
Craftax, a 2D open-world survival game, with two levels of control:

- A frozen open-weight vision-language model (VLM) sets sub-goals, and a
  symbolic planner chooses the skills that reach them.
- A goal-conditioned reinforcement-learning policy carries each skill out.

It trains and runs on one accessible GPU, such as an RTX 4090, L40, or A100,
and every experiment has a Jupyter notebook entry point.

Through Phase 4 the VLM chose each skill directly. Those results, the
development history, and the dated changelog are in
[ARCHIVE.md](ARCHIVE.md).

## Research question

Can a frozen VLM, acting as a hierarchical planner, turn a natural-language
objective into sub-goals that a symbolic planner and a learned executor can
reach, using only what the agent can observe and the game's published
mechanics?

Phase 4 asked whether the VLM could choose each skill directly, with a
measured model of the executor as the Can term. On the stone-pickaxe chain it
could not: the best per-turn condition solved 43 of 64 episodes. A plan veto
that searches the skill registry towards the chain's formal goal solves 63 and
64 of 64. That veto reads two things the agent does not have: the formal
goal, which we wrote by hand, and world facts scanned from the whole 64x64
map. It is therefore kept as an upper bound, not as the method.

Phase 4b keeps the registry, which only encodes the game's published crafting
rules, and removes both privileged inputs:

- The VLM supplies the goal, as sub-goals.
- An episode memory supplies the world facts, from what the agent has seen.

The comparisons that answer the question:

- Sub-goals against per-turn skill choice, which is what the hierarchy buys.
- VLM sub-goals against the formal goal given as one sub-goal, which is the
  VLM's decomposition error.
- Memory belief against the map scan, which is the cost of honest perception.
- The sub-goal planner against the plan veto, which is the distance left to
  the upper bound.
- On an open-ended task with no recipe to follow, VLM sub-goals against
  per-turn skill choice and against a scripted curriculum.
- A skill model the VLM writes from a manual and corrects from executor runs,
  against the hand-written registry.

## Status

As of October 28, 2026:

- **Phases 1 to 3 are complete.** The environment, PPO baseline, skill
  benchmark, and goal-conditioned executor are done. The current executor
  knows seven skills and completes the default scripted plan in 64 of 64
  closed-loop episodes.
- **Phase 4, the per-turn planner, is measured and closed.** No per-turn VLM
  condition reaches the collect-first scripted plan on the stone-pickaxe
  chain. The plan veto does, as an upper bound. See
  [Phase 4 baselines](#phase-4-baselines).
- **Phase 4b is implemented and tested, but not yet measured on the GPU.**
  This covers:
  - VLM sub-goals over a symbolic planner, with episode memory;
  - richer observations and replanning triggers;
  - failure causes;
  - an open-ended achievement task with three planner designs;
  - a VLM feasibility study;
  - SCALAR-style learning of the skill model.
- **Phase 5's structured memory is built**, as the Phase 4b episode memory.
  Phase 6, full evaluation, has not started.

## System overview

```text
Objective + observation (frame, text map, vitals, creatures, achievements)
        |                                   ^
        v                                   |
VLM: proposes up to four sub-goals          | asked again only on a trigger:
  {field, at_least} over inventory,         |   start, sub-goals done, skill failed,
  vitals and achievements                   |   sub-goal timed out, low vitals,
        |                                   |   threat, nightfall, unreachable
        v                                   |
Symbolic planner: breadth-first search over the skill model
  - unreachable sub-goal -> one-line reason back to the VLM (two retries)
  - otherwise the first step of a shortest plan, ties to the competence prior
        |                                   |
        v                                   |
Goal-conditioned PPO-RNN executor (JAX)     |
        |                                   |
        v                                   |
Craftax actions until success, timeout, or lost precondition
        |                                   |
        +-- outcome, failure cause, inventory change --> episode memory
                world facts from what was seen ----------^
```

### Sub-goal planner

`SubgoalPlanner` in [planner/subgoals.py](src/tiny_sima/planner/subgoals.py)
follows the LLM-Modulo pattern (Kambhampati et al., 2024): the VLM proposes,
and a sound external checker verifies and gives feedback.

- **The VLM sets sub-goals and never names a skill.** Outlines constrains its
  answer to up to four `{field, at_least}` pairs. The fields are the
  inventory and drink by default. The open-ended task adds the achievements
  the agent's skills unlock (`open_ended_subgoal_fields`). Offered all 22,
  the 2B model spent its retries on ones no skill produces.
- **The prompt.** The system prompt carries the game mechanics from the
  registry. Each request adds:
  - the objective;
  - the observation;
  - the memory summary;
  - the current sub-goals;
  - the last eight sub-goal events;
  - why the VLM is being asked now, for example `The last skill failed:
    collect_stone timeout, target not seen.`
- **The symbolic planner checks and follows.** `check_subgoal` in
  [planner/symbolic.py](src/tiny_sima/planner/symbolic.py) searches the skill
  model breadth first. A sub-goal it cannot reach goes back to the VLM with a
  reason, such as `inventory.iron_pickaxe needs 1 more: no available skill
  produces it`. Rejections are batched across episodes. Every chosen skill is
  the first step of a shortest plan to the current sub-goal. Ties go to the
  executor's competence prior.
- **Triggers.** The VLM is called only at these events:
  - **The episode starts.**
  - **Every sub-goal is reached** and the objective is not complete.
  - **A skill fails.**
  - **A sub-goal times out:** it is not reached within its plan length plus
    `subgoal_slack = 4` skills.
  - **A vital drops** to its threshold or below. This fires once per spell.
    The thresholds default to drink alone, and `ALL_VITAL_THRESHOLDS` covers
    health, food, drink and energy.
  - **A hostile creature comes close**, within `threat_radius`. Off by
    default.
  - **Night falls**, if `replan_at_night` is set. Off by default.
  - **The current sub-goal becomes unreachable.**

  On a scripted executor, a correct plan costs one call per episode, against
  one per decision for a per-turn planner.
- **The goal source is pluggable.** Anything with `propose_subgoals(requests)`
  (`SubgoalSource`) can replace the VLM. The planner checks its answers
  exactly as it checks the VLM's. `ScriptedGoalSelector` in
  [planner/goal_selector.py](src/tiny_sima/planner/goal_selector.py) is the
  scripted curriculum for the open-ended task.

### Observation

[planner/observation.py](src/tiny_sima/planner/observation.py) turns the
Craftax state into text the VLM can read beside, or instead of, the frame:

- **`local_map`:** the 7x9 view as characters, with a legend in the system
  prompt.
- **`vitals`:** health, food, drink and energy, plus sleeping, daylight and
  night.
- **`mobs_in_view`:** each creature's type and row and column offset from the
  agent, nearest first.
- **`achievements`:** the achievements unlocked so far.

The presets are the ablation: `image_only` (the default, and byte-identical
to the prompts before the text observation), `text_only`, `image_and_text`
and `image_and_state`. Text-only requests send no image, and the prompts say
"the local map" instead of "the screenshot".

### Episode memory

`EpisodeMemory` in [memory/episode.py](src/tiny_sima/memory/episode.py)
updates at every skill boundary. It keeps:

- every resource block seen in the 7x9 view: tree, stone, water, coal, iron,
  diamond, sand, lava and the rest, as well as placed tables, furnaces and
  plants;
- the creatures seen, forgotten after three boundaries out of view;
- a transition log of each skill's outcome, inventory change, inventory
  extremes during the skill, and failure cause;
- how many collect attempts in a row have failed.

The symbolic planner reads `world.<block>_reachable` facts from the memory,
not from the full map. A natural block counts as reachable until two failed
collects with none known rule it out. A placed block counts only once one has
been seen.

`classify_failure` labels every failed skill with one cause, in this order of
precedence:

1. died;
2. episode ended;
3. food, drink or energy depleted;
4. health low;
5. infeasible;
6. precondition lost;
7. target not seen;
8. target not reached;
9. timeout.

The memory summary and the trigger text both carry the cause, for example
`collect_stone timeout (target not seen): no change`.

### Skill model

The symbolic planner searches a `SkillModel`, which defaults to the registry.
The registry holds each skill's declared `requires` and `effects` over
inventory, vitals and world facts. Achievement unlocks are derived from the
same skills. A search keeps only the achievement fields that the goal reads,
so plans over the chain goals are unchanged.

[planner/model_learning.py](src/tiny_sima/planner/model_learning.py) and
[planner/model_rounds.py](src/tiny_sima/planner/model_rounds.py) build that
model the way SCALAR does (Zabounidis et al., arXiv 2603.09036):

1. **Induction.** The VLM reads
   [docs/craftax_classic_manual.md](docs/craftax_classic_manual.md), which
   names every recipe's ingredients but none of their amounts. Under a
   constrained schema, it writes each skill's requirements and effects.
2. **Validation.** SCALAR's feasibility and novelty checks:
   - every skill's requirements are reachable from the start by the other
     skills;
   - every skill changes something;
   - no two skills change the same things.

   A model that fails goes back to the VLM once, whole. Whatever still
   fails becomes narrow questions with fixed answers, for example "which
   skill makes the crafting table reachable?" or "must the player hold
   stone to craft a wood pickaxe?". Skills that still fail are dropped.

   A requirement that an item be absent, such as "collect stone without a
   wood pickaxe", holds at the start and can hide a deadlock from
   validation. So even a model that passes is asked "must the player hold
   a wood pickaxe first?". Yes makes the skill need the item, no drops the
   requirement, and validation then sees what the inversion hid.
3. **Correction.** Successful executor transitions fix each skill's
   inventory effects and requirements deterministically, using the
   inventory extremes recorded during the skill.
4. **Failure analysis.** A failure the model predicted to succeed, for a
   reason other than a missing target, goes back to the VLM. The VLM sees
   successful pre-states beside it and names at most one missing
   requirement. The requirement replaces any the skill has on that field
   that it contradicts.

`run_model_learning_rounds` alternates closed-loop episodes and correction.
It starts from the VLM's induced model, or from a registry with one wrong
cost (`corrupt_registry`). Each round reports:

- chain success;
- wasted skill requests;
- the model's distance from the registry.

### Executor

One recurrent PPO policy, conditioned on a learned embedding of the skill ID,
handles seven skills:

- `collect_wood`
- `collect_stone`
- `drink_water`
- `place_table`
- `craft_wood_pickaxe`
- `craft_wood_sword`
- `craft_stone_pickaxe`, which is opt-in through `num_skills: 7`

The first six keep their embedding indices, so six-skill checkpoints, priors
and prompts load unchanged.

Episodes pay one sparse reward on success. The executors are trained on the
`randomized` scenario. It undoes each simplification of `local_navigation`:
the cleared view, the single target, the minimal inventory and the dawn
start. That closed the plan-order gap of the baseline executor,
`runs/skill-ppo-local-1m`, from +0.359 to -0.016, and that executor is kept
as the paired reference. The selected seven-skill executor is recorded in
`runs/executor-improvement-stone_pickaxe/selected_executor.json` by notebook
07. On its own skills it is not the ceiling: on the stone-pickaxe chain the
scripted oracle succeeds in 62 of 64 episodes and the collect-first plan in
64.

The chain worlds clear creatures and switch spawning off, so the executor was
trained without them. The open-ended task runs with creatures off and on, so
their cost can be measured.

### Tasks

**Chains.** Each episode is one persistent world with a fixed objective, and
success is read from the final Craftax state. An episode ends:

- at success;
- after twice the reference plan's length in decisions;
- when one skill has failed twice.

The reference order is private to the scripted baselines. The chains are:

| Chain | Objective | Reference skills | Decision limit |
| --- | --- | ---: | ---: |
| `wood_to_stone` | wood, wood, table, wood, wood pickaxe, stone | 6 | 12 |
| `wood_to_stone_pickaxe` | the above, then a fourth wood and a stone pickaxe | 8 | 16 |
| `stone_pickaxe_terse` | the same chain, objective "Get a stone pickaxe." | 8 | 16 |
| `stone_pickaxe_and_sword` | a stone pickaxe and a wood sword | 10 | 20 |

**Open-ended achievements.**
[evaluation/open_ended.py](src/tiny_sima/evaluation/open_ended.py) runs an
unmodified Craftax-Classic world with the objective "Unlock as many
achievements as you can."

- **Budget:** 80 skill calls or 3,000 environment steps. An episode also ends
  on death, or when the planner has nothing left to propose.
- **Scoring:** as in Crafter, the per-achievement success rate and their
  geometric mean. The report also breaks results down by tier: basic, wood,
  stone, iron and diamond, survival, combat.
- **The three designs**, on matched worlds with creatures off and on:
  1. `saycan_recipe_aware`: the per-turn VLM, with the recipe-aware prompt
     and the competence prior.
  2. `subgoal_vlm`: VLM sub-goals over the achievements its skills unlock.
     It uses all four vitals, a threat radius of 3 and nightfall triggers.
  3. `subgoal_scripted`: the same sub-goal planner with
     `ScriptedGoalSelector` in place of the VLM. The selector gives survival
     first, then a fixed tech-tree order, and skips a goal after two
     failures. It is what the VLM has to beat.

With seven skills only seven achievements can be unlocked, so the score
ceiling is set by the skill set. For now, the designs differ in how
reliably, cheaply and safely they reach that ceiling. Hard-tier and survival
skills come next; see [Roadmap](#roadmap).

## Phase 4 baselines

The per-turn arm and the upper bound that Phase 4b is compared against. They
come from notebook 06 on seed 51, with the seven-skill `randomized-seed2`
executor, on the stone-pickaxe chain (8 skills, 16 decisions, 64 paired
episodes):

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

The full tables are in the archive:

- [the stone-pickaxe tables](ARCHIVE.md#stone-pickaxe-chain-seed-51), with
  the ladder, the search, surplus and detour states, and every paired test;
- [the six-skill ablations](ARCHIVE.md#planner-ablations-six-skill-chain);
- [the feasibility probes](ARCHIVE.md#feasibility-measurements-six-skill-chain);
- [the executor improvement](ARCHIVE.md#executor-improvement-result);
- [the Phase 4 closing diagnosis](ARCHIVE.md#closing-phase-4).

The notebook 06 cells that produced them were removed once Phase 4 closed.
They are at commit `282ef35`.

### Refuted approaches

| Approach | Measurement that refuted it | Status |
| --- | --- | --- |
| Executor critic `V(s, skill)` as the Can term | AUC 0.369 and 0.437 held out, below chance. Neither training scenario spawns an impossible target, so the critic learned which skill was asked for, not whether it is possible. | Fitted and scored on every probing run; retired from the ladder. |
| Competence prior under a saturated six-skill executor | Every cell p = 1.000: AUC 0.500, closed loop identical to reachability. | Refitted for each executor; informative again on the seven-skill one. |
| Grounded `available_first` prompt | 50 against 62 of 64 for the ungrounded prompt with reachability (p = 0.002); 162 of its 163 off-objective decisions were the VLM's own first choice. | `PlannerPromptStyle.AVAILABLE_FIRST` stays in the library. |
| Held-craft veto | 59 against 61 of 64 on seed 51: the vetoed craft was replaced by surplus wood. | `SkipHeldCraftFeasibility` stays in the library. |
| More training on `local_navigation` alone | A step-matched control left the plan-order gap at +0.318, the baseline's own. | Kept as the control run in notebook 07. |
| Symbolic preconditions as the Can term | They test the whole 64x64 map, which nearly always has the block. | Kept inside every Can term; retired as a condition. |
| Hard reachability gate | The scripted chain fell from 1.0 to 0.5 success, ending in `no_viable_skill`. | Replaced by grading, with unknown targets at 0.25. |
| Sampled decoding | `vlm_only` fell from 43 to 38 accepted of 48 on identical states. | Greedy decoding by default. |
| Per-skill advantage normalization | It collapsed the unsolved interaction skills in a debug ablation. | Available, off by default. |
| Larger PPO batches at equal steps | 256 environments lost 64% of return, and 512 lost 78%. | 64 environments remain the default. |

## Experiments to run

All of them are implemented and tested on CPU with stub backends. None has
run on the GPU yet.

1. **Phase 4b on the chains.** Notebook 06, section "Phase 4b", on seed 51
   with the seven-skill executor. It runs three conditions on the
   stone-pickaxe, terse and pickaxe-and-sword chains:
   - `subgoal_formal_goal`, the hierarchy's ceiling under the memory belief;
   - `subgoal_vlm_memory`;
   - `subgoal_vlm_map_scan`.

   Results go to `subgoal_loop_seed51_n64.json` beside the checkpoint. Read
   `subgoal_formal_goal` first. Then pair `subgoal_vlm_memory` against the
   per-turn arm and the plan veto.
2. **The feasibility study.** Notebook 06, section "Can the VLM judge when a
   skill will succeed?", after the paired probes. It is
   the reverse of the Can term: the VLM rates every skill from 0 to 10 for
   whether the executor can finish it in 64 steps, on the probe states of
   seeds 7 and 23. It is scored against the competence prior, the product of
   the two, and a logistic blend fitted on seed 7 and scored on 23, by AUC,
   ECE and Brier score, overall and per skill.
   - It runs with the image alone and with the text map.
   - `FEASIBILITY_INCLUDE_WORLD = False` drops the reachability readout from
     the prompt, which tests whether the VLM only reads back the distances
     the prior already uses.
3. **Learning the skill model.** Notebook 06, section "Learning the skill
   model (SCALAR-style)", after Phase 4b. It starts from three models:
   - the registry;
   - the VLM's induced model;
   - a registry corrupted so that a table costs one wood.

   Each is run under the plan veto and under `subgoal_formal_goal`, with
   and without correction, for three rounds of 16 episodes. The correction
   can also add harmless requirements that hold before every success, such
   as a wood pickaxe for the stone pickaxe. So even the registry does not
   stay exactly the registry.
4. **The open-ended task.** Notebook 08 runs the three designs with
   creatures off and on. It defaults to 4 episodes per design and setting,
   because a sub-goal design can ask the VLM up to three times per decision;
   raise `NUM_EPISODES` for the comparison. Results go to
   `runs/open-ended/`.
5. **The observation ablation.** Notebook 08's `OBSERVATION` setting picks
   the preset for the open-ended designs, and defaults to `image_and_state`.
   The feasibility study runs the image-only and text-only arms. The chain
   conditions in notebook 06 still use the image alone.

**The bar** stays the collect-first scripted plan on each chain, and
`subgoal_scripted` on the open-ended task.

### Seeds

Each seed is reserved for one purpose, so that selecting on one never biases
a result reported on another:

| Seed | Use |
| ---: | --- |
| 7 | Fitting the critic calibration, the competence prior and the feasibility blend |
| 23 | Held-out scoring of every feasibility head |
| 51 | Planner closed-loop and open-ended evaluation, and Phase 4's prefix ladder |
| 101 | Executor benchmark and run selection, in notebook 07 |
| 202 | Randomized-scenario evaluation in notebook 07; confirmation of planner choices in notebook 06 (`CONFIRM`) |

Boundary states mined for training must come from seeds disjoint from all of
these.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| 1. Environment and baseline | A reproducible Craftax PPO-RNN setup with throughput and memory measured | Complete |
| 2. Skill benchmark | Skills with preconditions, success predicates, sparse rewards, and scripted and random baselines | Complete |
| 3. Goal-conditioned executor | One policy for every skill, trained on `local_navigation`, then on the `randomized` scenario | Complete; seven skills |
| 4. Per-turn planner | A frozen VLM in a SayCan loop, with a grounded ablation ladder | Complete; no per-turn condition reaches the bar, and the plan veto is the upper bound |
| 4b. Sub-goal planner | VLM sub-goals, a symbolic planner, memory, richer observation, the open-ended task, and a learned skill model | Implemented and tested; GPU runs pending |
| 5. Memory and visual grounding | Structured episodic memory; symbolic against pixel inputs; frame history; paraphrased instructions | Memory and the image/text ablation built; the rest not started |
| 6. Full evaluation | The full Craftax-Classic achievement set, multi-seed ablations, compute accounting, and documented failures | Not started |

Next steps, in order:

1. **Run Experiments 1 to 4**, then confirm the chain results on seed 202
   with `CONFIRM = True`.
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

Deferred from the Phase 3 plan: learned skill termination, text-embedding
skill conditioning, and paraphrase evaluation.

## Quick start

Use Python 3.11 or 3.12. The pinned Flax version needs Python 3.11 or newer.

```bash
./scripts/setup_env.sh
source .venv/bin/activate
# On Linux with an NVIDIA GPU, plus the optional VLM planner:
pip install -e ".[dev,notebook,cuda12,vlm]"
```

Run the checks. Use the virtual environment's Python for pytest; the
real-model VLM test is opt-in.

```bash
python -m pytest
pyright
ruff check .
TINY_SIMA_RUN_VLM_SMOKE=1 python -m pytest -q tests/test_vlm_backend_smoke.py
```

Train the executors:

```bash
# Baseline executor, the paired reference: one million steps on local_navigation.
tiny-sima-train-skills --config configs/skill_ppo_local_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_local_1m.yaml

# Six skills on the randomized scenario: about 10M steps, 96 environments.
tiny-sima-train-skills --config configs/skill_ppo_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_randomized_10m.yaml

# Seven skills, the current executor: 112 environments, 16 per skill.
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_10m.yaml
```

Notebook 07's `stone_pickaxe` preset trains three seeds of the seven-skill
configuration and records the selected one.

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
| `06_evaluate_vlm_planner` | The competence prior, the VLM feasibility study, Phase 4b sub-goals, and skill-model learning |
| `07_improve_executor` | Executor diagnostics, randomized training, and run selection |
| `08_open_ended_achievements` | The three planner designs on the open-ended achievement task |

Notebooks 06 and 07 have presets:

- Notebook 06: `smoke`, `full`, and `incremental`.
- Notebook 07: `smoke`, `full`, and `stone_pickaxe`.

Both reuse completed work rather than repeating it.

Notebook 06's `incremental` preset is for checking a change to the VLM
sub-goal planner:

- It runs the sub-goal arms on the full run's episodes.
- It reloads the competence prior instead of probing.
- It skips the feasibility study and model learning.
- It reloads each baseline arm from the seed's saved results.

Notebook 08 does the same for the designs listed in `REUSE_DESIGNS`.

Every saved result records its provenance: the seed, the episode count, the
executor, the VLM configuration, the competence prior, and for notebook 08 the
observation and limits. A baseline is reused only when its provenance matches
the current run's, field for field; otherwise it is rerun. Results saved before
provenance was recorded are always rerun.

Provenance cannot see code. Each result also records the commit it ran at, so
rerun a baseline whose own code changed, such as the per-turn VLM prompt.

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

- **Separate planner errors from executor errors.** The scripted plans and
  `subgoal_formal_goal` bound what the executor and the hierarchy finish, so
  the gap below them is the planner's.
  Invalid planner output, no viable skill, skill failures and their causes
  are all reported separately.
- **Pair everything that can be paired.** Conditions share worlds, so
  comparisons use the discordant episodes rather than two success rates.
- **Fit and score on different seeds.** An isotonic fit's in-sample
  calibration error is zero by construction.
- **Keep privileged inputs out of the method.** The formal goal and the map
  scan are upper bounds; the method reads the objective, the observation and
  its own memory.
- **Keep refuted approaches runnable, with the number that refuted them.**
- **Report compute.** Every major run records wall-clock time and peak JAX
  memory, and the sub-goal conditions record VLM calls per episode.

## Compute scope and limitations

Everything runs on one GPU:

- The VLM is small and frozen, and runs only at triggers or skill boundaries.
- The executor uses symbolic Craftax observations.
- Parameter-efficient fine-tuning is reserved for the case where prompting
  proves insufficient.

Tiny-SIMA does not reproduce SIMA's scope:

- Craftax is turn-based and 2D.
- Its action space is simpler than keyboard and mouse.
- The VLM sets goals over a fixed skill catalog rather than issuing game
  actions.

The aim is a small system whose components and results can be understood and
reproduced.

## Repository structure

```text
tiny-sima/
├── README.md               current setup and status
├── ARCHIVE.md              superseded results, history, and changelog
├── pyproject.toml
├── configs/                PPO, skill-PPO, and VLM planner configurations
├── docs/                   the qualitative game manual for skill-model induction
├── notebooks/              00-08, as listed above
├── scripts/                setup and standalone baseline scripts
├── src/tiny_sima/
│   ├── envs/               Craftax wrappers, skill scenarios, chains, open-ended worlds
│   ├── goals/              skill registry, predicates, achievements, reachability
│   ├── executor/           PPO and goal-conditioned PPO-RNN, checkpoints
│   ├── planner/            SayCan and sub-goal planners, symbolic search, observation,
│   │                       goal selector, skill-model learning, VLM scorer, feasibility
│   ├── evaluation/         skill, chain, open-ended, planner, feasibility, diagnostics
│   └── memory/             episode memory and failure causes
└── tests/
```
