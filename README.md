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

As of October 31, 2026:

- **Phases 1 to 3 are complete.** The environment, PPO baseline, skill
  benchmark, and goal-conditioned executor are done. The executor in use
  knows seven skills and completes the default scripted plan in 64 of 64
  closed-loop episodes. The registry now has sixteen skills, and the
  sixteen-skill executor is in its smoke run.
- **Phase 4, the per-turn planner, is closed.** No per-turn VLM condition
  reached the collect-first scripted plan on the stone-pickaxe chain. The
  plan veto did, as an upper bound.
- **Phase 4b answers the research question on the chains.** With
  Qwen3-VL-2B, VLM sub-goals over episode memory solve 64 of 64 episodes on
  all three chains, on seeds 51 and 202. That matches the formal goal and does
  at least as well as the plan veto, without either privileged input. See
  [Phase 4b results](#phase-4b-results).
- **On the open-ended task, the seven skills cap the score.** The scripted
  curriculum reaches about 7 of the 7 achievements its skills unlock.
  Qwen3-VL-8B sub-goals come close but do not beat it. They do beat per-turn
  skill choice when creatures are on.
- **Not yet run:** the VLM feasibility study, skill-model learning, and the
  observation ablation.
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
handles the first `num_skills` skills of the registry:

- **Six original skills:** `collect_wood`, `collect_stone`, `drink_water`,
  `place_table`, `craft_wood_pickaxe` and `craft_wood_sword`.
- **`craft_stone_pickaxe`, with `num_skills: 7`.** This is the executor in
  use.
- **Nine more, with `num_skills: 16`:**
  - `craft_stone_sword`
  - `collect_coal`
  - `place_stone`
  - `place_furnace`
  - `collect_sapling`
  - `place_plant`
  - `eat_cow`
  - `sleep`
  - `eat_plant`

  Together they unlock 16 of the 22 Craftax-Classic achievements. Only the
  iron tier, diamond and combat stay out of reach.

Each skill keeps its embedding index as the catalog grows, so smaller
checkpoints, priors and prompts load unchanged. A larger catalog grows the
embedding, so it trains from scratch.

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

The planners offer every skill the executor knows, and each skill unlocks
one achievement, so the skill set caps the score:

- seven achievements with the seven-skill executor;
- sixteen with the sixteen-skill one.

The designs differ in how reliably, cheaply and safely they reach that cap.
Notebook 08 loads the executor that notebook 07 selected under
`EXECUTOR_PRESET`.

## Phase 4b results

### Chains

Notebook 06's `incremental` preset ran at commit `e50ea85`. It used
Qwen3-VL-2B and the seven-skill `randomized-seed2` executor, with 64 paired
episodes per condition. The table gives episodes solved on seed 51 and on
seed 202 (`CONFIRM = True`):

| Condition | Stone pickaxe | Terse | Pickaxe and sword |
| --- | ---: | ---: | ---: |
| `scripted_collect_first`, the bar | 64 / 64 | 64 / 64 | 63 / 63 |
| `recipe_aware_plus_competence_prior`, per-turn | 45 / 49 | 63 / 63 | 1 / 2 |
| `recipe_aware_plus_plan`, plan veto | 64 / 64 | 64 / 64 | 63 / 62 |
| `subgoal_formal_goal`, the hierarchy's ceiling | 64 / 64 | 64 / 64 | 64 / 64 |
| `subgoal_vlm_memory`, the method | 64 / 64 | 64 / 64 | 64 / 64 |
| `subgoal_vlm_map_scan` | 64 / 64 | 64 / 64 | 64 / 64 |
| VLM calls per episode, `subgoal_vlm_memory` | 3.9 / 3.9 | 2.7 / 2.4 | 1.2 / 1.2 |

What they show:

- **Sub-goals beat per-turn skill choice.** On the stone-pickaxe chain the
  sub-goal planner won 19 and 15 discordant episodes against 0. On the
  pickaxe-and-sword chain it won 63 and 62 against 0 (p < 0.001 on both
  chains and both seeds). The two tie on the terse chain.
- **The VLM's decomposition costs nothing.** `subgoal_vlm_memory` matches
  `subgoal_formal_goal` on every episode.
- **Honest perception costs nothing here.** The memory belief and the map
  scan produce identical traces on these chains.
- **The method reaches the upper bound without its privileged inputs.** It
  ties the plan veto on two chains. On pickaxe and sword it wins the only
  discordant episodes, 1 and 2 (p = 1.0 and 0.5).
- **It is cheap.** The sub-goal planner needs 1.2 to 3.9 VLM calls per
  episode. The per-turn arm needs one per decision: 9.3 to 19.7.
- **The 2B planner needed feedback on sub-goals it already held.** Before
  `a86fe31` and `e50ea85`, it re-proposed them until its rejection rounds
  ran out. The remaining rejections, about one per episode, are mostly
  proposals with `at_least` 0.

Phase 4b's chain comparisons are now at ceiling, so they no longer
discriminate. The open-ended task carries the comparison forward.

### Open-ended achievements

Notebook 08 ran the seven-skill executor with the `image_and_state`
observation, 16 episodes per design and setting, and an 80-skill budget.

**Achievements per episode, creatures off / on.** The cap is 7.

| Design | 2B, seed 51 | 8B, seed 51 | 8B, seed 202 |
| --- | ---: | ---: | ---: |
| `saycan_recipe_aware` | 5.06 / 3.94 | 6.88 / 5.75 | 6.94 / 5.62 |
| `subgoal_vlm` | 6.38 / 6.44 | 7.00 / 6.88 | 6.69 / 6.50 |
| `subgoal_scripted`, the bar | 6.88 / 6.94 | 6.88 / 6.94 | 7.06 / 6.88 |
| VLM calls per episode, `subgoal_vlm` | 7.1 / 7.0 | 6.8 / 6.2 | 7.6 / 6.2 |
| VLM calls per episode, `saycan_recipe_aware` | 38.9 / 20.8 | 41.2 / 23.2 | 37.0 / 17.9 |

The 2B runs are at `d8e4d70`, the 8B runs at `5209982` (seed 51) and
`7853c8d` (seed 202). The curriculum calls no VLM, and its seed-51 results
are identical across the 2B and 8B runs.

What they show:

- **The 8B model helps both VLM designs.** On seed 51, SayCan gains 1.8
  achievements per episode in both settings, and the sub-goal VLM gains 0.4
  to 0.6.
- **No VLM design beats the scripted curriculum.** Paired by world, the 8B
  sub-goal VLM against the curriculum is:
  - 2 better against 1 worse with creatures off, and 1 against 2 with them
    on, on seed 51;
  - 1 against 5 off and 1 against 4 on, on seed 202.

  None is significant (p ≥ 0.22). Both sit at the cap, which leaves the VLM
  no room to win.
- **With creatures on, sub-goals beat per-turn choice.** The 8B sub-goal VLM
  beats SayCan 11 against 1 on seed 51 (p = 0.006) and 11 against 2 on seed
  202 (p = 0.022). SayCan loses to the curriculum 1 against 13 and 1 against
  12 (p ≤ 0.003). With creatures off, no difference is significant.
- **SayCan dies in every episode.** It spends its decisions on surplus wood,
  stone and pickaxes: on seed 202 with creatures off, 277 of its 592
  requests were `collect_wood`, and 16 were `drink_water`. With creatures
  off it dies near step 340, and every episode's last skill ends on a lost
  precondition.
- **The sub-goal designs stop when nothing reachable is left.** The VLM's
  last proposal is usually an iron sword or iron pickaxe, which the planner
  rejects because no skill produces it: in 7 to 10 of 16 episodes per
  setting. The curriculum stops when its list runs out.
- **The Crafter score misleads at this cap.** Its geometric mean over 22
  achievements rewards rare incidental unlocks: a sapling, a placed stone, a
  furnace, coal or iron, picked up on the way to something else. SayCan's
  longer episodes collect more of these, so it has the highest score with
  creatures off (6.43 and 5.35). It does not have the most achievements per
  episode. We report achievements per episode first.

Results and manifests are in `runs/open-ended/seed<N>-<timestamp>/`.

## Refuted approaches

Most of these come from Phase 4. Its tables are in the archive:

- [the Phase 4 baselines](ARCHIVE.md#phase-4-baselines-as-of-october-28-2026);
- [the stone-pickaxe tables](ARCHIVE.md#stone-pickaxe-chain-seed-51);
- [the six-skill ablations](ARCHIVE.md#planner-ablations-six-skill-chain);
- [the feasibility probes](ARCHIVE.md#feasibility-measurements-six-skill-chain);
- [the executor improvement](ARCHIVE.md#executor-improvement-result);
- [the Phase 4 closing diagnosis](ARCHIVE.md#closing-phase-4).

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

All of them are implemented and tested on CPU with stub backends.

1. **Train the sixteen-skill executor.** In notebook 07, run
   `PRESET = "sixteen_skill_smoke"` first. Then run `sixteen_skill`: three
   seeds of 26M steps, with 256 environments, 16 per skill. Run selection
   uses the seven-skill chains, so check the nine new skills in the
   per-skill training table before going on. Compare each against its
   scripted baseline in notebook 03.
2. **The open-ended task with sixteen skills.** Run notebook 08 with
   `EXECUTOR_PRESET = "sixteen_skill"` and Qwen3-VL-8B, on seed 51, then on
   seed 202. The cap rises from 7 to 16, which gives the VLM room to beat the
   curriculum. The competence prior is refitted over the whole catalog and
   saved under its own name.
3. **The feasibility study.** Notebook 06, section "Can the VLM judge when a
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
4. **Learning the skill model.** Notebook 06, section "Learning the skill
   model (SCALAR-style)", after Phase 4b. It starts from three models:
   - the registry;
   - the VLM's induced model;
   - a registry corrupted so that a table costs one wood.

   Each is run under the plan veto and under `subgoal_formal_goal`, with
   and without correction, for three rounds of 16 episodes. The correction
   can also add harmless requirements that hold before every success, such
   as a wood pickaxe for the stone pickaxe. So even the registry does not
   stay exactly the registry.
5. **The observation ablation.** Notebook 08's `OBSERVATION` setting picks
   the preset for the open-ended designs. It defaults to `image_and_state`,
   the only preset run so far. The feasibility study runs the image-only and
   text-only arms. The chain conditions in notebook 06 still use the image
   alone.

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
| 3. Goal-conditioned executor | One policy for every skill, trained on `local_navigation`, then on the `randomized` scenario | Complete with seven skills; the sixteen-skill executor is training |
| 4. Per-turn planner | A frozen VLM in a SayCan loop, with a grounded ablation ladder | Complete; no per-turn condition reaches the bar, and the plan veto is the upper bound |
| 4b. Sub-goal planner | VLM sub-goals, a symbolic planner, memory, richer observation, the open-ended task, and a learned skill model | Chains at the upper bound on two seeds; open-ended task at the seven-skill cap; feasibility, model learning and the observation ablation pending |
| 5. Memory and visual grounding | Structured episodic memory; symbolic against pixel inputs; frame history; paraphrased instructions | Memory and the image/text ablation built; the rest not started |
| 6. Full evaluation | The full Craftax-Classic achievement set, multi-seed ablations, compute accounting, and documented failures | Not started |

Next steps, in order:

1. **Run Experiments 1 and 2:** train the sixteen-skill executor, then run
   the open-ended task with it.
2. **Add the remaining six achievements:**
   - collect iron;
   - make an iron pickaxe;
   - make an iron sword;
   - collect diamond;
   - defeat a zombie;
   - defeat a skeleton.

   The iron tools need a table and a furnace both nearby, plus coal and
   iron. Diamond needs the iron pickaxe. Skeletons live only in the dark
   tunnel areas. Each skill needs registry entries, scenario support and
   executor training.
3. **Run Experiments 3 to 5**, the feasibility study, model learning and the
   observation ablation.

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

# Sixteen skills: the seven above plus the stone sword, coal, stone, furnace,
# sapling, plant, cow, sleep and ripe plant. 256 environments, about 26M steps.
tiny-sima-train-skills --config configs/skill_ppo_16_skill_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_16_skill_randomized_26m.yaml
```

Notebook 07's `stone_pickaxe` preset trains three seeds of the seven-skill
configuration and records the selected one; `sixteen_skill_smoke` and
`sixteen_skill` do the same for the sixteen-skill catalog. The chain
evaluations keep their seven-skill recipes; notebook 08's open-ended
evaluation offers every skill the selected executor knows.

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
- Notebook 07: `smoke`, `full`, `stone_pickaxe_smoke`, `stone_pickaxe`,
  `sixteen_skill_smoke`, and `sixteen_skill`.

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
