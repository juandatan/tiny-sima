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

As of November 7, 2026:

- **Phases 1 to 3 are complete, but the executors must be retrained.**
  Executors with seven, sixteen and eighteen skills were trained and
  evaluated on a JupyterHub machine that has since been shut down, and its
  `runs/` went with it. The registry now has twenty-two skills, one per
  Craftax-Classic achievement, so the next step is to train the
  twenty-two-skill executor on Runpod (see [Runpod](#runpod)).
- **Phase 4, the per-turn planner, is closed.** No per-turn VLM condition
  reached the collect-first scripted plan on the stone-pickaxe chain. The
  plan veto did, as an upper bound.
- **Phase 4b answers the research question on the chains.** With
  Qwen3-VL-2B, VLM sub-goals over episode memory solve 64 of 64 episodes on
  all three chains, on seeds 51 and 202. That matches the formal goal and does
  at least as well as the plan veto, without either privileged input. See
  [Phase 4b results](#phase-4b-results).
- **On the open-ended task, sub-goals beat per-turn skill choice, and the
  VLM ties the curriculum.** With the eighteen-skill executor and
  Qwen3-VL-8B, both sub-goal designs reach about 13 achievements per episode
  with creatures off, of the 15 possible without them, about 3 more than
  SayCan on seeds 51 and 202. The VLM does not beat the scripted curriculum.
- **With creatures on, survival limits the score.** Sheltering at night and
  letting the goal source choose its fights raised the sub-goal designs
  from about 10.5 to about 12 achievements on seed 51, but 13 to 16 of 16
  episodes still die. Staged fights show the executor fights zombies as well
  as a scripted chase, so the deaths come from fighting at low health.
  Showing the VLM the measured fight odds did not change its choices in the
  one run so far. The odds of not fighting were added after it and have not
  run.
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
  produces it`. The reachable sub-goals of an answer are kept, and the VLM
  is asked again only when none is. Rejections are batched across episodes.
  Every chosen skill is the first step of a shortest plan to the current
  sub-goal. Ties go to the executor's competence prior.
- **Triggers.** The VLM is called only at these events:
  - **The episode starts.**
  - **Every sub-goal is reached** and the objective is not complete.
  - **A skill fails.**
  - **A sub-goal times out:** it is not reached within its plan length plus
    `subgoal_slack = 4` skills.
  - **A vital drops** to its threshold or below. This fires once per spell.
    The thresholds default to drink alone. `ALL_VITAL_THRESHOLDS` covers
    health, food and drink at 5, so there is time to restock before night,
    and energy at 3, since sleeping in the open is when zombies hit hardest.
  - **A hostile creature comes close**, within `threat_radius`. Off by
    default. Each kind of creature raises its own alert, so a zombie that
    arrives while a skeleton is close still asks again. A mob in `defend_against` within that radius is also fought
    with its hunting skill, outside the plan: the fight neither spends the
    sub-goal's budget nor counts as its failure. By default every such mob
    is fought at once. With `choose_fights`, each answer carries `fight`,
    true or false, and the prompt lists the close threats with the
    player's health, so the goal source decides, for example to drink or
    walk on rather than fight at low health. Given `fight_odds`, a
    `FightOdds` table fitted by `fit_fight_odds` from the staged fights,
    the prompt also gives how often the executor won, and died, at that
    health, sword and creature count, and of not fighting: how often each
    errand in the table, such as drinking, got done, and the player died,
    staged with the same creatures close. Each fight answer records the
    win rate it was shown. A fight holds while the mob
    stays close; the latest answer decides. The open-ended VLM design
    makes this choice, and the curriculum always fights.
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
- **`craft_stone_pickaxe`, with `num_skills: 7`.** The executor of the
  Phase 4b chain results.
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
- **Two combat skills, with `num_skills: 18`:** `defeat_zombie` and
  `defeat_skeleton`. Each scenario places one live mob of its kind, a
  skeleton on a tunnel path, and success is a kill, not a despawn.
- **The iron and diamond stage, with `num_skills: 22`:** `collect_iron`,
  `craft_iron_pickaxe`, `craft_iron_sword` and `collect_diamond`. Iron needs
  a stone pickaxe and diamond an iron one. The iron tools need one wood,
  stone, iron and coal, and both a crafting table and a furnace nearby, so
  their scenarios put a furnace beside the table, diagonal to the tile the
  player crafts from.

  Together they unlock all 22 Craftax-Classic achievements.

Each skill keeps its embedding index as the catalog grows, so smaller
checkpoints, priors and prompts load unchanged. A larger catalog grows the
embedding, so it trains from scratch.

Episodes pay one sparse reward on success. The executors are trained on the
`randomized` scenario. It undoes each simplification of `local_navigation`:
the cleared view, the single target, the minimal inventory and the dawn
start. That closed the plan-order gap of the baseline executor,
`runs/skill-ppo-local-1m`, from +0.359 to -0.016, and that executor is kept
as the paired reference. Notebook 07 records the executor it selects under
each preset in `runs/executor-improvement-<preset>/selected_executor.json`,
which notebook 08 loads. The seven-skill executor was not the ceiling on its
own skills: on the stone-pickaxe chain the scripted oracle succeeded in 62
of 64 episodes and the collect-first plan in 64.

The chain worlds clear creatures and switch spawning off, so the executor was
trained without them. Clearing zeroes each creature's health as well as its
mask: Craftax revives any slot with health left on every `DO` action, so
before this fix a cleared creature could come back near the map corner. The open-ended task runs with creatures off and on, so
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
- **Skill interruption:** a planner decides only between skills, and a skill
  runs up to 64 steps, so a zombie that turns up mid-skill used to hit the
  player until the skill ended. `SkillInterruption` in
  [evaluation/chains.py](src/tiny_sima/evaluation/chains.py) ends a skill
  `interrupted` when a zombie comes within 3 tiles or the player is hit while
  food, drink and energy are above zero. The planner then decides again.
  `defeat_zombie` is never interrupted. An interrupted skill takes a skill
  call, but it does not count as a failure for SayCan's penalty, the
  sub-goal trigger or budget, or memory's miss counts. It applies to all
  three designs, and with creatures off it never fires.
- **Shelter at night:** with `shelter_at_night`, the sub-goal designs'
  goal source decides whether to wall the player in with stone for the
  night. Zombies cannot walk through stone, and sleeping in the open was
  when they hit hardest. Each answer carries `shelter`: `now`, `after` or
  `no`. `now` shelters at once and the sub-goals wait until the player is
  out. `after` reaches the answer's sub-goals first, such as drinking up
  before a room with no water, and then shelters without asking again.
  The prompt says whether the shelter could start now and, if not, why,
  whether the room it would use has water beside it, and how many steps
  are left until morning. The VLM chooses, so when to hide, and what to do
  first, is part of what the comparison measures, while the curriculum
  shelters now whenever it can. A shelter holds for the rest of the night
  while it can run, and lapses at dawn. `shelter` is a scripted macro in
  [planner/shelter.py](src/tiny_sima/planner/shelter.py), not a learned
  skill, so the executor needs no retraining. It fights off an adjacent
  zombie, mines stone if it has too little, builds a 2x2 room, preferring
  one walled partly by water to drink from, and sleeps inside. In the
  morning, or once food falls to 2, or drink does with no water to hand,
  it mines a way out. The room is 2x2 because Craftax has no turn action:
  the player faces a free tile only by stepping towards it, so it can face
  every wall of a 2x2 room from inside, but not the last wall of a
  one-tile room. It can start at night when the player has a wood pickaxe
  and a room is within 8 steps. It runs to its own 256-step horizon, and
  like a zombie fight it spends no sub-goal budget. Notebook 08 offers it
  with creatures only.
- **Traces:** each decision records the player's vitals before and after
  its skill, as `vitals_before` and `vitals_after`.
- **Scoring:** as in Crafter, the per-achievement success rate and their
  geometric mean. The report also breaks results down by tier: basic, wood,
  stone, iron and diamond, survival, combat.
- **The three designs**, on matched worlds with creatures off and on:
  1. `saycan_recipe_aware`: the per-turn VLM, with the recipe-aware prompt
     and the competence prior.
  2. `subgoal_vlm`: VLM sub-goals over the achievements its skills unlock.
     It uses all four vitals, a threat radius of 3 and nightfall triggers,
     and fights off a zombie within that radius. Memory counts a skeleton
     reachable only once one has been seen, since skeletons keep to the dark
     tunnels.
     When every answer is rejected and no earlier sub-goal is still
     reachable, it takes the curriculum's next sub-goal instead of ending
     the episode. These fallback calls are counted apart from VLM calls.
     The report gives their share of all answers. It also gives how often
     the VLM's own shelter and fight answers differ from what the
     curriculum would answer in the same state, where it shelters whenever
     it can and always fights. The VLM is shown the executor's measured
     zombie-fight odds, and those of `drink_water` and `collect_wood` run
     with the zombies close (`ERRAND_SKILLS`), which notebook 08 stages
     once per executor and saves beside it, and the report counts its fight answers by the odds
     shown (`by_odds`).
  3. `subgoal_scripted`: the same sub-goal planner with
     `ScriptedGoalSelector` in place of the VLM. The selector gives survival
     first, drink and food at 5 with no sleep rule, then a fixed tech-tree
     order, and skips a goal after two failures. It shares the zombie
     defence. It is what the VLM has to beat.

The planners offer every skill the executor knows, and each skill unlocks
one achievement, so the skill set caps the score:

- seven achievements with the seven-skill executor;
- sixteen with the sixteen-skill one;
- eighteen with the eighteen-skill one;
- all twenty-two with the twenty-two-skill one.

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

Notebook 08 ran the eighteen-skill executor (`eighteen_skill`,
`randomized-seed0`) with Qwen3-VL-8B, the `image_and_state` observation, 16
episodes per design and setting, and an 80-skill budget. The seven- and
sixteen-skill results are in the
[archive](ARCHIVE.md#readme-material-moved-on-november-7-2026).

**Achievements per episode, creatures off / on.** With creatures off, 15 of
the 18 achievements are possible.

| Design | Seed 51 | Seed 202 |
| --- | ---: | ---: |
| `saycan_recipe_aware` | 10.12 / 8.00 | 10.12 / 7.81 |
| `subgoal_vlm` | 12.94 / 12.06 | 13.06 / 8.12 |
| `subgoal_scripted`, the bar | 13.19 / 11.38 | 13.19 / 8.75 |
| Deaths with creatures on, of 16 (SayCan, VLM, curriculum) | 16, 16, 15 | 16, 16, 16 |

Seed 202 ran at `efdbc67`, before the survival changes: the zombie defence,
restocking at 5, sheltering at night and the goal source's fight and shelter
choices. Seed 51's creatures-on runs are at `a072361`, with all of them, and
its creatures-off sub-goal runs at `dbed69e`; those changes act only on
creatures, apart from restocking. Seed 202 has not been rerun with them.

What they show:

- **Sub-goals beat per-turn choice with creatures off.** Paired by world,
  the sub-goal VLM gains +3.25 ± 0.71 achievements over SayCan on seed 51
  and +2.94 ± 0.52 on seed 202, on the earlier code, winning 14 worlds
  against 1 and 13 against 0. The curriculum gains about as much.
- **The VLM ties the curriculum.** VLM minus curriculum is +0.44 ± 0.33 and
  -0.12 ± 0.40 with creatures off. With creatures on at `a072361`, the VLM
  is better in 8 worlds and worse in 4 (p = 0.39).
- **Creatures on, the survival changes help the sub-goal designs.** On seed
  51 the VLM rose from 10.44 achievements per episode before them to 12.06,
  and the curriculum from 10.50 to 11.38. SayCan, which they do not touch,
  beats neither: the VLM wins 13 worlds against 2 (p = 0.007).
- **Shelter is where the VLM's choices differ.** On seed 51 its shelters
  lasted the night in 12 of 14 nights, against the curriculum's 4 of 11. It
  often sheltered straight after drinking, by water, so it could drink
  inside. It never chose to drink first and shelter after, and it accepted
  32 of 34 fights.
- **Fights are where the player dies.** In the `a072361` run, 12 of the
  curriculum's 16 deaths came during `defeat_zombie`, and 7 of the VLM's,
  with 5 more thirsty at night.
- **The executor fights as well as a scripted chase.** Notebook 08's staged
  fights (`11e8c0f`, 50 rollouts per cell) put the player near a zombie at
  set health, sword, distance and time of day. The executor won 0.73,
  against 0.75 for a scripted chase-and-hit, and matched it in every cell.
  At health 2 it always lost; from 5 up it won 0.8 to 1.0. Against
  skeletons it won 0.37 against the chase's 0.66, timing out more often
  than dying. Night changed nothing, since the environment spawns no extra
  creatures.
- **The VLM does not decide by the odds.** With the measured win and death
  rates in its prompt (`648eca2`, seed 51), it declined 12 of 71 fights, 10
  of them at a win rate of 0.75 or more, mostly to drink. The zombie
  followed, and deaths moved from fights (10 to 6) to drinking (2 to 6).
  Achievements per episode were 11.81, against 11.94 just before. The
  prompt then said walking away had not been measured; `623f115` adds the
  measured odds of drinking or collecting wood with zombies close, and has
  not run yet.
- **The Crafter score misleads at a cap.** Its geometric mean over 22
  achievements rewards rare incidental unlocks, so we report achievements
  per episode first.

Results and manifests are in `runs/open-ended/seed<N>-<timestamp>/`, and the
staged fights in `runs/open-ended/fights-<timestamp>/`, on the machine that
ran them.

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

1. **Train the twenty-two-skill executor.** In notebook 07, run
   `PRESET = "twenty_two_skill_smoke"` first. Then run `twenty_two_skill`:
   three seeds of 36M steps, with 352 environments, 16 per skill and 3,200
   updates, as `eighteen_skill` had. On a machine without `runs/`, the
   notebook first retrains the paired baseline from
   `configs/skill_ppo_local_1m.yaml`. Run selection uses the seven-skill
   chains, so check the combat and iron skills in the per-skill training
   table before going on. Compare each against its scripted baseline in
   notebook 03.
2. **The open-ended task with twenty-two skills.** Run notebook 08 with
   `EXECUTOR_PRESET = "twenty_two_skill"` and Qwen3-VL-8B, on seed 51, then
   on seed 202, which has not run with the survival changes. The cap rises
   to 22. Notebook 08 measures the competence prior and the fight and
   errand odds for the new executor first. Iron and diamond lie in caves
   and mountains, usually out of view, so the cap may stay out of reach for
   want of exploration rather than of skill. Check whether the errand odds
   change the VLM's fight choices.
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
| 3. Goal-conditioned executor | One policy for every skill, trained on `local_navigation`, then on the `randomized` scenario | Trained with seven, sixteen and eighteen skills, whose checkpoints are lost; the twenty-two-skill executor is next |
| 4. Per-turn planner | A frozen VLM in a SayCan loop, with a grounded ablation ladder | Complete; no per-turn condition reaches the bar, and the plan veto is the upper bound |
| 4b. Sub-goal planner | VLM sub-goals, a symbolic planner, memory, richer observation, the open-ended task, and a learned skill model | Chains at the upper bound on two seeds; on the open-ended task sub-goals beat per-turn choice and the VLM ties the curriculum; feasibility, model learning and the observation ablation pending |
| 5. Memory and visual grounding | Structured episodic memory; symbolic against pixel inputs; frame history; paraphrased instructions | Memory and the image/text ablation built; the rest not started |
| 6. Full evaluation | The full Craftax-Classic achievement set, multi-seed ablations, compute accounting, and documented failures | Every achievement has a skill; not started otherwise |

Next steps, in order:

1. **Train the twenty-two-skill executor** on Runpod with notebook 07's
   `twenty_two_skill_smoke` and `twenty_two_skill` presets (Experiment 1).
2. **Run the open-ended task with it** on seeds 51 and 202 (Experiment 2).
3. **Run Experiments 3 to 5**, the feasibility study, model learning and the
   observation ablation.

Deferred from the Phase 3 plan: learned skill termination (the open-ended
task uses the fixed `SkillInterruption` rule instead), text-embedding skill
conditioning, and paraphrase evaluation.

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

# Seven skills: 112 environments, 16 per skill.
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_10m.yaml

# Sixteen skills: the seven above plus the stone sword, coal, stone, furnace,
# sapling, plant, cow, sleep and ripe plant. 256 environments, about 26M steps.
tiny-sima-train-skills --config configs/skill_ppo_16_skill_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_16_skill_randomized_26m.yaml

# Eighteen skills: the sixteen above plus defeat_zombie and defeat_skeleton.
# 288 environments, about 29M steps.
tiny-sima-train-skills --config configs/skill_ppo_18_skill_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_18_skill_randomized_29m.yaml

# Twenty-two skills: the eighteen above plus iron, the iron pickaxe and sword,
# and diamond. 352 environments, about 36M steps.
tiny-sima-train-skills --config configs/skill_ppo_22_skill_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_22_skill_randomized_36m.yaml
```

Notebook 07's `stone_pickaxe` preset trains three seeds of the seven-skill
configuration and records the selected one; `sixteen_skill_smoke` and
`sixteen_skill` do the same for the sixteen-skill catalog, and
`eighteen_skill_smoke` and `eighteen_skill` for the eighteen-skill one, and
`twenty_two_skill_smoke` and `twenty_two_skill` for all twenty-two. The chain
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
  `sixteen_skill_smoke`, `sixteen_skill`, `eighteen_skill_smoke`,
  `eighteen_skill`, `twenty_two_skill_smoke`, and `twenty_two_skill`.

Both reuse completed work rather than repeating it.

Notebook 06's `incremental` preset is for checking a change to the VLM
sub-goal planner:

- It runs the sub-goal arms on the full run's episodes.
- It reloads the competence prior instead of probing.
- It skips the feasibility study and model learning.
- It reloads each baseline arm from the seed's saved results.

Notebook 08 does the same for the (design, mobs) runs listed in `REUSE_RUNS`.

Every saved result records its provenance: the seed, the episode count, the
executor, the VLM configuration, the competence prior, and for notebook 08 the
observation and limits. A baseline is reused only when its provenance matches
the current run's, field for field; otherwise it is rerun. Results saved before
provenance was recorded are always rerun.

Provenance cannot see code. Each result also records the commit it ran at, so
rerun a baseline whose own code changed, such as the per-turn VLM prompt.

### Notebook setup

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

### Runpod

The project runs on a Runpod pod:

- Use an image with Python 3.11 or 3.12 and a GPU with 40 to 48 GB, which
  the 8B VLM in notebook 08 needs.
- Clone the repository under `/workspace`, the persistent volume, so
  `runs/` and the Hugging Face cache survive the pod stopping.
- Run notebook 07 before 08. A fresh pod has no `runs/`, so notebook 07
  retrains the paired baseline, then trains and selects the executor that
  notebook 08 loads. Notebook 08 then measures the competence prior and the
  fight odds for that executor.

To run them headless, and stop the pod when they end, start them from a
terminal on the pod:

```bash
cd /workspace/tiny-sima
bash scripts/runpod_run.sh \
    notebooks/07_improve_executor.ipynb \
    notebooks/08_open_ended_achievements.ipynb
```

- It creates `/workspace/tiny-sima-venv` with Python 3.12, or 3.11, on the
  first run. A stopped pod loses its container disk, so packages installed
  into the image's Python would have to be reinstalled on every run.
- It runs `scripts/run_notebooks.py` under `nohup`, so closing the terminal
  or the browser does not end the run.
- Each notebook runs in a fresh kernel and is saved after every cell under
  `runs/notebook-runs/<timestamp>/`. Open the saved copy to follow progress.
  The log is in `runs/notebook-runs/<timestamp>.log`.
- The first failing notebook ends the run, with its outputs up to the error
  kept.
- However the run ends, by finishing, failing or being killed, it then runs
  `runpodctl stop pod $RUNPOD_POD_ID`. Pass `--keep-pod` to leave the pod
  running.
- `--set NAME=VALUE` replaces a one-line assignment in the notebooks before
  they run, and fails before running anything if no notebook has one. For
  example, `--set EVALUATION_SEED=202` runs notebook 08 on seed 202, and
  `--set PRESET='"twenty_two_skill_smoke"'` runs notebook 07's smoke preset.

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
