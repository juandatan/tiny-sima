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

As of October 26, 2026:

- **Phases 1 to 3 are complete.** The environment, PPO baseline, skill
  benchmark, and goal-conditioned executor are done.
- **Phase 4, the hierarchical planner, is measured.** The grounded VLM picks
  an objective-progressing skill at 45 of 48 controlled boundaries, which is
  every boundary where one is reachable. No planner response was invalid in
  any condition.
- **Executor improvement is complete.** An executor trained on the randomized
  scenario completes the default scripted plan in 64 of 64 closed-loop
  episodes, against 16 for the baseline executor (48 discordant pairs against
  0). The plan-order gap fell from +0.359 to -0.016, with no
  `local_navigation` regression.
- **The planner's sequencing is now the binding constraint.** The best VLM
  condition succeeds in 50 of 64 episodes against 63 for the collect-first
  scripted plan (p = 0.001). All 14 failures hit the 12-decision limit, and no
  skill failed in any of the 64 episodes. The ungrounded `baseline` prompt
  with the reachability Can term now beats the grounded `available_first`
  prompt, 62 against 50 (p = 0.002).
- **The sequencing diagnosis is done, and the main condition changed.** 162
  of the `available_first` prompt's 163 off-objective decisions were the
  VLM's own first choice, so `baseline_plus_reachability` is now the main
  condition and `available_first` an ablation. Its remaining waste is a
  second wood pickaxe (23 decisions over 64 episodes) and extra wood (25).
- **The held-craft veto is refuted, and the stall it hid is diagnosed.** On
  seed 51 the main condition solved 61 of 64 and the veto 59 (3 discordant
  episodes to 1 against it). The veto removed the second pickaxe, and the
  planner collected surplus wood instead. Every failure stalls before stone:
  the wood pickaxe is held, stone is out of view, and both the prompt and the
  Can term discount it.
- **Current work is the last Phase 4 measurement.** The `objective_first`
  prompt, a search-state test, a seventh skill (`craft_stone_pickaxe`), and a
  longer stone-pickaxe chain are implemented and tested, but not yet measured
  on the GPU.
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
the `available_first` prompt style, now an ablation, the model must echo which
skills are available before it may name a choice. The main condition uses the
`baseline` style, which leaves grounding to the Can term. It then emits a full ranking, and
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
the skill ID, handles all six skills. A seventh, `craft_stone_pickaxe`, is
opt-in through `num_skills: 7`; the first six keep their embedding indices, so
six-skill checkpoints, priors, and prompts load unchanged:

- `collect_wood`
- `collect_stone`
- `drink_water`
- `place_table`
- `craft_wood_pickaxe`
- `craft_wood_sword`

Episodes pay one sparse reward on success. The current executor,
`runs/executor-improvement-full/randomized-seed0`, trained for 9.8M steps on
the `randomized` scenario with a 64-step skill horizon. The baseline
executor, `runs/skill-ppo-local-1m`, trained for one million steps on
`local_navigation` with a 32-step horizon. It is kept as the reference that
every executor change is paired against.

**Evaluation chain.** The closed loop runs one persistent world per episode,
with the objective `wood -> wood -> table -> wood -> wood pickaxe -> stone`.
The reference order is private to the scripted baseline. The planner sees only
the objective and its own outcomes, and success is read from the final Craftax
state. An episode ends at success, after twice the reference plan's length in
decisions (12 here), or when one skill has failed twice.

The longer `wood_to_stone_pickaxe` chain adds a fourth wood and the stone
pickaxe: eight reference decisions, a limit of 16, and four trees per world,
since a mined tree is gone. It needs a seven-skill executor.

**Memory.** The memory package is still a placeholder. Recent outcomes in the
planner context are the only memory so far.

## Latest findings

### Planner ablations

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
[ARCHIVE.md](ARCHIVE.md#planner-ablations-with-the-baseline-executor).

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

### Feasibility measurements

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

### Refuted approaches, kept as ablations

| Approach | Measurement that refuted it | Status |
| --- | --- | --- |
| Executor critic `V(s, skill)` as the Can term | AUC 0.43 in the first run and 0.369 held out; 0.437 held out for the randomized executor. Neither training scenario spawns an impossible target, so the critic learned which skill was asked for, not whether it is possible. | Fitted and scored for each executor on every probing run, and retired from the ladder: identical closed loop at 2.8 times the cost. |
| Competence prior as the Can term, under a strong executor | Refitted on the randomized executor, every cell is p = 1.000: held-out base rate 0.995 and AUC 0.500. Its closed loop matched reachability, 62 of 64 each with no discordant pair. Under the baseline executor (AUC 0.738) it also matched, 17 each with 3 discordant pairs each way. | Kept as `baseline_plus_competence_prior`, refitted for each executor. |
| More training on `local_navigation` alone | The step-matched `local-matched-seed0` control raised stochastic default-plan success on seed 101 from 0.309 to 0.516, but left the plan-order gap at +0.318, the baseline's own gap. The randomized seeds closed it to between -0.002 and +0.002. | Kept as the control run in notebook 07. |
| Symbolic preconditions as the Can term | They test whether a block exists anywhere on the 64x64 map, which is true almost everywhere. `vlm_plus_symbolic` matched reachability exactly. | Kept inside every Can term; retired as a condition. |
| Held-craft veto (`baseline_plus_reachability_skip_held`) | 59 against 61 of 64 for the main condition on seed 51, 3 discordant episodes to 1 against it (p = 0.625). The pickaxe waste fell from 22 decisions to 0, and surplus wood rose from 39 to 74: the next-ranked skill in view took the vetoed one's place. | Kept in the ladder; `SkipHeldCraftFeasibility` stays in the library. |
| Hard reachability gate | The four-episode scripted chain fell from 1.0 to 0.5 success, with the lost episodes ending in `no_viable_skill`. | Replaced by grading, with unknown targets at 0.25. |
| Sampled decoding | `vlm_only` fell from 43 to 38 accepted of 48 on identical states. | Greedy decoding by default. |
| Per-skill advantage normalization | It collapsed the unsolved interaction skills in a debug ablation. | Available, off by default. |
| Larger PPO batches at equal steps | 256 environments lost 64% of return, and 512 lost 78%. | 64 environments remain the default. |

## Executor improvement: result

With the baseline executor, plan order alone moved the scripted plan from
0.250 to 0.609, so the gap was in the states each order hands the executor.
`local_navigation` trains on none of those states. Every episode starts at
the map centre in a cleared window, with one target, a reserved straight
route, the minimal inventory, and the clock at dawn. Notebook 07 attributed
the gap and trained it away. The full plan and diagnostics are in
[ARCHIVE.md](ARCHIVE.md#executor-improvement-plan-and-diagnostics).

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

## Current work: closing Phase 4

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
   conditions cannot beat what the executor finishes.

**The bar** stays the collect-first scripted plan on each chain.

### Seeds

Each seed is reserved for one purpose, so that selecting on one never biases
a result reported on another:

| Seed | Use |
| ---: | --- |
| 7 | Fitting the critic calibration and the competence prior |
| 23 | Held-out scoring of both feasibility heads |
| 51 | Planner prefix ladder and closed-loop evaluation |
| 101 | Executor benchmark and run selection, in notebook 07 |
| 202 | Randomized-scenario evaluation in notebook 07; confirmation of planner choices in notebook 06 (`CONFIRM`) |

Boundary states mined for training must come from seeds disjoint from all of
these.

## Roadmap

| Phase | Goal | Status |
| --- | --- | --- |
| 1. Environment and baseline | A reproducible Craftax PPO-RNN setup with throughput and memory measured | Complete |
| 2. Skill benchmark | Six skills with preconditions, success predicates, sparse rewards, and scripted and random baselines | Complete |
| 3. Goal-conditioned executor | One policy for every skill, trained on `local_navigation`, then on the `randomized` scenario | Complete |
| 4. Hierarchical planner | A frozen VLM in a SayCan loop, with a grounded ablation ladder | Held-craft veto refuted; `objective_first` prompt and stone-pickaxe chain awaiting measurement |
| 5. Memory and visual grounding | Structured episodic memory; symbolic against pixel inputs; frame history; paraphrased instructions | Not started |
| 6. Full evaluation | Long-horizon objectives, multi-seed ablations, compute accounting, and documented failures | Not started |

The Phase 3 plan had more items than were built. Learned skill termination,
text-embedding skill conditioning, and paraphrase evaluation were deferred.
The skill-success head became the measured competence prior, after the critic
failed as one.

Next steps, in order:

1. **Measure `objective_first`** and the search-state test in notebook 06 on
   seed 51, then confirm the chosen main condition on seed 202.
2. **Train the seven-skill executor** with notebook 07's `stone_pickaxe`
   preset, and check that it still pairs with the baseline on the six-skill
   benchmark.
3. **Run the stone-pickaxe chain** in notebook 06. That closes Phase 4.
4. **Cut evaluation cost.** Reuse the saved baseline-executor closed loops,
   which took 1,776 s of the October 22 run against 1,611 s for the
   candidate. `MEASURE_BASELINE` now defaults to off.
5. **Longer chains in Phase 6.** A furnace and iron need further skills; about
   3M training steps should suffice per added stage, since training macro
   success reached 0.988 in every randomized seed by 2.15M steps.

Phase 4 items still open:

- Replanning on danger or low vitals. The planner currently replans only at
  option boundaries.
- A planner that beats the best fixed plan. The executor now allows the
  comparison; no VLM condition has made it yet.

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
# Baseline executor, the paired reference: one million steps on local_navigation.
tiny-sima-train-skills --config configs/skill_ppo_local_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_local_1m.yaml

# Current executor: the randomized scenario, about 10M steps, 96 environments.
tiny-sima-train-skills --config configs/skill_ppo_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_randomized_10m.yaml

# Seven skills, for the stone-pickaxe chain: 112 environments, 16 per skill.
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_debug.yaml
tiny-sima-train-skills --config configs/skill_ppo_stone_pickaxe_randomized_10m.yaml
```

The current executor itself, `runs/executor-improvement-full/randomized-seed0`,
comes from notebook 07's `full` preset, which trains three seeds of this
configuration and the `local-matched` control.

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
