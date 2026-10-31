"""The open-ended achievement task: unlock as many achievements as possible.

Three planner designs share one budget of skill calls and environment steps
in an unmodified Craftax-Classic world, with creatures or without:

* ``saycan_recipe_aware`` ranks skills with the recipe-aware VLM prompt and
  multiplies by the competence prior.
* ``subgoal_vlm`` asks the VLM for formal sub-goals over the achievements
  its skills unlock, re-asking on low vitals, nearby threats and nightfall.
* ``subgoal_scripted`` is the same sub-goal planner with a fixed curriculum,
  :class:`~tiny_sima.planner.goal_selector.ScriptedGoalSelector`, in place
  of the VLM.

The episode runs through the chain loop, with a chain whose goal never
completes: the skill-call budget is the decision limit, and the step budget
and death end an episode early. Scores follow Crafter: the per-achievement
success rate over episodes, and their geometric mean.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
from craftax.craftax_classic.constants import Achievement

from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.open_ended import OpenEndedEnvironment
from tiny_sima.evaluation.chains import (
    ChainEpisodeResult,
    ChainTermination,
    LearnedSkillExecutor,
    SkillChainSpec,
    SkillExecutor,
    chain_episode_record,
    evaluate_chain_episodes,
)
from tiny_sima.goals import ALL_SKILL_IDS, SkillId
from tiny_sima.planner.competence import CompetencePrior
from tiny_sima.planner.goal_selector import ScriptedGoalSelector
from tiny_sima.planner.observation import ObservationConfig
from tiny_sima.planner.saycan import (
    PlannerMemory,
    SayCanPlanner,
    SkillPlanner,
    reachability_feasibility,
)
from tiny_sima.planner.subgoals import (
    ALL_VITAL_THRESHOLDS,
    SubgoalPlan,
    SubgoalPlanner,
    open_ended_subgoal_fields,
)
from tiny_sima.planner.symbolic import SkillModel
from tiny_sima.planner.vlm import (
    PlannerPromptStyle,
    VisionLanguageBackend,
    VLMUtilityScorer,
)

OPEN_ENDED_OBJECTIVE = "Unlock as many achievements as you can."
DEFAULT_MAX_SKILL_CALLS = 80
DEFAULT_MAX_ENVIRONMENT_STEPS = 3000
# Hostile creatures within this many tiles (Chebyshev) trigger a re-ask: the
# view's row radius, so anything visible above or below counts.
DEFAULT_THREAT_RADIUS = 3

ACHIEVEMENT_NAMES: tuple[str, ...] = tuple(
    achievement.name.lower() for achievement in Achievement
)

# Every achievement in exactly one tier. wake_up sits with eat_cow under
# survival rather than basic: it needs the agent to sleep through low energy
# and live until morning, a survival behaviour no skill covers, not a
# resource step on the way to tools.
ACHIEVEMENT_TIERS: dict[str, tuple[str, ...]] = {
    "basic": (
        "collect_wood",
        "place_table",
        "collect_sapling",
        "place_plant",
        "eat_plant",
        "collect_drink",
    ),
    "wood": ("make_wood_pickaxe", "make_wood_sword", "collect_stone"),
    "stone": (
        "place_stone",
        "make_stone_pickaxe",
        "make_stone_sword",
        "place_furnace",
        "collect_coal",
    ),
    "iron_diamond": (
        "collect_iron",
        "make_iron_pickaxe",
        "make_iron_sword",
        "collect_diamond",
    ),
    "survival": ("eat_cow", "wake_up"),
    "combat": ("defeat_zombie", "defeat_skeleton"),
}


def _never_complete(state: Any) -> bool:
    return False


# A chain with no end: every episode runs until a budget, a death or the
# planner stops it. No reference plan, so the decision limit is always given.
OPEN_ENDED_CHAIN = SkillChainSpec(
    name="open_ended",
    objective=OPEN_ENDED_OBJECTIVE,
    reference_skills=(),
    is_complete=_never_complete,
    skills=ALL_SKILL_IDS,
    tree_count=0,
)


def crafter_score(success_rates: Mapping[str, float]) -> float:
    """Return Crafter's score from success rates in ``[0, 1]``.

    It is ``exp(mean_i ln(1 + s_i)) - 1`` with ``s_i`` in percent, over all
    22 achievements; one missing from ``success_rates`` counts as zero. The
    geometric mean rewards unlocking a rare achievement sometimes over
    unlocking a common one more often.
    """

    unknown = set(success_rates) - set(ACHIEVEMENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown achievements: {sorted(unknown)}.")
    percents = [100.0 * success_rates.get(name, 0.0) for name in ACHIEVEMENT_NAMES]
    return math.exp(sum(math.log1p(value) for value in percents) / len(percents)) - 1


def episode_vlm_calls(episode: ChainEpisodeResult, planner: SkillPlanner) -> int:
    """Return the goal-source calls one episode made.

    A sub-goal planner counts them in its plan state, scripted answers
    included. A SayCan planner with a VLM utility asks once per selection,
    including a last one that ended the episode without a decision.
    """

    plan = episode.final_plan_state
    if isinstance(plan, SubgoalPlan):
        return plan.vlm_calls
    if isinstance(planner, SayCanPlanner) and isinstance(
        planner.utility_scorer, VLMUtilityScorer
    ):
        planner_stopped = episode.termination in (
            ChainTermination.NO_VIABLE_SKILL,
            ChainTermination.INVALID_PLANNER_OUTPUT,
        )
        return episode.decisions + int(planner_stopped)
    return 0


def open_ended_episode_record(
    episode: ChainEpisodeResult,
    planner: SkillPlanner,
    *,
    include_trace: bool = True,
) -> dict[str, Any]:
    """Return one episode as plain values: the chain record plus unlocks."""

    record = chain_episode_record(episode)
    if not include_trace:
        record.pop("trace", None)
    record["achievements"] = [
        unlock.achievement for unlock in episode.achievement_unlocks
    ]
    record["achievement_unlocks"] = [
        unlock.to_dict() for unlock in episode.achievement_unlocks
    ]
    record["vlm_calls"] = episode_vlm_calls(episode, planner)
    return record


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std())}


def summarize_open_ended(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate episode records into the task's metrics.

    Standard deviations are over episodes, not standard errors. A tier's
    success rate is the mean of its achievements' rates, and its count the
    mean number of its achievements an episode unlocks.
    """

    if not records:
        raise ValueError("No episodes to summarize.")
    unlocked = [set(record["achievements"]) for record in records]
    success_rates = {
        name: sum(name in episode for episode in unlocked) / len(records)
        for name in ACHIEVEMENT_NAMES
    }
    tiers = {
        tier: {
            "success_rate": float(np.mean([success_rates[name] for name in names])),
            "achievements_per_episode": _mean_std(
                [len(episode.intersection(names)) for episode in unlocked]
            ),
        }
        for tier, names in ACHIEVEMENT_TIERS.items()
    }
    return {
        "num_episodes": len(records),
        "achievements_per_episode": _mean_std([len(episode) for episode in unlocked]),
        "crafter_score": crafter_score(success_rates),
        "success_rates": success_rates,
        "tiers": tiers,
        "skill_calls": _mean_std([record["decisions"] for record in records]),
        "environment_steps": _mean_std(
            [record["environment_steps"] for record in records]
        ),
        "vlm_calls": _mean_std([record["vlm_calls"] for record in records]),
        "termination_counts": {
            termination.value: sum(
                record["termination"] == termination.value for record in records
            )
            for termination in ChainTermination
        },
    }


def evaluate_open_ended(
    planner: SkillPlanner,
    executor: SkillExecutor,
    environment: OpenEndedEnvironment,
    *,
    num_episodes: int,
    seed: int = 0,
    skill_horizon: int = 64,
    max_skill_calls: int = DEFAULT_MAX_SKILL_CALLS,
    max_environment_steps: int = DEFAULT_MAX_ENVIRONMENT_STEPS,
    max_failures_per_skill: int | None = None,
    memory_factory: Callable[[], PlannerMemory] | None = None,
    include_trace: bool = True,
) -> dict[str, Any]:
    """Run the open-ended task and return its metrics and episode records.

    The keys come from ``seed`` alone, so designs and mob settings run on
    the same worlds. Episodes end at ``max_skill_calls`` skills, at
    ``max_environment_steps`` steps (checked after each skill), on death, or
    when the planner has nothing to propose. ``max_failures_per_skill`` is
    off by default, since the budget already bounds a stuck episode; the
    chain evals' limit of two would end most open-ended episodes on the
    first hard resource.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    if isinstance(executor, LearnedSkillExecutor):
        executor.require_skills(
            getattr(planner, "candidate_skills", OPEN_ENDED_CHAIN.skills)
        )
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    episodes = evaluate_chain_episodes(
        planner,
        executor,
        environment,
        list(keys),
        chain=OPEN_ENDED_CHAIN,
        skill_horizon=skill_horizon,
        max_decisions=max_skill_calls,
        max_failures_per_skill=(
            max_failures_per_skill
            if max_failures_per_skill is not None
            else max_skill_calls + 1
        ),
        memory_factory=memory_factory,
        max_environment_steps=max_environment_steps,
        stop_on_death=True,
    )
    records = [
        open_ended_episode_record(episode, planner, include_trace=include_trace)
        for episode in episodes
    ]
    return {
        "objective": OPEN_ENDED_OBJECTIVE,
        "mobs": environment.mobs,
        "seed": seed,
        "skill_horizon": skill_horizon,
        "max_skill_calls": max_skill_calls,
        "max_environment_steps": max_environment_steps,
        "summary": summarize_open_ended(records),
        "episodes": records,
    }


def json_ready_open_ended_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a result with every value plain, ready for :func:`json.dump`."""

    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [plain(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return plain(result)


def open_ended_planners(
    candidate_skills: tuple[SkillId, ...],
    backend: VisionLanguageBackend,
    config: VLMPlannerConfig,
    prior: CompetencePrior | None,
    observation: ObservationConfig | None,
    *,
    model: SkillModel | None = None,
    threat_radius: int = DEFAULT_THREAT_RADIUS,
) -> dict[str, SkillPlanner]:
    """Return the three designs, built as notebook 06 builds its planners.

    Without a prior, the SayCan Can term and the sub-goal tie-break fall
    back to reachability, as in notebook 06. Both sub-goal planners share
    every setting but the goal source.
    """

    feasibility = prior if prior is not None else reachability_feasibility

    def subgoal_planner(source: Any) -> SubgoalPlanner:
        return SubgoalPlanner(
            source,
            candidate_skills=candidate_skills,
            config=config,
            model=model,
            tie_break=feasibility,
            fields=open_ended_subgoal_fields(candidate_skills),
            observation=observation,
            vital_thresholds=ALL_VITAL_THRESHOLDS,
            threat_radius=threat_radius,
            replan_at_night=True,
        )

    return {
        "saycan_recipe_aware": SayCanPlanner(
            utility_scorer=VLMUtilityScorer(
                backend=backend,
                config=config,
                prompt_style=PlannerPromptStyle.RECIPE_AWARE,
                observation=observation,
            ),
            feasibility_scorer=feasibility,
            candidate_skills=candidate_skills,
        ),
        "subgoal_vlm": subgoal_planner(backend),
        "subgoal_scripted": subgoal_planner(
            ScriptedGoalSelector(candidate_skills=candidate_skills, model=model)
        ),
    }
