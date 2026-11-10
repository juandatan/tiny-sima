"""The open-ended achievement task: unlock as many achievements as possible.

Three planner designs share one budget of skill calls and environment steps
in an unmodified Craftax-Classic world, with creatures or without:

* ``saycan_recipe_aware`` ranks skills with the recipe-aware VLM prompt and
  multiplies by the competence prior.
* ``subgoal_vlm`` asks the VLM for formal sub-goals over the achievements
  its skills unlock, re-asking on low vitals, nearby threats and nightfall.
  When every answer of a decision is unreachable, the scripted curriculum
  supplies the next sub-goal, so the episode goes on.
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
    SkillInterruption,
    chain_episode_record,
    evaluate_chain_episodes,
)
from tiny_sima.evaluation.reuse import fingerprint
from tiny_sima.goals import STONE_PICKAXE_SKILL_IDS, SkillId
from tiny_sima.planner.competence import CompetencePrior
from tiny_sima.planner.fight_odds import FightOdds
from tiny_sima.planner.goal_selector import SURVIVAL_VITAL_RULES, ScriptedGoalSelector
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
from tiny_sima.planner.vital_rates import VitalRates
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
# A skill hands back to the planner when a zombie comes that close or the
# player is hurt, so the planners can fight it off mid-skill.
DEFAULT_INTERRUPTION = SkillInterruption(threat_radius=DEFAULT_THREAT_RADIUS)

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
    skills=STONE_PICKAXE_SKILL_IDS,
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
    plan = episode.final_plan_state
    record["fallback_calls"] = (
        plan.fallback_calls if isinstance(plan, SubgoalPlan) else 0
    )
    record["choices"] = (
        [choice.to_dict() for choice in plan.choices]
        if isinstance(plan, SubgoalPlan)
        else []
    )
    return record


def _fallback_share(records: Sequence[Mapping[str, Any]]) -> float | None:
    """The share of goal-source answers the fallback gave, pooled."""

    fallback = sum(record.get("fallback_calls", 0) for record in records)
    total = fallback + sum(record["vlm_calls"] for record in records)
    return fallback / total if total else None


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "std": float(array.std())}


def curriculum_choice(choice: Mapping[str, Any]) -> str:
    """Return what the curriculum answers where a recorded choice was made.

    It shelters at night whenever the shelter could start, never rests by
    day, and always fights now; see
    :class:`~tiny_sima.planner.goal_selector.ScriptedGoalSelector`.
    """

    if choice["kind"] == "fight":
        return "now"
    if choice["kind"] == "rest":
        return "no"
    return "now" if choice["possible"] else "no"


# Bands of the measured win rate a fight answer was given, as (label, low).
FIGHT_ODDS_BANDS: tuple[tuple[str, float], ...] = (
    ("won under 25%", 0.0),
    ("won 25-75%", 0.25),
    ("won 75% or more", 0.75),
)


def _odds_band(win_rate: float) -> str:
    return [label for label, low in FIGHT_ODDS_BANDS if win_rate >= low][-1]


def summarize_choices(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Set the goal source's own shelter and fight answers against the curriculum.

    For each kind it pools the answers over episodes, leaving out the
    fallback's, which are the curriculum's own. ``counts`` breaks them down by
    whether the shelter could start and by answer, and ``differ`` counts the
    answers the curriculum would not have given in the same state. Fight
    answers given with odds are also counted by answer in each band of
    :data:`FIGHT_ODDS_BANDS`, as ``by_odds``.
    """

    summary: dict[str, Any] = {}
    for kind in ("shelter", "rest", "fight"):
        answers = [
            choice
            for record in records
            for choice in record.get("choices", ())
            if choice["kind"] == kind and not choice["fallback"]
        ]
        differ = sum(
            choice["answer"] != curriculum_choice(choice) for choice in answers
        )
        counts: dict[str, dict[str, int]] = {}
        for choice in answers:
            state = "possible" if choice["possible"] else "not possible"
            by_answer = counts.setdefault(state, {})
            by_answer[choice["answer"]] = by_answer.get(choice["answer"], 0) + 1
        summary[kind] = {
            "answers": len(answers),
            "differ": differ,
            "differ_rate": differ / len(answers) if answers else None,
            "counts": counts,
        }
        if kind == "fight":
            by_odds: dict[str, dict[str, int]] = {}
            for choice in answers:
                if choice.get("win_rate") is None:
                    continue
                by_answer = by_odds.setdefault(_odds_band(choice["win_rate"]), {})
                by_answer[choice["answer"]] = by_answer.get(choice["answer"], 0) + 1
            summary[kind]["by_odds"] = by_odds
    return summary


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
        # Records saved before the fallback existed have none.
        "fallback_calls": _mean_std(
            [record.get("fallback_calls", 0) for record in records]
        ),
        "fallback_share": _fallback_share(records),
        "choices": summarize_choices(records),
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
    interruption: SkillInterruption | None = DEFAULT_INTERRUPTION,
) -> dict[str, Any]:
    """Run the open-ended task and return its metrics and episode records.

    The keys come from ``seed`` alone, so designs and mob settings run on
    the same worlds. Episodes end at ``max_skill_calls`` skills, at
    ``max_environment_steps`` steps (checked after each skill), on death, or
    when the planner has nothing to propose. ``max_failures_per_skill`` is
    off by default, since the budget already bounds a stuck episode; the
    chain evals' limit of two would end most open-ended episodes on the
    first hard resource. ``interruption`` ends a skill early when a zombie
    comes close or the player is hurt; ``None`` runs every skill to its end.
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
        interruption=interruption,
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
        "interruption": None if interruption is None else interruption.to_dict(),
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
    shelter_at_night: bool = True,
    choose_fights: bool = True,
    fight_odds: FightOdds | None = None,
    vital_rates: VitalRates | None = None,
    informed: bool = True,
    fight_survival_weight: float | None = None,
) -> dict[str, SkillPlanner]:
    """Return the three designs, built as notebook 06 builds its planners.

    Without a prior, the SayCan Can term and the sub-goal tie-break fall
    back to reachability, as in notebook 06. Both sub-goal planners share
    every setting but the goal source, and ``subgoal_vlm`` falls back on the
    curriculum ``subgoal_scripted`` follows. Both can fight off a zombie
    within ``threat_radius`` before going on with their plan; with
    ``choose_fights`` their goal source chooses whether to. With
    ``shelter_at_night`` it chooses whether to wall the player in with stone
    for the night, now or after its sub-goals. The VLM decides both from
    what the prompt says, and may fight after its first sub-goal, such as
    drinking first; the curriculum always fights at once and shelters
    whenever it can. ``fight_odds``, the executor's measured fight odds,
    are shown to the VLM with the close threats.

    With ``informed`` both goal sources are asked again, for the fight and
    shelter choices alone, when the player is hit with a zombie close, and
    may shelter by day to rest; the VLM also gets the game's general rules
    and, with ``vital_rates``, the vital rates measured in staged play. A
    fight is asked about again at each hit only with an interruption with
    ``fights_on_hit``, which only the VLM needs: the curriculum fights on.

    With ``fight_survival_weight`` the VLM's fight is chosen SayCan-style:
    its probability of each choice weighed against the measured chance of
    surviving it (see :class:`SubgoalPlanner`).
    """

    feasibility = prior if prior is not None else reachability_feasibility

    def curriculum() -> ScriptedGoalSelector:
        return ScriptedGoalSelector(
            candidate_skills=candidate_skills,
            model=model,
            vital_rules=SURVIVAL_VITAL_RULES,
            shelter_at_night=shelter_at_night,
            choose_fights=choose_fights,
        )

    def subgoal_planner(source: Any, **options: Any) -> SubgoalPlanner:
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
            defend_against=("zombie",),
            shelter_at_night=shelter_at_night,
            choose_fights=choose_fights,
            replan_when_hurt=informed,
            rest_by_day=informed,
            **options,
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
        "subgoal_vlm": subgoal_planner(
            backend,
            fallback=curriculum(),
            fight_odds=fight_odds,
            manual=informed,
            vital_rates=vital_rates if informed else None,
            fight_survival_weight=fight_survival_weight,
        ),
        "subgoal_scripted": subgoal_planner(curriculum()),
    }


def goal_source_provenance(planner: SkillPlanner) -> dict[str, Any] | None:
    """What a sub-goal planner's goal source is told, for run reuse.

    ``None`` for any other planner, or one told none of it, so runs from
    before these options still match.
    """

    if not isinstance(planner, SubgoalPlanner):
        return None
    rates = planner.vital_rates
    told = {
        "manual": planner.manual,
        "replan_when_hurt": planner.replan_when_hurt,
        "rest_by_day": planner.rest_by_day,
        "vital_rates": None if rates is None else fingerprint(rates.to_dict()),
    }
    # Only when set, so runs from before the option still match.
    if planner.fight_survival_weight is not None:
        told["fight_survival_weight"] = planner.fight_survival_weight
        return told
    return told if any(told.values()) else None
