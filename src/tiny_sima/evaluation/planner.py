"""Planning-only evaluations at controlled persistent-world boundaries."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    ScriptedSkillExecutor,
    SkillChainSpec,
    chain_environment,
)
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_feasible,
    skill_metric,
    skill_precondition,
    travel_distance_map,
)
from tiny_sima.planner.saycan import (
    PlannerContext,
    PlannerSelectionStatus,
    SayCanPlanner,
    SkillAttempt,
    SkillOutcome,
)


@dataclass(frozen=True)
class PlannerBoundaryTrace:
    """Accepted and selected skills at one controlled chain prefix."""

    episode: int
    boundary: int
    reference_skill: SkillId
    accepted_skills: tuple[SkillId, ...]
    selected_skill: SkillId | None
    matched: bool
    selected_feasibility: float
    utility_valid: bool
    utility_error: str | None
    utility_ranking: tuple[SkillId, ...]
    utility_raw_response: str | None
    feasible_skills: tuple[SkillId, ...] = ()
    utility_top_skill: SkillId | None = None
    utility_top_feasible: bool = False
    selected_feasible: bool = False
    utility_repaired: bool = False
    label: str = ""


@dataclass
class _PrefixEpisode:
    """One episode's cursor while every episode advances in lockstep."""

    index: int
    state: Any
    context: PlannerContext
    rollout_key: Any


@dataclass(frozen=True)
class PrefixBoundary:
    """One oracle-reached state at which a planner decision is scored.

    ``label`` names how a derived state was reached from the oracle's, so its
    accuracy can be reported per kind of state.
    """

    episode: int
    boundary: int
    reference_skill: SkillId
    state: Any
    context: PlannerContext
    accepted_skills: tuple[SkillId, ...]
    label: str = ""


@dataclass(frozen=True)
class PrefixTrajectory:
    """The oracle-advanced boundaries every prefix-ablation planner is scored on.

    The scripted oracle ignores the planner's decisions, so these states are
    the same for every condition. Collecting them once and passing them to
    :func:`evaluate_planner_prefixes` removes the oracle rollout, which
    otherwise dominates a condition the response cache answers, from all but
    the first condition. ``boundaries[b]`` holds the episodes that reached
    boundary ``b``.
    """

    chain_name: str
    num_episodes: int
    seed: int
    skill_horizon: int
    distractor_count: int
    boundaries: tuple[tuple[PrefixBoundary, ...], ...]
    oracle_progress_failures: int


def collect_prefix_trajectory(
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 4,
    seed: int = 0,
    skill_horizon: int = 64,
    distractor_count: int = 3,
) -> PrefixTrajectory:
    """Advance every episode along the reference sequence with the oracle.

    Episodes advance in lockstep, one boundary at a time, so a planner backed
    by a batched utility model can decode every episode's decision at a
    boundary in one invocation. Each episode keeps its own rollout key, so the
    states visited are identical to advancing the episodes one after another.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    environment = chain_environment(chain, distractor_count=distractor_count)
    executor = ScriptedSkillExecutor()
    episode_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    boundaries: list[tuple[PrefixBoundary, ...]] = []
    oracle_progress_failures = 0

    episodes: list[_PrefixEpisode] = []
    for episode_index, episode_key in enumerate(episode_keys):
        reset_key, rollout_key = jax.random.split(episode_key)
        episodes.append(
            _PrefixEpisode(
                index=episode_index,
                state=environment.reset(reset_key).state,
                context=PlannerContext(objective=chain.objective),
                rollout_key=rollout_key,
            )
        )

    for boundary, reference_skill in enumerate(chain.reference_skills):
        if not episodes:
            break
        boundaries.append(
            tuple(
                PrefixBoundary(
                    episode=episode.index,
                    boundary=boundary,
                    reference_skill=reference_skill,
                    state=episode.state,
                    context=episode.context,
                    accepted_skills=(
                        chain.valid_next_skills(episode.state)
                        if chain.valid_next_skills is not None
                        else (reference_skill,)
                    ),
                )
                for episode in episodes
            )
        )
        advancing: list[_PrefixEpisode] = []
        for episode in episodes:
            episode.rollout_key, execution_key = jax.random.split(
                episode.rollout_key
            )
            result = executor.execute(
                environment,
                episode.state,
                reference_skill,
                execution_key,
                horizon=skill_horizon,
            )
            if result.outcome is not SkillOutcome.SUCCESS:
                # This episode stops here; the others keep going so one
                # oracle failure does not shorten the whole batch.
                oracle_progress_failures += 1
                continue
            episode.state = result.state
            episode.context = PlannerContext(
                objective=episode.context.objective,
                history=(
                    *episode.context.history,
                    SkillAttempt(
                        skill_id=reference_skill,
                        outcome=result.outcome,
                        steps=result.steps,
                        native_return=result.native_return,
                    ),
                ),
            )
            advancing.append(episode)
        episodes = advancing

    return PrefixTrajectory(
        chain_name=chain.name,
        num_episodes=num_episodes,
        seed=seed,
        skill_horizon=skill_horizon,
        distractor_count=distractor_count,
        boundaries=tuple(boundaries),
        oracle_progress_failures=oracle_progress_failures,
    )


# Skills whose target is a natural block the executor can walk off to find. A
# craft whose table is out of view is excluded: placing another table is as
# good a move as walking back, so the accepted set cannot judge it.
SEARCHABLE_SKILLS = frozenset(
    (SkillId.COLLECT_WOOD, SkillId.COLLECT_STONE, SkillId.DRINK_WATER)
)


def is_search_boundary(
    state: Any,
    accepted_skills: tuple[SkillId, ...],
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
) -> bool:
    """Whether the objective can only progress by searching out of view.

    True when some accepted skill exists, every accepted skill collects a
    natural resource and has its requirements met, and none has a target in
    view: the only correct request walks off to search.
    """

    if not accepted_skills or not SEARCHABLE_SKILLS.issuperset(accepted_skills):
        return False
    distances = travel_distance_map(state, reachability)
    for skill_id in accepted_skills:
        goal = SkillGoal(skill_id)
        if not bool(np.asarray(skill_precondition(goal, state))):
            return False
        if bool(
            np.asarray(
                skill_feasible(goal, state, reachability, distances=distances)
            )
        ):
            return False
    return True


def _run_detour(
    executor: ScriptedSkillExecutor,
    environment: Any,
    state: Any,
    context: PlannerContext,
    skill_id: SkillId,
    key: Any,
    horizon: int,
) -> tuple[Any, PlannerContext] | None:
    """Run one detour skill, recording it as the closed loop would have.

    Returns ``None`` when the skill does not succeed.
    """

    result = executor.execute(environment, state, skill_id, key, horizon=horizon)
    if result.outcome is not SkillOutcome.SUCCESS:
        return None
    return result.state, PlannerContext(
        objective=context.objective,
        history=(
            *context.history,
            SkillAttempt(
                skill_id=skill_id,
                outcome=result.outcome,
                steps=result.steps,
                native_return=result.native_return,
            ),
        ),
    )


def _check_derivable(
    trajectory: PrefixTrajectory,
    chain: SkillChainSpec,
    max_detours: int,
) -> None:
    if trajectory.chain_name != chain.name:
        raise ValueError(
            f"trajectory is for {trajectory.chain_name}, not {chain.name}."
        )
    if chain.valid_next_skills is None:
        raise ValueError(f"{chain.name} defines no accepted skills to compare.")
    if max_detours < 0:
        raise ValueError("max_detours must be non-negative.")


def _detour_trajectory(
    trajectory: PrefixTrajectory,
    *,
    chain: SkillChainSpec,
    detour_skill: SkillId,
    max_detours: int,
    seed: int,
    keep: Callable[[Any, tuple[SkillId, ...]], bool],
    keep_every: bool,
) -> PrefixTrajectory:
    """Run ``detour_skill`` from each oracle boundary and keep matching states.

    ``keep`` sees each state and its accepted set, the oracle state first.
    With ``keep_every`` every match along the detours is kept rather than
    the first. The detours are recorded in each kept state's history as the
    closed loop would have recorded them.
    """

    _check_derivable(trajectory, chain, max_detours)
    assert chain.valid_next_skills is not None
    environment = chain_environment(
        chain,
        distractor_count=trajectory.distractor_count,
    )
    executor = ScriptedSkillExecutor()
    rollout_key = jax.random.PRNGKey(seed)
    boundaries: list[tuple[PrefixBoundary, ...]] = []
    for points in trajectory.boundaries:
        found: list[PrefixBoundary] = []
        for point in points:
            state, context = point.state, point.context
            for detour in range(max_detours + 1):
                accepted = chain.valid_next_skills(state)
                if keep(state, accepted):
                    found.append(
                        PrefixBoundary(
                            episode=point.episode,
                            boundary=point.boundary,
                            reference_skill=point.reference_skill,
                            state=state,
                            context=context,
                            accepted_skills=accepted,
                        )
                    )
                    if not keep_every:
                        break
                if detour == max_detours or chain.is_complete(state):
                    break
                rollout_key, execution_key = jax.random.split(rollout_key)
                detoured = _run_detour(
                    executor,
                    environment,
                    state,
                    context,
                    detour_skill,
                    execution_key,
                    trajectory.skill_horizon,
                )
                if detoured is None:
                    break
                state, context = detoured
        boundaries.append(tuple(found))
    return dataclasses.replace(
        trajectory,
        boundaries=tuple(boundaries),
        oracle_progress_failures=0,
    )


def collect_search_trajectory(
    trajectory: PrefixTrajectory,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    detour_skill: SkillId = SkillId.COLLECT_WOOD,
    max_detours: int = 4,
    seed: int = 0,
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
) -> PrefixTrajectory:
    """Derive the boundaries where the next step's target is out of view.

    The oracle's own boundaries rarely have one, because the default order
    reaches stone while it is still nearby, so the prefix ladder cannot see the
    closed loop's stall: once the wood pickaxe was held and stone was out of
    view, the planner collected surplus wood until the decision limit. From
    each oracle boundary this runs ``detour_skill`` -- the off-objective choice
    the closed loop made -- up to ``max_detours`` times with the scripted
    executor and keeps the first state that :func:`is_search_boundary`
    accepts, the detours recorded in its history as the closed loop would
    have. The result has the input's settings, so
    :func:`evaluate_planner_prefixes` scores it unchanged.
    """

    return _detour_trajectory(
        trajectory,
        chain=chain,
        detour_skill=detour_skill,
        max_detours=max_detours,
        seed=seed,
        keep=lambda state, accepted: is_search_boundary(
            state, accepted, reachability
        ),
        keep_every=False,
    )


# Skills that add one unit of a resource to the inventory, so that repeating
# one is progress exactly while some remaining recipe is short of it.
COLLECTION_SKILLS = frozenset((SkillId.COLLECT_WOOD, SkillId.COLLECT_STONE))


def is_surplus_boundary(
    state: Any,
    accepted_skills: tuple[SkillId, ...],
    surplus_skill: SkillId = SkillId.COLLECT_STONE,
) -> bool:
    """Whether collecting more of a resource already held is off-objective.

    True when some accepted skill exists, every accepted skill collects a
    resource, ``surplus_skill`` is not among them, and it could run now on a
    player who already holds some of what it collects: requesting it again
    would succeed and add nothing the objective needs.
    """

    if surplus_skill not in COLLECTION_SKILLS:
        raise ValueError(f"{surplus_skill.value} collects no resource.")
    if (
        not accepted_skills
        or surplus_skill in accepted_skills
        or not COLLECTION_SKILLS.issuperset(accepted_skills)
    ):
        return False
    if int(np.asarray(skill_metric(surplus_skill, state))) < 1:
        return False
    return bool(np.asarray(skill_precondition(SkillGoal(surplus_skill), state)))


def collect_surplus_trajectory(
    trajectory: PrefixTrajectory,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_PICKAXE_CHAIN,
    surplus_skill: SkillId = SkillId.COLLECT_STONE,
    max_detours: int = 3,
    seed: int = 0,
) -> PrefixTrajectory:
    """Derive the boundaries where a resource is held and another is short.

    The closed loop's commonest off-objective choice under the
    objective-first prompt on the stone-pickaxe chain: stone held, wood spent
    on the wood pickaxe, collect wood the only correct step, and the planner
    collecting stone again, mostly as a repeat. The oracle never holds stone
    before its last wood, so neither the prefix ladder nor the search test
    reaches that state. From each oracle boundary this runs ``surplus_skill`` up to
    ``max_detours`` times and keeps every state :func:`is_surplus_boundary`
    accepts, so a boundary after one surplus stone and one after three are
    both scored, each with the stones in its history.
    """

    return _detour_trajectory(
        trajectory,
        chain=chain,
        detour_skill=surplus_skill,
        max_detours=max_detours,
        seed=seed,
        keep=lambda state, accepted: is_surplus_boundary(
            state, accepted, surplus_skill
        ),
        keep_every=True,
    )


def collect_detour_trajectory(
    trajectory: PrefixTrajectory,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    detour_skills: tuple[SkillId, ...] | None = None,
    max_detours: int = 2,
    seed: int = 0,
) -> PrefixTrajectory:
    """Derive every state one repeated off-objective request reaches.

    The search and surplus tests each derive one class of state the closed
    loop found the planner failing in. This derives them all at once: from
    each oracle boundary it requests each of ``detour_skills`` (the chain's
    catalog by default) repeatedly with the scripted executor, and keeps the
    state after each request the chain's accepted set did not contain, up to
    ``max_detours`` of them, labelled ``"<skill>x<k>"`` after the ``k``-th.
    Requests the accepted set did contain advance the world without being
    kept, so collecting wood past what the recipes need is reached too. Every
    kept state is labelled by the chain's exact accepted set, so a planner's
    accuracy on it needs no hand-written predicate. A request that fails ends
    that skill's detour, since the failure penalty, not the state, is then
    what the planner should read.
    """

    _check_derivable(trajectory, chain, max_detours)
    assert chain.valid_next_skills is not None
    skills = chain.skills if detour_skills is None else detour_skills
    # Long enough to finish the reference plan by repetition and still make
    # ``max_detours`` off-objective requests after it.
    max_requests = len(chain.reference_skills) + max_detours
    environment = chain_environment(
        chain,
        distractor_count=trajectory.distractor_count,
    )
    executor = ScriptedSkillExecutor()
    rollout_key = jax.random.PRNGKey(seed)
    boundaries: list[tuple[PrefixBoundary, ...]] = []
    for points in trajectory.boundaries:
        found: list[PrefixBoundary] = []
        for point in points:
            for skill_id in skills:
                state, context = point.state, point.context
                detours = 0
                for _ in range(max_requests):
                    accepted = chain.valid_next_skills(state)
                    if not accepted:
                        break
                    rollout_key, execution_key = jax.random.split(rollout_key)
                    detoured = _run_detour(
                        executor,
                        environment,
                        state,
                        context,
                        skill_id,
                        execution_key,
                        trajectory.skill_horizon,
                    )
                    if detoured is None:
                        break
                    state, context = detoured
                    if skill_id in accepted:
                        continue
                    detours += 1
                    following = chain.valid_next_skills(state)
                    if following:
                        found.append(
                            PrefixBoundary(
                                episode=point.episode,
                                boundary=point.boundary,
                                reference_skill=point.reference_skill,
                                state=state,
                                context=context,
                                accepted_skills=following,
                                label=f"{skill_id.value}x{detours}",
                            )
                        )
                    if detours == max_detours:
                        break
        boundaries.append(tuple(found))
    return dataclasses.replace(
        trajectory,
        boundaries=tuple(boundaries),
        oracle_progress_failures=0,
    )


def evaluate_planner_prefixes(
    planner: SayCanPlanner,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 4,
    seed: int = 0,
    skill_horizon: int = 64,
    distractor_count: int = 3,
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
    trajectory: PrefixTrajectory | None = None,
) -> dict[str, Any]:
    """Evaluate decisions while an oracle advances through valid prefixes.

    The planner sees only the objective, screenshot-derived state, and outcome
    history. The scripted executor follows the private reference sequence
    after each decision so planner errors do not alter later test states.

    Pass a ``trajectory`` from :func:`collect_prefix_trajectory` to score
    several planners on one oracle rollout. It must have been collected with
    the same chain and settings, which is checked rather than assumed, so a
    trajectory left over from another preset cannot silently be reused.
    """

    settings = {
        "chain_name": chain.name,
        "num_episodes": num_episodes,
        "seed": seed,
        "skill_horizon": skill_horizon,
        "distractor_count": distractor_count,
    }
    if trajectory is None:
        trajectory = collect_prefix_trajectory(
            chain=chain,
            num_episodes=num_episodes,
            seed=seed,
            skill_horizon=skill_horizon,
            distractor_count=distractor_count,
        )
    else:
        mismatched = {
            name: (getattr(trajectory, name), value)
            for name, value in settings.items()
            if getattr(trajectory, name) != value
        }
        if mismatched:
            raise ValueError(
                "trajectory was collected with different settings "
                f"(trajectory, requested): {mismatched}"
            )

    traces: list[PlannerBoundaryTrace] = []
    for points in trajectory.boundaries:
        # A derived trajectory can leave a boundary with no states.
        if not points:
            continue
        selections = planner.select_many(
            [point.state for point in points],
            [point.context for point in points],
        )
        for point, selection in zip(points, selections, strict=True):
            state = point.state
            accepted_skills = point.accepted_skills
            decision = selection.decision
            selected_skill = decision.skill_id if decision is not None else None
            selected_feasibility = 0.0
            if decision is not None:
                selected_feasibility = next(
                    score.feasibility
                    for score in decision.scores
                    if score.skill_id is decision.skill_id
                )

            # Ground truth, computed independently of whichever feasibility
            # model the planner uses, so the diagnostics below compare scorers
            # on the same footing: which skills were actually usable from here
            # without first exploring.
            distances = travel_distance_map(state, reachability)
            feasible_skills = tuple(
                skill_id
                for skill_id in planner.candidate_skills
                if bool(
                    np.asarray(
                        skill_feasible(
                            SkillGoal(skill_id),
                            state,
                            reachability,
                            distances=distances,
                        )
                    )
                )
            )
            utility_top_skill = (
                selection.utility_ranking[0]
                if selection.utility_ranking
                else None
            )
            utility_top_feasible = utility_top_skill in feasible_skills
            traces.append(
                PlannerBoundaryTrace(
                    episode=point.episode,
                    boundary=point.boundary,
                    reference_skill=point.reference_skill,
                    accepted_skills=accepted_skills,
                    selected_skill=selected_skill,
                    matched=selected_skill in accepted_skills,
                    selected_feasibility=selected_feasibility,
                    utility_valid=(
                        selection.status
                        is not PlannerSelectionStatus.INVALID_UTILITY
                    ),
                    utility_error=selection.utility_error,
                    utility_ranking=selection.utility_ranking,
                    utility_raw_response=selection.utility_raw_response,
                    feasible_skills=feasible_skills,
                    utility_top_skill=utility_top_skill,
                    utility_top_feasible=utility_top_feasible,
                    selected_feasible=selected_skill in feasible_skills,
                    utility_repaired=selection.utility_repaired,
                    label=point.label,
                )
            )

    # Report in episode order even though the boundaries were visited in
    # lockstep, so traces read as episode transcripts.
    traces.sort(key=lambda trace: (trace.episode, trace.boundary))

    matches = np.asarray([trace.matched for trace in traces], dtype=np.bool_)
    repaired = np.asarray(
        [trace.utility_repaired for trace in traces],
        dtype=np.bool_,
    )
    # Under strict parsing a repaired response is invalid and selects nothing,
    # so the strict accuracy follows from the same traces without regenerating.
    strict_matches = matches & ~repaired
    selections = [trace.selected_skill for trace in traces]
    utility_validity = np.asarray(
        [trace.utility_valid for trace in traces],
        dtype=np.bool_,
    )
    utility_errors = Counter(
        trace.utility_error
        for trace in traces
        if not trace.utility_valid and trace.utility_error is not None
    )
    # "Grounded" means the chosen skill was usable from this very state, not
    # merely that its precondition held. The old precondition-based definition
    # was satisfied at essentially every boundary and so measured nothing.
    grounded = np.asarray(
        [trace.selected_feasible for trace in traces],
        dtype=np.bool_,
    )

    # Boundaries where a feasible skill existed and the utility model still
    # ranked an infeasible one first. This is the failure the feasibility score
    # exists to catch, and it is invisible in the selection metrics above.
    ranked_traces = [
        trace
        for trace in traces
        if trace.utility_top_skill is not None and trace.feasible_skills
    ]
    ungrounded_top_choice = np.asarray(
        [not trace.utility_top_feasible for trace in ranked_traces],
        dtype=np.bool_,
    )
    rescued = np.asarray(
        [
            not trace.utility_top_feasible
            and trace.selected_feasible
            and trace.matched
            for trace in ranked_traces
        ],
        dtype=np.bool_,
    )
    label_matches: dict[str, list[bool]] = {}
    for trace in traces:
        label_matches.setdefault(trace.label, []).append(trace.matched)
    mean_feasible_skills = float(
        np.mean([len(trace.feasible_skills) for trace in traces])
    ) if traces else 0.0
    return {
        "chain": chain.name,
        "num_episodes": num_episodes,
        "num_boundaries": len(traces),
        "accepted_next_skill_accuracy": float(matches.mean())
        if len(matches)
        else 0.0,
        "strict_accepted_next_skill_accuracy": float(strict_matches.mean())
        if len(strict_matches)
        else 0.0,
        "repaired_utility_response_rate": float(repaired.mean())
        if len(repaired)
        else 0.0,
        "valid_utility_response_rate": float(utility_validity.mean())
        if len(utility_validity)
        else 0.0,
        "invalid_utility_errors": dict(utility_errors),
        "valid_selection_rate": float(
            np.mean([selection is not None for selection in selections])
        )
        if selections
        else 0.0,
        "grounded_selection_rate": float(grounded.mean())
        if len(grounded)
        else 0.0,
        "mean_feasible_skills": mean_feasible_skills,
        "ungrounded_top_choice_rate": float(ungrounded_top_choice.mean())
        if len(ungrounded_top_choice)
        else 0.0,
        "feasibility_rescue_rate": float(rescued.mean())
        if len(rescued)
        else 0.0,
        "oracle_progress_failures": trajectory.oracle_progress_failures,
        "accuracy_by_label": {
            label: float(np.mean(matched))
            for label, matched in sorted(label_matches.items())
        },
        "traces": tuple(traces),
    }
