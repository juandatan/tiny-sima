"""Planning-only evaluations at controlled persistent-world boundaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    ScriptedSkillExecutor,
    SkillChainSpec,
)
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_feasible,
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


@dataclass
class _PrefixEpisode:
    """One episode's cursor while every episode advances in lockstep."""

    index: int
    state: Any
    context: PlannerContext
    rollout_key: Any


@dataclass(frozen=True)
class PrefixBoundary:
    """One oracle-reached state at which a planner decision is scored."""

    episode: int
    boundary: int
    reference_skill: SkillId
    state: Any
    context: PlannerContext
    accepted_skills: tuple[SkillId, ...]


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
    environment = make_skill_chain_environment(
        distractor_count=distractor_count,
    )
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
        "traces": tuple(traces),
    }
