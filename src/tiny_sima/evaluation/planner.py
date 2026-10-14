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
from tiny_sima.goals import SkillId
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


def evaluate_planner_prefixes(
    planner: SayCanPlanner,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 4,
    seed: int = 0,
    skill_horizon: int = 64,
    distractor_count: int = 3,
) -> dict[str, Any]:
    """Evaluate decisions while an oracle advances through valid prefixes.

    The planner sees only the objective, screenshot-derived state, and outcome
    history. The scripted executor follows the private reference sequence
    after each decision so planner errors do not alter later test states.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    environment = make_skill_chain_environment(
        distractor_count=distractor_count,
    )
    executor = ScriptedSkillExecutor()
    episode_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    traces: list[PlannerBoundaryTrace] = []
    oracle_progress_failures = 0

    for episode_index, episode_key in enumerate(episode_keys):
        reset_key, rollout_key = jax.random.split(episode_key)
        state = environment.reset(reset_key).state
        context = PlannerContext(objective=chain.objective)
        for boundary, reference_skill in enumerate(chain.reference_skills):
            accepted_skills = (
                chain.valid_next_skills(state)
                if chain.valid_next_skills is not None
                else (reference_skill,)
            )
            selection = planner.select_with_diagnostics(state, context)
            decision = selection.decision
            selected_skill = decision.skill_id if decision is not None else None
            selected_feasibility = 0.0
            if decision is not None:
                selected_feasibility = next(
                    score.feasibility
                    for score in decision.scores
                    if score.skill_id is decision.skill_id
                )
            traces.append(
                PlannerBoundaryTrace(
                    episode=episode_index,
                    boundary=boundary,
                    reference_skill=reference_skill,
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
                )
            )

            rollout_key, execution_key = jax.random.split(rollout_key)
            result = executor.execute(
                environment,
                state,
                reference_skill,
                execution_key,
                horizon=skill_horizon,
            )
            if result.outcome is not SkillOutcome.SUCCESS:
                oracle_progress_failures += 1
                break
            state = result.state
            context = PlannerContext(
                objective=context.objective,
                history=(
                    *context.history,
                    SkillAttempt(
                        skill_id=reference_skill,
                        outcome=result.outcome,
                        steps=result.steps,
                        native_return=result.native_return,
                    ),
                ),
            )

    matches = np.asarray([trace.matched for trace in traces], dtype=np.bool_)
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
    grounded = np.asarray(
        [
            trace.selected_skill is not None
            and trace.selected_feasibility > 0.0
            for trace in traces
        ],
        dtype=np.bool_,
    )
    return {
        "chain": chain.name,
        "num_episodes": num_episodes,
        "num_boundaries": len(traces),
        "accepted_next_skill_accuracy": float(matches.mean())
        if len(matches)
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
        "oracle_progress_failures": oracle_progress_failures,
        "traces": tuple(traces),
    }
