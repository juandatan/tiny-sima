"""Fit and score the critic-as-affordance model against real rollouts.

The planner needs ``P(skill succeeds | state)``, and the executor's critic
already encodes a discounted version of it. This module collects the pairs
needed to correct the scale: at each option boundary it records the critic's
value for every skill in the catalog and then actually runs that skill from the
same state to observe whether it succeeds.

Boundary states come from the same scripted-oracle prefix walk used by
:func:`tiny_sima.evaluation.planner.evaluate_planner_prefixes`, so the
calibration is fitted on the distribution of states the planner is evaluated
on. Each candidate rollout starts from a copy of the boundary state and its
result is discarded, so probing a skill never perturbs the states that follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    SkillChainSpec,
)
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    SkillGoal,
    SkillId,
    skill_feasible,
    travel_distance_map,
)
from tiny_sima.planner.affordance import CriticAffordanceScorer
from tiny_sima.planner.calibration import (
    IsotonicCalibration,
    calibration_metrics,
    fit_isotonic_calibration,
)
from tiny_sima.planner.saycan import PlannerContext, SkillOutcome


@dataclass(frozen=True)
class AffordanceSample:
    """One critic prediction paired with the outcome of running the skill."""

    episode: int
    boundary: int
    skill_id: SkillId
    critic_value: float
    reachability_gate: float
    reachable: bool
    succeeded: bool
    outcome: SkillOutcome
    steps: int


def collect_affordance_samples(
    scorer: CriticAffordanceScorer,
    executor: LearnedSkillExecutor,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 8,
    seed: int = 0,
    skill_horizon: int = 64,
    distractor_count: int = 3,
    candidate_skills: Sequence[SkillId] = SKILL_IDS,
) -> tuple[AffordanceSample, ...]:
    """Probe every candidate skill at every boundary of a prefix walk."""

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    skills = tuple(candidate_skills)
    environment = make_skill_chain_environment(distractor_count=distractor_count)
    oracle = ScriptedSkillExecutor()
    # The reachability gate reads only the state, but the scorer interface takes
    # a planner context, so a bare objective stands in for one here.
    context = PlannerContext(objective=chain.objective)
    episode_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    samples: list[AffordanceSample] = []

    for episode_index, episode_key in enumerate(episode_keys):
        reset_key, rollout_key = jax.random.split(episode_key)
        state = environment.reset(reset_key).state
        for boundary, reference_skill in enumerate(chain.reference_skills):
            values = scorer.critic_values(state, skills)
            gates = (
                scorer.reachability_gate.score_all_feasibility(
                    state,
                    context,
                    skills,
                )
                if scorer.reachability_gate is not None
                else dict.fromkeys(skills, 1.0)
            )
            # The strict predicate, recorded alongside the graded gate so a
            # veto that cost a real success stays visible.
            reachability = (
                scorer.reachability_gate.config
                if scorer.reachability_gate is not None
                else DEFAULT_REACHABILITY
            )
            distances = travel_distance_map(state, reachability)
            for skill_id in skills:
                rollout_key, probe_key = jax.random.split(rollout_key)
                probe = executor.execute(
                    environment,
                    state,
                    skill_id,
                    probe_key,
                    horizon=skill_horizon,
                )
                samples.append(
                    AffordanceSample(
                        episode=episode_index,
                        boundary=boundary,
                        skill_id=skill_id,
                        critic_value=values[skill_id],
                        reachability_gate=gates.get(skill_id, 0.0),
                        reachable=bool(
                            np.asarray(
                                skill_feasible(
                                    SkillGoal(skill_id),
                                    state,
                                    reachability,
                                    distances=distances,
                                )
                            )
                        ),
                        succeeded=probe.outcome is SkillOutcome.SUCCESS,
                        outcome=probe.outcome,
                        steps=probe.steps,
                    )
                )

            rollout_key, advance_key = jax.random.split(rollout_key)
            result = oracle.execute(
                environment,
                state,
                reference_skill,
                advance_key,
                horizon=skill_horizon,
            )
            if result.outcome is not SkillOutcome.SUCCESS:
                break
            state = result.state

    return tuple(samples)


def fit_affordance_calibration(
    samples: Sequence[AffordanceSample],
) -> IsotonicCalibration:
    """Fit a monotone critic-value-to-success map from collected samples."""

    if not samples:
        raise ValueError("Cannot fit a calibration from zero samples.")
    scores = np.asarray([sample.critic_value for sample in samples])
    labels = np.asarray(
        [float(sample.succeeded) for sample in samples],
        dtype=np.float64,
    )
    return fit_isotonic_calibration(scores, labels)


def evaluate_affordance_calibration(
    samples: Sequence[AffordanceSample],
    calibration: IsotonicCalibration | None = None,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Report how well the critic predicts success, overall and per skill.

    When ``calibration`` is omitted a fit is made on these same samples, which
    measures separability rather than generalization. Pass a calibration fitted
    on a different seed for a held-out estimate.
    """

    if not samples:
        raise ValueError("Cannot evaluate zero samples.")
    scores = np.asarray([sample.critic_value for sample in samples])
    labels = np.asarray(
        [float(sample.succeeded) for sample in samples],
        dtype=np.float64,
    )
    fitted = (
        fit_isotonic_calibration(scores, labels)
        if calibration is None
        else calibration
    )
    reachable = np.asarray(
        [sample.reachable for sample in samples],
        dtype=np.bool_,
    )

    per_skill: dict[str, dict[str, float]] = {}
    for skill_id in sorted({sample.skill_id for sample in samples}):
        selected = [sample for sample in samples if sample.skill_id is skill_id]
        skill_scores = np.asarray([sample.critic_value for sample in selected])
        skill_labels = np.asarray(
            [float(sample.succeeded) for sample in selected],
            dtype=np.float64,
        )
        per_skill[skill_id.value] = {
            "mean_critic_value": float(skill_scores.mean()),
            "success_rate": float(skill_labels.mean()),
            **calibration_metrics(
                skill_scores,
                skill_labels,
                fitted,
                num_bins=num_bins,
            ),
        }

    # How often a skill judged unreachable was nevertheless completed. A hard
    # gate would have thrown each of these away, which is the measurement that
    # argues for grading reachability instead of vetoing on it.
    unreachable_successes = int(np.count_nonzero(~reachable & (labels > 0.5)))
    return {
        "num_samples": len(samples),
        "overall": calibration_metrics(scores, labels, fitted, num_bins=num_bins),
        "per_skill": per_skill,
        "unreachable_rate": float((~reachable).mean()),
        "unreachable_successes": unreachable_successes,
        "reachable_success_rate": float(labels[reachable].mean())
        if np.any(reachable)
        else 0.0,
        "unreachable_success_rate": float(labels[~reachable].mean())
        if np.any(~reachable)
        else 0.0,
        "calibration": fitted,
        "samples": tuple(samples),
    }
