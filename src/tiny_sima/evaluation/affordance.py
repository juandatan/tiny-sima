"""Label boundary states by probing skills, then fit a competence head to them.

At each option boundary this records, for every skill in the catalog, what the
executor's critic predicted, whether the target was reachable and how far away
it was -- and then actually runs that skill from the same state to observe
whether it succeeds. Each candidate rollout starts from a copy of the boundary
state and its result is discarded, so probing never perturbs the states that
follow.

Boundary states come from the same scripted-oracle prefix walk used by
:func:`tiny_sima.evaluation.planner.evaluate_planner_prefixes`, so a head fitted
here sees the distribution of states the planner is evaluated on.

Two heads are fitted from the same labels. The isotonic recalibration in
:func:`fit_affordance_calibration` is the critic-as-affordance hypothesis, kept
because it is the code that measured its own failure: the critic separates
success at chance level, so the recalibration collapses to the base rate and
multiplies the SayCan score by a near-constant. The replacement is
:func:`fit_affordance_competence_prior`, which conditions on the features the
same labels show carry signal -- see :mod:`tiny_sima.planner.competence`.

Probes are the expensive part, so they run batched on device. Fit on one seed
and evaluate on another: a calibration scored against the samples it was fitted
on measures separability, not generalization.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    SkillChainSpec,
    chain_environment,
    probe_skills_batched,
)
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_feasible,
    skill_precondition,
    skill_reachability_report,
    travel_distance_map,
)
from tiny_sima.planner.affordance import CriticAffordanceScorer
from tiny_sima.planner.calibration import (
    IsotonicCalibration,
    calibration_metrics,
    fit_isotonic_calibration,
)
from tiny_sima.planner.competence import (
    DEFAULT_DISTANCE_EDGES,
    DEFAULT_FLOOR,
    DEFAULT_PRIOR_STRENGTH,
    CompetencePrior,
    distance_cell,
    fit_competence_prior,
    num_distance_cells,
)
from tiny_sima.planner.saycan import PlannerContext, SkillOutcome


@dataclass(frozen=True)
class AffordanceSample:
    """One set of boundary-state features paired with a probe outcome."""

    episode: int
    boundary: int
    skill_id: SkillId
    critic_value: float
    reachability_gate: float
    reachable: bool
    precondition: bool
    steps_to_target: int | None
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
    """Probe every candidate skill at every boundary of a prefix walk.

    Episodes advance in lockstep over the boundary index so that one batched
    probe covers every ``(episode, skill)`` pair at that boundary. The batch
    length stays ``num_episodes * len(candidate_skills)`` even after episodes
    stop early, because a fixed shape reuses the compiled probe; the surplus
    entries re-probe a finished episode's last state and are discarded.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    skills = tuple(candidate_skills)
    if not skills:
        raise ValueError("At least one candidate skill is required.")
    environment = chain_environment(chain, distractor_count=distractor_count)
    oracle = ScriptedSkillExecutor()
    # The reachability gate reads only the state, but the scorer interface takes
    # a planner context, so a bare objective stands in for one here.
    context = PlannerContext(objective=chain.objective)
    reachability = (
        scorer.reachability_gate.config
        if scorer.reachability_gate is not None
        else DEFAULT_REACHABILITY
    )

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_episodes)
    states = [environment.reset(reset_key).state for reset_key in reset_keys]
    alive = [True] * num_episodes
    samples: list[AffordanceSample] = []

    for boundary, reference_skill in enumerate(chain.reference_skills):
        if not any(alive):
            break

        features = [
            _boundary_features(scorer, context, reachability, states[index], skills)
            if alive[index]
            else None
            for index in range(num_episodes)
        ]

        rng, probe_rng = jax.random.split(rng)
        probes = probe_skills_batched(
            executor,
            [states[index] for index in range(num_episodes) for _ in skills],
            [skill_id for _ in range(num_episodes) for skill_id in skills],
            probe_rng,
            skill_horizon=skill_horizon,
            distractor_count=distractor_count,
        )
        for index in range(num_episodes):
            boundary_features = features[index]
            if boundary_features is None:
                continue
            values, gates, reports, preconditions, feasible = boundary_features
            for offset, skill_id in enumerate(skills):
                probe = probes[index * len(skills) + offset]
                samples.append(
                    AffordanceSample(
                        episode=index,
                        boundary=boundary,
                        skill_id=skill_id,
                        critic_value=values[skill_id],
                        reachability_gate=gates.get(skill_id, 0.0),
                        # The strict predicate -- precondition and reachability
                        # together -- recorded so a veto that would have cost a
                        # real success stays visible.
                        reachable=feasible[skill_id],
                        precondition=preconditions[skill_id],
                        steps_to_target=reports[skill_id],
                        succeeded=probe.succeeded,
                        outcome=probe.outcome,
                        steps=probe.steps,
                    )
                )

        for index in range(num_episodes):
            if not alive[index]:
                continue
            rng, advance_key = jax.random.split(rng)
            result = oracle.execute(
                environment,
                states[index],
                reference_skill,
                advance_key,
                horizon=skill_horizon,
            )
            if result.outcome is not SkillOutcome.SUCCESS:
                alive[index] = False
                continue
            states[index] = result.state

    return tuple(samples)


def _boundary_features(
    scorer: CriticAffordanceScorer,
    context: PlannerContext,
    reachability: ReachabilityConfig,
    state: Any,
    skills: tuple[SkillId, ...],
) -> tuple[
    dict[SkillId, float],
    dict[SkillId, float],
    dict[SkillId, int | None],
    dict[SkillId, bool],
    dict[SkillId, bool],
]:
    """Read every scorer feature from one boundary state."""

    values = scorer.critic_values(state, skills)
    gates = (
        scorer.reachability_gate.score_all_feasibility(state, context, skills)
        if scorer.reachability_gate is not None
        else dict.fromkeys(skills, 1.0)
    )
    reports = skill_reachability_report(state, reachability, skills)
    distances = travel_distance_map(state, reachability)
    preconditions = {
        skill_id: bool(np.asarray(skill_precondition(SkillGoal(skill_id), state)))
        for skill_id in skills
    }
    feasible = {
        skill_id: bool(
            np.asarray(
                skill_feasible(
                    SkillGoal(skill_id),
                    state,
                    reachability,
                    distances=distances,
                )
            )
        )
        for skill_id in skills
    }
    return values, gates, reports, preconditions, feasible


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

    Pass a ``calibration`` fitted on samples collected under a *different* seed
    for a held-out estimate; the returned ``held_out`` flag records whether that
    happened. When ``calibration`` is omitted a fit is made on these same
    samples, which measures separability rather than generalization and makes
    the reported calibration error meaningless.
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
        "held_out": calibration is not None,
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


def fit_affordance_competence_prior(
    samples: Sequence[AffordanceSample],
    *,
    distance_edges: Sequence[int] = DEFAULT_DISTANCE_EDGES,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    floor: float = DEFAULT_FLOOR,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    skills: Sequence[SkillId] = SKILL_IDS,
) -> CompetencePrior:
    """Fit the ``(skill, reachable, steps)`` competence head from probes.

    Samples whose symbolic precondition failed are dropped: the precondition
    vetoes at scoring time, so a skill should not be charged for states where
    the planner never consults its competence.
    """

    usable = [sample for sample in samples if sample.precondition]
    if not usable:
        raise ValueError(
            "Cannot fit a competence prior: no sample satisfied its precondition."
        )
    return fit_competence_prior(
        [sample.skill_id for sample in usable],
        [sample.steps_to_target for sample in usable],
        [sample.succeeded for sample in usable],
        distance_edges=distance_edges,
        prior_strength=prior_strength,
        floor=floor,
        config=config,
        skills=skills,
    )


def evaluate_competence_prior(
    samples: Sequence[AffordanceSample],
    prior: CompetencePrior,
    *,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Score a fitted competence head against probe outcomes.

    Unlike the critic, the head emits probabilities directly, so the reported
    Brier score and calibration error need no recalibration step. Collect these
    samples under a different seed than the one the head was fitted on.
    """

    if not samples:
        raise ValueError("Cannot evaluate zero samples.")
    usable = [sample for sample in samples if sample.precondition]
    if not usable:
        raise ValueError("No sample satisfied its precondition.")
    probabilities = np.asarray(
        [prior.success_probability(s.skill_id, s.steps_to_target) for s in usable],
        dtype=np.float64,
    )
    labels = np.asarray([float(s.succeeded) for s in usable], dtype=np.float64)

    per_skill: dict[str, dict[str, float]] = {}
    for skill_id in sorted({sample.skill_id for sample in usable}):
        selected = [
            index
            for index, sample in enumerate(usable)
            if sample.skill_id is skill_id
        ]
        per_skill[skill_id.value] = {
            "mean_probability": float(probabilities[selected].mean()),
            "success_rate": float(labels[selected].mean()),
            **calibration_metrics(
                probabilities[selected],
                labels[selected],
                num_bins=num_bins,
            ),
        }

    width = num_distance_cells(prior.distance_edges)
    cells: list[dict[str, Any]] = []
    for skill_id in prior.candidate_skills:
        for cell in range(width):
            in_cell = [
                index
                for index, sample in enumerate(usable)
                if sample.skill_id is skill_id
                and distance_cell(sample.steps_to_target, prior.distance_edges) == cell
            ]
            if not in_cell:
                continue
            cells.append(
                {
                    "skill": skill_id.value,
                    "cell": cell,
                    "label": _cell_label(cell, prior.distance_edges),
                    "fit_support": prior.support_by_cell(skill_id, cell),
                    "num_samples": len(in_cell),
                    "probability": prior.rate_by_cell(skill_id, cell),
                    "success_rate": float(labels[in_cell].mean()),
                }
            )

    return {
        "num_samples": len(usable),
        "num_dropped_precondition": len(samples) - len(usable),
        "base_rate": float(labels.mean()),
        "overall": calibration_metrics(probabilities, labels, num_bins=num_bins),
        "per_skill": per_skill,
        "cells": cells,
        "prior": prior,
        "samples": tuple(samples),
    }


def _cell_label(cell: int, distance_edges: Sequence[int]) -> str:
    """Describe one distance band for readable reports."""

    edges = tuple(distance_edges)
    if cell == 0:
        return "unreachable"
    if cell == 1:
        return f"steps<={edges[0]}" if edges else "reachable"
    if cell - 1 < len(edges):
        return f"{edges[cell - 2] + 1}<=steps<={edges[cell - 1]}"
    return f"steps>{edges[-1]}"
