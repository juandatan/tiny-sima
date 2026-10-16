"""Small SayCan-style skill ranking with closed-loop outcome feedback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_feasibility_score,
    skill_precondition,
    travel_distance_map,
)


class SkillOutcome(str, Enum):
    """Terminal reasons reported by a low-level skill execution."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    PRECONDITION_LOST = "precondition_lost"
    ENVIRONMENT_TERMINATED = "environment_terminated"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class SkillAttempt:
    """Compact planner memory for one completed skill request."""

    skill_id: SkillId
    outcome: SkillOutcome
    steps: int
    native_return: float

    @property
    def succeeded(self) -> bool:
        return self.outcome is SkillOutcome.SUCCESS


@dataclass(frozen=True)
class PlannerContext:
    """Objective state supplied to a planner at an option boundary."""

    objective: str
    history: tuple[SkillAttempt, ...] = ()


@dataclass(frozen=True)
class SkillScore:
    """Utility and feasibility factors for one candidate skill."""

    skill_id: SkillId
    utility: float
    feasibility: float
    failure_penalty: float
    combined: float


@dataclass(frozen=True)
class PlannerDecision:
    """Selected skill and the complete candidate ranking."""

    skill_id: SkillId
    scores: tuple[SkillScore, ...]


class PlannerSelectionStatus(str, Enum):
    """Reason a planner did or did not return a skill."""

    SELECTED = "selected"
    INVALID_UTILITY = "invalid_utility"
    NO_VIABLE_SKILL = "no_viable_skill"


@dataclass(frozen=True)
class PlannerSelection:
    """Planner decision together with utility-model diagnostics."""

    decision: PlannerDecision | None
    status: PlannerSelectionStatus
    utility_error: str | None = None
    utility_ranking: tuple[SkillId, ...] = ()
    utility_raw_response: str | None = None


@dataclass(frozen=True)
class UtilityScoreBatch:
    """Whole-catalog utility scores and their validation status."""

    utilities: dict[SkillId, float]
    valid: bool = True
    error: str | None = None
    ranking: tuple[SkillId, ...] = ()
    raw_response: str | None = None


UtilityScorer = Callable[[Any, PlannerContext, SkillId], float]
FeasibilityScorer = Callable[[Any, PlannerContext, SkillId], float]


@runtime_checkable
class BatchedUtilityScorer(Protocol):
    """Score the complete skill catalog in one planner-model invocation."""

    def score_all(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> UtilityScoreBatch:
        """Return utility in the range zero to one for each candidate."""

        ...


@runtime_checkable
class BatchedFeasibilityScorer(Protocol):
    """Score feasibility for the whole catalog from shared intermediate work.

    Reachability and critic affordances both derive every candidate's score
    from one expensive computation per state, so scoring the catalog together
    avoids repeating it once per skill.
    """

    def score_all_feasibility(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        """Return feasibility in the range zero to one for each candidate."""

        ...


def zero_utility(
    _state: Any,
    _context: PlannerContext,
    _skill_id: SkillId,
) -> float:
    """Provide a safe default until a planner utility model is configured."""

    return 0.0


def constant_utility(
    _state: Any,
    _context: PlannerContext,
    _skill_id: SkillId,
) -> float:
    """Give every skill equal utility for feasibility-only ablations."""

    return 1.0


def constant_feasibility(
    _state: Any,
    _context: PlannerContext,
    _skill_id: SkillId,
) -> float:
    """Treat every skill as feasible for utility-only ablations."""

    return 1.0


@dataclass(frozen=True)
class SequentialUtilityScorer:
    """Keep an oracle reference sequence private to the scripted baseline."""

    reference_skills: tuple[SkillId, ...]

    def __call__(
        self,
        _state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        reference_index = 0
        for attempt in context.history:
            if (
                reference_index < len(self.reference_skills)
                and attempt.succeeded
                and attempt.skill_id is self.reference_skills[reference_index]
            ):
                reference_index += 1
        if reference_index >= len(self.reference_skills):
            return 0.0
        return float(skill_id is self.reference_skills[reference_index])


def symbolic_feasibility(
    state: Any,
    _context: PlannerContext,
    skill_id: SkillId,
) -> float:
    """Use explicit Craftax preconditions as an initial affordance model.

    Resource preconditions test whether a block exists anywhere on the map, so
    this never rejects an unreachable target. It is retained as the Phase 4
    ablation baseline; prefer :class:`ReachabilityFeasibilityScorer`.
    """

    feasible = skill_precondition(SkillGoal(skill_id), state)
    return float(bool(np.asarray(feasible)))


@dataclass(frozen=True)
class ReachabilityFeasibilityScorer:
    """Grade skills by preconditions and horizon-bounded reachability.

    This refines the world-wide existence test in :func:`symbolic_feasibility`
    with a lower bound on the steps needed to stand next to a target and
    interact with it. Skills usable from here score one and skills whose target
    is out of reach score ``config.unknown_feasibility``, which demotes them
    without making them unselectable -- walking toward a distant target is a
    legitimate plan, so a hard veto here costs real episodes.
    """

    config: ReachabilityConfig = DEFAULT_REACHABILITY

    def score_all_feasibility(
        self,
        state: Any,
        _context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        """Score every candidate from one shared breadth-first distance map."""

        if not candidate_skills:
            return {}
        distances = travel_distance_map(state, self.config)
        return {
            skill_id: float(
                np.asarray(
                    skill_feasibility_score(
                        SkillGoal(skill_id),
                        state,
                        self.config,
                        distances=distances,
                    )
                )
            )
            for skill_id in candidate_skills
        }

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        """Score one candidate, recomputing the distance map on demand."""

        return self.score_all_feasibility(state, context, (skill_id,))[skill_id]


reachability_feasibility = ReachabilityFeasibilityScorer()


@dataclass(frozen=True)
class SayCanPlanner:
    """Rank skills by utility times feasibility and re-rank after outcomes."""

    utility_scorer: UtilityScorer | BatchedUtilityScorer = zero_utility
    feasibility_scorer: FeasibilityScorer | BatchedFeasibilityScorer = (
        reachability_feasibility
    )
    candidate_skills: tuple[SkillId, ...] = SKILL_IDS

    def _feasibilities(
        self,
        state: Any,
        context: PlannerContext,
    ) -> dict[SkillId, float]:
        if isinstance(self.feasibility_scorer, BatchedFeasibilityScorer):
            scores = self.feasibility_scorer.score_all_feasibility(
                state,
                context,
                self.candidate_skills,
            )
            return {
                skill_id: float(scores.get(skill_id, 0.0))
                for skill_id in self.candidate_skills
            }
        return {
            skill_id: float(self.feasibility_scorer(state, context, skill_id))
            for skill_id in self.candidate_skills
        }

    def _score_with_utility_status(
        self,
        state: Any,
        context: PlannerContext,
    ) -> tuple[tuple[SkillScore, ...], UtilityScoreBatch]:
        if isinstance(self.utility_scorer, BatchedUtilityScorer):
            utility_batch = self.utility_scorer.score_all(
                state,
                context,
                self.candidate_skills,
            )
        else:
            utility_batch = UtilityScoreBatch(
                utilities={
                    skill_id: float(
                        self.utility_scorer(state, context, skill_id)
                    )
                    for skill_id in self.candidate_skills
                }
            )

        feasibilities = self._feasibilities(state, context)
        scores = []
        for skill_id in self.candidate_skills:
            utility = float(utility_batch.utilities.get(skill_id, 0.0))
            feasibility = feasibilities[skill_id]
            failures = sum(
                attempt.skill_id is skill_id and not attempt.succeeded
                for attempt in context.history
            )
            failure_penalty = 1.0 / (1.0 + failures)
            scores.append(
                SkillScore(
                    skill_id=skill_id,
                    utility=utility,
                    feasibility=feasibility,
                    failure_penalty=failure_penalty,
                    combined=utility * feasibility * failure_penalty,
                )
            )
        ranking = tuple(
            sorted(
                scores,
                key=lambda score: score.combined,
                reverse=True,
            )
        )
        return ranking, utility_batch

    def score(
        self,
        state: Any,
        context: PlannerContext,
    ) -> tuple[SkillScore, ...]:
        """Return a descending candidate ranking with inspectable factors."""

        scores, _ = self._score_with_utility_status(state, context)
        return scores

    def select_with_diagnostics(
        self,
        state: Any,
        context: PlannerContext,
    ) -> PlannerSelection:
        """Select a skill while preserving utility-model failure details."""

        scores, utility_batch = self._score_with_utility_status(state, context)
        if not utility_batch.valid:
            return PlannerSelection(
                decision=None,
                status=PlannerSelectionStatus.INVALID_UTILITY,
                utility_error=utility_batch.error,
                utility_ranking=utility_batch.ranking,
                utility_raw_response=utility_batch.raw_response,
            )
        if not scores or scores[0].combined <= 0.0:
            return PlannerSelection(
                decision=None,
                status=PlannerSelectionStatus.NO_VIABLE_SKILL,
                utility_ranking=utility_batch.ranking,
                utility_raw_response=utility_batch.raw_response,
            )
        return PlannerSelection(
            decision=PlannerDecision(
                skill_id=scores[0].skill_id,
                scores=scores,
            ),
            status=PlannerSelectionStatus.SELECTED,
            utility_ranking=utility_batch.ranking,
            utility_raw_response=utility_batch.raw_response,
        )

    def select(
        self,
        state: Any,
        context: PlannerContext,
    ) -> PlannerDecision | None:
        """Select the best grounded skill, or stop when none is viable."""

        return self.select_with_diagnostics(state, context).decision
