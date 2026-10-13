"""Small SayCan-style skill ranking with closed-loop outcome feedback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from tiny_sima.goals import SKILL_IDS, SkillGoal, SkillId, skill_precondition


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


UtilityScorer = Callable[[Any, PlannerContext, SkillId], float]
FeasibilityScorer = Callable[[Any, PlannerContext, SkillId], float]


def zero_utility(
    _state: Any,
    _context: PlannerContext,
    _skill_id: SkillId,
) -> float:
    """Provide a safe default until a planner utility model is configured."""

    return 0.0


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
    """Use explicit Craftax preconditions as an initial affordance model."""

    feasible = skill_precondition(SkillGoal(skill_id), state)
    return float(bool(np.asarray(feasible)))


@dataclass(frozen=True)
class SayCanPlanner:
    """Rank skills by utility times feasibility and re-rank after outcomes."""

    utility_scorer: UtilityScorer = zero_utility
    feasibility_scorer: FeasibilityScorer = symbolic_feasibility
    candidate_skills: tuple[SkillId, ...] = SKILL_IDS

    def score(
        self,
        state: Any,
        context: PlannerContext,
    ) -> tuple[SkillScore, ...]:
        """Return a descending candidate ranking with inspectable factors."""

        scores = []
        for skill_id in self.candidate_skills:
            utility = float(self.utility_scorer(state, context, skill_id))
            feasibility = float(self.feasibility_scorer(state, context, skill_id))
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
        return tuple(
            sorted(
                scores,
                key=lambda score: score.combined,
                reverse=True,
            )
        )

    def select(
        self,
        state: Any,
        context: PlannerContext,
    ) -> PlannerDecision | None:
        """Select the best grounded skill, or stop when none is viable."""

        scores = self.score(state, context)
        if not scores or scores[0].combined <= 0.0:
            return None
        return PlannerDecision(skill_id=scores[0].skill_id, scores=scores)
