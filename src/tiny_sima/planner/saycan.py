"""Small SayCan-style skill ranking with closed-loop outcome feedback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    SKILL_REGISTRY,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    get_skill_spec,
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


class PlannerMemory(Protocol):
    """Episode memory the closed loop updates at every option boundary.

    Both methods return the updated memory rather than mutating it, so a
    context keeps the memory it was planned with.
    """

    def observe(self, state: Any) -> PlannerMemory:
        """Return the memory after seeing ``state``, the episode's reset."""

        ...

    def record(
        self,
        pre_state: Any,
        skill_id: SkillId,
        outcome: SkillOutcome,
        steps: int,
        post_state: Any,
    ) -> PlannerMemory:
        """Return the memory after one executed skill request."""

        ...


@dataclass(frozen=True)
class PlannerContext:
    """Objective state supplied to a planner at an option boundary.

    ``plan_state`` is opaque per-episode planner state, such as a sub-goal
    plan: the closed loop hands each selection's ``plan_state`` to the next
    context. Neither it nor ``memory`` takes part in equality.
    """

    objective: str
    history: tuple[SkillAttempt, ...] = ()
    memory: PlannerMemory | None = field(default=None, compare=False)
    plan_state: Any = field(default=None, compare=False)


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
    utility_repaired: bool = False
    # The planner's per-episode state after this selection, which the closed
    # loop passes on as the next context's plan_state.
    plan_state: Any = None


@dataclass(frozen=True)
class UtilityScoreBatch:
    """Whole-catalog utility scores and their validation status."""

    utilities: dict[SkillId, float]
    valid: bool = True
    error: str | None = None
    ranking: tuple[SkillId, ...] = ()
    raw_response: str | None = None
    # True when the ranking was accepted only after a parse-time repair, so the
    # strict-parse result stays recoverable from the same generations.
    repaired: bool = False


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
class MultiStateUtilityScorer(Protocol):
    """Score several option boundaries in one planner-model invocation.

    Generation dominates the wall clock of a planner evaluation, and the
    boundaries reached by independent episodes are independent of each other,
    so a scorer able to decode them together replaces one call per episode
    with one call per boundary.
    """

    def score_all_batched(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
        candidate_skills: tuple[SkillId, ...],
    ) -> list[UtilityScoreBatch]:
        """Return one whole-catalog utility batch per state, in order."""

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


def _held_tool_thresholds() -> dict[str, int]:
    """Return each tool field with the most any requirement asks of it.

    A tool is an inventory field some skill increases and none decreases: no
    recipe spends it, so once a requirement on it holds it holds for good.
    """

    increased: set[str] = set()
    decreased: set[str] = set()
    thresholds: dict[str, int] = {}
    for spec in SKILL_REGISTRY.values():
        for effect in spec.mechanics.effects:
            if effect.operation == "increase":
                increased.add(effect.field)
            elif effect.operation == "decrease":
                decreased.add(effect.field)
        for requirement in spec.mechanics.requires:
            if requirement.comparison == "at_least":
                thresholds[requirement.field] = max(
                    thresholds.get(requirement.field, 1),
                    int(requirement.value),
                )
    return {
        field: thresholds.get(field, 1)
        for field in sorted(increased - decreased)
        if field.startswith("inventory.")
    }


def redundant_craft(state: Any, skill_id: SkillId) -> bool:
    """Return whether a skill only makes tools the player already holds.

    The rule reads nothing but the static skill mechanics the planner prompt
    already shows: a second wood pickaxe satisfies no requirement the first
    did not. Skills that also gather or spend a consumable are never
    redundant, so it cannot veto wood, stone, a table or a drink.
    """

    thresholds = _HELD_TOOL_THRESHOLDS
    increased = [
        effect.field
        for effect in get_skill_spec(skill_id).mechanics.effects
        if effect.operation == "increase"
    ]
    if not increased or any(field not in thresholds for field in increased):
        return False
    inventory = state.inventory
    return all(
        int(np.asarray(getattr(inventory, field.removeprefix("inventory."))))
        >= thresholds[field]
        for field in increased
    )


def catalog_feasibility(
    scorer: FeasibilityScorer | BatchedFeasibilityScorer,
    state: Any,
    context: PlannerContext,
    candidate_skills: tuple[SkillId, ...],
) -> dict[SkillId, float]:
    """Score every candidate with either kind of feasibility scorer."""

    if isinstance(scorer, BatchedFeasibilityScorer):
        scores = scorer.score_all_feasibility(state, context, candidate_skills)
        return {
            skill_id: float(scores.get(skill_id, 0.0))
            for skill_id in candidate_skills
        }
    return {
        skill_id: float(scorer(state, context, skill_id))
        for skill_id in candidate_skills
    }


@dataclass(frozen=True)
class SkipHeldCraftFeasibility:
    """Veto crafts whose result is already held, and defer the rest.

    Under the reachability Can term the remaining waste was a second wood
    pickaxe: 23 decisions over 64 seed-51 episodes. This wrapper scores such
    a craft zero and otherwise returns the wrapped scorer's value.
    """

    base: FeasibilityScorer | BatchedFeasibilityScorer = reachability_feasibility

    def score_all_feasibility(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        scores = catalog_feasibility(self.base, state, context, candidate_skills)
        return {
            skill_id: 0.0 if redundant_craft(state, skill_id) else value
            for skill_id, value in scores.items()
        }

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return self.score_all_feasibility(state, context, (skill_id,))[skill_id]


_HELD_TOOL_THRESHOLDS = _held_tool_thresholds()


def _planner_selection(
    scores: tuple[SkillScore, ...],
    utility_batch: UtilityScoreBatch,
) -> PlannerSelection:
    """Turn one boundary's ranking into a decision plus its diagnostics."""

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
            utility_repaired=utility_batch.repaired,
        )
    return PlannerSelection(
        decision=PlannerDecision(
            skill_id=scores[0].skill_id,
            scores=scores,
        ),
        status=PlannerSelectionStatus.SELECTED,
        utility_ranking=utility_batch.ranking,
        utility_raw_response=utility_batch.raw_response,
        utility_repaired=utility_batch.repaired,
    )


class SkillPlanner(Protocol):
    """A planner the closed loop can drive, SayCan or otherwise."""

    def select_many(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
    ) -> list[PlannerSelection]:
        """Return one selection per boundary, in order."""

        ...


@dataclass(frozen=True)
class SayCanPlanner:
    """Rank skills by utility times feasibility and re-rank after outcomes.

    The default Can term is reachability alone. A calibrated
    :class:`~tiny_sima.planner.competence.CompetencePrior` replaces its single
    hand-set constant with measured success rates once one has been fitted;
    :class:`~tiny_sima.planner.affordance.CriticAffordanceScorer` is retained
    only as the ablation that showed the critic cannot do this job.
    """

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

    def _utility_batch(
        self,
        state: Any,
        context: PlannerContext,
    ) -> UtilityScoreBatch:
        if isinstance(self.utility_scorer, BatchedUtilityScorer):
            return self.utility_scorer.score_all(
                state,
                context,
                self.candidate_skills,
            )
        return UtilityScoreBatch(
            utilities={
                skill_id: float(self.utility_scorer(state, context, skill_id))
                for skill_id in self.candidate_skills
            }
        )

    def _utility_batches(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
    ) -> list[UtilityScoreBatch]:
        if isinstance(self.utility_scorer, MultiStateUtilityScorer):
            batches = list(
                self.utility_scorer.score_all_batched(
                    states,
                    contexts,
                    self.candidate_skills,
                )
            )
            if len(batches) != len(states):
                raise ValueError(
                    f"score_all_batched returned {len(batches)} batches for "
                    f"{len(states)} states."
                )
            return batches
        return [
            self._utility_batch(state, context)
            for state, context in zip(states, contexts, strict=True)
        ]

    def _scores(
        self,
        state: Any,
        context: PlannerContext,
        utility_batch: UtilityScoreBatch,
    ) -> tuple[SkillScore, ...]:
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
        return tuple(
            sorted(
                scores,
                key=lambda score: score.combined,
                reverse=True,
            )
        )

    def score(
        self,
        state: Any,
        context: PlannerContext,
    ) -> tuple[SkillScore, ...]:
        """Return a descending candidate ranking with inspectable factors."""

        utility_batch = self._utility_batch(state, context)
        return self._scores(state, context, utility_batch)

    def select_with_diagnostics(
        self,
        state: Any,
        context: PlannerContext,
    ) -> PlannerSelection:
        """Select a skill while preserving utility-model failure details."""

        utility_batch = self._utility_batch(state, context)
        return _planner_selection(
            self._scores(state, context, utility_batch),
            utility_batch,
        )

    def select_many(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
    ) -> list[PlannerSelection]:
        """Select at several boundaries with one utility-model invocation.

        Feasibility is still computed per state: a breadth-first distance map
        costs nothing next to a generation, and batching it would tie the
        planner to a fixed batch shape.
        """

        if len(states) != len(contexts):
            raise ValueError("states and contexts must have equal length.")
        utility_batches = self._utility_batches(states, contexts)
        return [
            _planner_selection(
                self._scores(state, context, utility_batch),
                utility_batch,
            )
            for state, context, utility_batch in zip(
                states,
                contexts,
                utility_batches,
                strict=True,
            )
        ]

    def select(
        self,
        state: Any,
        context: PlannerContext,
    ) -> PlannerDecision | None:
        """Select the best grounded skill, or stop when none is viable."""

        return self.select_with_diagnostics(state, context).decision
