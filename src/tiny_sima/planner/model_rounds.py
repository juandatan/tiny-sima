"""Closed-loop rounds that correct a skill model from its own executor runs.

Each round runs a planner built on the current model for a few chain
episodes, harvests every transition the episodes record, and corrects the
model from them with :func:`~tiny_sima.planner.model_learning.correct_model`,
optionally followed by :func:`~tiny_sima.planner.model_learning.
analyse_failures`. Rounds report the chain's success rate, the skill requests
the model wasted and how far the model still is from the registry, so a model
started from the VLM's induction or from a corrupted registry can be followed
back towards the registry, or not.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax

from tiny_sima.goals import SkillId, SkillMechanics, StateEffect, StateRequirement
from tiny_sima.memory import EpisodeMemory, TransitionRecord
from tiny_sima.planner.model_learning import (
    DEFAULT_FAILURE_CAUSES,
    DEFAULT_MIN_SUPPORT,
    analyse_failures,
    correct_model,
    model_diff,
    model_to_dict,
)
from tiny_sima.planner.saycan import SayCanPlanner, SkillPlanner, constant_utility
from tiny_sima.planner.subgoals import SubgoalPlanner, SubgoalSource
from tiny_sima.planner.symbolic import (
    REGISTRY_MODEL,
    PlanProgressFeasibility,
    SkillModel,
    shortest_plan,
)
from tiny_sima.planner.vlm import VisionLanguageBackend

if TYPE_CHECKING:
    from tiny_sima.evaluation.chains import (
        ChainEpisodeResult,
        SkillChainSpec,
        SkillExecutor,
    )

PlannerFactory = Callable[[SkillModel], SkillPlanner]


# ---------------------------------------------------------------------------
# Corrupted registries


@dataclass(frozen=True)
class Corruption:
    """One wrong cost in a skill model.

    ``amount`` replaces both the skill's ``at_least`` requirement on
    ``field`` and its decrease of it, adding either if missing. ``None``
    removes both instead, so the skill seems to need none of the field.
    """

    skill_id: SkillId
    field: str
    amount: int | None

    def apply(self, model: SkillModel) -> SkillModel:
        entries: list[tuple[SkillId, SkillMechanics]] = []
        for skill_id, mechanics in model:
            if skill_id is self.skill_id:
                mechanics = self._apply(mechanics)
            entries.append((skill_id, mechanics))
        return tuple(entries)

    def _apply(self, mechanics: SkillMechanics) -> SkillMechanics:
        requires = [
            requirement
            for requirement in mechanics.requires
            if (requirement.field, requirement.comparison) != (self.field, "at_least")
        ]
        decreases = any(
            effect.field == self.field and effect.operation == "decrease"
            for effect in mechanics.effects
        )
        effects = [
            effect
            for effect in mechanics.effects
            if not (effect.field == self.field and effect.operation == "decrease")
        ]
        if self.amount is not None:
            requires.insert(0, StateRequirement(self.field, "at_least", self.amount))
            if decreases or self.field.startswith("inventory."):
                effects.insert(0, StateEffect(self.field, "decrease", self.amount))
        return dataclasses.replace(
            mechanics, requires=tuple(requires), effects=tuple(effects)
        )


CORRUPTIONS: dict[str, tuple[Corruption, ...]] = {
    "table_costs_1_wood": (Corruption(SkillId.PLACE_TABLE, "inventory.wood", 1),),
    "table_costs_4_wood": (Corruption(SkillId.PLACE_TABLE, "inventory.wood", 4),),
    "wood_pickaxe_costs_2_wood": (
        Corruption(SkillId.CRAFT_WOOD_PICKAXE, "inventory.wood", 2),
    ),
    "stone_pickaxe_costs_2_stone": (
        Corruption(SkillId.CRAFT_STONE_PICKAXE, "inventory.stone", 2),
    ),
}


def _requirement_dropped(
    model: SkillModel, skill_id: SkillId, field_name: str
) -> SkillModel:
    return tuple(
        (
            entry_id,
            dataclasses.replace(
                mechanics,
                requires=tuple(
                    requirement
                    for requirement in mechanics.requires
                    if requirement.field != field_name
                ),
            )
            if entry_id is skill_id
            else mechanics,
        )
        for entry_id, mechanics in model
    )


def corrupt_registry(
    corruption: str | Sequence[Corruption],
    model: SkillModel = REGISTRY_MODEL,
) -> SkillModel:
    """Return ``model`` with a named corruption, or the given ones, applied.

    The names are the keys of :data:`CORRUPTIONS`, plus
    ``"stone_without_pickaxe"``, which drops the wood-pickaxe requirement of
    collect_stone: a missing requirement that correction from successes
    cannot add back but failure analysis can.
    """

    if isinstance(corruption, str):
        if corruption == "stone_without_pickaxe":
            return _requirement_dropped(
                model, SkillId.COLLECT_STONE, "inventory.wood_pickaxe"
            )
        if corruption not in CORRUPTIONS:
            names = sorted((*CORRUPTIONS, "stone_without_pickaxe"))
            raise ValueError(
                f"Unknown corruption {corruption!r}; expected one of {names}."
            )
        corruption = CORRUPTIONS[corruption]
    for item in corruption:
        model = item.apply(model)
    return model


# ---------------------------------------------------------------------------
# Harvesting transitions


@dataclass(frozen=True)
class HarvestingMemory(EpisodeMemory):
    """An episode memory that also appends each transition to ``sink``.

    :func:`~tiny_sima.evaluation.chains.evaluate_chain_episodes` keeps each
    episode's memory to itself, so a factory handing every episode the same
    sink collects the transitions of all of them. Being an
    :class:`EpisodeMemory`, it is given the executor's inventory extremes.
    """

    sink: list[TransitionRecord] = field(
        default_factory=list, compare=False, repr=False
    )

    def record(
        self,
        pre_state: Any,
        skill_id: SkillId | str,
        outcome: Any,
        steps: int,
        post_state: Any,
        *,
        inventory_extremes: Any = None,
    ) -> EpisodeMemory:
        memory = super().record(
            pre_state,
            skill_id,
            outcome,
            steps,
            post_state,
            inventory_extremes=inventory_extremes,
        )
        self.sink.append(memory.transitions[-1])
        return memory


def harvesting_memory_factory(
    sink: list[TransitionRecord],
) -> Callable[[], HarvestingMemory]:
    """Return a memory factory whose memories all append to ``sink``."""

    return lambda: HarvestingMemory(sink=sink)


def run_harvested_episodes(
    planner: SkillPlanner,
    executor: SkillExecutor,
    rngs: Sequence[jax.Array],
    *,
    chain: SkillChainSpec,
    skill_horizon: int = 64,
    max_decisions: int | None = None,
    max_failures_per_skill: int = 2,
    distractor_count: int = 3,
) -> tuple[list[ChainEpisodeResult], list[TransitionRecord]]:
    """Run chain episodes under episode memory; return them and every transition."""

    from tiny_sima.evaluation.chains import (
        LearnedSkillExecutor,
        chain_decision_limit,
        chain_environment,
        evaluate_chain_episodes,
    )

    if isinstance(executor, LearnedSkillExecutor):
        executor.require_skills(chain.skills)
    sink: list[TransitionRecord] = []
    episodes = evaluate_chain_episodes(
        planner,
        executor,
        chain_environment(chain, distractor_count=distractor_count),
        list(rngs),
        chain=chain,
        skill_horizon=skill_horizon,
        max_decisions=(
            max_decisions if max_decisions is not None else chain_decision_limit(chain)
        ),
        max_failures_per_skill=max_failures_per_skill,
        memory_factory=harvesting_memory_factory(sink),
    )
    return episodes, sink


# ---------------------------------------------------------------------------
# Planners over a model


def plan_veto_planner(chain: SkillChainSpec) -> PlannerFactory:
    """Return a factory for the plan-only planner: constant utility, plan veto.

    Every skill that starts a shortest plan under the model is equally good,
    and every other is vetoed, so the model alone decides what runs.
    """

    if chain.goal is None:
        raise ValueError(f"Chain {chain.name!r} has no formal goal to plan to.")
    goal = chain.goal

    def build(model: SkillModel) -> SkillPlanner:
        return SayCanPlanner(
            candidate_skills=chain.skills,
            utility_scorer=constant_utility,
            feasibility_scorer=PlanProgressFeasibility(
                goal, skills=chain.skills, model=model
            ),
        )

    return build


def subgoal_planner_factory(
    backend: VisionLanguageBackend | SubgoalSource,
    chain: SkillChainSpec,
    **options: Any,
) -> PlannerFactory:
    """Return a factory for :class:`SubgoalPlanner` over the model.

    ``options`` are passed on to every planner; each round gets a fresh one,
    so no response cache carries over between models.
    """

    def build(model: SkillModel) -> SkillPlanner:
        return SubgoalPlanner(
            backend, candidate_skills=chain.skills, model=model, **options
        )

    return build


# ---------------------------------------------------------------------------
# Rounds


@dataclass(frozen=True)
class RoundSummary:
    """What one round's episodes did under the model they were planned with.

    ``extra_successes`` counts, over successful episodes, the successful
    skills beyond the registry's shortest plan from the episode's start;
    ``wasted_skills`` adds them to the failed requests.
    """

    episodes: int
    successes: int
    skill_requests: int
    failed_requests: int
    extra_successes: int
    termination_counts: dict[str, int]
    failure_causes: dict[str, int]

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def wasted_skills(self) -> int:
        return self.failed_requests + self.extra_successes

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate,
            "skill_requests": self.skill_requests,
            "failed_requests": self.failed_requests,
            "extra_successes": self.extra_successes,
            "wasted_skills": self.wasted_skills,
            "termination_counts": self.termination_counts,
            "failure_causes": self.failure_causes,
        }


def summarize_round(
    episodes: Sequence[ChainEpisodeResult],
    transitions: Sequence[TransitionRecord],
    rngs: Sequence[jax.Array],
    *,
    chain: SkillChainSpec,
    distractor_count: int = 3,
) -> RoundSummary:
    """Summarize one round; ``rngs`` are the episodes' keys, in order."""

    from tiny_sima.evaluation.chains import chain_environment

    extra = 0
    if chain.goal is not None:
        environment = chain_environment(chain, distractor_count=distractor_count)
        for episode, rng in zip(episodes, rngs, strict=True):
            if not episode.succeeded:
                continue
            plan = shortest_plan(environment.reset(rng).state, chain.goal, chain.skills)
            if plan is not None:
                done = sum(trace.outcome.value == "success" for trace in episode.trace)
                extra += max(0, done - plan[1])
    requests = sum(len(episode.trace) for episode in episodes)
    failed = sum(
        trace.outcome.value != "success"
        for episode in episodes
        for trace in episode.trace
    )
    return RoundSummary(
        episodes=len(episodes),
        successes=sum(episode.succeeded for episode in episodes),
        skill_requests=requests,
        failed_requests=failed,
        extra_successes=extra,
        termination_counts=dict(
            Counter(episode.termination.value for episode in episodes)
        ),
        failure_causes=dict(
            Counter(
                record.cause
                for record in transitions
                if record.outcome != "success" and record.cause is not None
            )
        ),
    )


def run_model_learning_rounds(
    model: SkillModel,
    planner_factory: PlannerFactory,
    executor: SkillExecutor,
    *,
    chain: SkillChainSpec,
    rounds: int = 3,
    episodes_per_round: int = 8,
    seed: int = 0,
    correct: bool = True,
    failure_backend: VisionLanguageBackend | None = None,
    failure_causes: Sequence[str] = DEFAULT_FAILURE_CAUSES,
    accumulate: bool = True,
    min_support: int = DEFAULT_MIN_SUPPORT,
    reference: SkillModel = REGISTRY_MODEL,
    skill_horizon: int = 64,
    max_decisions: int | None = None,
    max_failures_per_skill: int = 2,
    distractor_count: int = 3,
) -> dict[str, Any]:
    """Run ``rounds`` rounds of ``episodes_per_round`` episodes, learning between.

    Round ``r`` plans with the model left by round ``r - 1`` on the worlds of
    ``jax.random.fold_in(PRNGKey(seed), r)``, so a run with ``correct=False``
    and no ``failure_backend`` sees the same worlds under the starting model
    throughout, as the uncorrected baseline. After each round the model is
    corrected from the transitions of every round so far (only that round's
    with ``accumulate=False``), then, with ``failure_backend``, its
    unexplained failures are analysed.

    Returns a JSON-ready dict: the settings, one entry per round with the
    model it planned with, its diff against ``reference``, its
    :class:`RoundSummary` and what was learned after it, and the final model
    and diff.
    """

    if rounds < 1 or episodes_per_round < 1:
        raise ValueError("rounds and episodes_per_round must be positive.")
    harvested: list[TransitionRecord] = []
    round_results: list[dict[str, Any]] = []
    for round_index in range(rounds):
        rngs = list(
            jax.random.split(
                jax.random.fold_in(jax.random.PRNGKey(seed), round_index),
                episodes_per_round,
            )
        )
        episodes, transitions = run_harvested_episodes(
            planner_factory(model),
            executor,
            rngs,
            chain=chain,
            skill_horizon=skill_horizon,
            max_decisions=max_decisions,
            max_failures_per_skill=max_failures_per_skill,
            distractor_count=distractor_count,
        )
        summary = summarize_round(
            episodes, transitions, rngs, chain=chain, distractor_count=distractor_count
        )
        harvested = [*harvested, *transitions] if accumulate else list(transitions)
        entry: dict[str, Any] = {
            "round": round_index,
            "model": model_to_dict(model),
            "diff_vs_reference": [
                change.to_dict() for change in model_diff(reference, model)
            ],
            **summary.to_dict(),
            "transitions": len(transitions),
            "correction": None,
            "failure_analysis": [],
        }
        if correct:
            correction = correct_model(model, harvested, min_support=min_support)
            model = correction.model
            entry["correction"] = {
                "changes": [change.to_dict() for change in correction.changes],
                "support": {
                    skill_id.value: count for skill_id, count in correction.support
                },
            }
        if failure_backend is not None:
            model, analyses = analyse_failures(
                failure_backend, model, harvested, causes=failure_causes
            )
            entry["failure_analysis"] = [analysis.to_dict() for analysis in analyses]
        round_results.append(entry)
    return {
        "chain": chain.name,
        "settings": {
            "rounds": rounds,
            "episodes_per_round": episodes_per_round,
            "seed": seed,
            "correct": correct,
            "failure_analysis": failure_backend is not None,
            "accumulate": accumulate,
            "min_support": min_support,
            "skill_horizon": skill_horizon,
        },
        "rounds": round_results,
        "final_model": model_to_dict(model),
        "final_diff_vs_reference": [
            change.to_dict() for change in model_diff(reference, model)
        ],
    }
