"""Plan over the skill registry's declared mechanics.

Every skill already declares what it requires and what it changes, as typed
conditions over planner-visible state. That is a STRIPS-style model, so the
questions the VLM kept getting wrong -- how much wood a recipe still needs,
whether a tool is already held, which step comes next -- have exact answers a
breadth-first search can read off it. This module answers them from the
registry and a formal goal condition alone, never from a chain's accepted set.

The action model is an input. :data:`REGISTRY_MODEL` is the hand-written
registry; a model induced by a VLM or learned from the executor's outcomes is
the same type, so every source plans through the same search.

Each skill also unlocks a Craftax achievement, so ``achievement.*`` fields
are planner-visible and a sub-goal can ask for one. The registry model adds
that effect to each skill; a search tracks an achievement only when the goal
or a requirement reads it, so a plan for any other goal is searched over the
same states as before.

World facts come from a :class:`WorldBelief` when one is given, built from
what the agent has seen. Without one they are read as whether such a target
exists anywhere on the map, the same test the chains' accepted sets use. That
reading is privileged -- the agent has not seen most of the map -- and is kept
for :class:`PlanProgressFeasibility`, the upper bound. How far a target is
belongs to the Can term the plan is combined with, and the collect skills
search for what is out of view.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
from craftax.craftax_classic.constants import Achievement, BlockType
from craftax.craftax_classic.envs.craftax_state import Inventory

from tiny_sima.goals import (
    ACHIEVEMENT_FIELDS,
    ALL_SKILL_IDS,
    SKILL_ACHIEVEMENTS,
    STONE_PICKAXE_SKILL_IDS,
    SkillId,
    SkillMechanics,
    StateEffect,
    StateRequirement,
    achievement_field,
    get_skill_spec,
    walkable_mask,
)
from tiny_sima.planner.saycan import (
    BatchedFeasibilityScorer,
    FeasibilityScorer,
    PlannerContext,
    catalog_feasibility,
    reachability_feasibility,
)

GoalCondition = tuple[StateRequirement, ...]

# One entry per skill the planner may use, in catalog order. Hashable, so the
# search can be cached on it.
SkillModel = tuple[tuple[SkillId, SkillMechanics], ...]

# Plans on the recorded chains are at most eight skills long. The bound only
# stops the search when the goal cannot be reached at all.
DEFAULT_MAX_PLAN_LENGTH = 16

_STONE_PICKAXE_BLOCKS = (
    BlockType.TREE,
    BlockType.STONE,
    BlockType.WATER,
    BlockType.CRAFTING_TABLE,
)
# Blocks the later skills mine, use or eat from, read like those above.
_LATER_BLOCKS = (
    BlockType.COAL,
    BlockType.FURNACE,
    BlockType.RIPE_PLANT,
    BlockType.GRASS,
)


def _world_field(block: BlockType) -> str:
    return f"world.{block.name.lower()}_reachable"


_WORLD_BLOCKS = {
    _world_field(block): block for block in (*_STONE_PICKAXE_BLOCKS, *_LATER_BLOCKS)
}

# The fields the seven stone-pickaxe-stage skills read and change. The
# induction prompts list these, so they stay as they were recorded.
STONE_PICKAXE_FIELDS: tuple[str, ...] = (
    *(f"inventory.{item.name}" for item in dataclasses.fields(Inventory)),
    "survival.drink",
    "world.placement_space_reachable",
    *map(_world_field, _STONE_PICKAXE_BLOCKS),
    *ACHIEVEMENT_FIELDS,
)

PLANNER_FIELDS: tuple[str, ...] = (
    *STONE_PICKAXE_FIELDS,
    "survival.food",
    "survival.energy",
    *map(_world_field, _LATER_BLOCKS),
    "world.cow_reachable",
)

_ACHIEVEMENTS = {
    achievement_field(achievement): achievement for achievement in Achievement
}

_COMPARISONS = {"at_least": ">=", "less_than": "<", "equals": "=="}

# A planning state: each field the plan reads, sorted by name, with its value.
_Values = tuple[tuple[str, int], ...]


def _with_achievement(skill_id: SkillId, mechanics: SkillMechanics) -> SkillMechanics:
    achievement = SKILL_ACHIEVEMENTS.get(skill_id)
    if achievement is None:
        return mechanics
    unlock = StateEffect(achievement_field(achievement), "set", 1)
    return dataclasses.replace(mechanics, effects=(*mechanics.effects, unlock))


def registry_model(skills: Sequence[SkillId] = ALL_SKILL_IDS) -> SkillModel:
    """Return the hand-written registry's mechanics for ``skills``.

    Each skill's effects end with the achievement its success unlocks, from
    :data:`SKILL_ACHIEVEMENTS`. The registry's own mechanics leave it out,
    since they are what the planner prompts show.
    """

    return tuple(
        (skill_id, _with_achievement(skill_id, get_skill_spec(skill_id).mechanics))
        for skill_id in skills
    )


# The stone-pickaxe stage's catalog, which the recorded model-learning runs
# corrupt and compare against.
REGISTRY_MODEL: SkillModel = registry_model(STONE_PICKAXE_SKILL_IDS)


class WorldBelief(Protocol):
    """What the agent believes about the world, from what it has observed."""

    def world_fact(self, field: str) -> int | None:
        """Return a ``world.*`` field's value, or ``None`` to read the map."""
        ...


def read_field(state: Any, field: str, belief: WorldBelief | None = None) -> int:
    """Return one registry field of a world state as an integer.

    Booleans are read as zero or one, so every field compares and updates the
    same way during the search. A ``world.*`` field comes from ``belief``
    when it has a value for it, and from the whole map otherwise.
    """

    if field.startswith("inventory."):
        name = field.removeprefix("inventory.")
        return int(np.asarray(getattr(state.inventory, name)))
    if field == "survival.drink":
        return int(np.asarray(state.player_drink))
    if field == "survival.food":
        return int(np.asarray(state.player_food))
    if field == "survival.energy":
        return int(np.asarray(state.player_energy))
    if field in _ACHIEVEMENTS:
        return int(np.asarray(state.achievements[_ACHIEVEMENTS[field].value]))
    if belief is not None and field.startswith("world."):
        believed = belief.world_fact(field)
        if believed is not None:
            return int(believed)
    if field == "world.placement_space_reachable":
        return int(bool(np.any(np.asarray(walkable_mask(state)))))
    if field in _WORLD_BLOCKS:
        return int(bool(np.any(np.asarray(state.map) == _WORLD_BLOCKS[field].value)))
    if field == "world.cow_reachable":
        return int(bool(np.any(np.asarray(state.cows.mask))))
    raise ValueError(f"No symbolic reading for registry field {field!r}.")


def _holds(requirement: StateRequirement, values: dict[str, int]) -> bool:
    value = values[requirement.field]
    target = int(requirement.value)
    if requirement.comparison == "at_least":
        return value >= target
    if requirement.comparison == "less_than":
        return value < target
    return value == target


def _apply(effect: StateEffect, values: dict[str, int]) -> None:
    amount = int(effect.value)
    if effect.operation == "increase":
        values[effect.field] += amount
    elif effect.operation == "decrease":
        values[effect.field] -= amount
    else:
        values[effect.field] = amount


@lru_cache(maxsize=1024)
def _planning_problem(
    goal: GoalCondition,
    model: SkillModel,
) -> tuple[tuple[str, ...], SkillModel]:
    """Return the fields a search tracks, and ``model`` projected onto them.

    Every field the goal or a requirement reads is tracked, and so is every
    other effect field except an achievement: one that nothing reads only
    tells apart states that agree on everything else, so tracking it would
    grow the search without changing a plan. Effects on untracked fields are
    dropped from the projected model.
    """

    read = {requirement.field for requirement in goal}
    for _, mechanics in model:
        read.update(requirement.field for requirement in mechanics.requires)
    fields = set(read)
    for _, mechanics in model:
        fields.update(
            effect.field
            for effect in mechanics.effects
            if effect.field in read or effect.field not in _ACHIEVEMENTS
        )
    projected = tuple(
        (
            skill_id,
            dataclasses.replace(
                mechanics,
                effects=tuple(
                    effect for effect in mechanics.effects if effect.field in fields
                ),
            ),
        )
        for skill_id, mechanics in model
    )
    return tuple(sorted(fields)), projected


@lru_cache(maxsize=65536)
def _first_steps(
    start: _Values,
    goal: GoalCondition,
    model: SkillModel,
    max_plan_length: int,
) -> tuple[frozenset[SkillId], int] | None:
    """Breadth-first search from ``start``, tracking each state's first skills.

    A state reached at its shortest depth along several paths keeps the first
    skill of each, so the goal states of the shallowest layer hold every first
    step of every shortest plan.
    """

    if all(_holds(requirement, dict(start)) for requirement in goal):
        return frozenset(), 0
    seen = {start}
    layer: dict[_Values, frozenset[SkillId]] = {start: frozenset()}
    for depth in range(1, max_plan_length + 1):
        following: dict[_Values, frozenset[SkillId]] = {}
        for node, firsts in layer.items():
            values = dict(node)
            for skill_id, mechanics in model:
                if not all(
                    _holds(requirement, values) for requirement in mechanics.requires
                ):
                    continue
                successor = dict(values)
                for effect in mechanics.effects:
                    _apply(effect, successor)
                key = tuple(sorted(successor.items()))
                if key in seen:
                    continue
                reached = firsts or frozenset((skill_id,))
                following[key] = following.get(key, frozenset()) | reached
        reached_goal = [
            firsts
            for node, firsts in following.items()
            if all(_holds(requirement, dict(node)) for requirement in goal)
        ]
        if reached_goal:
            return frozenset[SkillId]().union(*reached_goal), depth
        if not following:
            return None
        seen.update(following)
        layer = following
    return None


def _restrict(
    model: SkillModel | None,
    candidate_skills: Sequence[SkillId],
) -> SkillModel:
    if model is None:
        return registry_model(tuple(candidate_skills))
    allowed = set(candidate_skills)
    return tuple(entry for entry in model if entry[0] in allowed)


def shortest_plan(
    state: Any,
    goal: GoalCondition,
    candidate_skills: Sequence[SkillId] = STONE_PICKAXE_SKILL_IDS,
    *,
    model: SkillModel | None = None,
    belief: WorldBelief | None = None,
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> tuple[frozenset[SkillId], int] | None:
    """Return the first steps of every shortest plan to ``goal``, and its length.

    Empty first steps and length zero mean the goal already holds. ``None``
    means no plan of at most ``max_plan_length`` skills reaches it -- for
    instance when wood is short and no tree is left.

    ``model`` defaults to the registry's mechanics for ``candidate_skills``;
    a given model is restricted to the skills in ``candidate_skills``.
    """

    fields, planned = _planning_problem(goal, _restrict(model, candidate_skills))
    start = tuple((field, read_field(state, field, belief)) for field in fields)
    return _first_steps(start, goal, planned, max_plan_length)


def plan_next_skills(
    state: Any,
    goal: GoalCondition,
    candidate_skills: Sequence[SkillId] = STONE_PICKAXE_SKILL_IDS,
    *,
    model: SkillModel | None = None,
    belief: WorldBelief | None = None,
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> frozenset[SkillId] | None:
    """Return the skills that start some shortest plan to ``goal``.

    A craft of a tool already held, a resource beyond what the remaining
    recipes consume, and a skill no recipe needs are never on a shortest plan,
    so none of them is returned.
    """

    plan = shortest_plan(
        state,
        goal,
        candidate_skills,
        model=model,
        belief=belief,
        max_plan_length=max_plan_length,
    )
    return None if plan is None else plan[0]


@dataclass(frozen=True)
class SubgoalCheck:
    """Whether a proposed sub-goal can be reached, and if not, why.

    ``reason`` is empty when the sub-goal is reachable, and otherwise a short
    explanation meant to be shown to the planner that proposed it.
    """

    reachable: bool
    already_holds: bool
    first_steps: frozenset[SkillId]
    plan_length: int | None
    reason: str


def _rejected(reason: str) -> SubgoalCheck:
    return SubgoalCheck(
        reachable=False,
        already_holds=False,
        first_steps=frozenset(),
        plan_length=None,
        reason=reason,
    )


def _requirement_text(requirement: StateRequirement) -> str:
    comparison = _COMPARISONS[requirement.comparison]
    return f"{requirement.field} {comparison} {int(requirement.value)}"


def _invalid_requirement(requirement: Any) -> str | None:
    field = getattr(requirement, "field", None)
    if not isinstance(field, str) or field not in PLANNER_FIELDS:
        return f"unknown field {field}"
    comparison = getattr(requirement, "comparison", None)
    if comparison not in _COMPARISONS:
        return f"{field}: unknown comparison {comparison}"
    value = getattr(requirement, "value", None)
    integral = isinstance(value, int) or (
        isinstance(value, float) and value.is_integer()
    )
    if not integral:
        return f"{field}: value {value!r} is not an integer"
    return None


def _moves_toward(
    effect: StateEffect,
    requirement: StateRequirement,
    value: int,
) -> bool:
    if effect.field != requirement.field:
        return False
    if effect.operation == "set":
        return _holds(requirement, {requirement.field: int(effect.value)})
    if effect.operation == "increase":
        return value < int(requirement.value)
    return value >= int(requirement.value)


def _blocked_reason(
    requirement: StateRequirement,
    values: dict[str, int],
    model: SkillModel,
) -> str:
    """Explain one unmet requirement from the skills that could change it."""

    field = requirement.field
    value = values[field]
    producers = [
        (skill_id, mechanics)
        for skill_id, mechanics in model
        if any(
            _moves_toward(effect, requirement, value) for effect in mechanics.effects
        )
    ]
    if requirement.comparison == "at_least":
        need = f"{field} needs {int(requirement.value) - value} more"
    else:
        need = f"{field} is {value}, needs {_requirement_text(requirement)}"
    if not producers:
        change = "produces" if requirement.comparison == "at_least" else "changes"
        return f"{need}: no available skill {change} it"
    for skill_id, mechanics in producers:
        for unmet in mechanics.requires:
            if unmet.field.startswith("world.") and not _holds(unmet, values):
                return f"{need} but {unmet.field} is false for {skill_id.value}"
    skill_id, mechanics = producers[0]
    unmet = [
        _requirement_text(item)
        for item in mechanics.requires
        if not _holds(item, values)
    ]
    if unmet:
        return f"{need}; {skill_id.value} needs {', '.join(unmet)}"
    return f"{need}; no plan reaches it"


def check_subgoal(
    state: Any,
    subgoal: GoalCondition,
    candidate_skills: Sequence[SkillId] = STONE_PICKAXE_SKILL_IDS,
    *,
    model: SkillModel | None = None,
    belief: WorldBelief | None = None,
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> SubgoalCheck:
    """Check a proposed sub-goal against the model from ``state``.

    The sub-goal usually comes from a VLM, so malformed requirements -- an
    unknown field, an unknown comparison, a value that is not an integer, or
    no requirement at all -- are rejected with a reason rather than raised.
    When no plan reaches it, the reason names up to three unmet requirements
    and what blocks each: no skill that changes the field, a world fact a
    producing skill needs, or the producing skill's own unmet requirements.
    """

    if not subgoal:
        return _rejected("empty sub-goal")
    for requirement in subgoal:
        invalid = _invalid_requirement(requirement)
        if invalid is not None:
            return _rejected(invalid)
    goal = tuple(
        StateRequirement(item.field, item.comparison, int(item.value))
        for item in subgoal
    )
    try:
        fields, planned = _planning_problem(goal, _restrict(model, candidate_skills))
        values = {field: read_field(state, field, belief) for field in fields}
    except (AttributeError, ValueError) as error:
        return _rejected(f"model error: {error}")
    plan = _first_steps(
        tuple(sorted(values.items())),
        goal,
        planned,
        max_plan_length,
    )
    if plan is not None:
        first_steps, length = plan
        return SubgoalCheck(
            reachable=True,
            already_holds=length == 0,
            first_steps=first_steps,
            plan_length=length,
            reason="",
        )
    unmet = [requirement for requirement in goal if not _holds(requirement, values)]
    reasons = [_blocked_reason(item, values, planned) for item in unmet[:3]]
    if not reasons:
        reasons = [f"no plan of at most {max_plan_length} skills reaches it"]
    return _rejected("; ".join(reasons))


@dataclass(frozen=True)
class PlanProgressFeasibility:
    """Veto skills that start no shortest plan to the goal, and defer the rest.

    The generalization of :class:`SkipHeldCraftFeasibility`: rather than one
    rule per waste the diagnosis finds, every skill that does not start a
    shortest plan over the registry's mechanics is scaled by ``off_plan``.
    Where no plan exists, or every skill that starts one scores zero under the
    base, the base scores are returned unchanged, so the planner is never left
    with nothing to choose.

    It reads world facts from the whole map, which the agent has not seen, so
    it is the upper bound for a planner working from a :class:`WorldBelief`,
    not a condition that planner can match on observation alone.

    The plan is searched over ``skills``, the skills the executor offers,
    whichever candidates are being scored, so scoring one skill alone gives
    the same value as scoring it among the catalog.

    A hard veto cannot recover from a step the executor keeps failing, since
    the failure penalty only reorders the skills on the plan; a positive
    ``off_plan`` lets a repeatedly failed step fall below an alternative.
    """

    goal: GoalCondition
    base: FeasibilityScorer | BatchedFeasibilityScorer = reachability_feasibility
    off_plan: float = 0.0
    skills: tuple[SkillId, ...] = STONE_PICKAXE_SKILL_IDS
    model: SkillModel | None = None

    def score_all_feasibility(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        plan = plan_next_skills(state, self.goal, self.skills, model=self.model)
        if not plan:
            return catalog_feasibility(self.base, state, context, candidate_skills)
        scored = (*candidate_skills, *sorted(plan - set(candidate_skills)))
        scores = catalog_feasibility(self.base, state, context, scored)
        if all(scores[skill_id] <= 0.0 for skill_id in plan):
            return {skill_id: scores[skill_id] for skill_id in candidate_skills}
        return {
            skill_id: scores[skill_id] * (1.0 if skill_id in plan else self.off_plan)
            for skill_id in candidate_skills
        }

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return self.score_all_feasibility(state, context, (skill_id,))[skill_id]
