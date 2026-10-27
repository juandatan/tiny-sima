"""Plan over the skill registry's declared mechanics.

Every skill already declares what it requires and what it changes, as typed
conditions over planner-visible state. That is a STRIPS-style model, so the
questions the VLM kept getting wrong -- how much wood a recipe still needs,
whether a tool is already held, which step comes next -- have exact answers a
breadth-first search can read off it. This module answers them from the
registry and a formal goal condition alone, never from a chain's accepted set.

World facts are read as whether such a target exists anywhere on the map, the
same test the chains' accepted sets use: how far it is belongs to the Can term
the plan is combined with, and the collect skills search for what is out of
view.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from craftax.craftax_classic.constants import BlockType

from tiny_sima.goals import (
    ALL_SKILL_IDS,
    SkillId,
    StateEffect,
    StateRequirement,
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

# Plans on the recorded chains are at most eight skills long. The bound only
# stops the search when the goal cannot be reached at all.
DEFAULT_MAX_PLAN_LENGTH = 16

_WORLD_BLOCKS = {
    "world.tree_reachable": BlockType.TREE,
    "world.stone_reachable": BlockType.STONE,
    "world.water_reachable": BlockType.WATER,
    "world.crafting_table_reachable": BlockType.CRAFTING_TABLE,
}

# A planning state: each field the plan reads, sorted by name, with its value.
_Values = tuple[tuple[str, int], ...]


def read_field(state: Any, field: str) -> int:
    """Return one registry field of a world state as an integer.

    Booleans are read as zero or one, so every field compares and updates the
    same way during the search.
    """

    if field.startswith("inventory."):
        name = field.removeprefix("inventory.")
        return int(np.asarray(getattr(state.inventory, name)))
    if field == "survival.drink":
        return int(np.asarray(state.player_drink))
    if field == "world.placement_space_reachable":
        return int(bool(np.any(np.asarray(walkable_mask(state)))))
    if field in _WORLD_BLOCKS:
        return int(bool(np.any(np.asarray(state.map) == _WORLD_BLOCKS[field].value)))
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


def _plan_fields(
    goal: GoalCondition,
    candidate_skills: tuple[SkillId, ...],
) -> tuple[str, ...]:
    fields = {requirement.field for requirement in goal}
    for skill_id in candidate_skills:
        mechanics = get_skill_spec(skill_id).mechanics
        fields.update(requirement.field for requirement in mechanics.requires)
        fields.update(effect.field for effect in mechanics.effects)
    return tuple(sorted(fields))


@lru_cache(maxsize=65536)
def _first_steps(
    start: _Values,
    goal: GoalCondition,
    candidate_skills: tuple[SkillId, ...],
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
            for skill_id in candidate_skills:
                mechanics = get_skill_spec(skill_id).mechanics
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


def shortest_plan(
    state: Any,
    goal: GoalCondition,
    candidate_skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> tuple[frozenset[SkillId], int] | None:
    """Return the first steps of every shortest plan to ``goal``, and its length.

    Empty first steps and length zero mean the goal already holds. ``None``
    means no plan of at most ``max_plan_length`` skills reaches it -- for
    instance when wood is short and no tree is left.
    """

    candidates = tuple(candidate_skills)
    fields = _plan_fields(goal, candidates)
    start = tuple((field, read_field(state, field)) for field in fields)
    return _first_steps(start, goal, candidates, max_plan_length)


def plan_next_skills(
    state: Any,
    goal: GoalCondition,
    candidate_skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
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
        max_plan_length=max_plan_length,
    )
    return None if plan is None else plan[0]


@dataclass(frozen=True)
class PlanProgressFeasibility:
    """Veto skills that start no shortest plan to the goal, and defer the rest.

    The generalization of :class:`SkipHeldCraftFeasibility`: rather than one
    rule per waste the diagnosis finds, every skill that does not start a
    shortest plan over the registry's mechanics is scaled by ``off_plan``.
    Where no plan exists, or every skill that starts one scores zero under the
    base, the base scores are returned unchanged, so the planner is never left
    with nothing to choose.

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
    skills: tuple[SkillId, ...] = ALL_SKILL_IDS

    def score_all_feasibility(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        plan = plan_next_skills(state, self.goal, self.skills)
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
