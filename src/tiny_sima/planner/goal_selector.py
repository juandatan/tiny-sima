"""A scripted goal source for the sub-goal planner on the open-ended task.

The selector stands where the VLM stands in :class:`SubgoalPlanner`, so the
comparison isolates what the VLM adds over a fixed curriculum: the same
planner checks the same sub-goals, follows them with the same symbolic
search and re-asks on the same triggers. It answers in the VLM's JSON format.

Its rule is short. A vital at or below its threshold, when a skill restores
it, comes first. Otherwise it asks for the next locked achievement, in a
fixed tech-tree order, that the symbolic planner can reach. An achievement
that has timed out, been abandoned or been rejected ``max_failures`` times is
skipped for the rest of the episode, and one rejected earlier in the same
decision is skipped at once, so a re-ask moves on to the next candidate.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from craftax.craftax_classic.constants import Achievement

from tiny_sima.goals import ALL_SKILL_IDS, SkillId, StateRequirement
from tiny_sima.planner.observation import vitals
from tiny_sima.planner.subgoals import SubgoalRequest, render_subgoal
from tiny_sima.planner.symbolic import (
    DEFAULT_MAX_PLAN_LENGTH,
    GoalCondition,
    SkillModel,
    WorldBelief,
    check_subgoal,
)

# Every Craftax-Classic achievement once: the tool line first, from wood to
# diamond, then the side achievements, cheapest first. Only the order among
# reachable achievements matters, since the rest are skipped.
TECH_TREE_ORDER: tuple[Achievement, ...] = (
    Achievement.COLLECT_WOOD,
    Achievement.PLACE_TABLE,
    Achievement.MAKE_WOOD_PICKAXE,
    Achievement.MAKE_WOOD_SWORD,
    Achievement.COLLECT_STONE,
    Achievement.MAKE_STONE_PICKAXE,
    Achievement.MAKE_STONE_SWORD,
    Achievement.PLACE_STONE,
    Achievement.PLACE_FURNACE,
    Achievement.COLLECT_COAL,
    Achievement.COLLECT_IRON,
    Achievement.MAKE_IRON_PICKAXE,
    Achievement.MAKE_IRON_SWORD,
    Achievement.COLLECT_DIAMOND,
    Achievement.COLLECT_DRINK,
    Achievement.COLLECT_SAPLING,
    Achievement.PLACE_PLANT,
    Achievement.EAT_PLANT,
    Achievement.EAT_COW,
    Achievement.WAKE_UP,
    Achievement.DEFEAT_ZOMBIE,
    Achievement.DEFEAT_SKELETON,
)

# The planner field that restores each vital, or None while no skill does.
# Food and energy are hooks: an eating or a sleeping skill fills one in, and
# a matching rule in ``vital_rules`` turns it on.
VITAL_SUBGOAL_FIELDS: dict[str, str | None] = {
    "drink": "survival.drink",
    "food": None,
    "energy": None,
    "health": None,
}
# (vital, threshold, target): at or below the threshold, ask for the target.
DEFAULT_VITAL_RULES: tuple[tuple[str, int, int], ...] = (("drink", 3, 7),)
# Plan events that count against a sub-goal.
DEFAULT_FAILURE_STATUSES: tuple[str, ...] = ("timed_out", "abandoned", "rejected")
DEFAULT_MAX_FAILURES = 2
# Holds in every state, so the planner always turns it down.
NOTHING_TO_DO: GoalCondition = (StateRequirement("inventory.wood", "at_least", 0),)


def achievement_subgoal(achievement: Achievement) -> GoalCondition:
    """Return the sub-goal that holds once ``achievement`` is unlocked."""

    field = f"achievement.{achievement.name.lower()}"
    return (StateRequirement(field, "at_least", 1),)


def _answer(subgoal: GoalCondition, assessment: str) -> str:
    return json.dumps(
        {
            "assessment": assessment,
            "subgoals": [
                {"field": requirement.field, "at_least": int(requirement.value)}
                for requirement in subgoal
            ],
        }
    )


@dataclass(frozen=True)
class ScriptedGoalSelector:
    """Propose one sub-goal per request from a fixed curriculum.

    ``candidate_skills`` and ``model`` should be the planner's, so that the
    selector only proposes what the planner will accept. With
    ``use_memory_belief`` the world facts come from the episode memory when
    it has any, as in the planner.

    When no candidate is left it answers with :data:`NOTHING_TO_DO`, which
    always holds, so the planner rejects it as it would a VLM that has run
    out of ideas: the episode keeps a plan that is still reachable, and one
    with nothing left to do stops.
    """

    candidate_skills: tuple[SkillId, ...] = ALL_SKILL_IDS
    model: SkillModel | None = None
    use_memory_belief: bool = True
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH
    order: tuple[Achievement, ...] = TECH_TREE_ORDER
    vital_rules: tuple[tuple[str, int, int], ...] = DEFAULT_VITAL_RULES
    max_failures: int = DEFAULT_MAX_FAILURES
    failure_statuses: tuple[str, ...] = DEFAULT_FAILURE_STATUSES

    def candidates(self, request: SubgoalRequest) -> list[GoalCondition]:
        """Return the sub-goals worth proposing, in priority order."""

        readings = vitals(request.state)
        candidates: list[GoalCondition] = []
        for vital, threshold, target in self.vital_rules:
            field = VITAL_SUBGOAL_FIELDS.get(vital)
            if field is not None and readings[vital] <= threshold:
                candidates.append((StateRequirement(field, "at_least", target),))
        candidates.extend(
            achievement_subgoal(achievement)
            for achievement in self.order
            if not bool(request.state.achievements[achievement.value])
        )
        return candidates

    def select(self, request: SubgoalRequest) -> str:
        """Return the JSON answer for one request."""

        plan = request.plan
        rejected_now = {
            event.subgoal
            for event in plan.events
            if event.status == "rejected"
            and event.decision_index == plan.decision_index
        }
        failures = Counter(
            event.subgoal
            for event in plan.events
            if event.status in self.failure_statuses
        )
        memory = request.context.memory
        belief: WorldBelief | None = None
        if self.use_memory_belief and hasattr(memory, "world_fact"):
            belief = cast(WorldBelief, memory)
        for subgoal in self.candidates(request):
            text = render_subgoal(subgoal)
            if text in rejected_now or failures[text] >= self.max_failures:
                continue
            check = check_subgoal(
                request.state,
                subgoal,
                self.candidate_skills,
                model=self.model,
                belief=belief,
                max_plan_length=self.max_plan_length,
            )
            if check.reachable and not check.already_holds:
                return _answer(subgoal, f"next: {text}")
        return _answer(NOTHING_TO_DO, "nothing left to reach")

    def propose_subgoals(self, requests: Sequence[SubgoalRequest]) -> list[str]:
        return [self.select(request) for request in requests]
