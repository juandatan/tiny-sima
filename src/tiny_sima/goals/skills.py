"""Typed Craftax-Classic skills and state-based completion predicates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Action, BlockType


class SkillId(str, Enum):
    """Stable identifiers shared by the planner, executor, and evaluations."""

    COLLECT_WOOD = "collect_wood"
    COLLECT_STONE = "collect_stone"
    DRINK_WATER = "drink_water"
    PLACE_TABLE = "place_table"
    CRAFT_WOOD_PICKAXE = "craft_wood_pickaxe"
    CRAFT_WOOD_SWORD = "craft_wood_sword"


RequirementComparison = Literal["at_least", "less_than", "equals"]
EffectOperation = Literal["increase", "decrease", "set"]
StateValue = int | bool


@dataclass(frozen=True)
class StateRequirement:
    """One typed precondition expressed over planner-visible state."""

    field: str
    comparison: RequirementComparison
    value: StateValue

    def to_dict(self) -> dict[str, StateValue | str]:
        """Return a JSON-friendly representation."""

        return {
            "field": self.field,
            "comparison": self.comparison,
            "value": self.value,
        }


@dataclass(frozen=True)
class StateEffect:
    """One expected state change after successful skill execution."""

    field: str
    operation: EffectOperation
    value: StateValue

    def to_dict(self) -> dict[str, StateValue | str]:
        """Return a JSON-friendly representation."""

        return {
            "field": self.field,
            "operation": self.operation,
            "value": self.value,
        }


@dataclass(frozen=True)
class SkillMechanics:
    """Static game knowledge supplied to the high-level planner."""

    requires: tuple[StateRequirement, ...]
    effects: tuple[StateEffect, ...]
    repeatable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "requires": [
                requirement.to_dict() for requirement in self.requires
            ],
            "effects": [effect.to_dict() for effect in self.effects],
            "repeatable": self.repeatable,
        }


@dataclass(frozen=True)
class SkillSpec:
    """Static metadata for one skill exposed to an agent."""

    skill_id: SkillId
    instruction: str
    description: str
    mechanics: SkillMechanics
    default_horizon: int
    interaction_action: Action


@dataclass(frozen=True, init=False)
class SkillGoal:
    """One concrete short-horizon objective."""

    skill_id: SkillId
    quantity: int = 1
    horizon: int | None = None

    def __init__(
        self,
        skill_id: SkillId | str,
        quantity: int = 1,
        horizon: int | None = None,
    ) -> None:
        normalized_skill_id = SkillId(skill_id)
        if quantity < 1:
            raise ValueError("Skill quantity must be positive.")
        if horizon is not None and horizon < 1:
            raise ValueError("Skill horizon must be positive.")
        object.__setattr__(self, "skill_id", normalized_skill_id)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "horizon", horizon)

    @property
    def effective_horizon(self) -> int:
        return self.horizon or get_skill_spec(self.skill_id).default_horizon


SKILL_REGISTRY: dict[SkillId, SkillSpec] = {
    SkillId.COLLECT_WOOD: SkillSpec(
        skill_id=SkillId.COLLECT_WOOD,
        instruction="Collect wood from a tree.",
        description="Increase the wood inventory by interacting with trees.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.tree_available", "equals", True),
            ),
            effects=(
                StateEffect("inventory.wood", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.DO,
    ),
    SkillId.COLLECT_STONE: SkillSpec(
        skill_id=SkillId.COLLECT_STONE,
        instruction="Collect stone with a wood pickaxe.",
        description="Mine stone using a wooden pickaxe.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement(
                    "inventory.wood_pickaxe",
                    "at_least",
                    1,
                ),
                StateRequirement("world.stone_available", "equals", True),
            ),
            effects=(
                StateEffect("inventory.stone", "increase", 1),
            ),
        ),
        default_horizon=96,
        interaction_action=Action.DO,
    ),
    SkillId.DRINK_WATER: SkillSpec(
        skill_id=SkillId.DRINK_WATER,
        instruction="Drink water.",
        description="Increase the player's drink level at a water tile.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.water_available", "equals", True),
                StateRequirement("survival.drink", "less_than", 9),
            ),
            effects=(
                StateEffect("survival.drink", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.DO,
    ),
    SkillId.PLACE_TABLE: SkillSpec(
        skill_id=SkillId.PLACE_TABLE,
        instruction="Place a crafting table.",
        description="Spend two wood to place a crafting table.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood", "at_least", 2),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 2),
                StateEffect(
                    "world.crafting_table_present",
                    "set",
                    True,
                ),
            ),
        ),
        default_horizon=48,
        interaction_action=Action.PLACE_TABLE,
    ),
    SkillId.CRAFT_WOOD_PICKAXE: SkillSpec(
        skill_id=SkillId.CRAFT_WOOD_PICKAXE,
        instruction="Craft a wood pickaxe at a crafting table.",
        description="Spend one wood at a crafting table to make a wood pickaxe.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood", "at_least", 1),
                StateRequirement(
                    "world.crafting_table_present",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 1),
                StateEffect(
                    "inventory.wood_pickaxe",
                    "increase",
                    1,
                ),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.MAKE_WOOD_PICKAXE,
    ),
    SkillId.CRAFT_WOOD_SWORD: SkillSpec(
        skill_id=SkillId.CRAFT_WOOD_SWORD,
        instruction="Craft a wood sword at a crafting table.",
        description="Spend one wood at a crafting table to make a wood sword.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood", "at_least", 1),
                StateRequirement(
                    "world.crafting_table_present",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 1),
                StateEffect("inventory.wood_sword", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.MAKE_WOOD_SWORD,
    ),
}

SKILL_IDS: tuple[SkillId, ...] = tuple(SKILL_REGISTRY)
SKILL_TO_INDEX: dict[SkillId, int] = {
    skill_id: index for index, skill_id in enumerate(SKILL_IDS)
}


def get_skill_spec(skill_id: SkillId | str) -> SkillSpec:
    """Return metadata for a known skill."""

    return SKILL_REGISTRY[SkillId(skill_id)]


def skill_index(skill_id: SkillId | str) -> int:
    """Return the stable categorical index used by the executor."""

    return SKILL_TO_INDEX[SkillId(skill_id)]


def skill_id_from_index(index: int) -> SkillId:
    """Return the skill associated with a categorical executor index."""

    try:
        return SKILL_IDS[index]
    except IndexError as error:
        raise ValueError(f"Unknown skill index: {index}") from error


def skill_catalog() -> list[dict[str, Any]]:
    """Return JSON-friendly metadata suitable for prompts and notebooks."""

    return [
        {
            "skill_id": spec.skill_id.value,
            "skill_index": skill_index(spec.skill_id),
            "instruction": spec.instruction,
            "description": spec.description,
            "mechanics": spec.mechanics.to_dict(),
            "default_horizon": spec.default_horizon,
            "interaction_action": spec.interaction_action.value,
        }
        for spec in SKILL_REGISTRY.values()
    ]


def planner_skill_catalog() -> list[dict[str, Any]]:
    """Return only the static semantics needed for high-level planning."""

    return [
        {
            "skill": spec.skill_id.value,
            "description": spec.description,
            **spec.mechanics.to_dict(),
        }
        for spec in SKILL_REGISTRY.values()
    ]


def _block_count(state: Any, block_type: BlockType) -> jax.Array:
    return jnp.count_nonzero(state.map == block_type.value)


def skill_metric(skill_id: SkillId | str, state: Any) -> jax.Array:
    """Measure cumulative progress using repeatable state, not achievements."""

    skill_id = SkillId(skill_id)
    if skill_id is SkillId.COLLECT_WOOD:
        return jnp.asarray(state.inventory.wood)
    if skill_id is SkillId.COLLECT_STONE:
        return jnp.asarray(state.inventory.stone)
    if skill_id is SkillId.DRINK_WATER:
        return jnp.asarray(state.player_drink)
    if skill_id is SkillId.PLACE_TABLE:
        return _block_count(state, BlockType.CRAFTING_TABLE)
    if skill_id is SkillId.CRAFT_WOOD_PICKAXE:
        return jnp.asarray(state.inventory.wood_pickaxe)
    if skill_id is SkillId.CRAFT_WOOD_SWORD:
        return jnp.asarray(state.inventory.wood_sword)
    raise AssertionError(f"Unhandled skill: {skill_id}")


def skill_precondition(goal: SkillGoal, state: Any) -> jax.Array:
    """Return whether a goal is feasible from the supplied world state."""

    alive = jnp.asarray(state.player_health) > 0
    skill_id = goal.skill_id
    if skill_id is SkillId.COLLECT_WOOD:
        return alive & jnp.any(state.map == BlockType.TREE.value)
    if skill_id is SkillId.COLLECT_STONE:
        return (
            alive
            & (jnp.asarray(state.inventory.wood_pickaxe) > 0)
            & jnp.any(state.map == BlockType.STONE.value)
        )
    if skill_id is SkillId.DRINK_WATER:
        return (
            alive
            & (jnp.asarray(state.player_drink) + goal.quantity <= 9)
            & jnp.any(state.map == BlockType.WATER.value)
        )
    if skill_id is SkillId.PLACE_TABLE:
        return alive & (jnp.asarray(state.inventory.wood) >= 2 * goal.quantity)
    if skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
    }:
        return (
            alive
            & (jnp.asarray(state.inventory.wood) >= goal.quantity)
            & jnp.any(state.map == BlockType.CRAFTING_TABLE.value)
        )
    raise AssertionError(f"Unhandled skill: {skill_id}")


def skill_progress(
    goal: SkillGoal,
    initial_state: Any,
    current_state: Any,
) -> jax.Array:
    """Return normalized progress in the range zero to one."""

    initial = skill_metric(goal.skill_id, initial_state)
    return skill_progress_from_metric(goal, initial, current_state)


def skill_progress_from_metric(
    goal: SkillGoal,
    initial_metric: jax.Array,
    current_state: Any,
) -> jax.Array:
    """Return normalized progress from a stored initial metric."""

    current = skill_metric(goal.skill_id, current_state)
    gained = jnp.maximum(current - initial_metric, 0)
    return jnp.clip(gained / goal.quantity, 0.0, 1.0).astype(jnp.float32)


def skill_success(
    goal: SkillGoal,
    initial_state: Any,
    current_state: Any,
) -> jax.Array:
    """Return whether the requested state change has occurred."""

    initial = skill_metric(goal.skill_id, initial_state)
    return skill_success_from_metric(goal, initial, current_state)


def skill_success_from_metric(
    goal: SkillGoal,
    initial_metric: jax.Array,
    current_state: Any,
) -> jax.Array:
    """Return success from a stored initial metric."""

    current = skill_metric(goal.skill_id, current_state)
    return current - initial_metric >= goal.quantity


def sparse_skill_reward(
    goal: SkillGoal,
    initial_state: Any,
    previous_state: Any,
    current_state: Any,
) -> jax.Array:
    """Pay one only on the transition that first completes a skill."""

    was_complete = skill_success(goal, initial_state, previous_state)
    is_complete = skill_success(goal, initial_state, current_state)
    return (is_complete & ~was_complete).astype(jnp.float32)
