"""Typed Craftax-Classic skills and state-based completion predicates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Achievement, Action, BlockType


class SkillId(str, Enum):
    """Stable identifiers shared by the planner, executor, and evaluations."""

    COLLECT_WOOD = "collect_wood"
    COLLECT_STONE = "collect_stone"
    DRINK_WATER = "drink_water"
    PLACE_TABLE = "place_table"
    CRAFT_WOOD_PICKAXE = "craft_wood_pickaxe"
    CRAFT_WOOD_SWORD = "craft_wood_sword"
    # Appended last so the six indices above keep their meaning in existing
    # checkpoints, priors and recorded results.
    CRAFT_STONE_PICKAXE = "craft_stone_pickaxe"
    # Stages A and B of the full achievement set, appended in turn so every
    # earlier index keeps its meaning.
    CRAFT_STONE_SWORD = "craft_stone_sword"
    COLLECT_COAL = "collect_coal"
    PLACE_STONE = "place_stone"
    PLACE_FURNACE = "place_furnace"
    COLLECT_SAPLING = "collect_sapling"
    PLACE_PLANT = "place_plant"
    EAT_COW = "eat_cow"
    SLEEP = "sleep"
    EAT_PLANT = "eat_plant"


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


# Health, food, drink and energy all top out here.
MAX_VITAL = 9


SKILL_REGISTRY: dict[SkillId, SkillSpec] = {
    SkillId.COLLECT_WOOD: SkillSpec(
        skill_id=SkillId.COLLECT_WOOD,
        instruction="Collect wood from a tree.",
        description="Increase the wood inventory by interacting with trees.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.tree_reachable", "equals", True),
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
                StateRequirement("world.stone_reachable", "equals", True),
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
                StateRequirement("world.water_reachable", "equals", True),
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
                StateRequirement(
                    "world.placement_space_reachable",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 2),
                StateEffect(
                    "world.crafting_table_reachable",
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
                    "world.crafting_table_reachable",
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
                    "world.crafting_table_reachable",
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
    SkillId.CRAFT_STONE_PICKAXE: SkillSpec(
        skill_id=SkillId.CRAFT_STONE_PICKAXE,
        instruction="Craft a stone pickaxe at a crafting table.",
        description=(
            "Spend one wood and one stone at a crafting table to make a "
            "stone pickaxe."
        ),
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood", "at_least", 1),
                StateRequirement("inventory.stone", "at_least", 1),
                StateRequirement(
                    "world.crafting_table_reachable",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 1),
                StateEffect("inventory.stone", "decrease", 1),
                StateEffect("inventory.stone_pickaxe", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.MAKE_STONE_PICKAXE,
    ),
    SkillId.CRAFT_STONE_SWORD: SkillSpec(
        skill_id=SkillId.CRAFT_STONE_SWORD,
        instruction="Craft a stone sword at a crafting table.",
        description=(
            "Spend one wood and one stone at a crafting table to make a "
            "stone sword."
        ),
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood", "at_least", 1),
                StateRequirement("inventory.stone", "at_least", 1),
                StateRequirement(
                    "world.crafting_table_reachable",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.wood", "decrease", 1),
                StateEffect("inventory.stone", "decrease", 1),
                StateEffect("inventory.stone_sword", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.MAKE_STONE_SWORD,
    ),
    SkillId.COLLECT_COAL: SkillSpec(
        skill_id=SkillId.COLLECT_COAL,
        instruction="Collect coal with a wood pickaxe.",
        description="Mine coal using a wooden pickaxe.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.wood_pickaxe", "at_least", 1),
                StateRequirement("world.coal_reachable", "equals", True),
            ),
            effects=(
                StateEffect("inventory.coal", "increase", 1),
            ),
        ),
        default_horizon=96,
        interaction_action=Action.DO,
    ),
    SkillId.PLACE_STONE: SkillSpec(
        skill_id=SkillId.PLACE_STONE,
        instruction="Place a stone.",
        description="Spend one stone to place a stone block.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.stone", "at_least", 1),
                StateRequirement(
                    "world.placement_space_reachable",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.stone", "decrease", 1),
            ),
        ),
        default_horizon=48,
        interaction_action=Action.PLACE_STONE,
    ),
    SkillId.PLACE_FURNACE: SkillSpec(
        skill_id=SkillId.PLACE_FURNACE,
        instruction="Place a furnace.",
        description="Spend one stone to place a furnace.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.stone", "at_least", 1),
                StateRequirement(
                    "world.placement_space_reachable",
                    "equals",
                    True,
                ),
            ),
            effects=(
                StateEffect("inventory.stone", "decrease", 1),
                StateEffect("world.furnace_reachable", "set", True),
            ),
        ),
        default_horizon=48,
        interaction_action=Action.PLACE_FURNACE,
    ),
    SkillId.COLLECT_SAPLING: SkillSpec(
        skill_id=SkillId.COLLECT_SAPLING,
        instruction="Collect a sapling from grass.",
        description=(
            "Interact with grass until it yields a sapling, which it does "
            "one time in ten."
        ),
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.grass_reachable", "equals", True),
            ),
            effects=(
                StateEffect("inventory.sapling", "increase", 1),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.DO,
    ),
    SkillId.PLACE_PLANT: SkillSpec(
        skill_id=SkillId.PLACE_PLANT,
        instruction="Plant a sapling on grass.",
        description="Spend one sapling to plant it on a grass tile.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("inventory.sapling", "at_least", 1),
                StateRequirement("world.grass_reachable", "equals", True),
            ),
            effects=(
                StateEffect("inventory.sapling", "decrease", 1),
            ),
        ),
        default_horizon=48,
        interaction_action=Action.PLACE_PLANT,
    ),
    SkillId.EAT_COW: SkillSpec(
        skill_id=SkillId.EAT_COW,
        instruction="Hunt and eat a cow.",
        description="Hit a cow until it dies, which restores food.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.cow_reachable", "equals", True),
                StateRequirement("survival.food", "less_than", MAX_VITAL),
            ),
            # Craftax adds six food, capped at the maximum.
            effects=(
                StateEffect("survival.food", "increase", 6),
            ),
        ),
        default_horizon=96,
        interaction_action=Action.DO,
    ),
    SkillId.SLEEP: SkillSpec(
        skill_id=SkillId.SLEEP,
        instruction="Sleep until rested.",
        description=(
            "Sleep until energy is full and wake up; each missing energy "
            "point takes about eleven steps."
        ),
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("survival.energy", "less_than", MAX_VITAL),
            ),
            effects=(
                StateEffect("survival.energy", "set", MAX_VITAL),
            ),
        ),
        default_horizon=128,
        interaction_action=Action.SLEEP,
    ),
    SkillId.EAT_PLANT: SkillSpec(
        skill_id=SkillId.EAT_PLANT,
        instruction="Eat a ripe plant.",
        description="Eat a ripe plant, which restores food.",
        mechanics=SkillMechanics(
            requires=(
                StateRequirement("world.ripe_plant_reachable", "equals", True),
                StateRequirement("survival.food", "less_than", MAX_VITAL),
            ),
            # Craftax adds four food, capped at the maximum.
            effects=(
                StateEffect("survival.food", "increase", 4),
            ),
        ),
        default_horizon=64,
        interaction_action=Action.DO,
    ),
}

# Every skill the registry defines, in categorical index order.
ALL_SKILL_IDS: tuple[SkillId, ...] = tuple(SKILL_REGISTRY)
# The six-skill catalog every executor before the stone-pickaxe stage was
# trained on, and the default candidate set of the planner. Keeping it the
# default leaves those checkpoints, priors and planner prompts unchanged.
SKILL_IDS: tuple[SkillId, ...] = ALL_SKILL_IDS[:6]
# The seven-skill catalog of the stone-pickaxe stage. The chain and
# open-ended evaluations offer it, so the executors trained on it, their
# priors and the recorded results stay valid as the registry grows.
STONE_PICKAXE_SKILL_IDS: tuple[SkillId, ...] = ALL_SKILL_IDS[:7]
SKILL_TO_INDEX: dict[SkillId, int] = {
    skill_id: index for index, skill_id in enumerate(ALL_SKILL_IDS)
}

# The achievement each skill's success unlocks, read off Craftax-Classic's
# game logic: the action that changes the skill's metric is the one that sets
# the achievement. It is kept apart from the mechanics, which are rendered
# verbatim into the planner prompts; a new skill adds its entry here.
SKILL_ACHIEVEMENTS: dict[SkillId, Achievement] = {
    SkillId.COLLECT_WOOD: Achievement.COLLECT_WOOD,
    SkillId.COLLECT_STONE: Achievement.COLLECT_STONE,
    SkillId.DRINK_WATER: Achievement.COLLECT_DRINK,
    SkillId.PLACE_TABLE: Achievement.PLACE_TABLE,
    SkillId.CRAFT_WOOD_PICKAXE: Achievement.MAKE_WOOD_PICKAXE,
    SkillId.CRAFT_WOOD_SWORD: Achievement.MAKE_WOOD_SWORD,
    SkillId.CRAFT_STONE_PICKAXE: Achievement.MAKE_STONE_PICKAXE,
    SkillId.CRAFT_STONE_SWORD: Achievement.MAKE_STONE_SWORD,
    SkillId.COLLECT_COAL: Achievement.COLLECT_COAL,
    SkillId.PLACE_STONE: Achievement.PLACE_STONE,
    SkillId.PLACE_FURNACE: Achievement.PLACE_FURNACE,
    SkillId.COLLECT_SAPLING: Achievement.COLLECT_SAPLING,
    SkillId.PLACE_PLANT: Achievement.PLACE_PLANT,
    SkillId.EAT_COW: Achievement.EAT_COW,
    SkillId.SLEEP: Achievement.WAKE_UP,
    SkillId.EAT_PLANT: Achievement.EAT_PLANT,
}


def achievement_field(achievement: Achievement) -> str:
    """Return the planner field that reads whether ``achievement`` is unlocked."""

    return f"achievement.{achievement.name.lower()}"


# One planner field per Craftax-Classic achievement, in achievement order.
ACHIEVEMENT_FIELDS: tuple[str, ...] = tuple(
    achievement_field(achievement) for achievement in Achievement
)


def executor_skill_ids(num_skills: int) -> tuple[SkillId, ...]:
    """Return the skills an executor trained on ``num_skills`` indices knows."""

    if not 1 <= num_skills <= len(ALL_SKILL_IDS):
        raise ValueError(
            f"num_skills must be between 1 and {len(ALL_SKILL_IDS)}."
        )
    return ALL_SKILL_IDS[:num_skills]


def get_skill_spec(skill_id: SkillId | str) -> SkillSpec:
    """Return metadata for a known skill."""

    return SKILL_REGISTRY[SkillId(skill_id)]


def skill_index(skill_id: SkillId | str) -> int:
    """Return the stable categorical index used by the executor."""

    return SKILL_TO_INDEX[SkillId(skill_id)]


def skill_id_from_index(index: int) -> SkillId:
    """Return the skill associated with a categorical executor index."""

    try:
        return ALL_SKILL_IDS[index]
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


def planner_skill_catalog(
    skill_ids: Sequence[SkillId] = SKILL_IDS,
) -> list[dict[str, Any]]:
    """Return only the static semantics needed for high-level planning."""

    return [
        {
            "skill": spec.skill_id.value,
            "description": spec.description,
            **spec.mechanics.to_dict(),
        }
        for spec in (get_skill_spec(skill_id) for skill_id in skill_ids)
    ]


def skill_target_block(skill_id: SkillId | str) -> BlockType | None:
    """Return the block a skill faces to act on, from its registry entry.

    It is the block named by the skill's ``world.<block>_reachable``
    requirement, so the crafting skills target the crafting table. ``None``
    for a skill without one: the placements, which need only free ground,
    sleep, and eating a cow, whose target is a mob.
    """

    for requirement in get_skill_spec(skill_id).mechanics.requires:
        field = requirement.field
        if field.startswith("world.") and field.endswith("_reachable"):
            name = field[len("world.") : -len("_reachable")].upper()
            if name in BlockType.__members__:
                return BlockType[name]
    return None


def is_crafting_skill(skill_id: SkillId | str) -> bool:
    """Return whether a skill crafts at a table, which works diagonally too."""

    return skill_target_block(skill_id) is BlockType.CRAFTING_TABLE


def interacts_by_doing(skill_id: SkillId | str) -> bool:
    """Return whether a skill acts by facing its target and pressing DO."""

    return get_skill_spec(skill_id).interaction_action is Action.DO


def _block_count(state: Any, block_type: BlockType) -> jax.Array:
    return jnp.count_nonzero(state.map == block_type.value)


def _has_block(state: Any, block_type: BlockType) -> jax.Array:
    return jnp.any(state.map == block_type.value)


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
    if skill_id is SkillId.CRAFT_STONE_PICKAXE:
        return jnp.asarray(state.inventory.stone_pickaxe)
    if skill_id is SkillId.CRAFT_STONE_SWORD:
        return jnp.asarray(state.inventory.stone_sword)
    if skill_id is SkillId.COLLECT_COAL:
        return jnp.asarray(state.inventory.coal)
    if skill_id is SkillId.PLACE_STONE:
        return _block_count(state, BlockType.STONE)
    if skill_id is SkillId.PLACE_FURNACE:
        return _block_count(state, BlockType.FURNACE)
    if skill_id is SkillId.COLLECT_SAPLING:
        return jnp.asarray(state.inventory.sapling)
    if skill_id is SkillId.PLACE_PLANT:
        # A placed sapling may ripen within the skill.
        return jnp.count_nonzero(
            (state.map == BlockType.PLANT.value)
            | (state.map == BlockType.RIPE_PLANT.value)
        )
    if skill_id in {SkillId.EAT_COW, SkillId.EAT_PLANT}:
        # Either meal counts for both. The scenarios offer only the one food
        # source, and the planner re-reads the achievements at each boundary.
        return jnp.asarray(state.player_food)
    if skill_id is SkillId.SLEEP:
        # Energy only rises asleep, and the player wakes once it is full,
        # so this turns from zero to one exactly on waking rested.
        return (
            (jnp.asarray(state.player_energy) >= MAX_VITAL)
            & ~jnp.asarray(state.is_sleeping)
        ).astype(jnp.int32)
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
    if skill_id in {SkillId.CRAFT_STONE_PICKAXE, SkillId.CRAFT_STONE_SWORD}:
        return (
            alive
            & (jnp.asarray(state.inventory.wood) >= goal.quantity)
            & (jnp.asarray(state.inventory.stone) >= goal.quantity)
            & jnp.any(state.map == BlockType.CRAFTING_TABLE.value)
        )
    if skill_id is SkillId.COLLECT_COAL:
        return (
            alive
            & (jnp.asarray(state.inventory.wood_pickaxe) > 0)
            & _has_block(state, BlockType.COAL)
        )
    if skill_id in {SkillId.PLACE_STONE, SkillId.PLACE_FURNACE}:
        return alive & (jnp.asarray(state.inventory.stone) >= goal.quantity)
    if skill_id is SkillId.COLLECT_SAPLING:
        return alive & _has_block(state, BlockType.GRASS)
    if skill_id is SkillId.PLACE_PLANT:
        return (
            alive
            & (jnp.asarray(state.inventory.sapling) >= goal.quantity)
            & _has_block(state, BlockType.GRASS)
        )
    if skill_id is SkillId.EAT_COW:
        return (
            alive
            & (jnp.asarray(state.player_food) + goal.quantity <= MAX_VITAL)
            & jnp.any(state.cows.mask)
        )
    if skill_id is SkillId.SLEEP:
        return alive & (jnp.asarray(state.player_energy) < MAX_VITAL)
    if skill_id is SkillId.EAT_PLANT:
        return (
            alive
            & (jnp.asarray(state.player_food) + goal.quantity <= MAX_VITAL)
            & _has_block(state, BlockType.RIPE_PLANT)
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
