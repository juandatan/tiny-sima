from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.envs.craftax import make_environment
from tiny_sima.goals import (
    SKILL_REGISTRY,
    SkillGoal,
    SkillId,
    get_skill_spec,
    skill_catalog,
    skill_precondition,
    skill_progress,
    skill_success,
    sparse_skill_reward,
)


def _initial_state() -> Any:
    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        max_episode_steps=128,
    )
    _, state = bundle.env.reset(jax.random.PRNGKey(0), bundle.params)
    return state


@pytest.mark.parametrize(
    ("skill_id", "inventory_field"),
    [
        (SkillId.COLLECT_WOOD, "wood"),
        (SkillId.COLLECT_STONE, "stone"),
        (SkillId.CRAFT_WOOD_PICKAXE, "wood_pickaxe"),
        (SkillId.CRAFT_WOOD_SWORD, "wood_sword"),
    ],
)
def test_inventory_skill_success_uses_state_delta(
    skill_id: SkillId,
    inventory_field: str,
) -> None:
    initial = _initial_state()
    inventory = initial.inventory.replace(**{inventory_field: 2})
    current = initial.replace(inventory=inventory)
    goal = SkillGoal(skill_id, quantity=2)

    assert bool(np.asarray(skill_success(goal, initial, current)))
    assert float(np.asarray(skill_progress(goal, initial, current))) == 1.0


def test_drink_water_success_uses_drink_level_delta() -> None:
    initial = _initial_state().replace(player_drink=5)
    current = initial.replace(player_drink=6)
    goal = SkillGoal(SkillId.DRINK_WATER)

    assert bool(np.asarray(skill_success(goal, initial, current)))


def test_place_table_success_counts_new_map_tiles() -> None:
    initial = _initial_state()
    current_map = initial.map.at[0, 0].set(BlockType.CRAFTING_TABLE.value)
    current = initial.replace(map=current_map)
    goal = SkillGoal(SkillId.PLACE_TABLE)

    assert bool(np.asarray(skill_success(goal, initial, current)))


def test_stone_precondition_requires_wood_pickaxe() -> None:
    state = _initial_state()
    state = state.replace(
        map=state.map.at[0, 0].set(BlockType.STONE.value),
        inventory=state.inventory.replace(wood_pickaxe=0),
    )
    goal = SkillGoal(SkillId.COLLECT_STONE)

    assert not bool(np.asarray(skill_precondition(goal, state)))

    state = state.replace(inventory=state.inventory.replace(wood_pickaxe=1))
    assert bool(np.asarray(skill_precondition(goal, state)))


def test_crafting_precondition_requires_wood_and_table() -> None:
    state = _initial_state()
    state_without_table = state.replace(
        map=jnp.where(
            state.map == BlockType.CRAFTING_TABLE.value,
            BlockType.GRASS.value,
            state.map,
        ),
        inventory=state.inventory.replace(wood=1),
    )
    goal = SkillGoal(SkillId.CRAFT_WOOD_PICKAXE)

    assert not bool(np.asarray(skill_precondition(goal, state_without_table)))

    state_with_table = state_without_table.replace(
        map=state_without_table.map.at[0, 0].set(
            BlockType.CRAFTING_TABLE.value
        )
    )
    assert bool(np.asarray(skill_precondition(goal, state_with_table)))


def test_sparse_reward_is_paid_only_on_completion_transition() -> None:
    initial = _initial_state()
    previous = initial.replace(inventory=initial.inventory.replace(wood=0))
    complete = initial.replace(inventory=initial.inventory.replace(wood=1))
    goal = SkillGoal(SkillId.COLLECT_WOOD)

    reward = sparse_skill_reward(goal, initial, previous, complete)
    repeated_reward = sparse_skill_reward(goal, initial, complete, complete)

    assert float(np.asarray(reward)) == 1.0
    assert float(np.asarray(repeated_reward)) == 0.0


def test_skill_catalog_covers_registry_and_actions() -> None:
    catalog = skill_catalog()

    assert {item["skill_id"] for item in catalog} == {
        skill_id.value for skill_id in SKILL_REGISTRY
    }
    assert get_skill_spec(SkillId.PLACE_TABLE).interaction_action is Action.PLACE_TABLE


def test_skill_goal_validates_quantity_and_horizon() -> None:
    assert SkillGoal("collect_wood").skill_id is SkillId.COLLECT_WOOD

    with pytest.raises(ValueError, match="quantity"):
        SkillGoal(SkillId.COLLECT_WOOD, quantity=0)
    with pytest.raises(ValueError, match="horizon"):
        SkillGoal(SkillId.COLLECT_WOOD, horizon=0)
