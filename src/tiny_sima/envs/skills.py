"""Short, reproducible Craftax skill episodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import Inventory
from flax import struct

from tiny_sima.envs.craftax import CraftaxBundle, make_environment
from tiny_sima.goals.skills import (
    SkillGoal,
    SkillId,
    skill_metric,
    skill_precondition,
    skill_progress_from_metric,
    skill_success_from_metric,
)


@struct.dataclass
class SkillEpisodeState:
    """Dynamic state needed by a short skill episode."""

    env_state: Any
    initial_metric: jax.Array
    steps: jax.Array
    target_direction: jax.Array


@dataclass(frozen=True)
class SkillEnvironment:
    """A Craftax environment configured for one short objective."""

    bundle: CraftaxBundle
    goal: SkillGoal

    def reset(self, rng: jax.Array) -> tuple[jax.Array, SkillEpisodeState]:
        world_rng, scenario_rng = jax.random.split(rng)
        _, state = self.bundle.env.reset(world_rng, self.bundle.params)
        state, target_direction = prepare_skill_scenario(
            state,
            self.goal,
            scenario_rng,
        )
        observation = self.bundle.env.get_obs(state)
        episode_state = SkillEpisodeState(
            env_state=state,
            initial_metric=skill_metric(self.goal.skill_id, state),
            steps=jnp.array(0, dtype=jnp.int32),
            target_direction=target_direction,
        )
        return observation, episode_state

    def step(
        self,
        rng: jax.Array,
        state: SkillEpisodeState,
        action: jax.Array,
    ) -> tuple[
        jax.Array,
        SkillEpisodeState,
        jax.Array,
        jax.Array,
        dict[str, jax.Array],
    ]:
        previous_metric = skill_metric(self.goal.skill_id, state.env_state)
        precondition = skill_precondition(self.goal, state.env_state)
        observation, env_state, native_reward, env_done, info = self.bundle.env.step(
            rng,
            state.env_state,
            action,
            self.bundle.params,
        )
        steps = state.steps + 1
        success = skill_success_from_metric(
            self.goal,
            state.initial_metric,
            env_state,
        )
        previous_success = previous_metric - state.initial_metric >= self.goal.quantity
        reward = (success & ~previous_success).astype(jnp.float32)
        timeout = (steps >= self.goal.effective_horizon) & ~success
        done = env_done | success | timeout
        next_state = state.replace(env_state=env_state, steps=steps)
        skill_info = {
            **info,
            "native_reward": native_reward,
            "skill_success": success,
            "skill_timeout": timeout,
            "skill_progress": skill_progress_from_metric(
                self.goal,
                state.initial_metric,
                env_state,
            ),
            "skill_precondition": precondition,
        }
        return observation, next_state, reward, done, skill_info


def _clear_mobs(state: Any) -> Any:
    def clear(mobs: Any) -> Any:
        return mobs.replace(mask=jnp.zeros_like(mobs.mask))

    return state.replace(
        mob_map=jnp.zeros_like(state.mob_map),
        zombies=clear(state.zombies),
        cows=clear(state.cows),
        skeletons=clear(state.skeletons),
        arrows=clear(state.arrows),
    )


def prepare_skill_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
) -> tuple[Any, jax.Array]:
    """Inject a valid nearby target while retaining real Craftax dynamics."""

    if goal.quantity != 1:
        raise ValueError("The nearby-target curriculum currently supports quantity=1.")

    target_direction = jax.random.randint(
        rng,
        shape=(),
        minval=Action.LEFT.value,
        maxval=Action.DOWN.value + 1,
        dtype=jnp.int32,
    )
    player_position = jnp.array(
        [state.map.shape[0] // 2, state.map.shape[1] // 2],
        dtype=jnp.int32,
    )
    clean_map = jnp.where(
        state.map == BlockType.CRAFTING_TABLE.value,
        BlockType.GRASS.value,
        state.map,
    )
    clean_map = clean_map.at[player_position[0], player_position[1]].set(
        BlockType.GRASS.value
    )
    for direction in range(Action.LEFT.value, Action.DOWN.value + 1):
        position = player_position + DIRECTIONS[direction]
        clean_map = clean_map.at[position[0], position[1]].set(
            BlockType.GRASS.value
        )

    target_position = player_position + DIRECTIONS[target_direction]
    skill_id = goal.skill_id
    inventory = Inventory()
    player_drink = 9
    player_direction = target_direction

    if skill_id is SkillId.COLLECT_WOOD:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.TREE.value
        )
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.COLLECT_STONE:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.STONE.value
        )
        inventory = inventory.replace(wood_pickaxe=1)
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.DRINK_WATER:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.WATER.value
        )
        player_drink = 5
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.PLACE_TABLE:
        inventory = inventory.replace(wood=2)
    elif skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
    }:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.CRAFTING_TABLE.value
        )
        inventory = inventory.replace(wood=1)
    else:
        raise AssertionError(f"Unhandled skill: {skill_id}")

    state = _clear_mobs(state)
    state = state.replace(
        map=clean_map,
        player_position=player_position,
        player_direction=player_direction,
        player_health=9,
        player_food=9,
        player_drink=player_drink,
        player_energy=9,
        is_sleeping=False,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        inventory=inventory,
        achievements=jnp.zeros_like(state.achievements),
        timestep=0,
        light_level=1.0,
    )
    return state, target_direction


def make_skill_environment(
    skill_id: SkillId | str,
    *,
    horizon: int = 8,
) -> SkillEnvironment:
    """Create the first-stage nearby-target benchmark for one skill."""

    goal = SkillGoal(SkillId(skill_id), horizon=horizon)
    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        max_episode_steps=max(128, horizon + 1),
    )
    params = bundle.params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    return SkillEnvironment(bundle=replace(bundle, params=params), goal=goal)


def reset_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    num_envs: int,
) -> tuple[jax.Array, SkillEpisodeState]:
    """Reset a vector of independent skill scenarios."""

    reset_keys = jax.random.split(rng, num_envs)
    return jax.vmap(environment.reset)(reset_keys)


def step_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    state: SkillEpisodeState,
    action: jax.Array,
) -> tuple[
    jax.Array,
    SkillEpisodeState,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
]:
    """Step a vector of independent skill scenarios."""

    step_keys = jax.random.split(rng, action.shape[0])
    return jax.vmap(environment.step)(step_keys, state, action)
