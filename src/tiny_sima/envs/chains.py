"""Persistent controlled Craftax worlds for evaluating skill composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import Inventory

from tiny_sima.envs.craftax import CraftaxBundle, make_environment


class SkillChainReset(NamedTuple):
    """Initial observation and persistent state for one chain episode."""

    observation: jax.Array
    state: Any


@dataclass(frozen=True)
class SkillChainEnvironment:
    """Craftax environment whose state persists across requested skills."""

    bundle: CraftaxBundle
    distractor_count: int = 3

    def reset(self, rng: jax.Array) -> SkillChainReset:
        world_rng, scenario_rng = jax.random.split(rng)
        _, state = self.bundle.env.reset(world_rng, self.bundle.params)
        state = prepare_chain_scenario(
            state,
            scenario_rng,
            distractor_count=self.distractor_count,
        )
        return SkillChainReset(
            observation=self.bundle.env.get_obs(state),
            state=state,
        )


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


def prepare_chain_scenario(
    state: Any,
    rng: jax.Array,
    *,
    distractor_count: int = 3,
) -> Any:
    """Create a randomized but solvable wood-to-stone crafting world."""

    if not 0 <= distractor_count <= 4:
        raise ValueError("distractor_count must be between zero and four.")

    position_rng, direction_rng = jax.random.split(rng)
    center_row = state.map.shape[0] // 2
    center_column = state.map.shape[1] // 2
    player_position = jnp.asarray([center_row, center_column], dtype=jnp.int32)

    row_radius = OBS_DIM[0] // 2
    column_radius = OBS_DIM[1] // 2
    clean_map = state.map.at[
        center_row - row_radius : center_row + row_radius + 1,
        center_column - column_radius : center_column + column_radius + 1,
    ].set(BlockType.GRASS.value)

    candidate_offsets = jnp.asarray(
        [
            (-2, -2),
            (-2, 0),
            (-2, 2),
            (0, -3),
            (0, 3),
            (2, -2),
            (2, 0),
            (2, 2),
            (-1, 3),
            (1, -3),
        ],
        dtype=jnp.int32,
    )
    shuffled_offsets = jax.random.permutation(position_rng, candidate_offsets)
    required_blocks = [
        BlockType.TREE,
        BlockType.TREE,
        BlockType.TREE,
        BlockType.STONE,
        BlockType.WATER,
    ]
    distractor_blocks = [
        BlockType.COAL,
        BlockType.FURNACE,
        BlockType.SAND,
        BlockType.PLANT,
    ]
    blocks = required_blocks + distractor_blocks[:distractor_count]
    for index, block in enumerate(blocks):
        position = player_position + shuffled_offsets[index]
        clean_map = clean_map.at[position[0], position[1]].set(block.value)

    state = _clear_mobs(state)
    return state.replace(
        map=clean_map,
        player_position=player_position,
        player_direction=jax.random.randint(
            direction_rng,
            shape=(),
            minval=Action.LEFT.value,
            maxval=Action.DOWN.value + 1,
            dtype=jnp.int32,
        ),
        player_health=9,
        player_food=9,
        player_drink=5,
        player_energy=9,
        is_sleeping=False,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        inventory=Inventory(),
        achievements=jnp.zeros_like(state.achievements),
        timestep=0,
        light_level=1.0,
    )


def make_skill_chain_environment(
    *,
    max_episode_steps: int = 512,
    distractor_count: int = 3,
) -> SkillChainEnvironment:
    """Create the controlled persistent environment used by chain evals."""

    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        max_episode_steps=max_episode_steps,
    )
    params = bundle.params.replace(
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    return SkillChainEnvironment(
        bundle=CraftaxBundle(
            env=bundle.env,
            params=params,
            spec=bundle.spec,
        ),
        distractor_count=distractor_count,
    )
