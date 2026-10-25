"""Short, reproducible Craftax skill episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import (
    DIRECTIONS,
    OBS_DIM,
    Action,
    BlockType,
)
from craftax.craftax_classic.envs.craftax_state import EnvParams, Inventory
from craftax.craftax_classic.game_logic import calculate_light_level

from tiny_sima.envs.craftax import (
    CraftaxBundle,
    EnvironmentSpec,
    make_environment,
)
from tiny_sima.goals.reachability import (
    DEFAULT_REACHABILITY,
    interaction_distance,
    walkable_mask,
)
from tiny_sima.goals.skills import (
    SKILL_IDS,
    SkillGoal,
    SkillId,
    executor_skill_ids,
    skill_index,
    skill_metric,
    skill_precondition,
    skill_progress_from_metric,
    skill_success_from_metric,
)

if TYPE_CHECKING:
    from tiny_sima.config import SkillPPOConfig


class SkillEpisodeState(NamedTuple):
    """Dynamic state needed by a short skill episode."""

    env_state: Any
    initial_metric: jax.Array
    steps: jax.Array
    target_direction: jax.Array
    target_position: jax.Array
    approach_position: jax.Array


class SkillReset(NamedTuple):
    """Output of a single or batched skill reset."""

    observation: jax.Array
    state: SkillEpisodeState


class SkillStep(NamedTuple):
    """Output of a single or batched skill step."""

    observation: jax.Array
    state: SkillEpisodeState
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]


class PreparedSkillScenario(NamedTuple):
    """Craftax state and target metadata produced by scenario preparation."""

    state: Any
    target_direction: jax.Array
    target_position: jax.Array
    approach_position: jax.Array


class GoalConditionedSkillReset(NamedTuple):
    """Batched observations, state, and categorical goals after reset."""

    observation: jax.Array
    state: SkillEpisodeState
    skill_id: jax.Array


class GoalConditionedSkillStep(NamedTuple):
    """Batched transition returned during goal-conditioned training."""

    observation: jax.Array
    state: SkillEpisodeState
    skill_id: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]


class SkillScenario(str, Enum):
    """Supported controlled distributions for short skill episodes."""

    NEARBY = "nearby"
    LOCAL_NAVIGATION = "local_navigation"
    RANDOMIZED = "randomized"


@dataclass(frozen=True)
class ScenarioRandomization:
    """Knobs that widen local navigation toward the worlds a chain hands over.

    ``local_navigation`` starts every episode at the map centre in a cleared
    window, with one target, a reserved straight route, the minimal inventory
    and the clock at dawn. A chain boundary differs in each of those ways, and
    each knob below undoes one of them, so any one can be switched off as an
    ablation. The defaults are the full randomized distribution. Every world
    stays solvable: a target the in-view search cannot reach inside the horizon
    gets its L-route cleared, and ``place_table`` always keeps a free neighbour.
    """

    # Obstacles may land on the shortest route, so reaching a target can need
    # a detour, as it does once a chain has filled the window.
    reserve_route: bool = False
    # Further copies of a collect or drink target, as the chain's three trees.
    extra_target_count: int = 2
    # Chance of a crafting table in view for skills that do not use one, as
    # after place_table.
    table_probability: float = 0.5
    # The player stands up to this many tiles from the map centre on each
    # axis, so the view can include terrain the window clearing missed.
    max_player_offset: int = 3
    # Chance of clearing the window around the map centre at all; otherwise
    # the target sits in natural terrain.
    clear_probability: float = 0.5
    # Wood, saplings, stone and wood tools beyond the skill's minimum.
    random_inventory: bool = True
    # Health, food, drink and energy anywhere from three to nine.
    random_vitals: bool = True
    # The clock starts anywhere up to this step, so light_level, which the
    # symbolic observation carries, covers dusk and night as well as dawn.
    max_start_timestep: int = 300

    def __post_init__(self) -> None:
        if not 0 <= self.extra_target_count <= 8:
            raise ValueError("extra_target_count must be between 0 and 8.")
        if not 0.0 <= self.table_probability <= 1.0:
            raise ValueError("table_probability must lie in [0, 1].")
        if not 0 <= self.max_player_offset <= 8:
            raise ValueError("max_player_offset must be between 0 and 8.")
        if not 0.0 <= self.clear_probability <= 1.0:
            raise ValueError("clear_probability must lie in [0, 1].")
        if self.max_start_timestep < 0:
            raise ValueError("max_start_timestep cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, init=False)
class SkillScenarioConfig:
    """Static controls for reproducible skill-scenario generation.

    ``randomization`` applies only to the randomized kind, which defaults to
    the full :class:`ScenarioRandomization`.
    """

    kind: SkillScenario
    min_distance: int
    max_distance: int
    distractor_count: int
    randomization: ScenarioRandomization | None

    def __init__(
        self,
        kind: SkillScenario | str = SkillScenario.NEARBY,
        *,
        min_distance: int = 2,
        max_distance: int = 4,
        distractor_count: int = 6,
        randomization: ScenarioRandomization | None = None,
    ) -> None:
        normalized_kind = SkillScenario(kind)
        if normalized_kind is SkillScenario.RANDOMIZED:
            randomization = randomization or ScenarioRandomization()
        elif randomization is not None:
            raise ValueError("randomization applies only to the randomized kind.")
        if min_distance < 1:
            raise ValueError("min_distance must be positive.")
        if max_distance < min_distance:
            raise ValueError("max_distance must be at least min_distance.")
        max_visible_distance = OBS_DIM[0] // 2 + OBS_DIM[1] // 2
        if max_distance > max_visible_distance:
            raise ValueError(
                "max_distance must keep targets inside the initial observation."
            )
        if distractor_count < 0:
            raise ValueError("distractor_count cannot be negative.")
        if distractor_count > 32:
            raise ValueError("distractor_count cannot exceed 32.")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "min_distance", min_distance)
        object.__setattr__(self, "max_distance", max_distance)
        object.__setattr__(self, "distractor_count", distractor_count)
        object.__setattr__(self, "randomization", randomization)

    @property
    def max_start_timestep(self) -> int:
        if self.randomization is None:
            return 0
        return self.randomization.max_start_timestep


@dataclass(frozen=True)
class SkillEnvironment:
    """A Craftax environment configured for one short objective."""

    bundle: CraftaxBundle
    goal: SkillGoal
    scenario: SkillScenarioConfig

    def reset(self, rng: jax.Array) -> SkillReset:
        world_rng, scenario_rng = jax.random.split(rng)
        _, state = self.bundle.env.reset(world_rng, self.bundle.params)
        scenario = prepare_skill_scenario(
            state,
            self.goal,
            scenario_rng,
            config=self.scenario,
        )
        observation = self.bundle.env.get_obs(scenario.state)
        episode_state = SkillEpisodeState(
            env_state=scenario.state,
            initial_metric=skill_metric(self.goal.skill_id, scenario.state),
            steps=jnp.array(0, dtype=jnp.int32),
            target_direction=scenario.target_direction,
            target_position=scenario.target_position,
            approach_position=scenario.approach_position,
        )
        return SkillReset(observation=observation, state=episode_state)

    def step(
        self,
        rng: jax.Array,
        state: SkillEpisodeState,
        action: jax.Array,
    ) -> SkillStep:
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
        precondition_after = skill_precondition(self.goal, env_state)
        previous_success = previous_metric - state.initial_metric >= self.goal.quantity
        reward = (success & ~previous_success).astype(jnp.float32)
        timeout = (steps >= self.goal.effective_horizon) & ~success
        done = env_done | success | timeout
        next_state = state._replace(env_state=env_state, steps=steps)
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
            "skill_precondition_after": precondition_after,
        }
        return SkillStep(
            observation=observation,
            state=next_state,
            reward=reward,
            done=done,
            info=skill_info,
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


LOCAL_NAVIGATION_SKILLS: tuple[SkillId, ...] = (
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_STONE,
    SkillId.DRINK_WATER,
    SkillId.CRAFT_WOOD_PICKAXE,
    SkillId.CRAFT_WOOD_SWORD,
    SkillId.CRAFT_STONE_PICKAXE,
)


def skills_for_scenario(
    scenario: SkillScenario | str,
    skill_ids: tuple[SkillId, ...] = SKILL_IDS,
) -> tuple[SkillId, ...]:
    """Return the skills to which a controlled scenario meaningfully applies.

    ``skill_ids`` is the executor's catalog; the result keeps its order.
    """

    if SkillScenario(scenario) is SkillScenario.LOCAL_NAVIGATION:
        return tuple(
            skill_id for skill_id in skill_ids if skill_id in LOCAL_NAVIGATION_SKILLS
        )
    return skill_ids


def _target_block(skill_id: SkillId) -> BlockType | None:
    if skill_id is SkillId.COLLECT_WOOD:
        return BlockType.TREE
    if skill_id is SkillId.COLLECT_STONE:
        return BlockType.STONE
    if skill_id is SkillId.DRINK_WATER:
        return BlockType.WATER
    if skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
        SkillId.CRAFT_STONE_PICKAXE,
    }:
        return BlockType.CRAFTING_TABLE
    if skill_id is SkillId.PLACE_TABLE:
        return None
    raise AssertionError(f"Unhandled skill: {skill_id}")


def _scenario_inventory(skill_id: SkillId) -> Inventory:
    inventory = Inventory()
    if skill_id is SkillId.COLLECT_STONE:
        return replace(inventory, wood_pickaxe=1)
    if skill_id is SkillId.PLACE_TABLE:
        return replace(inventory, wood=2)
    if skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
    }:
        return replace(inventory, wood=1)
    if skill_id is SkillId.CRAFT_STONE_PICKAXE:
        # The stone could only have been mined with a wood pickaxe, so a
        # chain always reaches this craft holding one.
        return replace(inventory, wood=1, stone=1, wood_pickaxe=1)
    return inventory


def _local_offsets(config: SkillScenarioConfig) -> jax.Array:
    row_radius = OBS_DIM[0] // 2
    column_radius = OBS_DIM[1] // 2
    offsets = [
        (row, column)
        for row in range(-row_radius, row_radius + 1)
        for column in range(-column_radius, column_radius + 1)
        if config.min_distance
        <= abs(row) + abs(column)
        <= config.max_distance
    ]
    if not offsets:
        raise ValueError("No local target positions satisfy the distance bounds.")
    return jnp.asarray(offsets, dtype=jnp.int32)


def _direction_to_target(offset: jax.Array) -> jax.Array:
    row, column = offset[0], offset[1]
    use_horizontal = (column != 0) & (jnp.abs(column) >= jnp.abs(row))
    horizontal = jnp.where(column < 0, Action.LEFT.value, Action.RIGHT.value)
    vertical = jnp.where(row < 0, Action.UP.value, Action.DOWN.value)
    return jnp.where(use_horizontal, horizontal, vertical).astype(jnp.int32)


def _between(value: jax.Array, start: jax.Array, stop: jax.Array) -> jax.Array:
    lower = jnp.minimum(start, stop)
    upper = jnp.maximum(start, stop)
    return (value >= lower) & (value <= upper)


def _distractor_blocks(target: BlockType | None) -> jax.Array:
    candidates = (
        BlockType.TREE,
        BlockType.STONE,
        BlockType.WATER,
        BlockType.COAL,
        BlockType.SAND,
        BlockType.FURNACE,
    )
    return jnp.asarray(
        [block.value for block in candidates if block is not target],
        dtype=jnp.int32,
    )


def _prepare_nearby_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
) -> PreparedSkillScenario:
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
        clean_map = clean_map.at[position[0], position[1]].set(BlockType.GRASS.value)

    target_position = player_position + DIRECTIONS[target_direction]
    target = _target_block(goal.skill_id)
    if target is not None:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            target.value
        )

    player_direction = target_direction
    if goal.skill_id in {
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_STONE,
        SkillId.DRINK_WATER,
    }:
        player_direction = target_direction % 4 + 1

    return PreparedSkillScenario(
        state=state.replace(
            map=clean_map,
            player_position=player_position,
            player_direction=player_direction,
            inventory=_scenario_inventory(goal.skill_id),
        ),
        target_direction=target_direction,
        target_position=target_position,
        approach_position=player_position,
    )


def _prepare_local_navigation_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
    config: SkillScenarioConfig,
) -> PreparedSkillScenario:
    target_rng, direction_rng, distractor_rng = jax.random.split(rng, 3)
    player_position = jnp.array(
        [state.map.shape[0] // 2, state.map.shape[1] // 2],
        dtype=jnp.int32,
    )
    offsets = _local_offsets(config)
    target_offset = offsets[
        jax.random.randint(target_rng, (), 0, offsets.shape[0])
    ]
    target_direction = _direction_to_target(target_offset)
    target_position = player_position + target_offset
    approach_position = target_position - DIRECTIONS[target_direction]

    row_radius = OBS_DIM[0] // 2
    column_radius = OBS_DIM[1] // 2
    row_start = state.map.shape[0] // 2 - row_radius
    row_stop = state.map.shape[0] // 2 + row_radius + 1
    column_start = state.map.shape[1] // 2 - column_radius
    column_stop = state.map.shape[1] // 2 + column_radius + 1
    clean_map = state.map.at[row_start:row_stop, column_start:column_stop].set(
        BlockType.GRASS.value
    )

    candidate_offsets = jnp.asarray(
        [
            (row, column)
            for row in range(-row_radius, row_radius + 1)
            for column in range(-column_radius, column_radius + 1)
            if row != 0 or column != 0
        ],
        dtype=jnp.int32,
    )
    approach_offset = approach_position - player_position
    candidate_rows = candidate_offsets[:, 0]
    candidate_columns = candidate_offsets[:, 1]
    vertical_path = (candidate_columns == 0) & _between(
        candidate_rows,
        jnp.asarray(0),
        approach_offset[0],
    )
    horizontal_path = (candidate_rows == approach_offset[0]) & _between(
        candidate_columns,
        jnp.asarray(0),
        approach_offset[1],
    )
    reserved = (
        vertical_path
        | horizontal_path
        | jnp.all(candidate_offsets == target_offset, axis=-1)
        | jnp.all(candidate_offsets == approach_offset, axis=-1)
    )
    random_scores = jax.random.uniform(
        distractor_rng,
        shape=(candidate_offsets.shape[0],),
    )
    selected = jnp.argsort(jnp.where(reserved, jnp.inf, random_scores))[
        : config.distractor_count
    ]
    distractor_positions = player_position + candidate_offsets[selected]
    distractor_blocks = _distractor_blocks(_target_block(goal.skill_id))
    for index in range(config.distractor_count):
        position = distractor_positions[index]
        block = distractor_blocks[index % distractor_blocks.shape[0]]
        clean_map = clean_map.at[position[0], position[1]].set(block)

    target = _target_block(goal.skill_id)
    if target is not None:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            target.value
        )

    player_direction = jax.random.randint(
        direction_rng,
        shape=(),
        minval=Action.LEFT.value,
        maxval=Action.DOWN.value + 1,
        dtype=jnp.int32,
    )
    return PreparedSkillScenario(
        state=state.replace(
            map=clean_map,
            player_position=player_position,
            player_direction=player_direction,
            inventory=_scenario_inventory(goal.skill_id),
        ),
        target_direction=target_direction,
        target_position=target_position,
        approach_position=approach_position,
    )


_RESOURCE_SKILLS = frozenset(
    {SkillId.COLLECT_WOOD, SkillId.COLLECT_STONE, SkillId.DRINK_WATER}
)
_TABLE_SKILLS = frozenset(
    {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
        SkillId.CRAFT_STONE_PICKAXE,
    }
)
_DEFAULT_PARAMS = EnvParams()


def _randomized_distractor_blocks(target: BlockType | None) -> jax.Array:
    candidates = (
        BlockType.TREE,
        BlockType.STONE,
        BlockType.WATER,
        BlockType.COAL,
        BlockType.SAND,
        BlockType.FURNACE,
        BlockType.PLANT,
    )
    return jnp.asarray(
        [block.value for block in candidates if block is not target],
        dtype=jnp.int32,
    )


def _view_offsets() -> jax.Array:
    row_radius = OBS_DIM[0] // 2
    column_radius = OBS_DIM[1] // 2
    return jnp.asarray(
        [
            (row, column)
            for row in range(-row_radius, row_radius + 1)
            for column in range(-column_radius, column_radius + 1)
            if row != 0 or column != 0
        ],
        dtype=jnp.int32,
    )


def _route_mask(
    shape: tuple[int, ...],
    start: jax.Array,
    stop: jax.Array,
) -> jax.Array:
    """Return the L-route from ``start`` down or up to ``stop``'s row, then across."""

    rows = jnp.arange(shape[0])[:, None]
    columns = jnp.arange(shape[1])[None, :]
    vertical = (columns == start[1]) & _between(rows, start[0], stop[0])
    horizontal = (rows == stop[0]) & _between(columns, start[1], stop[1])
    return vertical | horizontal


def _random_inventory(skill_id: SkillId, rng: jax.Array) -> Inventory:
    """Add chain-like holdings on top of what the skill needs."""

    base = _scenario_inventory(skill_id)
    wood, sapling, stone, pickaxe, sword = jax.random.randint(
        rng,
        (5,),
        jnp.asarray([0, 0, 0, 0, 0]),
        jnp.asarray([4, 2, 3, 2, 2]),
    )
    return replace(
        base,
        wood=base.wood + wood,
        sapling=sapling,
        stone=base.stone + stone,
        wood_pickaxe=jnp.maximum(base.wood_pickaxe, pickaxe),
        wood_sword=sword,
    )


def _prepare_randomized_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
    config: SkillScenarioConfig,
) -> PreparedSkillScenario:
    randomization = config.randomization
    assert randomization is not None
    (
        offset_rng,
        clear_rng,
        target_rng,
        direction_rng,
        placement_rng,
        table_rng,
        block_rng,
        neighbour_rng,
        inventory_rng,
    ) = jax.random.split(rng, 9)
    centre = jnp.array(
        [state.map.shape[0] // 2, state.map.shape[1] // 2],
        dtype=jnp.int32,
    )
    player_position = centre + jax.random.randint(
        offset_rng,
        (2,),
        -randomization.max_player_offset,
        randomization.max_player_offset + 1,
        dtype=jnp.int32,
    )

    # The chain world clears the same window around the map centre, so an
    # offset player sees cleared ground on one side and terrain on the other.
    row_radius = OBS_DIM[0] // 2
    column_radius = OBS_DIM[1] // 2
    centre_row = state.map.shape[0] // 2
    centre_column = state.map.shape[1] // 2
    cleared = state.map.at[
        centre_row - row_radius : centre_row + row_radius + 1,
        centre_column - column_radius : centre_column + column_radius + 1,
    ].set(BlockType.GRASS.value)
    world_map = jnp.where(
        jax.random.bernoulli(clear_rng, randomization.clear_probability),
        cleared,
        state.map,
    )
    world_map = world_map.at[player_position[0], player_position[1]].set(
        BlockType.GRASS.value
    )

    offsets = _local_offsets(config)
    target_offset = offsets[jax.random.randint(target_rng, (), 0, offsets.shape[0])]
    target_direction = _direction_to_target(target_offset)
    target_position = player_position + target_offset
    approach_position = target_position - DIRECTIONS[target_direction]
    route = _route_mask(world_map.shape, player_position, approach_position)

    view_offsets = _view_offsets()
    candidates = player_position + view_offsets
    scores = jax.random.uniform(placement_rng, (candidates.shape[0],))
    reserved = jnp.all(candidates == target_position, axis=-1)
    if randomization.reserve_route:
        reserved = reserved | route[candidates[:, 0], candidates[:, 1]]
    used = reserved

    def place(
        world_map: jax.Array,
        used: jax.Array,
        block: Any,
        *,
        eligible: Any = True,
        keep: Any = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Put ``block`` on the lowest-scoring eligible tile not yet used."""

        allowed = eligible & ~used
        index = jnp.argmin(jnp.where(allowed, scores, jnp.inf))
        keep = keep & allowed[index]
        position = candidates[index]
        return (
            world_map.at[position[0], position[1]].set(
                jnp.where(keep, block, world_map[position[0], position[1]])
            ),
            used.at[index].set(used[index] | keep),
        )

    target = _target_block(goal.skill_id)
    if target is not None and goal.skill_id in _RESOURCE_SKILLS:
        # Copies no closer than the sampled target, so the distance band
        # still names the nearest target rather than a lucky extra one.
        far = jnp.sum(jnp.abs(view_offsets), axis=-1) >= jnp.sum(
            jnp.abs(target_offset)
        )
        for _ in range(randomization.extra_target_count):
            world_map, used = place(world_map, used, target.value, eligible=far)
    if goal.skill_id not in _TABLE_SKILLS and randomization.table_probability > 0:
        world_map, used = place(
            world_map,
            used,
            BlockType.CRAFTING_TABLE.value,
            keep=jax.random.bernoulli(table_rng, randomization.table_probability),
        )
    blocks = jax.random.choice(
        block_rng,
        _randomized_distractor_blocks(target),
        (config.distractor_count,),
    )
    for index in range(config.distractor_count):
        world_map, used = place(world_map, used, blocks[index])

    probe = state.replace(map=world_map, player_position=player_position)
    if target is not None:
        world_map = world_map.at[target_position[0], target_position[1]].set(
            target.value
        )
        target_mask = (
            jnp.zeros(world_map.shape, dtype=jnp.bool_)
            .at[target_position[0], target_position[1]]
            .set(True)
        )
        # Orthogonal adjacency even for crafting, which is the stricter test.
        bound = interaction_distance(probe.replace(map=world_map), target_mask)
        blocked = (bound > goal.effective_horizon) | (
            bound >= DEFAULT_REACHABILITY.unreachable
        )
        world_map = jnp.where(blocked & route, BlockType.GRASS.value, world_map)
    else:
        neighbours = player_position + jnp.asarray(
            DIRECTIONS[Action.LEFT.value : Action.DOWN.value + 1]
        )
        free = walkable_mask(probe)[neighbours[:, 0], neighbours[:, 1]]
        fallback = neighbours[jax.random.randint(neighbour_rng, (), 0, 4)]
        world_map = world_map.at[fallback[0], fallback[1]].set(
            jnp.where(
                jnp.any(free),
                world_map[fallback[0], fallback[1]],
                BlockType.GRASS.value,
            )
        )

    inventory = (
        _random_inventory(goal.skill_id, inventory_rng)
        if randomization.random_inventory
        else _scenario_inventory(goal.skill_id)
    )
    return PreparedSkillScenario(
        state=state.replace(
            map=world_map,
            player_position=player_position,
            player_direction=jax.random.randint(
                direction_rng,
                shape=(),
                minval=Action.LEFT.value,
                maxval=Action.DOWN.value + 1,
                dtype=jnp.int32,
            ),
            inventory=inventory,
        ),
        target_direction=target_direction,
        target_position=target_position,
        approach_position=approach_position,
    )


def _randomize_vitals_and_clock(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
    randomization: ScenarioRandomization,
) -> Any:
    vitals_rng, clock_rng = jax.random.split(rng)
    if randomization.random_vitals:
        health, food, drink, energy = jax.random.randint(vitals_rng, (4,), 3, 10)
        if goal.skill_id is SkillId.DRINK_WATER:
            drink = jnp.minimum(drink, 8)
        state = state.replace(
            player_health=health,
            player_food=food,
            player_drink=drink,
            player_energy=energy,
        )
    if randomization.max_start_timestep:
        timestep = jax.random.randint(
            clock_rng,
            (),
            0,
            randomization.max_start_timestep + 1,
        )
        state = state.replace(
            timestep=timestep,
            light_level=calculate_light_level(timestep, _DEFAULT_PARAMS).astype(
                jnp.float32
            ),
        )
    return state


def prepare_skill_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
    *,
    config: SkillScenarioConfig | None = None,
) -> PreparedSkillScenario:
    """Inject a controlled target while retaining real Craftax dynamics."""

    if goal.quantity != 1:
        raise ValueError("Controlled skill scenarios currently support quantity=1.")

    scenario_config = config or SkillScenarioConfig()
    if (
        scenario_config.kind is SkillScenario.LOCAL_NAVIGATION
        and goal.skill_id not in LOCAL_NAVIGATION_SKILLS
    ):
        raise ValueError(
            f"{goal.skill_id.value} does not define a local-navigation target."
        )
    state_rng = None
    if scenario_config.kind is SkillScenario.NEARBY:
        scenario = _prepare_nearby_scenario(state, goal, rng)
    elif scenario_config.kind is SkillScenario.LOCAL_NAVIGATION:
        scenario = _prepare_local_navigation_scenario(
            state,
            goal,
            rng,
            scenario_config,
        )
    else:
        # Split only here, so the older scenarios keep their exact worlds.
        rng, state_rng = jax.random.split(rng)
        scenario = _prepare_randomized_scenario(
            state,
            goal,
            rng,
            scenario_config,
        )

    player_drink = 5 if goal.skill_id is SkillId.DRINK_WATER else 9
    prepared_state = _clear_mobs(scenario.state)
    prepared_state = prepared_state.replace(
        player_health=9,
        player_food=9,
        player_drink=player_drink,
        player_energy=9,
        is_sleeping=False,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        achievements=jnp.zeros_like(state.achievements),
        timestep=0,
        light_level=1.0,
    )
    if state_rng is not None and scenario_config.randomization is not None:
        prepared_state = _randomize_vitals_and_clock(
            prepared_state,
            goal,
            state_rng,
            scenario_config.randomization,
        )
    return scenario._replace(state=prepared_state)


def make_skill_environment(
    skill_id: SkillId | str,
    *,
    horizon: int = 8,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
    randomization: ScenarioRandomization | None = None,
) -> SkillEnvironment:
    """Create one controlled skill benchmark environment."""

    goal = SkillGoal(SkillId(skill_id), horizon=horizon)
    scenario_config = SkillScenarioConfig(
        scenario,
        min_distance=min_distance,
        max_distance=max_distance,
        distractor_count=distractor_count,
        randomization=randomization,
    )
    if (
        scenario_config.kind is SkillScenario.LOCAL_NAVIGATION
        and goal.skill_id not in LOCAL_NAVIGATION_SKILLS
    ):
        raise ValueError(
            f"{goal.skill_id.value} does not define a local-navigation target."
        )
    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        # A randomized clock must not end the episode before the skill does.
        max_episode_steps=max(128, horizon + 1) + scenario_config.max_start_timestep,
    )
    params = replace(
        bundle.params,
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    return SkillEnvironment(
        bundle=replace(bundle, params=params),
        goal=goal,
        scenario=scenario_config,
    )


def reset_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    num_envs: int,
) -> SkillReset:
    """Reset a vector of independent skill scenarios."""

    reset_keys = jax.random.split(rng, num_envs)
    return jax.vmap(environment.reset)(reset_keys)


def step_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    state: SkillEpisodeState,
    action: jax.Array,
) -> SkillStep:
    """Step a vector of independent skill scenarios."""

    step_keys = jax.random.split(rng, action.shape[0])
    return jax.vmap(environment.step)(step_keys, state, action)


def _concatenate_pytrees(*values: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.concatenate(leaves, axis=0),
        *values,
    )


def _select_completed(
    done: jax.Array,
    reset_value: Any,
    stepped_value: Any,
) -> Any:
    def select_leaf(reset_leaf: jax.Array, stepped_leaf: jax.Array) -> jax.Array:
        mask_shape = (done.shape[0],) + (1,) * (reset_leaf.ndim - 1)
        return jnp.where(done.reshape(mask_shape), reset_leaf, stepped_leaf)

    return jax.tree_util.tree_map(select_leaf, reset_value, stepped_value)


@dataclass(frozen=True)
class VectorizedSkillEnvironment:
    """Balanced multi-skill environment with automatic episode resets."""

    environments: tuple[SkillEnvironment, ...]
    envs_per_skill: int

    @property
    def num_envs(self) -> int:
        return len(self.environments) * self.envs_per_skill

    @property
    def num_skills(self) -> int:
        return len(self.environments)

    @property
    def spec(self) -> EnvironmentSpec:
        return self.environments[0].bundle.spec

    @property
    def skill_ids(self) -> jax.Array:
        return jnp.repeat(
            jnp.arange(self.num_skills, dtype=jnp.int32),
            self.envs_per_skill,
        )

    def reset(self, rng: jax.Array) -> GoalConditionedSkillReset:
        reset_keys = jax.random.split(rng, self.num_skills)
        resets = tuple(
            reset_skill_batch(environment, reset_keys[index], self.envs_per_skill)
            for index, environment in enumerate(self.environments)
        )
        return GoalConditionedSkillReset(
            observation=jnp.concatenate(
                tuple(reset.observation for reset in resets),
                axis=0,
            ),
            state=_concatenate_pytrees(*(reset.state for reset in resets)),
            skill_id=self.skill_ids,
        )

    def step(
        self,
        rng: jax.Array,
        state: SkillEpisodeState,
        action: jax.Array,
    ) -> GoalConditionedSkillStep:
        keys = jax.random.split(rng, self.num_skills * 2)
        steps: list[SkillStep] = []
        next_states: list[SkillEpisodeState] = []
        next_observations: list[jax.Array] = []

        for index, environment in enumerate(self.environments):
            start = index * self.envs_per_skill
            stop = start + self.envs_per_skill

            def slice_group(
                value: jax.Array,
                group_start: int = start,
                group_stop: int = stop,
            ) -> jax.Array:
                return value[group_start:group_stop]

            group_state = jax.tree_util.tree_map(
                slice_group,
                state,
            )
            result = step_skill_batch(
                environment,
                keys[2 * index],
                group_state,
                action[start:stop],
            )
            reset = reset_skill_batch(
                environment,
                keys[2 * index + 1],
                self.envs_per_skill,
            )
            steps.append(result)
            next_states.append(
                _select_completed(result.done, reset.state, result.state)
            )
            next_observations.append(
                _select_completed(
                    result.done,
                    reset.observation,
                    result.observation,
                )
            )

        skill_ids = self.skill_ids
        info = _concatenate_pytrees(*(step.info for step in steps))
        info = {**info, "skill_index": skill_ids}
        return GoalConditionedSkillStep(
            observation=jnp.concatenate(tuple(next_observations), axis=0),
            state=_concatenate_pytrees(*next_states),
            skill_id=skill_ids,
            reward=jnp.concatenate(tuple(step.reward for step in steps), axis=0),
            done=jnp.concatenate(tuple(step.done for step in steps), axis=0),
            info=info,
        )


def make_vectorized_skill_environment(
    *,
    num_envs: int,
    horizon: int = 8,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
    randomization: ScenarioRandomization | None = None,
    num_skills: int = len(SKILL_IDS),
) -> VectorizedSkillEnvironment:
    """Create a balanced batch covering the executor's first ``num_skills``.

    Local navigation applies to skills with a visible target, so there
    ``place_table`` remains a nearby scenario. The randomized scenario covers
    every skill, ``place_table`` included.
    """

    skill_ids = executor_skill_ids(num_skills)
    if num_envs < num_skills or num_envs % num_skills:
        raise ValueError(
            f"num_envs must be divisible by the {num_skills} executor skills."
        )
    normalized_scenario = SkillScenario(scenario)
    environments = tuple(
        make_skill_environment(
            skill_id,
            horizon=horizon,
            scenario=(
                normalized_scenario
                if skill_id in skills_for_scenario(normalized_scenario, skill_ids)
                else SkillScenario.NEARBY
            ),
            min_distance=min_distance,
            max_distance=max_distance,
            distractor_count=distractor_count,
            randomization=(
                randomization
                if normalized_scenario is SkillScenario.RANDOMIZED
                else None
            ),
        )
        for skill_id in skill_ids
    )
    expected_indices = tuple(
        skill_index(environment.goal.skill_id) for environment in environments
    )
    if expected_indices != tuple(range(num_skills)):
        raise AssertionError("Skill environment order does not match skill indices.")
    return VectorizedSkillEnvironment(
        environments=environments,
        envs_per_skill=num_envs // num_skills,
    )


def scenario_randomization_from_config(
    config: SkillPPOConfig,
) -> ScenarioRandomization | None:
    """Return the randomization a training config asks for, if any."""

    if SkillScenario(config.skill_scenario) is not SkillScenario.RANDOMIZED:
        return None
    return ScenarioRandomization(
        reserve_route=config.skill_reserve_route,
        extra_target_count=config.skill_extra_target_count,
        table_probability=config.skill_table_probability,
        max_player_offset=config.skill_max_player_offset,
        clear_probability=config.skill_clear_probability,
        random_inventory=config.skill_random_inventory,
        random_vitals=config.skill_random_vitals,
        max_start_timestep=config.skill_max_start_timestep,
    )
