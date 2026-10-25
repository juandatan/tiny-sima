"""Horizon-bounded reachability used as a necessary affordance condition.

The symbolic preconditions in :mod:`tiny_sima.goals.skills` answer "does this
resource exist anywhere in the world", which is almost always true on a
generated Craftax map and therefore carries no information at a planner
boundary. This module answers the question the executor actually faces: "can
the player stand next to a target and interact with it within the skill
horizon".

Every distance here is a *lower bound* on the number of environment steps the
skill needs from the current state, computed over the tiles the planner can
see. Within that window the bound is exact enough to be trusted; outside it the
answer is unknown, not negative. A target the player must walk toward over
several option boundaries leaves the window, so "unreachable in view" must not
become a veto -- doing so makes the planner refuse skills the executor could
finish, which shows up as spurious ``no_viable_skill`` terminations.

:func:`skill_feasibility_score` therefore grades instead of gating: a target
that can be reached and used inside the skill horizon scores one, an existing
but currently unreachable target scores
``ReachabilityConfig.unknown_feasibility``, and only a failed symbolic
precondition scores zero. Multiplied into a SayCan ranking this demotes
"walk somewhere far away" below "use what is in front of you" without ever
making it unselectable. Executor competence is a separate question, estimated
by the calibrated affordance model in :mod:`tiny_sima.planner.affordance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import OBS_DIM, BlockType

from tiny_sima.goals.skills import (
    SKILL_IDS,
    SkillGoal,
    SkillId,
    skill_precondition,
)

# Blocks the player can occupy. ``SOLID_BLOCKS`` covers everything that blocks
# movement; lava is walkable but lethal, so it is excluded here on purpose.
WALKABLE_BLOCKS: tuple[BlockType, ...] = (
    BlockType.GRASS,
    BlockType.SAND,
    BlockType.PATH,
    BlockType.WOOD,
)

# The player's own tile plus the four orthogonal neighbours reachable by one
# move. ``Action.DO`` and ``Action.PLACE_TABLE`` both apply to the faced tile.
_ORTHOGONAL_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)

# ``craftax.craftax_classic.game_logic.is_near_block`` uses the eight
# surrounding tiles, so crafting does not require facing the table.
_DIAGONAL_OFFSETS: tuple[tuple[int, int], ...] = (
    *_ORTHOGONAL_OFFSETS,
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

OBSERVATION_RADIUS: tuple[int, int] = (OBS_DIM[0] // 2, OBS_DIM[1] // 2)


@dataclass(frozen=True)
class ReachabilityConfig:
    """Static controls for horizon-bounded reachability queries.

    ``view_radius`` restricts reachability to a window centred on the player.
    It defaults to the Craftax observation window so the planner is grounded in
    the same information the renderer shows the VLM and the executor receives
    as input. Set it to ``None`` to search the whole map, which grants the
    planner privileged unobserved state and should only be used as an ablation.

    ``unknown_feasibility`` is the score given to a skill whose target exists
    but cannot be reached within the searched window. It must stay above zero,
    because such a skill may well be achievable after a few steps of
    exploration; setting it to zero recovers a hard gate and is only correct as
    an ablation on how much the soft score matters.
    """

    max_steps: int = 64
    view_radius: tuple[int, int] | None = OBSERVATION_RADIUS
    unknown_feasibility: float = 0.25

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if self.view_radius is not None and any(
            radius < 0 for radius in self.view_radius
        ):
            raise ValueError("view_radius entries cannot be negative.")
        if not 0.0 <= self.unknown_feasibility <= 1.0:
            raise ValueError("unknown_feasibility must lie in [0, 1].")

    @property
    def unreachable(self) -> int:
        """Return the saturating sentinel used for unreachable tiles."""

        return self.max_steps + 1

    @property
    def relaxation_steps(self) -> int:
        """Return how many breadth-first relaxations fully resolve the view."""

        if self.view_radius is None:
            return self.max_steps
        span = (2 * self.view_radius[0] + 1) + (2 * self.view_radius[1] + 1)
        return min(self.max_steps, span)


DEFAULT_REACHABILITY = ReachabilityConfig()


def walkable_mask(state: Any) -> jax.Array:
    """Return which tiles the player may occupy."""

    world_map = jnp.asarray(state.map)
    mask = jnp.zeros(world_map.shape, dtype=jnp.bool_)
    for block in WALKABLE_BLOCKS:
        mask = mask | (world_map == block.value)
    return mask


def view_mask(state: Any, view_radius: tuple[int, int] | None) -> jax.Array:
    """Return which tiles fall inside the planner's visible window."""

    world_map = jnp.asarray(state.map)
    if view_radius is None:
        return jnp.ones(world_map.shape, dtype=jnp.bool_)
    position = jnp.asarray(state.player_position)
    rows = jnp.arange(world_map.shape[0])[:, None]
    columns = jnp.arange(world_map.shape[1])[None, :]
    return (jnp.abs(rows - position[0]) <= view_radius[0]) & (
        jnp.abs(columns - position[1]) <= view_radius[1]
    )


def _player_mask(state: Any) -> jax.Array:
    world_map = jnp.asarray(state.map)
    position = jnp.asarray(state.player_position)
    rows = jnp.arange(world_map.shape[0])[:, None]
    columns = jnp.arange(world_map.shape[1])[None, :]
    return (rows == position[0]) & (columns == position[1])


def travel_distance_map(
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
) -> jax.Array:
    """Return the number of moves needed to stand on each tile.

    Unreachable tiles hold ``config.unreachable``. The player's own tile is
    always zero even when it is not itself walkable.
    """

    return _travel_distance_map(
        _MapView(
            map=jnp.asarray(state.map),
            player_position=jnp.asarray(state.player_position),
        ),
        config,
    )


class _MapView(NamedTuple):
    """The two state fields the search reads, as a pytree ``jit`` can take."""

    map: jax.Array
    player_position: jax.Array


# Compiled once per config. Run eagerly, the loop body is a fresh closure on
# every call, so JAX recompiled it at every planner boundary and the search
# dominated an evaluation that never touched the VLM.
@partial(jax.jit, static_argnames=("config",))
def _travel_distance_map(
    state: _MapView,
    config: ReachabilityConfig,
) -> jax.Array:
    unreachable = config.unreachable
    passable = walkable_mask(state) & view_mask(state, config.view_radius)
    start = _player_mask(state)
    distances = jnp.where(start, 0, unreachable).astype(jnp.int32)

    def relax(_index: Any, current: jax.Array) -> jax.Array:
        padded = jnp.pad(
            current,
            1,
            constant_values=unreachable,
        )
        height, width = current.shape
        neighbour = jnp.full_like(current, unreachable)
        for row_offset, column_offset in _ORTHOGONAL_OFFSETS:
            shifted = jax.lax.dynamic_slice(
                padded,
                (1 + row_offset, 1 + column_offset),
                (height, width),
            )
            neighbour = jnp.minimum(neighbour, shifted)
        candidate = jnp.where(
            passable,
            jnp.minimum(neighbour + 1, unreachable),
            unreachable,
        )
        return jnp.where(start, 0, jnp.minimum(current, candidate))

    return jax.lax.fori_loop(
        0,
        config.relaxation_steps,
        relax,
        distances,
    )


def _adjacent_distance(
    distances: jax.Array,
    *,
    unreachable: int,
    diagonal: bool,
) -> jax.Array:
    """Return the cheapest standing distance next to each tile."""

    padded = jnp.pad(distances, 1, constant_values=unreachable)
    height, width = distances.shape
    offsets = _DIAGONAL_OFFSETS if diagonal else _ORTHOGONAL_OFFSETS
    nearest = jnp.full_like(distances, unreachable)
    for row_offset, column_offset in offsets:
        shifted = jax.lax.dynamic_slice(
            padded,
            (1 + row_offset, 1 + column_offset),
            (height, width),
        )
        nearest = jnp.minimum(nearest, shifted)
    return nearest


def interaction_distance(
    state: Any,
    target_mask: jax.Array,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    diagonal: bool = False,
    distances: jax.Array | None = None,
) -> jax.Array:
    """Return a lower bound on the steps needed to interact with a target.

    The bound is the cheapest distance to a tile adjacent to a target plus one
    step for the interaction itself. Turning to face a target is free when the
    approach already ends in the right direction, so this never overestimates.
    """

    unreachable = config.unreachable
    if distances is None:
        distances = travel_distance_map(state, config)
    nearest = _adjacent_distance(
        distances,
        unreachable=unreachable,
        diagonal=diagonal,
    )
    visible_targets = target_mask & view_mask(state, config.view_radius)
    cost = jnp.where(
        visible_targets,
        jnp.minimum(nearest + 1, unreachable),
        unreachable,
    )
    return jnp.min(cost)


def block_mask(state: Any, block: BlockType) -> jax.Array:
    """Return which tiles hold one block type."""

    return jnp.asarray(state.map) == block.value


def _placement_mask(state: Any) -> jax.Array:
    """Return tiles a crafting table may be placed on.

    ``place_block`` only requires that the faced tile is not solid. Restricting
    this to walkable blocks keeps lava and out-of-bounds tiles out of the bound.
    """

    return walkable_mask(state)


def _skill_target(
    skill_id: SkillId,
    state: Any,
) -> tuple[jax.Array, bool]:
    """Return the interaction target mask and adjacency rule for a skill."""

    if skill_id is SkillId.COLLECT_WOOD:
        return block_mask(state, BlockType.TREE), False
    if skill_id is SkillId.COLLECT_STONE:
        return block_mask(state, BlockType.STONE), False
    if skill_id is SkillId.DRINK_WATER:
        return block_mask(state, BlockType.WATER), False
    if skill_id is SkillId.PLACE_TABLE:
        return _placement_mask(state), False
    if skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
        SkillId.CRAFT_STONE_PICKAXE,
    }:
        return block_mask(state, BlockType.CRAFTING_TABLE), True
    raise AssertionError(f"Unhandled skill: {skill_id}")


def skill_interaction_distance(
    goal: SkillGoal,
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    distances: jax.Array | None = None,
) -> jax.Array:
    """Return a lower bound on the steps one skill needs from this state."""

    target_mask, diagonal = _skill_target(goal.skill_id, state)
    return interaction_distance(
        state,
        target_mask,
        config,
        diagonal=diagonal,
        distances=distances,
    )


def skill_reachable(
    goal: SkillGoal,
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    distances: jax.Array | None = None,
) -> jax.Array:
    """Return whether a skill's target can be reached inside the horizon.

    This is the strict predicate, true only when the lower bound fits in the
    horizon. It is the right question for diagnostics -- "could the executor
    have finished this from here without exploring" -- and the wrong question
    for ranking, which should use :func:`skill_feasibility_score`.
    """

    horizon = min(goal.effective_horizon, config.max_steps)
    bound = skill_interaction_distance(
        goal,
        state,
        config,
        distances=distances,
    )
    return bound <= horizon


def skill_feasible(
    goal: SkillGoal,
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    distances: jax.Array | None = None,
) -> jax.Array:
    """Return whether a skill passes both its precondition and reachability."""

    return skill_precondition(goal, state) & skill_reachable(
        goal,
        state,
        config,
        distances=distances,
    )


def skill_feasibility_score(
    goal: SkillGoal,
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    distances: jax.Array | None = None,
) -> jax.Array:
    """Return a graded affordance in ``[0, 1]`` for one skill.

    Zero means the symbolic precondition fails, which is the only case this
    module treats as genuinely impossible. One means the target is reachable
    and usable within the skill horizon. In between sits
    ``config.unknown_feasibility`` for a target that exists but lies outside
    the reachable window, where exploration may still find it.
    """

    precondition = skill_precondition(goal, state)
    reachable = skill_reachable(goal, state, config, distances=distances)
    score = jnp.where(reachable, 1.0, config.unknown_feasibility)
    return jnp.where(precondition, score, 0.0)


def skill_reachability_report(
    state: Any,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    skill_ids: tuple[SkillId, ...] | None = None,
) -> dict[SkillId, int | None]:
    """Return per-skill interaction distances, sharing one distance map.

    Unreachable skills map to ``None`` so callers never confuse a sentinel with
    a real distance.
    """

    candidates = SKILL_IDS if skill_ids is None else skill_ids
    distances = travel_distance_map(state, config)
    report: dict[SkillId, int | None] = {}
    for skill_id in candidates:
        bound = int(
            jnp.asarray(
                skill_interaction_distance(
                    SkillGoal(skill_id),
                    state,
                    config,
                    distances=distances,
                )
            ).item()
        )
        report[skill_id] = None if bound >= config.unreachable else bound
    return report
