"""Controlled fights: how well the executor attacks a creature, and at what cost.

The open-ended runs die mostly inside ``defeat_zombie``, but a closed-loop
trace cannot say whether a fight was lost to the odds or to the executor. This
module stages the fight instead. Each cell fixes the distance to the creature,
the player's health, the sword held, one creature or two, and day or night, in
an open arena carved around the player, so only the fight varies. The learned
executor and a scripted chase-and-hit reference run on the same worlds and
environment keys through the batched probe loop, which records the player and
the creatures before the first step and after every step.

:func:`fight_rollout_rows` turns those traces into one row per rollout: the
outcome, the health it cost, and what the player did while next to the
creature -- whether it faced it, swung at nothing, or walked away.
:func:`summarize_fights` groups rows by any cell fields; grouped by health
and sword it is the win-rate table a feasibility term for fights needs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from itertools import product
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.game_logic import calculate_light_level

from tiny_sima.envs.chains import clear_mobs, make_skill_chain_environment, reset_player
from tiny_sima.envs.skills import add_mob, direction_to_target, scenario_vitals
from tiny_sima.evaluation.chains import (
    PROBE_OUTCOMES,
    BatchedProbeOutput,
    LearnedSkillExecutor,
    run_skill_probe,
)
from tiny_sima.goals import (
    MAX_VITAL,
    SKILL_MOBS,
    SkillId,
    skill_index,
    target_mob_slots,
)
from tiny_sima.planner.observation import NIGHT_LIGHT_THRESHOLD
from tiny_sima.planner.saycan import SkillOutcome

FIGHT_SKILLS: tuple[SkillId, ...] = (
    SkillId.DEFEAT_ZOMBIE,
    SkillId.DEFEAT_SKELETON,
    SkillId.EAT_COW,
)
SWORDS: tuple[str, ...] = ("none", "wood", "stone")
FIGHT_BENCHMARK_SEED = 211
# Creatures start at most this many steps from the player.
MAX_DISTANCE = 4
# The arena is the square of this radius around the player, all one ground
# block: wider than the view, so nothing but the creatures stands between
# them and the player while they close in or wander.
ARENA_RADIUS = 8
# A second creature stands no nearer than the first, and within this many
# steps of the player, or as far as the first if that is farther.
SECOND_MOB_MAX_DISTANCE = 3
# A night fight starts this many steps after dark, as Craftax and the planner
# define it, so it stays dark for the longest horizon the notebooks use.
_DARK = np.flatnonzero(
    np.asarray(calculate_light_level(np.arange(EnvParams().day_length), EnvParams()))
    < NIGHT_LIGHT_THRESHOLD
)
NIGHT_TIMESTEP = int(_DARK[0]) + 12
# Skeletons walk only on path; the others are placed on grass, as in training.
_MOB_GROUND: dict[str, BlockType] = {
    "zombie": BlockType.GRASS,
    "cow": BlockType.GRASS,
    "skeleton": BlockType.PATH,
}
_OFFSETS = np.asarray(
    [
        (row, column)
        for row in range(-MAX_DISTANCE, MAX_DISTANCE + 1)
        for column in range(-MAX_DISTANCE, MAX_DISTANCE + 1)
        if 1 <= abs(row) + abs(column) <= MAX_DISTANCE
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class FightCell:
    """One controlled fight setting."""

    skill_id: SkillId
    distance: int
    health: int
    sword: str
    mobs: int
    night: bool

    def as_row(self) -> dict[str, Any]:
        return {**asdict(self), "skill_id": self.skill_id.value}


def fight_cells(
    skill_ids: Iterable[SkillId] = FIGHT_SKILLS,
    *,
    distances: Iterable[int] = (1, 2, 3),
    healths: Iterable[int] = range(2, MAX_VITAL + 1),
    swords: Iterable[str] = SWORDS,
    mob_counts: Iterable[int] = (1, 2),
    nights: Iterable[bool] = (False, True),
) -> tuple[FightCell, ...]:
    """Return every combination of the given settings, skill-major."""

    skill_ids = tuple(SkillId(skill_id) for skill_id in skill_ids)
    distances, healths, swords = tuple(distances), tuple(healths), tuple(swords)
    mob_counts, nights = tuple(mob_counts), tuple(nights)
    for skill_id in skill_ids:
        if skill_id not in SKILL_MOBS:
            raise ValueError(f"{skill_id.value} attacks no creature.")
    for distance in distances:
        if not 1 <= distance <= MAX_DISTANCE:
            raise ValueError(f"Distance {distance} is outside the arena.")
    for health in healths:
        if not 1 <= health <= MAX_VITAL:
            raise ValueError(f"Health {health} is not a player health.")
    for sword in swords:
        if sword not in SWORDS:
            raise ValueError(f"Unknown sword {sword!r}.")
    if not set(mob_counts) <= {1, 2}:
        raise ValueError("A fight stages one creature or two.")
    return tuple(
        FightCell(*setting)
        for setting in product(
            skill_ids, distances, healths, swords, mob_counts, nights
        )
    )


def _fight_world(
    state: Any,
    rng: jax.Array,
    distance: jax.Array,
    health: jax.Array,
    sword: jax.Array,
    mob_count: jax.Array,
    night: jax.Array,
    *,
    skill_id: SkillId,
    params: Any,
) -> Any:
    """Stage one fight in ``state``: carve the arena, then place the creatures.

    Food, drink and energy start as in the skill's training scenario, so
    eating is possible but nothing runs low; by day the light is full, as
    in training.
    """

    mob = SKILL_MOBS[skill_id]
    food, drink, energy = scenario_vitals(skill_id)

    side = 2 * ARENA_RADIUS + 1
    position = jnp.asarray(state.player_position, dtype=jnp.int32)
    world = jax.lax.dynamic_update_slice(
        state.map,
        jnp.full((side, side), _MOB_GROUND[mob].value, dtype=state.map.dtype),
        position - ARENA_RADIUS,
    )

    first_rng, second_rng = jax.random.split(rng)
    offsets = jnp.asarray(_OFFSETS)
    steps_away = jnp.abs(offsets).sum(axis=1)
    first = jax.random.choice(
        first_rng,
        offsets.shape[0],
        p=(steps_away == distance) / jnp.sum(steps_away == distance),
    )
    others = (
        (steps_away >= distance)
        & (steps_away <= jnp.maximum(distance, SECOND_MOB_MAX_DISTANCE))
        & (jnp.arange(offsets.shape[0]) != first)
    )
    second = jax.random.choice(second_rng, offsets.shape[0], p=others / jnp.sum(others))

    state = clear_mobs(state.replace(map=world))
    state = add_mob(state, mob, position + offsets[first])
    state = add_mob(state, mob, position + offsets[second], slot=1, live=mob_count >= 2)
    timestep = jnp.where(night, NIGHT_TIMESTEP, 0)
    return reset_player(
        state.replace(
            inventory=state.inventory.replace(
                wood_sword=(sword == SWORDS.index("wood")).astype(jnp.int32),
                stone_sword=(sword == SWORDS.index("stone")).astype(jnp.int32),
            )
        ),
        health=health,
        food=food,
        drink=drink,
        energy=energy,
        timestep=timestep,
        light_level=jnp.where(night, calculate_light_level(timestep, params), 1.0),
    )


@cache
def _world_stager(skill_id: SkillId) -> Any:
    """Compile, once per skill, a batched reset that stages each world."""

    bundle = make_skill_chain_environment().bundle

    def stage(key: jax.Array, *setting: jax.Array) -> Any:
        reset_key, stage_key = jax.random.split(key)
        _, state = bundle.env.reset(reset_key, bundle.params)
        return _fight_world(
            state, stage_key, *setting, skill_id=skill_id, params=bundle.params
        )

    return jax.jit(jax.vmap(stage))


def fight_worlds(entries: Sequence[FightCell], keys: jax.Array) -> Any:
    """Stage one world per entry from its own key, as a stacked pytree.

    Repeat a cell to roll it out more than once. All entries must share one
    skill, since the creature placed is a static choice.
    """

    skills = {entry.skill_id for entry in entries}
    if len(skills) != 1:
        raise ValueError("Stage one skill's cells at a time.")
    if len(keys) != len(entries):
        raise ValueError("Stage each entry from its own key.")
    return _world_stager(skills.pop())(
        keys,
        jnp.asarray([entry.distance for entry in entries], dtype=jnp.int32),
        jnp.asarray([entry.health for entry in entries], dtype=jnp.int32),
        jnp.asarray([SWORDS.index(entry.sword) for entry in entries], dtype=jnp.int32),
        jnp.asarray([entry.mobs for entry in entries], dtype=jnp.int32),
        jnp.asarray([entry.night for entry in entries], dtype=bool),
    )


def _chase_action(skill_id: jax.Array, state: Any) -> jax.Array:
    """Hit a target creature if facing one, else step toward the nearest.

    Stepping into an occupied tile only turns the player, so a creature
    beside the player is faced before it is hit.
    """

    positions, mask = target_mob_slots(skill_id, state)
    offset = positions - jnp.asarray(state.player_position, dtype=jnp.int32)
    front = jnp.asarray(DIRECTIONS)[state.player_direction]
    facing = jnp.any(mask & jnp.all(offset == front, axis=1))
    steps_away = jnp.where(mask, jnp.abs(offset).sum(axis=1), jnp.iinfo(jnp.int32).max)
    move = direction_to_target(offset[jnp.argmin(steps_away)])
    action = jnp.where(facing, Action.DO.value, move)
    return jnp.where(jnp.any(mask), action, Action.NOOP.value).astype(jnp.int32)


def chase_policy(skill_ids: jax.Array, states: Any) -> jax.Array:
    """The scripted reference: hit a target creature, or chase the nearest."""

    return jax.vmap(_chase_action)(skill_ids, states)


class FightSnapshot(NamedTuple):
    """The player and the target creatures, as the probe records them."""

    position: jax.Array
    direction: jax.Array
    health: jax.Array
    mob_positions: jax.Array
    mob_mask: jax.Array


def fight_snapshot(skill_ids: jax.Array, states: Any) -> FightSnapshot:
    """Read a batch of fights: the probe recorder the benchmark runs with."""

    mob_positions, mob_mask = jax.vmap(target_mob_slots)(skill_ids, states)
    return FightSnapshot(
        position=jnp.asarray(states.player_position, dtype=jnp.int32),
        direction=jnp.asarray(states.player_direction, dtype=jnp.int32),
        health=jnp.asarray(states.player_health, dtype=jnp.int32),
        mob_positions=mob_positions,
        mob_mask=mob_mask,
    )


def _steps_apart(mobs: np.ndarray, player: np.ndarray) -> np.ndarray:
    return np.abs(mobs - player[..., None, :]).sum(axis=-1)


def fight_rollout_rows(
    entries: Sequence[FightCell],
    output: BatchedProbeOutput,
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Describe each probed fight: how it ended and how the player fought.

    ``output`` must carry a :func:`fight_snapshot` record, and ``entries``
    names the cell of each probed world; worlds past the last entry are
    padding and get no row. A step is *adjacent* when a live target creature
    stands one step from the player before the action, and *facing* when one
    stands on the tile the player faces. A *hit* is a DO while facing; a
    *miss* a DO while not; a *retreat* a step from beside a creature to a
    tile beside none. Health lost is start minus end, so regeneration
    offsets it; ``damage_taken`` sums only the drops. ``outcome`` is the
    probe's, so a death reads as a lost precondition, since every skill
    needs the player alive; ``died`` says so, as a failed skill's cause
    would, and a kill that costs the last health still counts as a win.
    """

    count = len(entries)
    record = jax.tree_util.tree_map(lambda leaf: leaf[:, :count], output.record)
    actions, steps = output.actions[:, :count], output.steps[:count]
    # Row t of "before" is the world before action t, of "after" after it.
    player, after = record.position[:-1], record.position[1:]
    mobs, mask = record.mob_positions[:-1], record.mob_mask[:-1]

    valid = np.arange(actions.shape[0])[:, None] < steps[None, :]
    beside = mask & (_steps_apart(mobs, player) == 1)
    adjacent = valid & beside.any(axis=-1)
    front = player + np.asarray(DIRECTIONS)[record.direction[:-1]]
    facing = valid & np.any(
        mask & np.all(mobs == front[..., None, :], axis=-1), axis=-1
    )
    swing = valid & (actions == Action.DO.value)
    hits = swing & facing
    moved = np.any(after != player, axis=-1)
    still_beside = np.any(mask & (_steps_apart(mobs, after) <= 1), axis=-1)
    retreats = adjacent & moved & ~still_beside
    drops = np.maximum(record.health[:-1] - record.health[1:], 0) * valid

    final_health = record.health[steps, np.arange(count)]
    outcomes = output.outcome[:count]
    died = (final_health <= 0) & (
        outcomes != PROBE_OUTCOMES.index(SkillOutcome.SUCCESS)
    )
    columns = zip(
        entries,
        outcomes.tolist(),
        died.tolist(),
        steps.tolist(),
        final_health.tolist(),
        drops.sum(axis=0).tolist(),
        adjacent.sum(axis=0).tolist(),
        facing.sum(axis=0).tolist(),
        hits.sum(axis=0).tolist(),
        (swing & ~facing).sum(axis=0).tolist(),
        retreats.sum(axis=0).tolist(),
        np.where(hits.any(axis=0), hits.argmax(axis=0), -1).tolist(),
        strict=True,
    )
    return [
        {
            **cell.as_row(),
            "policy": policy,
            "outcome": PROBE_OUTCOMES[outcome].value,
            "steps": rollout_steps,
            "final_health": health,
            "died": death,
            "health_lost": cell.health - health,
            "damage_taken": damage,
            "adjacent_steps": adjacent_steps,
            "facing_steps": facing_steps,
            "hits": hit_count,
            "misses": miss_count,
            "retreats": retreat_count,
            "first_hit_step": None if first_hit < 0 else first_hit,
        }
        for (
            cell,
            outcome,
            death,
            rollout_steps,
            health,
            damage,
            adjacent_steps,
            facing_steps,
            hit_count,
            miss_count,
            retreat_count,
            first_hit,
        ) in columns
    ]


def benchmark_fights(
    executor: LearnedSkillExecutor,
    cells: Sequence[FightCell] | None = None,
    *,
    rollouts_per_cell: int = 50,
    seed: int = FIGHT_BENCHMARK_SEED,
    skill_horizon: int = 96,
    batch_size: int = 1024,
) -> list[dict[str, Any]]:
    """Run the executor and the chase reference on every cell's worlds.

    Worlds are staged, run and summarized ``batch_size`` at a time, so
    memory stays flat however many cells there are; the last chunk is padded
    with copies of its last world, so every chunk reuses one compilation.
    Each world is staged from its own key, split from the skill's key, so
    the worlds do not depend on the batch size, and adding a skill leaves the
    others' worlds unchanged. Both policies face the same worlds and
    environment keys.
    """

    if rollouts_per_cell < 1:
        raise ValueError("rollouts_per_cell must be positive.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    cells = fight_cells() if cells is None else tuple(cells)
    rows: list[dict[str, Any]] = []
    for skill_id in dict.fromkeys(cell.skill_id for cell in cells):
        entries = [
            cell
            for cell in cells
            if cell.skill_id is skill_id
            for _ in range(rollouts_per_cell)
        ]
        skill_rng = jax.random.fold_in(jax.random.PRNGKey(seed), skill_index(skill_id))
        world_rng, rollout_rng = jax.random.split(skill_rng)
        keys = jax.random.split(world_rng, len(entries))
        size = min(batch_size, len(entries))
        for chunk, start in enumerate(range(0, len(entries), size)):
            index = np.minimum(np.arange(start, start + size), len(entries) - 1)
            states = fight_worlds([entries[i] for i in index], keys[index])
            for name, policy in (("executor", None), ("chase", chase_policy)):
                output = run_skill_probe(
                    executor,
                    states,
                    [skill_id] * size,
                    jax.random.fold_in(rollout_rng, chunk),
                    skill_horizon=skill_horizon,
                    policy=policy,
                    recorder=fight_snapshot,
                )
                rows.extend(
                    fight_rollout_rows(
                        entries[start : start + size], output, policy=name
                    )
                )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_fights(
    rows: Sequence[dict[str, Any]],
    by: Sequence[str] = ("skill_id", "policy"),
) -> list[dict[str, Any]]:
    """Group rollout rows by the ``by`` fields and summarize each group.

    Per-win costs are averaged over won fights only, and the first hit
    over rollouts that landed one; everything else is over the whole group.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in by)].append(row)
    summaries = []
    for key in sorted(groups):
        group = groups[key]
        wins = [row for row in group if row["outcome"] == SkillOutcome.SUCCESS.value]
        adjacent = sum(row["adjacent_steps"] for row in group)
        steps = sum(row["steps"] for row in group)
        first_hits = [
            row["first_hit_step"] for row in group if row["first_hit_step"] is not None
        ]
        summaries.append(
            {
                **dict(zip(by, key, strict=True)),
                "count": len(group),
                "win_rate": len(wins) / len(group),
                "death_rate": sum(row["died"] for row in group) / len(group),
                "outcomes": dict(Counter(row["outcome"] for row in group)),
                "win_steps": _mean([row["steps"] for row in wins]),
                "win_health_lost": _mean([row["health_lost"] for row in wins]),
                "damage_taken": _mean([row["damage_taken"] for row in group]),
                "adjacent_share": adjacent / steps if steps else None,
                "facing_when_adjacent": (
                    sum(row["facing_steps"] for row in group) / adjacent
                    if adjacent
                    else None
                ),
                "hits": _mean([row["hits"] for row in group]),
                "misses": _mean([row["misses"] for row in group]),
                "retreats": _mean([row["retreats"] for row in group]),
                "first_hit_step": _mean(first_hits),
            }
        )
    return summaries
