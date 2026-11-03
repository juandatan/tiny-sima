"""Controlled fights: how well the executor attacks a creature, and at what cost.

The open-ended runs die mostly inside ``defeat_zombie``, but a closed-loop
trace cannot say whether a fight was lost to the odds or to the executor. This
module stages the fight instead. Each cell fixes the distance to the creature,
the player's health, the sword held, one creature or two, and day or night, in
an open arena carved around the player, so only the fight varies. The learned
executor and a scripted chase-and-hit reference run on the same worlds and
environment keys through the batched probe loop, which also records health and
the creatures' positions after every step.

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
from functools import cache, partial
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType
from craftax.craftax_classic.game_logic import calculate_light_level

from tiny_sima.envs.chains import clear_mobs, make_skill_chain_environment
from tiny_sima.envs.skills import scenario_vitals
from tiny_sima.evaluation.chains import (
    BatchedProbeOutput,
    LearnedSkillExecutor,
    run_skill_probe,
    target_mob_slots,
)
from tiny_sima.goals import (
    MAX_VITAL,
    MOB_TYPES,
    SKILL_MOBS,
    SkillId,
    mob_instances,
    skill_index,
)
from tiny_sima.planner.saycan import SkillOutcome

FIGHT_SKILLS: tuple[SkillId, ...] = (
    SkillId.DEFEAT_ZOMBIE,
    SkillId.DEFEAT_SKELETON,
    SkillId.EAT_COW,
)
SWORDS: tuple[str, ...] = ("none", "wood", "stone")
FIGHT_BENCHMARK_SEED = 211
# The arena is the square of this radius around the player, all one ground
# block, so nothing but the creatures stands between them and the player.
ARENA_RADIUS = 4
# A second creature stands anywhere within this many steps of the player.
SECOND_MOB_MAX_DISTANCE = 3
# Night runs from timestep 148 to 272 of each 300-step day, so a fight that
# starts here stays dark for 112 steps.
NIGHT_TIMESTEP = 160
# Skeletons walk only on path; the others are placed on grass, as in training.
_MOB_GROUND: dict[str, BlockType] = {
    "zombie": BlockType.GRASS,
    "cow": BlockType.GRASS,
    "skeleton": BlockType.PATH,
}
_OFFSETS = np.asarray(
    [
        (row, column)
        for row in range(-ARENA_RADIUS, ARENA_RADIUS + 1)
        for column in range(-ARENA_RADIUS, ARENA_RADIUS + 1)
        if 1 <= abs(row) + abs(column) <= ARENA_RADIUS
    ],
    dtype=np.int32,
)
_MOVES = np.asarray(
    [Action.LEFT.value, Action.RIGHT.value, Action.UP.value, Action.DOWN.value]
)
# The probe's outcome codes index this tuple.
_OUTCOMES = tuple(SkillOutcome)


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

    cells = tuple(
        FightCell(SkillId(skill_id), *setting)
        for skill_id, *setting in product(
            skill_ids, distances, healths, swords, mob_counts, nights
        )
    )
    for cell in cells:
        if cell.skill_id not in SKILL_MOBS:
            raise ValueError(f"{cell.skill_id.value} attacks no creature.")
        if not 1 <= cell.distance <= ARENA_RADIUS:
            raise ValueError(f"Distance {cell.distance} is outside the arena.")
        if not 1 <= cell.health <= MAX_VITAL:
            raise ValueError(f"Health {cell.health} is not a player health.")
        if cell.sword not in SWORDS:
            raise ValueError(f"Unknown sword {cell.sword!r}.")
        if cell.mobs not in (1, 2):
            raise ValueError("A fight stages one creature or two.")
    return cells


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
    eating is possible but nothing runs low.
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
    others = (steps_away <= SECOND_MOB_MAX_DISTANCE) & (
        jnp.arange(offsets.shape[0]) != first
    )
    second = jax.random.choice(second_rng, offsets.shape[0], p=others / jnp.sum(others))
    placed = position + offsets[jnp.stack([first, second])]
    live = jnp.stack([jnp.asarray(True), mob_count >= 2])

    state = clear_mobs(state)
    mobs = mob_instances(mob, state)
    full_health = getattr(params, f"{mob}_health")
    mobs = mobs.replace(
        position=mobs.position.at[:2].set(placed),
        health=mobs.health.at[:2].set(jnp.where(live, full_health, 0)),
        mask=mobs.mask.at[:2].set(live),
        attack_cooldown=mobs.attack_cooldown.at[:2].set(0),
    )
    mob_map = state.mob_map.at[placed[0, 0], placed[0, 1]].set(True)
    mob_map = mob_map.at[placed[1, 0], placed[1, 1]].set(
        mob_map[placed[1, 0], placed[1, 1]] | live[1]
    )
    timestep = jnp.where(night, NIGHT_TIMESTEP, 0)
    inventory = state.inventory.replace(
        wood_sword=(sword == SWORDS.index("wood")).astype(jnp.int32),
        stone_sword=(sword == SWORDS.index("stone")).astype(jnp.int32),
    )
    return state.replace(
        map=world,
        mob_map=mob_map,
        inventory=inventory,
        player_health=health,
        player_food=food,
        player_drink=drink,
        player_energy=energy,
        is_sleeping=False,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        achievements=jnp.zeros_like(state.achievements),
        timestep=timestep,
        light_level=calculate_light_level(timestep, params),
        **{MOB_TYPES[mob]: mobs},
    )


@cache
def _world_stager(skill_id: SkillId) -> Any:
    """Compile, once per skill, a batched reset that stages each world."""

    bundle = make_skill_chain_environment().bundle

    def stage(keys, distance, health, sword, mobs, night):
        def one(key, *setting):
            reset_key, stage_key = jax.random.split(key)
            _, state = bundle.env.reset(reset_key, bundle.params)
            return _fight_world(
                state, stage_key, *setting, skill_id=skill_id, params=bundle.params
            )

        return jax.vmap(one)(keys, distance, health, sword, mobs, night)

    return jax.jit(stage)


def _stage_worlds(entries: Sequence[FightCell], keys: jax.Array) -> Any:
    skills = {entry.skill_id for entry in entries}
    if len(skills) != 1:
        raise ValueError("Stage one skill's cells at a time.")

    def column(values: Iterable[Any], dtype: Any) -> jax.Array:
        return jnp.asarray(list(values), dtype=dtype)

    return _world_stager(skills.pop())(
        keys,
        column((entry.distance for entry in entries), jnp.int32),
        column((entry.health for entry in entries), jnp.int32),
        column((SWORDS.index(entry.sword) for entry in entries), jnp.int32),
        column((entry.mobs for entry in entries), jnp.int32),
        column((entry.night for entry in entries), bool),
    )


def fight_worlds(entries: Sequence[FightCell], rng: jax.Array) -> Any:
    """Return one staged world per entry, as a stacked pytree.

    Every entry gets its own map, so repeat a cell to roll it out more than
    once. All entries must share one skill, since the creature placed is a
    static choice.
    """

    return _stage_worlds(entries, jax.random.split(rng, len(entries)))


def _chase_action(skill_id: jax.Array, state: Any) -> jax.Array:
    """Hit the nearest target creature if facing it, else step toward it.

    Stepping into an occupied tile only turns the player, so a creature
    beside the player is faced before it is hit.
    """

    positions, mask = target_mob_slots(skill_id, state)
    offset = positions - jnp.asarray(state.player_position, dtype=jnp.int32)
    steps_away = jnp.where(mask, jnp.abs(offset).sum(axis=1), jnp.iinfo(jnp.int32).max)
    nearest = jnp.argmin(steps_away)
    row, column = offset[nearest, 0], offset[nearest, 1]
    horizontal = (column != 0) & (jnp.abs(column) >= jnp.abs(row))
    move = jnp.where(
        horizontal,
        jnp.where(column < 0, Action.LEFT.value, Action.RIGHT.value),
        jnp.where(row < 0, Action.UP.value, Action.DOWN.value),
    )
    facing = jnp.all(jnp.asarray(DIRECTIONS)[state.player_direction] == offset[nearest])
    action = jnp.where(facing, Action.DO.value, move)
    return jnp.where(jnp.any(mask), action, Action.NOOP.value).astype(jnp.int32)


def chase_policy(skill_ids: jax.Array, states: Any) -> jax.Array:
    """The scripted reference: chase the nearest target creature and hit it."""

    return jax.vmap(_chase_action)(skill_ids, states)


def fight_rollout_rows(
    entries: Sequence[FightCell],
    states: Any,
    output: BatchedProbeOutput,
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Describe each staged rollout: how it ended and how the player fought.

    ``entries`` names the cell of each staged world; worlds past the last
    entry are padding and get no row.

    A step is *adjacent* when a live target creature stands one step from the
    player before the action, and *facing* when one stands on the tile the
    player faces. A *hit* is a DO while facing; a *miss* a DO while not; a
    *retreat* a step that moves the player while adjacent. Health lost is
    start minus end, so regeneration offsets it; ``damage_taken`` sums only
    the drops. A creature is unreachable from a dead player, so the probe
    calls a death a lost precondition; the row's outcome says ``died``.
    """

    initial = jax.device_get(states)
    skill = entries[0].skill_id
    initial_mobs = jax.device_get(
        jax.vmap(partial(target_mob_slots, jnp.asarray(skill_index(skill))))(states)
    )
    horizon = output.actions.shape[0]
    # Row t of each "before" array is the world before action t.
    player = np.concatenate(
        [np.asarray(initial.player_position)[None], output.positions[:-1]]
    )
    direction = np.concatenate(
        [np.asarray(initial.player_direction)[None], output.directions[:-1]]
    )
    mob_positions = np.concatenate(
        [np.asarray(initial_mobs[0])[None], output.mob_positions[:-1]]
    )
    mob_mask = np.concatenate([np.asarray(initial_mobs[1])[None], output.mob_mask[:-1]])
    health = np.concatenate(
        [np.asarray(initial.player_health, dtype=np.int32)[None], output.health]
    )

    valid = np.arange(horizon)[:, None] < output.steps[None, :]
    steps_away = np.abs(mob_positions - player[:, :, None, :]).sum(axis=-1)
    adjacent = valid & np.any(mob_mask & (steps_away == 1), axis=-1)
    front = player + np.asarray(DIRECTIONS)[direction]
    facing = valid & np.any(
        mob_mask & np.all(mob_positions == front[:, :, None, :], axis=-1), axis=-1
    )
    swing = valid & (output.actions == Action.DO.value)
    hits = swing & facing
    moved = valid & np.any(output.positions != player, axis=-1)
    retreats = adjacent & moved & np.isin(output.actions, _MOVES)
    drops = np.maximum(health[:-1] - health[1:], 0) * valid

    rows = []
    for index, cell in enumerate(entries):
        steps = int(output.steps[index])
        final_health = int(health[steps, index])
        hit_steps = np.flatnonzero(hits[:, index])
        outcome = _OUTCOMES[int(output.outcome[index])]
        died = final_health <= 0 and outcome is not SkillOutcome.SUCCESS
        rows.append(
            {
                **cell.as_row(),
                "policy": policy,
                "outcome": "died" if died else outcome.value,
                "steps": steps,
                "final_health": final_health,
                "died": died,
                "health_lost": cell.health - final_health,
                "damage_taken": int(drops[:, index].sum()),
                "adjacent_steps": int(adjacent[:, index].sum()),
                "facing_steps": int(facing[:, index].sum()),
                "hits": int(hits[:, index].sum()),
                "misses": int((swing & ~facing)[:, index].sum()),
                "retreats": int(retreats[:, index].sum()),
                "first_hit_step": int(hit_steps[0]) if hit_steps.size else None,
            }
        )
    return rows


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
    with copies of its first world, so every chunk reuses one compilation.
    Each world is staged from its own key, split from the skill's key, so
    the worlds do not depend on the batch size, and adding a skill leaves the
    others' worlds unchanged. Both policies face the same worlds and
    environment keys.
    """

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
            chunk_entries = entries[start : start + size]
            padding = size - len(chunk_entries)
            states = _stage_worlds(
                chunk_entries + [chunk_entries[0]] * padding,
                jnp.concatenate(
                    [
                        keys[start : start + size],
                        jnp.repeat(keys[start : start + 1], padding, axis=0),
                    ]
                ),
            )
            for name, policy in (("executor", None), ("chase", chase_policy)):
                output = run_skill_probe(
                    executor,
                    states,
                    [skill_id] * size,
                    jax.random.fold_in(rollout_rng, chunk),
                    skill_horizon=skill_horizon,
                    policy=policy,
                )
                rows.extend(
                    fight_rollout_rows(chunk_entries, states, output, policy=name)
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
