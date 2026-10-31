"""What one episode has seen and done, kept for the planner.

Three stores, each read by the part that uses it reliably:

- **Transitions:** the inventory before and after every skill request, with a
  failure cause, for the competence prior and for learning the skills'
  mechanics from experience.
- **Block sightings:** where resources, stations and plants have been seen.
  :meth:`EpisodeMemory.world_fact` answers the planner's world facts from them
  instead of from a scan of the whole map, which the agent has never seen.
- **Mob sightings:** where zombies, skeletons and cows were last seen, and
  when. Mobs move, so these are forgotten after a few boundaries.

Memory only observes at skill boundaries. The executor's intermediate frames
are not returned to the planner, so a resource passed mid-skill and out of view
again by the boundary is never recorded.

Every update returns a new :class:`EpisodeMemory`, so a memory can be stored in
a frozen planner context and shared between conditions without copying.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from craftax.craftax_classic.constants import BlockType

from tiny_sima.goals import OBSERVATION_RADIUS, SkillId, get_skill_spec, view_mask

if TYPE_CHECKING:
    from tiny_sima.planner.saycan import SkillOutcome

Position = tuple[int, int]
Readings = tuple[tuple[str, int], ...]
Extremes = tuple[tuple[str, int, int], ...]

# Every block worth remembering in Craftax-Classic. The first four are the
# ones the skills use, and the prompt always reports them.
RESOURCE_BLOCKS: dict[str, BlockType] = {
    "tree": BlockType.TREE,
    "stone": BlockType.STONE,
    "water": BlockType.WATER,
    "crafting_table": BlockType.CRAFTING_TABLE,
    "coal": BlockType.COAL,
    "iron": BlockType.IRON,
    "diamond": BlockType.DIAMOND,
    "sand": BlockType.SAND,
    "lava": BlockType.LAVA,
    "furnace": BlockType.FURNACE,
    "plant": BlockType.PLANT,
    "ripe_plant": BlockType.RIPE_PLANT,
}

# Blocks only the agent creates. A generated map holds none, so not having
# seen one is evidence there is none to use.
PLACED_BLOCKS = frozenset(("crafting_table", "furnace", "plant", "ripe_plant"))

PROMPT_BLOCKS = ("tree", "stone", "water", "crafting_table")

# Mob type to the state field holding its instances.
MOB_TYPES: dict[str, str] = {
    "zombie": "zombies",
    "skeleton": "skeletons",
    "cow": "cows",
}

# The skills that search for a resource. Failing one while no instance of the
# resource is known is the evidence that rules the resource out.
COLLECT_RESOURCES: dict[SkillId, str] = {
    SkillId.COLLECT_WOOD: "tree",
    SkillId.COLLECT_STONE: "stone",
    SkillId.DRINK_WATER: "water",
    SkillId.COLLECT_COAL: "coal",
}

_WORLD_FACTS = {f"world.{block}_reachable": block for block in RESOURCE_BLOCKS}


class FailureCause(str, Enum):
    """Why a skill request failed, as far as the boundary states tell.

    :func:`classify_failure` tries the causes in this order and returns the
    first that applies, so the state of the player explains a failure before
    the target does.
    """

    DIED = "died"
    EPISODE_ENDED = "episode_ended"
    FOOD_DEPLETED = "food_depleted"
    DRINK_DEPLETED = "drink_depleted"
    ENERGY_DEPLETED = "energy_depleted"
    HEALTH_LOW = "health_low"
    INFEASIBLE = "infeasible"
    PRECONDITION_LOST = "precondition_lost"
    TARGET_NOT_SEEN = "target_not_seen"
    TARGET_NOT_REACHED = "target_not_reached"
    TIMEOUT = "timeout"


_VITAL_CAUSES = (
    ("food", FailureCause.FOOD_DEPLETED),
    ("drink", FailureCause.DRINK_DEPLETED),
    ("energy", FailureCause.ENERGY_DEPLETED),
)


@dataclass(frozen=True)
class TransitionRecord:
    """One skill request: the readings before and after, and its outcome.

    ``cause`` is the :class:`FailureCause` value, ``None`` for a success.
    ``inventory_extremes`` holds ``(field, min, max)`` per inventory field
    over the skill's steps, when the executor reports them.
    """

    decision_index: int
    skill_id: SkillId
    outcome: str
    steps: int
    before: Readings
    after: Readings
    cause: str | None = None
    inventory_extremes: Extremes | None = None


@dataclass(frozen=True)
class MobSighting:
    """Where one mob was seen, at which boundary and environment timestep.

    ``boundary`` counts the transitions recorded when it was seen, so the
    reset is boundary 0 and the state after the first skill boundary 1.
    """

    position: Position
    boundary: int
    timestep: int


def readings(state: Any) -> Readings:
    """Return the nonzero inventory counts and the drink level, by field name.

    Field names are the skill registry's, so a transition can be compared
    against declared requirements and effects directly.
    """

    inventory = tuple(
        (f"inventory.{field.name}", value)
        for field in dataclasses.fields(state.inventory)
        if (value := int(np.asarray(getattr(state.inventory, field.name)))) != 0
    )
    return (*inventory, ("survival.drink", int(np.asarray(state.player_drink))))


def skill_target(skill_id: SkillId | str) -> str | None:
    """Return the tracked block a skill must reach, from its registry entry.

    It is the block named by the skill's ``world.<block>_reachable``
    requirement, so the crafting skills target the crafting table. ``None``
    for a skill without one, such as placing a table.
    """

    for requirement in get_skill_spec(skill_id).mechanics.requires:
        block = _WORLD_FACTS.get(requirement.field)
        if block is not None:
            return block
    return None


def _outcome_value(outcome: Any) -> str:
    return str(getattr(outcome, "value", outcome))


def _player(state: Any) -> Position:
    row, column = np.asarray(state.player_position).tolist()
    return int(row), int(column)


def _vital(state: Any, name: str) -> int:
    return int(np.asarray(getattr(state, f"player_{name}")))


def _died(state: Any) -> bool:
    row, column = _player(state)
    return (
        _vital(state, "health") <= 0
        or int(np.asarray(state.map)[row, column]) == BlockType.LAVA.value
    )


def classify_failure(
    pre_state: Any,
    skill_id: SkillId | str,
    outcome: SkillOutcome | str,
    steps: int,
    post_state: Any,
    memory: EpisodeMemory,
    *,
    low_health: int = 3,
) -> FailureCause | None:
    """Return why one skill request failed, or ``None`` if it succeeded.

    The first cause that applies wins:

    1. ``died``: health is gone, or the player stands in lava, after it.
    2. ``episode_ended``: the environment ended the episode otherwise.
    3. ``food_depleted``, ``drink_depleted``, ``energy_depleted``: the vital
       ran out during the request, in that order.
    4. ``health_low``: health fell during the request to ``low_health`` or
       below, with no vital run out to explain it.
    5. ``infeasible`` and ``precondition_lost``, from the outcome.
    6. For a timeout, ``target_not_seen`` when no instance of the skill's
       target block was known before the request, ``target_not_reached``
       when one was, and ``timeout`` for a skill without a target.

    ``memory`` is the belief the skill was chosen with. Blocks in view in
    ``pre_state`` count as known whether or not it has observed them.
    ``steps`` is accepted for a uniform signature with :meth:`EpisodeMemory.
    record` and does not affect the cause.
    """

    outcome_value = _outcome_value(outcome)
    if outcome_value == "success":
        return None
    if _died(post_state):
        return FailureCause.DIED
    if outcome_value == "environment_terminated":
        return FailureCause.EPISODE_ENDED
    for vital, cause in _VITAL_CAUSES:
        if _vital(post_state, vital) <= 0 < _vital(pre_state, vital):
            return cause
    health = _vital(post_state, "health")
    if health <= low_health and health < _vital(pre_state, "health"):
        return FailureCause.HEALTH_LOW
    if outcome_value == "infeasible":
        return FailureCause.INFEASIBLE
    if outcome_value == "precondition_lost":
        return FailureCause.PRECONDITION_LOST
    target = skill_target(skill_id)
    if target is None:
        return FailureCause.TIMEOUT
    if memory.observe(pre_state).positions(target):
        return FailureCause.TARGET_NOT_REACHED
    return FailureCause.TARGET_NOT_SEEN


def _extremes(
    extremes: Mapping[str, tuple[int, int]] | Iterable[tuple[str, int, int]] | None,
) -> Extremes | None:
    if extremes is None:
        return None
    items = (
        ((field, *bounds) for field, bounds in extremes.items())
        if isinstance(extremes, Mapping)
        else extremes
    )
    return tuple(
        sorted((str(field), int(low), int(high)) for field, low, high in items)
    )


def _render_change(record: TransitionRecord) -> str:
    before, after = dict(record.before), dict(record.after)
    changes = [
        f"{field.split('.', 1)[1]} {before.get(field, 0)}->{after.get(field, 0)}"
        for field in sorted(before.keys() | after.keys())
        if before.get(field, 0) != after.get(field, 0)
    ]
    # A cause that only repeats the outcome, such as an infeasible request,
    # is left out.
    cause = (
        f" ({record.cause.replace('_', ' ')})"
        if record.cause is not None and record.cause != record.outcome
        else ""
    )
    return (
        f"{record.skill_id.value} {record.outcome}{cause}: "
        f"{', '.join(changes) if changes else 'no change'}"
    )


def _distance(position: Position, player: Position) -> tuple[int, Position]:
    # Manhattan distance, ties broken by position for a stable answer.
    return abs(position[0] - player[0]) + abs(position[1] - player[1]), position


def _offset(position: Position | None, player: Position | None) -> list[int] | None:
    if position is None or player is None:
        return None
    return [position[0] - player[0], position[1] - player[1]]


def _visible_mobs(state: Any, in_view: np.ndarray) -> dict[str, set[Position]]:
    rows, columns = in_view.shape
    visible: dict[str, set[Position]] = {}
    for mob, attribute in MOB_TYPES.items():
        mobs = getattr(state, attribute)
        positions = np.asarray(mobs.position).reshape(-1, 2).tolist()
        alive = np.asarray(mobs.mask).reshape(-1).tolist()
        visible[mob] = {
            (int(row), int(column))
            for (row, column), present in zip(positions, alive, strict=True)
            if present
            and 0 <= row < rows
            and 0 <= column < columns
            and in_view[row, column]
        }
    return visible


@dataclass(frozen=True)
class EpisodeMemory:
    """Transitions, block sightings and mob sightings for one episode.

    ``known`` holds the map positions where each block was last seen, and
    ``misses`` how many collect requests for a resource have failed in a row
    while none was known. A resource is ruled out after ``miss_limit`` such
    failures, and back in play as soon as one is seen.

    ``mobs`` holds the last sighting of each mob seen, and a sighting more
    than ``mob_forget_after`` boundaries old is forgotten.
    """

    view_radius: tuple[int, int] = OBSERVATION_RADIUS
    miss_limit: int = 2
    mob_forget_after: int = 3
    transitions: tuple[TransitionRecord, ...] = ()
    known: tuple[tuple[str, tuple[Position, ...]], ...] = tuple(
        (block, ()) for block in RESOURCE_BLOCKS
    )
    misses: tuple[tuple[str, int], ...] = tuple(
        (resource, 0) for resource in COLLECT_RESOURCES.values()
    )
    mobs: tuple[tuple[str, tuple[MobSighting, ...]], ...] = tuple(
        (mob, ()) for mob in MOB_TYPES
    )

    def positions(self, block: str) -> tuple[Position, ...]:
        """Return the positions where ``block`` was last seen, sorted."""

        return dict(self.known)[block]

    def mob_sightings(self, mob: str) -> tuple[MobSighting, ...]:
        """Return the remembered sightings of ``mob``, sorted by position."""

        return dict(self.mobs)[mob]

    def ruled_out(self, resource: str) -> bool:
        """Return whether repeated failed searches rule ``resource`` out."""

        return (
            dict(self.misses).get(resource, 0) >= self.miss_limit
            and not self.positions(resource)
        )

    def observe(self, state: Any) -> EpisodeMemory:
        """Return the memory updated with the blocks and mobs in view.

        Known positions inside the view that no longer hold the block, a
        mined tree or a removed table, are dropped, and so are remembered mobs
        whose tile is in view without that mob type on it. Positions out of
        view are kept as last seen, mobs only for ``mob_forget_after``
        boundaries.
        """

        world_map = np.asarray(state.map)
        in_view = np.asarray(view_mask(state, self.view_radius))
        known: list[tuple[str, tuple[Position, ...]]] = []
        misses = dict(self.misses)
        for block, positions in self.known:
            visible = {
                (int(row), int(column))
                for row, column in np.argwhere(
                    in_view & (world_map == RESOURCE_BLOCKS[block].value)
                )
            }
            kept = {position for position in positions if not in_view[position]}
            known.append((block, tuple(sorted(kept | visible))))
            if visible and block in misses:
                misses[block] = 0

        boundary = len(self.transitions)
        timestep = int(np.asarray(state.timestep))
        visible_mobs = _visible_mobs(state, in_view)
        mobs: list[tuple[str, tuple[MobSighting, ...]]] = []
        for mob, sightings in self.mobs:
            kept_sightings = [
                sighting
                for sighting in sightings
                if not in_view[sighting.position]
                and boundary - sighting.boundary <= self.mob_forget_after
            ]
            seen = [
                MobSighting(position, boundary, timestep)
                for position in visible_mobs[mob]
            ]
            mobs.append(
                (
                    mob,
                    tuple(
                        sorted(
                            (*kept_sightings, *seen),
                            key=lambda sighting: sighting.position,
                        )
                    ),
                )
            )
        return dataclasses.replace(
            self,
            known=tuple(known),
            misses=tuple(misses.items()),
            mobs=tuple(mobs),
        )

    def record(
        self,
        pre_state: Any,
        skill_id: SkillId | str,
        outcome: Any,
        steps: int,
        post_state: Any,
        *,
        inventory_extremes: (
            Mapping[str, tuple[int, int]] | Iterable[tuple[str, int, int]] | None
        ) = None,
    ) -> EpisodeMemory:
        """Return the memory with one skill request appended and observed.

        ``outcome`` may be a :class:`~tiny_sima.planner.saycan.SkillOutcome`
        or its string value; the string is stored, with the cause from
        :func:`classify_failure`. ``inventory_extremes`` maps each inventory
        field to its ``(min, max)`` over the skill's steps, or lists
        ``(field, min, max)`` triples; it is stored sorted by field.
        """

        skill_id = SkillId(skill_id)
        outcome_value = _outcome_value(outcome)
        memory = self.observe(pre_state)
        cause = classify_failure(
            pre_state, skill_id, outcome_value, steps, post_state, memory
        )
        transition = TransitionRecord(
            decision_index=len(self.transitions),
            skill_id=skill_id,
            outcome=outcome_value,
            steps=int(steps),
            before=readings(pre_state),
            after=readings(post_state),
            cause=None if cause is None else cause.value,
            inventory_extremes=_extremes(inventory_extremes),
        )
        misses = dict(memory.misses)
        resource = COLLECT_RESOURCES.get(skill_id)
        if resource is not None:
            if outcome_value == "success":
                misses[resource] = 0
            elif not memory.positions(resource):
                misses[resource] += 1
        return dataclasses.replace(
            memory,
            transitions=(*memory.transitions, transition),
            misses=tuple(misses.items()),
        ).observe(post_state)

    def world_fact(self, field: str) -> int | None:
        """Answer one registry world fact from what this episode has seen.

        ``world.<block>_reachable`` for a naturally generated block counts as
        obtainable unless it was ruled out, since never having seen it is no
        evidence it is absent. A block only the agent places, such as a
        crafting table, counts only once one is known, because a generated map
        holds none. ``None`` means the field is not a world fact this memory
        answers. Grass, like free ground, is everywhere, and cows keep spawning
        on it, so those count as obtainable too.
        """

        block = _WORLD_FACTS.get(field)
        if block is not None:
            if block in PLACED_BLOCKS:
                return int(bool(self.positions(block)))
            return int(not self.ruled_out(block))
        if field in {
            "world.placement_space_reachable",
            "world.grass_reachable",
            "world.cow_reachable",
        }:
            return 1
        return None

    def summary(
        self,
        state: Any = None,
        *,
        max_transitions: int = 6,
    ) -> dict[str, Any]:
        """Return a short JSON-ready digest for a planner prompt.

        Offsets are ``[row, column]`` from the player in ``state`` to the
        nearest known instance, by Manhattan distance, and ``None`` without a
        state or a known instance. The four blocks the skills use are always
        listed, other blocks only once seen or ruled out, and mobs only while
        remembered, with ``last_seen_ago`` in boundaries for the nearest.
        """

        player = None if state is None else _player(state)
        resources: dict[str, dict[str, Any]] = {}
        for block, positions in self.known:
            ruled_out = self.ruled_out(block)
            if block not in PROMPT_BLOCKS and not positions and not ruled_out:
                continue
            nearest = (
                None
                if player is None or not positions
                else min(positions, key=lambda position: _distance(position, player))
            )
            resources[block] = {
                "known": len(positions),
                "nearest_offset": _offset(nearest, player),
                "ruled_out": ruled_out,
            }
        mobs: dict[str, dict[str, Any]] = {}
        for mob, sightings in self.mobs:
            if not sightings:
                continue
            # Without a state, the age is that of the latest sighting.
            nearest = (
                max(sightings, key=lambda sighting: sighting.boundary)
                if player is None
                else min(
                    sightings,
                    key=lambda sighting: _distance(sighting.position, player),
                )
            )
            mobs[mob] = {
                "known": len(sightings),
                "nearest_offset": _offset(nearest.position, player),
                "last_seen_ago": len(self.transitions) - nearest.boundary,
            }
        recent = self.transitions[-max_transitions:] if max_transitions > 0 else ()
        return {
            "recent": [_render_change(record) for record in recent],
            "resources": resources,
            "mobs": mobs,
        }
