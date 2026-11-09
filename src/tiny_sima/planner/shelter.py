"""A scripted shelter: wall the player into a 2x2 room for the night.

Zombies walk only onto free tiles, so a room whose every neighbour is a
solid block keeps them out, and the player can sleep inside. The room is
2x2 because Craftax has no turn action: pressing towards a free tile steps
onto it, so the player faces a free tile only after stepping in that
direction. In a 2x2 room each wall tile ``tile + step`` is faced from
``tile`` on arriving there from ``tile - step``, another room tile, which a
one-tile room has no room for.

:func:`shelter_action` is the policy, one action per step; the chain
evaluator runs it as the ``shelter`` macro skill. Until morning it fights
off adjacent zombies, mines stone if it has too little for the open walls,
walks into the room, places stone on each open wall and then sleeps. It
prefers a room walled partly by water, which keeps zombies out as well and
which the player drinks from inside. In the morning, or once food runs low
or drink does with no water to hand, it mines a stone wall to get out.

By day the same room is a place to rest: with ``rest`` the player stays
walled in, asleep while it can, until its health is full, and then leaves.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from craftax.craftax_classic.constants import (
    DIRECTIONS,
    SOLID_BLOCKS,
    Action,
    BlockType,
)
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.game_logic import calculate_light_level

from tiny_sima.goals import MAX_VITAL, MOB_TYPES, mob_instances
from tiny_sima.planner.observation import NIGHT_LIGHT_THRESHOLD

Position = tuple[int, int]

# The player leaves for food or water once either falls this low.
SHELTER_NEED_FLOOR = 2
# How far, in steps, to look for a room.
SHELTER_SEARCH_RADIUS = 8
# The steps a water wall is worth: the player drinks from it inside.
_WATER_BONUS = 4

_MOVES: dict[int, Position] = {
    action.value: (int(DIRECTIONS[action.value][0]), int(DIRECTIONS[action.value][1]))
    for action in (Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN)
}
# Neither the player nor a mob can walk onto these.
_CLOSED = frozenset(np.asarray(SOLID_BLOCKS).tolist()) | {BlockType.LAVA.value}


def _add(position: Position, step: Position) -> Position:
    return position[0] + step[0], position[1] + step[1]


def _block(world_map: np.ndarray, position: Position) -> int | None:
    row, column = position
    if not (0 <= row < world_map.shape[0] and 0 <= column < world_map.shape[1]):
        return None
    return int(world_map[row, column])


def _closed(world_map: np.ndarray, position: Position) -> bool:
    block = _block(world_map, position)
    return block is None or block in _CLOSED


@dataclass(frozen=True)
class Room:
    """The 2x2 room whose top-left tile is ``corner``."""

    corner: Position

    @property
    def tiles(self) -> tuple[Position, ...]:
        row, column = self.corner
        return (
            (row, column),
            (row, column + 1),
            (row + 1, column),
            (row + 1, column + 1),
        )

    @property
    def walls(self) -> tuple[tuple[Position, int], ...]:
        """Each ``(tile, move)`` whose wall is the tile one ``move`` beyond."""

        return tuple(
            (tile, move)
            for tile in self.tiles
            for move, step in _MOVES.items()
            if _add(tile, step) not in self.tiles
        )

    def open_walls(self, world_map: np.ndarray) -> list[tuple[Position, int]]:
        return [
            (tile, move)
            for tile, move in self.walls
            if not _closed(world_map, _add(tile, _MOVES[move]))
        ]

    def usable(self, world_map: np.ndarray) -> bool:
        return not any(_closed(world_map, tile) for tile in self.tiles)


def _player(state: Any) -> Position:
    row, column = np.asarray(state.player_position).tolist()
    return int(row), int(column)


def is_night(state: Any) -> bool:
    return float(np.asarray(state.light_level)) < NIGHT_LIGHT_THRESHOLD


def steps_until_morning(state: Any, day_length: int = EnvParams().day_length) -> int:
    """Steps until the light comes back, as Craftax computes it; 0 by day."""

    timesteps = int(np.asarray(state.timestep)) + np.arange(day_length)
    light = np.asarray(
        calculate_light_level(timesteps, EnvParams(day_length=day_length))
    )
    day = np.flatnonzero(light >= NIGHT_LIGHT_THRESHOLD)
    return int(day[0]) if day.size else day_length


def needs_low(state: Any) -> bool:
    """Whether food or drink is too low to stay in."""

    return min(_food(state), int(np.asarray(state.player_drink))) <= SHELTER_NEED_FLOOR


def _health(state: Any) -> int:
    return int(np.asarray(state.player_health))


def _food(state: Any) -> int:
    return int(np.asarray(state.player_food))


def _stone(state: Any) -> int:
    return int(np.asarray(state.inventory.stone))


def _mob_positions(state: Any, mobs: tuple[str, ...]) -> set[Position]:
    found: set[Position] = set()
    for mob in mobs:
        instances = mob_instances(mob, state)
        for position in np.asarray(instances.position)[np.asarray(instances.mask)]:
            found.add((int(position[0]), int(position[1])))
    return found


def _distances(
    world_map: np.ndarray, start: Position, radius: int
) -> dict[Position, int]:
    """Steps from ``start`` to each open tile within ``radius`` of them."""

    distances = {start: 0}
    queue = deque([start])
    while queue:
        position = queue.popleft()
        if distances[position] == radius:
            continue
        for step in _MOVES.values():
            neighbour = _add(position, step)
            if neighbour not in distances and not _closed(world_map, neighbour):
                distances[neighbour] = distances[position] + 1
                queue.append(neighbour)
    return distances


def _first_move(
    world_map: np.ndarray, start: Position, goal: Callable[[Position], bool]
) -> int | None:
    """The first move of a shortest walk to an open tile ``goal`` accepts."""

    first: dict[Position, int | None] = {start: None}
    queue = deque([start])
    while queue:
        position = queue.popleft()
        if goal(position) and position != start:
            return first[position]
        for move, step in _MOVES.items():
            neighbour = _add(position, step)
            if neighbour not in first and not _closed(world_map, neighbour):
                first[neighbour] = move if first[position] is None else first[position]
                queue.append(neighbour)
    return None


def choose_room(state: Any, radius: int = SHELTER_SEARCH_RADIUS) -> Room | None:
    """Return the nearby room that needs the least stone, then walking.

    Its tiles must be reachable, and it must have an open wall or a stone
    one, so the player can get in and back out.
    """

    world_map = np.asarray(state.map)
    distances = _distances(world_map, _player(state), radius)
    stone = _stone(state)
    best: tuple[tuple[int, int], Room] | None = None
    for corner in distances:
        room = Room(corner)
        if not all(tile in distances for tile in room.tiles):
            continue
        open_count = len(room.open_walls(world_map))
        if open_count == 0 and not _walls_of(room, world_map, BlockType.STONE):
            continue
        cost = (
            max(open_count - stone, 0),
            open_count
            + min(distances[tile] for tile in room.tiles)
            - _WATER_BONUS * bool(_walls_of(room, world_map, BlockType.WATER)),
        )
        if best is None or cost < best[0]:
            best = cost, room
    return None if best is None else best[1]


def _walls_of(
    room: Room, world_map: np.ndarray, block: BlockType
) -> list[tuple[Position, int]]:
    return [
        (tile, move)
        for tile, move in room.walls
        if _block(world_map, _add(tile, _MOVES[move])) == block.value
    ]


def _use_wall(state: Any, walls: list[tuple[Position, int]]) -> int:
    """Walk to the first of ``walls`` inside the room, face it and DO."""

    player = _player(state)
    for tile, move in walls:
        if tile == player:
            return _face(state, move, Action.DO)
    return _towards(player, walls[0][0])


def shelter_blocker(state: Any, by_day: bool = False) -> str | None:
    """Why the shelter cannot start now, or ``None`` when it can.

    With ``by_day`` it may also start by day, to rest while health is not
    full.
    """

    return shelter_status(state, by_day)[0]


def shelter_time(state: Any, by_day: bool = False) -> str | None:
    """Why it is not a time to shelter, or ``None`` when it is.

    It is at night, and with ``by_day`` by day too while health is not full.
    """

    if is_night(state):
        return None
    if not by_day:
        return "it is day"
    if _health(state) >= MAX_VITAL:
        return "health is full"
    return None


def shelter_status(state: Any, by_day: bool = False) -> tuple[str | None, Room | None]:
    """:func:`shelter_blocker` with the room the shelter would use, if any.

    The room is looked for only at a :func:`shelter_time`.
    """

    not_now = shelter_time(state, by_day)
    if not_now is not None:
        return not_now, None
    room = choose_room(state)
    if _food(state) <= SHELTER_NEED_FLOOR:
        return "food is too low to stay in", room
    if int(np.asarray(state.inventory.wood_pickaxe)) < 1:
        return "it needs a wood pickaxe", room
    if room is None:
        return "there is no room to wall in nearby", None
    world_map = np.asarray(state.map)
    # Low drink is fine in a room the player can drink inside.
    if needs_low(state) and not room_has_water(state, room):
        return "drink is too low to stay in", room
    if _stone(state) < len(room.open_walls(world_map)) and (
        _mine_stone(state, world_map, _player(state), _walls(room)) is None
    ):
        return "there is too little stone and none to mine nearby", room
    return None, room


def room_has_water(state: Any, room: Room) -> bool:
    """Whether ``room`` has water beside it to drink from."""

    return bool(_walls_of(room, np.asarray(state.map), BlockType.WATER))


def shelter_feasible(state: Any, by_day: bool = False) -> bool:
    """Whether the shelter can start: night, a pickaxe, food, drink, a room.

    With ``by_day`` a day with health below full serves as well as night.
    """

    return shelter_blocker(state, by_day) is None


def _face(state: Any, move: int, action: Action) -> int:
    """``action`` if the player faces ``move``, else ``move`` to turn.

    Pressing towards a blocked tile turns without stepping.
    """

    if int(np.asarray(state.player_direction)) == move:
        return action.value
    return move


def _towards(start: Position, goal: Position) -> int:
    """The move one step from ``start`` towards ``goal``, row first."""

    if start[0] != goal[0]:
        return Action.UP.value if goal[0] < start[0] else Action.DOWN.value
    return Action.LEFT.value if goal[1] < start[1] else Action.RIGHT.value


def shelter_action(
    state: Any, room: Room | None, rest: bool = False
) -> tuple[int | None, Room | None]:
    """Return the next shelter action and the room it works on.

    The action is ``None`` once there is nothing left to do: the player is
    out in the morning, or out with food or drink low, or there is no room
    to build or stone to build it with. With ``rest`` the day keeps the
    player in, as the night does, until its health is full.
    """

    world_map = np.asarray(state.map)
    player = _player(state)
    zombies = _mob_positions(state, ("zombie",))
    for move, step in _MOVES.items():
        if _add(player, step) in zombies:
            return _face(state, move, Action.DO), room
    if room is None or not room.usable(world_map):
        room = choose_room(state)
    if room is None:
        return None, room
    inside = player in room.tiles
    open_walls = room.open_walls(world_map)
    sleeping = bool(np.asarray(state.is_sleeping))
    water = _walls_of(room, world_map, BlockType.WATER)
    # Water walls the room, so the player drinks without leaving.
    drink = int(np.asarray(state.player_drink))
    if inside and not open_walls and not sleeping and water and drink < MAX_VITAL:
        return _use_wall(state, water), room
    thirsty = needs_low(state) and (_food(state) <= SHELTER_NEED_FLOOR or not water)
    stay = sleeping or shelter_time(state, rest) is None
    if not stay or thirsty:
        if not inside or open_walls:
            return None, room
        exits = _walls_of(room, world_map, BlockType.STONE)
        return (_use_wall(state, exits) if exits else None), room
    if not open_walls:
        if not inside:
            return None, room
        energy = int(np.asarray(state.player_energy))
        return (Action.SLEEP if energy < MAX_VITAL else Action.NOOP).value, room
    # Clear any other creature out of the room before walling it in.
    mobs = _mob_positions(state, tuple(MOB_TYPES))
    for move, step in _MOVES.items():
        if _add(player, step) in mobs and _add(player, step) in room.tiles:
            return _face(state, move, Action.DO), room
    if _stone(state) < len(open_walls):
        return _mine_stone(state, world_map, player, _walls(room)), room
    if not inside:
        move = _first_move(world_map, player, lambda tile: tile in room.tiles)
        return (Action.NOOP.value if move is None else move), room
    for tile, move in open_walls:
        if tile == player and int(np.asarray(state.player_direction)) == move:
            return Action.PLACE_STONE.value, room

    def behind(wall: tuple[Position, int]) -> Position:
        step = _MOVES[wall[1]]
        return wall[0][0] - step[0], wall[0][1] - step[1]

    tile, move = min(
        open_walls,
        key=lambda wall: (
            abs(behind(wall)[0] - player[0]) + abs(behind(wall)[1] - player[1])
        ),
    )
    if player == behind((tile, move)):
        # Stepping into the wall's tile leaves the player facing the wall.
        return move, room
    return _towards(player, behind((tile, move))), room


def _mine_stone(
    state: Any, world_map: np.ndarray, player: Position, keep: set[Position]
) -> int | None:
    """Step towards, face and mine the nearest stone outside ``keep``.

    ``None`` when no stone can be reached.
    """

    def stone_beside(position: Position) -> int | None:
        for move, step in _MOVES.items():
            neighbour = _add(position, step)
            if (
                neighbour not in keep
                and _block(world_map, neighbour) == BlockType.STONE.value
            ):
                return move
        return None

    move = stone_beside(player)
    if move is not None:
        return _face(state, move, Action.DO)
    return _first_move(world_map, player, lambda tile: stone_beside(tile) is not None)


def _walls(room: Room) -> set[Position]:
    return {_add(tile, _MOVES[move]) for tile, move in room.walls}
