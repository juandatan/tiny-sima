"""Text renderings of the planner's observation for image/text ablations.

The VLM planner has so far seen the world only through the rendered
screenshot plus the structured inventory and reachability summary. This module
renders the same egocentric window as text, alongside the vitals, creatures
and achievements the screenshot shows, so that a prompt can carry the image,
the text, or both. Every reading is restricted to what the Craftax-Classic
observation shows the player; nothing outside the view window is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from craftax.craftax_classic.constants import OBS_DIM, Achievement, BlockType

# ``OBS_DIM`` is (rows, columns): the renderers slice ``state.map`` with it
# directly, axis 0 is the image's vertical axis, and ``DIRECTIONS`` moves UP
# along -axis 0. The map is therefore 7 rows tall and 9 columns wide.
_VIEW_RADIUS: tuple[int, int] = (OBS_DIM[0] // 2, OBS_DIM[1] // 2)

# One character per block, chosen so that no block, creature or player glyph
# shares a character.
_BLOCK_CHARS: dict[BlockType, str] = {
    BlockType.INVALID: "?",
    BlockType.OUT_OF_BOUNDS: "#",
    BlockType.GRASS: ".",
    BlockType.WATER: "~",
    BlockType.STONE: "S",
    BlockType.TREE: "T",
    BlockType.WOOD: "W",
    BlockType.PATH: "_",
    BlockType.COAL: "C",
    BlockType.IRON: "I",
    BlockType.DIAMOND: "D",
    BlockType.CRAFTING_TABLE: "B",
    BlockType.FURNACE: "F",
    BlockType.SAND: ":",
    BlockType.LAVA: "L",
    BlockType.PLANT: "p",
    BlockType.RIPE_PLANT: "P",
}

# Later entries are drawn over earlier ones, the order the pixel renderer
# draws them in, so a zombie standing on an arrow shows as the zombie.
_MOB_CHARS: dict[str, str] = {
    "arrow": "*",
    "cow": "c",
    "skeleton": "k",
    "zombie": "z",
}

# Indexed by ``state.player_direction``; see ``craftax_classic.constants``
# ``DIRECTIONS``, where 1-4 are left, right, up and down.
_PLAYER_CHARS: dict[int, str] = {1: "<", 2: ">", 3: "^", 4: "v"}

_BLOCK_LUT = np.array([_BLOCK_CHARS[block] for block in BlockType])

MAP_LEGEND: str = (
    "Map legend: "
    + ", ".join(
        f"'{char}' {block.name.lower().replace('_', ' ')}"
        for block, char in _BLOCK_CHARS.items()
    )
    + "; creatures: "
    + ", ".join(f"'{char}' {name}" for name, char in _MOB_CHARS.items())
    + "; the player is drawn as '^', 'v', '<' or '>' pointing the way it faces."
)

# The pixel renderer switches to its night texture below this light level,
# so ``night`` is true exactly when the screenshot is drawn as night.
NIGHT_LIGHT_THRESHOLD: float = 0.5


def _int(value: Any) -> int:
    return int(np.asarray(value).item())


def _local_blocks(state: Any) -> np.ndarray:
    """Return the block ids in view, padded with out-of-bounds tiles."""

    world_map = np.asarray(state.map)
    rows, columns = _VIEW_RADIUS
    padded = np.pad(
        world_map,
        ((rows, rows), (columns, columns)),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    row, column = np.asarray(state.player_position).tolist()
    # Padding shifts every index by the radius, so the slice starting at the
    # player's own index is centred on the player.
    return padded[row : row + OBS_DIM[0], column : column + OBS_DIM[1]]


def _mobs_with_offsets(state: Any) -> list[tuple[str, int, int]]:
    """Return ``(type, d_row, d_col)`` for every live mob inside the window."""

    player = np.asarray(state.player_position)
    found: list[tuple[str, int, int]] = []
    for name in _MOB_CHARS:
        mobs = getattr(state, f"{name}s")
        positions = np.asarray(mobs.position).reshape(-1, 2)
        mask = np.asarray(mobs.mask).reshape(-1)
        for position, alive in zip(positions, mask, strict=True):
            if not alive:
                continue
            d_row, d_col = (position - player).tolist()
            if abs(d_row) <= _VIEW_RADIUS[0] and abs(d_col) <= _VIEW_RADIUS[1]:
                found.append((name, int(d_row), int(d_col)))
    return found


def render_local_map(state: Any) -> str:
    """Render the player's view as text, one character per tile.

    The grid matches the Craftax-Classic observation window: ``OBS_DIM[0]``
    rows from top to bottom, each ``OBS_DIM[1]`` characters wide, with the
    player at the centre. See :data:`MAP_LEGEND` for the characters.
    """

    grid = _BLOCK_LUT[_local_blocks(state)]
    for name, d_row, d_col in _mobs_with_offsets(state):
        grid[_VIEW_RADIUS[0] + d_row, _VIEW_RADIUS[1] + d_col] = _MOB_CHARS[name]
    grid[_VIEW_RADIUS] = _PLAYER_CHARS.get(_int(state.player_direction), "?")
    return "\n".join("".join(row) for row in grid)


def vitals(state: Any) -> dict[str, Any]:
    """Return the survival readings and time of day the screenshot shows."""

    daylight = float(np.asarray(state.light_level).item())
    return {
        "health": _int(state.player_health),
        "food": _int(state.player_food),
        "drink": _int(state.player_drink),
        "energy": _int(state.player_energy),
        "sleeping": bool(np.asarray(state.is_sleeping).item()),
        "daylight": round(daylight, 2),
        "night": daylight < NIGHT_LIGHT_THRESHOLD,
    }


def mobs_in_view(state: Any) -> list[dict[str, Any]]:
    """Return each visible creature and its ``[d_row, d_col]`` offset.

    Negative ``d_row`` is up and negative ``d_col`` is left. Entries are
    sorted by grid (Manhattan) distance from the player, then by offset and
    type so that ties keep a stable order.
    """

    found = sorted(
        _mobs_with_offsets(state),
        key=lambda mob: (abs(mob[1]) + abs(mob[2]), mob[1], mob[2], mob[0]),
    )
    return [{"type": name, "offset": [d_row, d_col]} for name, d_row, d_col in found]


def achievements(state: Any) -> list[str]:
    """Return the names of the achievements unlocked so far, in enum order."""

    unlocked = np.asarray(state.achievements).reshape(-1)
    return [
        achievement.name.lower()
        for achievement in Achievement
        if achievement.value < unlocked.size and bool(unlocked[achievement.value])
    ]


@dataclass(frozen=True)
class ObservationConfig:
    """Which parts of the observation a planner prompt carries.

    ``image`` sends the rendered screenshot; the other flags add text fields
    to the prompt. A planner must see the map one way or the other, so at
    least one of ``image`` and ``text_map`` has to be on.
    """

    image: bool = True
    text_map: bool = False
    vitals: bool = False
    mobs: bool = False
    achievements: bool = False

    def __post_init__(self) -> None:
        if not (self.image or self.text_map):
            raise ValueError("ObservationConfig needs image or text_map enabled.")

    @property
    def has_text(self) -> bool:
        """Return whether any text field is enabled."""

        return self.text_map or self.vitals or self.mobs or self.achievements


# The ablation arms. ``IMAGE_ONLY`` renders the same prompts as no config.
IMAGE_ONLY = ObservationConfig()
TEXT_ONLY = ObservationConfig(
    image=False, text_map=True, vitals=True, mobs=True, achievements=True
)
IMAGE_AND_TEXT = ObservationConfig(
    image=True, text_map=True, vitals=True, mobs=True, achievements=True
)
# The screenshot plus everything except the map, which separates the value of
# the text map from that of the non-spatial readings.
IMAGE_AND_STATE = ObservationConfig(
    image=True, vitals=True, mobs=True, achievements=True
)
OBSERVATION_PRESETS: dict[str, ObservationConfig] = {
    "image_only": IMAGE_ONLY,
    "text_only": TEXT_ONLY,
    "image_and_text": IMAGE_AND_TEXT,
    "image_and_state": IMAGE_AND_STATE,
}


def observation_block(state: Any, config: ObservationConfig) -> dict[str, Any]:
    """Return the enabled text fields as a JSON-ready dict.

    The map is a list of row strings rather than one string, so that
    ``json.dumps`` does not escape it into a single line of ``\\n``.
    """

    block: dict[str, Any] = {}
    if config.text_map:
        block["local_map"] = render_local_map(state).split("\n")
    if config.vitals:
        block["vitals"] = vitals(state)
    if config.mobs:
        block["mobs_in_view"] = mobs_in_view(state)
    if config.achievements:
        block["achievements"] = achievements(state)
    return block


def observation_system_text(config: ObservationConfig) -> str:
    """Return system-prompt lines explaining the enabled text fields.

    Returns an empty string when no text field is enabled, so an image-only
    configuration leaves the system prompt unchanged.
    """

    if not config.has_text:
        return ""
    fields: list[str] = []
    if config.text_map:
        fields.append(
            f"local_map is the visible {OBS_DIM[0]}x{OBS_DIM[1]} window as "
            "rows of characters, top row first, with the player at the centre"
        )
    if config.vitals:
        fields.append(
            "vitals are health, food, drink and energy out of 9, whether the "
            "player is asleep, and daylight from 0 (night) to 1 (day)"
        )
    if config.mobs:
        fields.append(
            "mobs_in_view lists each visible creature with its [row, column] "
            "offset from the player, negative meaning up or left, nearest first"
        )
    if config.achievements:
        fields.append("achievements lists the achievements unlocked so far")
    lines = ["The text observation is a JSON object: " + "; ".join(fields) + "."]
    if config.text_map:
        lines.append(MAP_LEGEND)
    return "\n".join(lines)
