import json
import sys
import types
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import OBS_DIM, Achievement, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.game_logic import calculate_light_level
from craftax.craftax_classic.renderer import render_craftax_symbolic

import tiny_sima.planner.vlm as vlm_module
from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import OBSERVATION_RADIUS, SKILL_IDS
from tiny_sima.planner import (
    IMAGE_AND_STATE,
    IMAGE_AND_TEXT,
    IMAGE_ONLY,
    MAP_LEGEND,
    OBSERVATION_PRESETS,
    TEXT_ONLY,
    ObservationConfig,
    achievements,
    mobs_in_view,
    observation_block,
    observation_system_text,
    render_local_map,
    vitals,
)
from tiny_sima.planner.observation import _BLOCK_CHARS
from tiny_sima.planner.saycan import PlannerContext
from tiny_sima.planner.vlm import (
    PlannerPromptStyle,
    TransformersVLMBackend,
    VLMRequest,
    VLMUtilityScorer,
    _planner_chat,
    build_planner_prompt,
    build_planner_system_prompt,
)


def _state(seed: int = 0):
    return make_skill_chain_environment().reset(jax.random.PRNGKey(seed)).state


def _grass_world(state=None, **blocks: list[tuple[int, int]]):
    # Every tile grass, then each named block at offsets from the player.
    state = _state() if state is None else state
    world_map = np.full(np.asarray(state.map).shape, BlockType.GRASS.value)
    row, column = np.asarray(state.player_position).tolist()
    for name, offsets in blocks.items():
        for d_row, d_column in offsets:
            world_map[row + d_row, column + d_column] = BlockType[name.upper()].value
    return state.replace(map=jnp.asarray(world_map, dtype=state.map.dtype))


def _with_mobs(state, kind: str, *offsets: tuple[int, int]):
    # Fill the first slots of one mob array, relative to the player.
    mobs = getattr(state, kind)
    positions = np.asarray(mobs.position).copy()
    mask = np.zeros_like(np.asarray(mobs.mask))
    player = np.asarray(state.player_position)
    for slot, offset in enumerate(offsets):
        positions[slot] = player + np.asarray(offset)
        mask[slot] = True
    return state.replace(
        **{
            kind: mobs.replace(
                position=jnp.asarray(positions, dtype=mobs.position.dtype),
                mask=jnp.asarray(mask),
            )
        }
    )


def _rows(state) -> list[str]:
    return render_local_map(state).split("\n")


def _at(state, d_row: int, d_col: int) -> str:
    return _rows(state)[OBSERVATION_RADIUS[0] + d_row][OBSERVATION_RADIUS[1] + d_col]


def test_local_map_matches_the_observation_window() -> None:
    state = _grass_world(
        tree=[(-1, 0), (3, 4)], stone=[(0, 2)], water=[(-3, -4)], lava=[(4, 0)]
    )
    rows = _rows(state)

    assert OBS_DIM == (7, 9)
    assert len(rows) == OBS_DIM[0]
    assert all(len(row) == OBS_DIM[1] for row in rows)
    assert _at(state, -1, 0) == "T"
    assert _at(state, 3, 4) == "T"
    assert _at(state, 0, 2) == "S"
    assert _at(state, -3, -4) == "~"
    # Lava four rows down lies outside the seven-row window.
    assert "L" not in render_local_map(state)
    assert rows[0][1] == "."


def test_local_map_agrees_with_the_craftax_symbolic_renderer() -> None:
    # The symbolic observation slices the map with OBS_DIM directly, so it
    # settles which axis is rows: an unrotated text map must decode to it.
    state = _with_mobs(_state(3), "cows", (2, -3))
    char_to_block = {char: block.value for block, char in _BLOCK_CHARS.items()}
    observation = np.asarray(render_craftax_symbolic(state))
    channels = len(BlockType) + 4
    grid = observation[: OBS_DIM[0] * OBS_DIM[1] * channels].reshape(
        (*OBS_DIM, channels)
    )
    expected_blocks = grid[..., : len(BlockType)].argmax(axis=-1)
    rows = _rows(state)
    for row in range(OBS_DIM[0]):
        for column in range(OBS_DIM[1]):
            char = rows[row][column]
            if (row, column) == OBSERVATION_RADIUS:
                continue
            if grid[row, column, len(BlockType) + 1]:
                assert char == "c"
            else:
                assert char_to_block[char] == expected_blocks[row, column]


@pytest.mark.parametrize(
    ("direction", "arrow"), [(1, "<"), (2, ">"), (3, "^"), (4, "v")]
)
def test_player_arrow_shows_the_facing_direction(direction: int, arrow: str) -> None:
    state = _grass_world().replace(player_direction=jnp.int32(direction))

    assert _at(state, 0, 0) == arrow
    assert render_local_map(state).count(arrow) == 1


def test_mobs_are_overlaid_and_listed_by_distance() -> None:
    state = _grass_world(stone=[(1, 1)])
    state = _with_mobs(state, "zombies", (-2, 3), (0, 1))
    state = _with_mobs(state, "cows", (1, 1), (0, 9))
    state = _with_mobs(state, "skeletons", (3, -4))
    state = _with_mobs(state, "arrows", (-1, 0))

    assert _at(state, -2, 3) == "z"
    assert _at(state, 0, 1) == "z"
    # A cow standing on a stone tile hides the stone, as in the image.
    assert _at(state, 1, 1) == "c"
    assert _at(state, 3, -4) == "k"
    assert _at(state, -1, 0) == "*"
    assert mobs_in_view(state) == [
        {"type": "arrow", "offset": [-1, 0]},
        {"type": "zombie", "offset": [0, 1]},
        {"type": "cow", "offset": [1, 1]},
        {"type": "zombie", "offset": [-2, 3]},
        {"type": "skeleton", "offset": [3, -4]},
    ]


def test_masked_mobs_are_not_shown() -> None:
    state = _with_mobs(_grass_world(), "zombies", (1, 0))
    hidden = state.replace(
        zombies=state.zombies.replace(mask=jnp.zeros_like(state.zombies.mask))
    )

    assert mobs_in_view(hidden) == []
    assert "z" not in render_local_map(hidden)


def test_out_of_bounds_tiles_are_padded_at_a_map_corner() -> None:
    state = _grass_world().replace(player_position=jnp.array([0, 0], dtype=jnp.int32))
    rows = _rows(state)

    assert len(rows) == OBS_DIM[0] and all(len(row) == OBS_DIM[1] for row in rows)
    # Rows above and columns left of the player lie off the map.
    for row in range(OBSERVATION_RADIUS[0]):
        assert rows[row] == "#" * OBS_DIM[1]
    for row in rows[OBSERVATION_RADIUS[0] :]:
        assert row[: OBSERVATION_RADIUS[1]] == "#" * OBSERVATION_RADIUS[1]
    assert rows[OBSERVATION_RADIUS[0] + 1][OBSERVATION_RADIUS[1] + 1] == "."


def test_vitals_and_night_follow_the_light_level() -> None:
    params = EnvParams()
    day = _state().replace(
        timestep=jnp.int32(0),
        light_level=calculate_light_level(0, params),
        player_drink=jnp.int32(4),
    )
    night = day.replace(
        timestep=jnp.int32(210),
        light_level=calculate_light_level(210, params),
        is_sleeping=jnp.bool_(True),
    )

    day_vitals = vitals(day)
    assert day_vitals == {
        "health": int(day.player_health),
        "food": int(day.player_food),
        "drink": 4,
        "energy": int(day.player_energy),
        "sleeping": False,
        "daylight": round(float(calculate_light_level(0, params)), 2),
        "night": False,
    }
    assert day_vitals["daylight"] > 0.5
    night_vitals = vitals(night)
    assert night_vitals["night"] is True
    assert night_vitals["sleeping"] is True
    assert night_vitals["daylight"] < 0.5
    json.dumps(night_vitals)


def test_achievements_lists_unlocked_names() -> None:
    state = _state()
    assert achievements(state) == []
    unlocked = state.achievements.at[Achievement.COLLECT_WOOD.value].set(True)
    unlocked = unlocked.at[Achievement.MAKE_WOOD_PICKAXE.value].set(True)

    assert achievements(state.replace(achievements=unlocked)) == [
        "collect_wood",
        "make_wood_pickaxe",
    ]


def test_config_needs_an_image_or_a_text_map() -> None:
    with pytest.raises(ValueError, match="image or text_map"):
        ObservationConfig(image=False)
    with pytest.raises(ValueError, match="image or text_map"):
        ObservationConfig(image=False, vitals=True, mobs=True)
    assert ObservationConfig(image=False, text_map=True).has_text
    assert not IMAGE_ONLY.has_text
    assert set(OBSERVATION_PRESETS.values()) == {
        IMAGE_ONLY,
        TEXT_ONLY,
        IMAGE_AND_TEXT,
        IMAGE_AND_STATE,
    }


def test_observation_block_contains_only_enabled_parts() -> None:
    state = _with_mobs(_grass_world(tree=[(1, 0)]), "cows", (0, 2))

    assert observation_block(state, IMAGE_ONLY) == {}
    assert observation_system_text(IMAGE_ONLY) == ""
    full = observation_block(state, TEXT_ONLY)
    assert set(full) == {"local_map", "vitals", "mobs_in_view", "achievements"}
    assert full["local_map"] == _rows(state)
    assert full["mobs_in_view"] == [{"type": "cow", "offset": [0, 2]}]
    assert observation_block(state, IMAGE_AND_TEXT) == full
    assert set(observation_block(state, IMAGE_AND_STATE)) == {
        "vitals",
        "mobs_in_view",
        "achievements",
    }
    # Each row stays one readable JSON string.
    assert json.dumps(full["local_map"][0]) == f'"{full["local_map"][0]}"'

    assert MAP_LEGEND in observation_system_text(TEXT_ONLY)
    assert MAP_LEGEND not in observation_system_text(IMAGE_AND_STATE)
    assert "mobs_in_view" in observation_system_text(IMAGE_AND_STATE)


@dataclass
class _FakeBackend:
    response: str
    requests: list[VLMRequest] = field(default_factory=list)

    def generate(self, request: VLMRequest) -> str:
        self.requests.append(request)
        return self.response


def _response() -> str:
    ranking = [skill_id.value for skill_id in SKILL_IDS]
    return json.dumps(
        {
            "progress_assessment": "Start.",
            "selected_skill": ranking[0],
            "ranking": ranking,
        }
    )


@pytest.mark.parametrize(
    "style", [PlannerPromptStyle.BASELINE, PlannerPromptStyle.RECIPE_AWARE]
)
@pytest.mark.parametrize("observation", [None, IMAGE_ONLY])
def test_default_observation_leaves_prompts_byte_identical(
    style: PlannerPromptStyle, observation: ObservationConfig | None
) -> None:
    state = _state()
    context = PlannerContext(objective="Collect stone.")
    backend = _FakeBackend(_response())

    VLMUtilityScorer(
        backend, prompt_style=style, observation=observation
    ).score_all(state, context, SKILL_IDS)

    (request,) = backend.requests
    assert request.system_prompt == build_planner_system_prompt(style, SKILL_IDS)
    assert request.user_prompt == build_planner_prompt(
        state, context, SKILL_IDS, style=style
    )
    assert request.image is not None and request.image.ndim == 3


def test_text_observation_is_added_to_both_prompts() -> None:
    state = _with_mobs(_state(), "zombies", (1, 2))
    context = PlannerContext(objective="Collect stone.")
    backend = _FakeBackend(_response())
    style = PlannerPromptStyle.BASELINE

    VLMUtilityScorer(
        backend, prompt_style=style, observation=IMAGE_AND_TEXT
    ).score_all(state, context, SKILL_IDS)

    (request,) = backend.requests
    default_system = build_planner_system_prompt(style, SKILL_IDS)
    assert request.system_prompt == (
        default_system + "\n" + observation_system_text(IMAGE_AND_TEXT)
    )
    line = "Text observation: " + json.dumps(
        observation_block(state, IMAGE_AND_TEXT)
    )
    lines = request.user_prompt.split("\n")
    assert lines[3] == line
    assert "\n".join(lines[:3] + lines[4:]) == build_planner_prompt(
        state, context, SKILL_IDS, style=style
    )
    assert request.image is not None


def test_text_only_request_skips_the_image(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_render(*_: object, **__: object) -> None:
        raise AssertionError("text-only requests must not render the image")

    monkeypatch.setattr(vlm_module, "render_state", fail_render)
    state = _state()
    backend = _FakeBackend(_response())
    scorer = VLMUtilityScorer(
        backend,
        prompt_style=PlannerPromptStyle.AVAILABLE_FIRST,
        observation=TEXT_ONLY,
    )

    scores = scorer.score_all_batched(
        (state, state), (PlannerContext(objective="Collect stone."),) * 2, SKILL_IDS
    )

    assert all(score.valid for score in scores)
    # Identical text-only boundaries still share one generation.
    (request,) = backend.requests
    assert request.image is None
    assert "screenshot" not in request.system_prompt
    assert "screenshot" not in request.user_prompt
    assert "local_map" in request.user_prompt
    assert scorer.last_response is not None
    assert scorer.last_response.selected_skill is SKILL_IDS[0]


def test_text_only_chat_holds_the_prompt_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Outlines is an optional dependency, so stand in for its input types.
    # Outlines renders a list of the prompt alone through the chat template's
    # text entry and hands the processor no images.
    @dataclass
    class Chat:
        messages: list[dict[str, object]]

    @dataclass
    class Image:
        image: object

    inputs = types.ModuleType("outlines.inputs")
    inputs.Chat = Chat  # type: ignore[attr-defined]
    inputs.Image = Image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "outlines", types.ModuleType("outlines"))
    monkeypatch.setitem(sys.modules, "outlines.inputs", inputs)
    request = VLMRequest(
        image=None,
        system_prompt="Plan.",
        user_prompt="Choose a skill.",
        candidate_skills=SKILL_IDS,
    )

    chat = _planner_chat(request)

    assert chat.messages == [
        {"role": "system", "content": "Plan."},
        {"role": "user", "content": ["Choose a skill."]},
    ]
    with_image = _planner_chat(
        request._replace(image=np.zeros((4, 4, 3), dtype=np.uint8))
    )
    prompt, image = with_image.messages[1]["content"]
    assert prompt == "Choose a skill."
    assert isinstance(image, Image)


def test_transformers_backend_batches_text_only_requests_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches: list[list[VLMRequest]] = []

    class RecordingGenerator:
        def batch(self, prompts: list[VLMRequest], **_: object) -> list[str]:
            batches.append(prompts)
            return ["{}"] * len(prompts)

    backend = TransformersVLMBackend(VLMPlannerConfig())
    monkeypatch.setattr(
        backend, "_load", lambda: (None, object(), lambda *_: object())
    )
    monkeypatch.setattr(
        backend, "_generator_for", lambda *_, **__: RecordingGenerator()
    )
    monkeypatch.setattr(vlm_module, "_planner_chat", lambda request: request)
    with_image = VLMRequest(
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        system_prompt="Plan.",
        user_prompt="Choose a skill.",
        candidate_skills=SKILL_IDS,
    )
    text_only = with_image._replace(image=None)

    responses = backend.generate_batch([with_image, text_only, with_image])

    assert responses == ["{}"] * 3
    assert [[request.image is None for request in batch] for batch in batches] == [
        [False, False],
        [True],
    ]
