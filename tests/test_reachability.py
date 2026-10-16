from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import OBS_DIM, BlockType

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    WALKABLE_BLOCKS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    block_mask,
    interaction_distance,
    skill_feasibility_score,
    skill_feasible,
    skill_interaction_distance,
    skill_reachability_report,
    travel_distance_map,
    view_mask,
    walkable_mask,
)


class _FakeState:
    """Minimal stand-in exposing only what reachability reads."""

    def __init__(self, world_map: np.ndarray, position: tuple[int, int]) -> None:
        self.map = jnp.asarray(world_map)
        self.player_position = jnp.asarray(position)
        self.player_health = jnp.asarray(9.0)


def _grid(layout: list[str]) -> np.ndarray:
    """Build a map from a character sketch: ``.`` grass, ``#`` stone, ``T`` tree."""

    blocks = {
        ".": BlockType.GRASS.value,
        "#": BlockType.STONE.value,
        "T": BlockType.TREE.value,
        "~": BlockType.WATER.value,
        "C": BlockType.CRAFTING_TABLE.value,
    }
    return np.asarray(
        [[blocks[character] for character in row] for row in layout],
        dtype=np.int32,
    )


def _reference_distances(
    world_map: np.ndarray,
    position: tuple[int, int],
    unreachable: int,
    walkable: set[int],
    inside_view: np.ndarray,
) -> np.ndarray:
    """Independent queue-based breadth-first search used as an oracle."""

    distances = np.full(world_map.shape, unreachable, dtype=np.int64)
    distances[position] = 0
    queue: deque[tuple[int, int]] = deque([position])
    while queue:
        row, column = queue.popleft()
        for row_offset, column_offset in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            next_row = row + row_offset
            next_column = column + column_offset
            if not (
                0 <= next_row < world_map.shape[0]
                and 0 <= next_column < world_map.shape[1]
            ):
                continue
            if not inside_view[next_row, next_column]:
                continue
            if world_map[next_row, next_column] not in walkable:
                continue
            if distances[next_row, next_column] <= distances[row, column] + 1:
                continue
            distances[next_row, next_column] = distances[row, column] + 1
            queue.append((next_row, next_column))
    return np.minimum(distances, unreachable)


def _chain_state() -> Any:
    return make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state


def test_distance_map_routes_around_a_wall() -> None:
    state = _FakeState(
        _grid(
            [
                ".........",
                ".........",
                "####.####",
                ".........",
                ".........",
            ]
        ),
        position=(0, 0),
    )
    config = ReachabilityConfig(max_steps=64, view_radius=None)

    distances = np.asarray(travel_distance_map(state, config))

    assert distances[0, 0] == 0
    assert distances[0, 4] == 4
    # The only gap is at column four, so the far side costs the detour.
    assert distances[3, 4] == 7
    assert distances[3, 0] == 11
    # Stone is not walkable and never gets a finite standing distance.
    assert distances[2, 0] == config.unreachable


def test_distance_map_matches_an_independent_search() -> None:
    state = _chain_state()
    config = ReachabilityConfig(max_steps=128, view_radius=None)
    world_map = np.asarray(state.map)
    position = (
        int(np.asarray(state.player_position)[0]),
        int(np.asarray(state.player_position)[1]),
    )

    distances = np.asarray(travel_distance_map(state, config))
    reference = _reference_distances(
        world_map,
        position,
        config.unreachable,
        {block.value for block in WALKABLE_BLOCKS},
        np.asarray(view_mask(state, config.view_radius)),
    )

    # Relaxation is capped at ``max_steps`` iterations, so compare only the
    # tiles the oracle itself places inside that budget.
    resolved = reference <= config.max_steps
    np.testing.assert_array_equal(distances[resolved], reference[resolved])


def test_distance_map_is_confined_to_the_view_window() -> None:
    state = _chain_state()
    config = DEFAULT_REACHABILITY

    distances = np.asarray(travel_distance_map(state, config))
    inside = np.asarray(view_mask(state, config.view_radius))

    assert config.view_radius == (OBS_DIM[0] // 2, OBS_DIM[1] // 2)
    assert np.all(distances[~inside] == config.unreachable)
    assert np.any(distances[inside] < config.unreachable)


def test_whole_map_search_sees_at_least_as_much_as_the_window() -> None:
    state = _chain_state()
    windowed = skill_reachability_report(state, DEFAULT_REACHABILITY)
    whole = skill_reachability_report(
        state,
        ReachabilityConfig(max_steps=128, view_radius=None),
    )

    for skill_id in SKILL_IDS:
        windowed_steps = windowed[skill_id]
        if windowed_steps is None:
            continue
        whole_steps = whole[skill_id]
        assert whole_steps is not None
        assert whole_steps <= windowed_steps


def test_interaction_distance_counts_the_interaction_step() -> None:
    state = _FakeState(
        _grid(
            [
                "....",
                ".T..",
                "....",
            ]
        ),
        position=(1, 0),
    )
    config = ReachabilityConfig(max_steps=16, view_radius=None)

    # The player already stands next to the tree, so only the DO step remains.
    assert int(
        np.asarray(
            interaction_distance(
                state,
                block_mask(state, BlockType.TREE),
                config,
            )
        )
    ) == 1
    # A tree is not walkable, so it has no standing distance of its own.
    assert np.asarray(travel_distance_map(state, config))[1, 1] == (
        config.unreachable
    )


def test_crafting_uses_the_eight_neighbourhood() -> None:
    state = _FakeState(
        _grid(
            [
                "...",
                ".C.",
                "...",
            ]
        ),
        position=(0, 0),
    )
    config = ReachabilityConfig(max_steps=16, view_radius=None)

    # The player stands diagonally adjacent, which Craftax counts as near.
    assert int(
        np.asarray(
            skill_interaction_distance(
                SkillGoal(SkillId.CRAFT_WOOD_PICKAXE),
                state,
                config,
            )
        )
    ) == 1
    # Collecting requires facing the target, so it needs an orthogonal tile.
    assert int(
        np.asarray(
            skill_interaction_distance(SkillGoal(SkillId.COLLECT_WOOD), state, config)
        )
    ) >= config.unreachable


def test_missing_target_is_unreachable_not_merely_distant() -> None:
    state = _FakeState(_grid(["...", "...", "..."]), position=(1, 1))
    config = ReachabilityConfig(max_steps=16, view_radius=None)

    report = skill_reachability_report(state, config)

    assert report[SkillId.COLLECT_WOOD] is None
    assert report[SkillId.COLLECT_STONE] is None
    # Any walkable tile accepts a crafting table, including the faced one.
    assert report[SkillId.PLACE_TABLE] == 1


def test_feasibility_score_grades_rather_than_vetoes() -> None:
    state = _chain_state()
    config = DEFAULT_REACHABILITY
    distances = travel_distance_map(state, config)

    scores = {
        skill_id: float(
            np.asarray(
                skill_feasibility_score(
                    SkillGoal(skill_id),
                    state,
                    config,
                    distances=distances,
                )
            )
        )
        for skill_id in SKILL_IDS
    }
    reachable = {
        skill_id: bool(
            np.asarray(
                skill_feasible(
                    SkillGoal(skill_id),
                    state,
                    config,
                    distances=distances,
                )
            )
        )
        for skill_id in SKILL_IDS
    }

    # No crafting table exists yet, so crafting fails its precondition outright.
    assert scores[SkillId.CRAFT_WOOD_PICKAXE] == 0.0
    assert not reachable[SkillId.CRAFT_WOOD_PICKAXE]
    # A reachable target scores one; a far one is demoted but still selectable.
    assert scores[SkillId.COLLECT_WOOD] == 1.0
    assert reachable[SkillId.COLLECT_WOOD]
    for skill_id, score in scores.items():
        if score > 0.0 and not reachable[skill_id]:
            assert score == pytest.approx(config.unknown_feasibility)


def test_hard_gate_remains_available_as_an_ablation() -> None:
    state = _FakeState(
        _grid(
            [
                "....T",
                ".....",
                ".....",
            ]
        ),
        position=(2, 0),
    )
    graded = ReachabilityConfig(max_steps=2, view_radius=None)
    strict = ReachabilityConfig(
        max_steps=2,
        view_radius=None,
        unknown_feasibility=0.0,
    )
    goal = SkillGoal(SkillId.COLLECT_WOOD, horizon=2)

    # The tree is six steps away but the horizon is two.
    assert float(
        np.asarray(skill_feasibility_score(goal, state, graded))
    ) == pytest.approx(graded.unknown_feasibility)
    assert float(np.asarray(skill_feasibility_score(goal, state, strict))) == 0.0


def test_config_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        ReachabilityConfig(max_steps=0)
    with pytest.raises(ValueError, match="view_radius"):
        ReachabilityConfig(view_radius=(-1, 2))
    with pytest.raises(ValueError, match="unknown_feasibility"):
        ReachabilityConfig(unknown_feasibility=1.5)


def test_walkable_mask_excludes_lava_and_solids() -> None:
    state = _chain_state()
    mask = np.asarray(walkable_mask(state))
    world_map = np.asarray(state.map)

    assert not np.any(mask[world_map == BlockType.LAVA.value])
    assert not np.any(mask[world_map == BlockType.STONE.value])
    assert not np.any(mask[world_map == BlockType.WATER.value])
    assert np.all(mask[world_map == BlockType.GRASS.value])
