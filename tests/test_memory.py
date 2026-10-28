import json

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import BlockType

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import OBSERVATION_RADIUS, SkillId
from tiny_sima.memory import MOB_TYPES, EpisodeMemory
from tiny_sima.planner.saycan import SkillOutcome


def _state(seed: int = 0):
    return make_skill_chain_environment().reset(jax.random.PRNGKey(seed)).state


def _grass_world(**blocks: list[tuple[int, int]]):
    # Every tile grass, then each named block at offsets from the player.
    state = _state()
    world_map = np.full(np.asarray(state.map).shape, BlockType.GRASS.value)
    row, column = np.asarray(state.player_position).tolist()
    for name, offsets in blocks.items():
        for d_row, d_column in offsets:
            world_map[row + d_row, column + d_column] = BlockType[name.upper()].value
    return state.replace(map=jnp.asarray(world_map, dtype=state.map.dtype))


def _absolute(state, offset: tuple[int, int]) -> tuple[int, int]:
    row, column = np.asarray(state.player_position).tolist()
    return row + offset[0], column + offset[1]


def _with_inventory(state, **counts):
    return state.replace(inventory=state.inventory.replace(**counts))


def _with_mobs(state, **mobs: list[tuple[int, int]]):
    # Place each named mob type at offsets from the player; the rest are absent.
    mob_map = np.zeros(np.asarray(state.mob_map).shape, dtype=bool)
    updates = {}
    for mob, attribute in MOB_TYPES.items():
        instances = getattr(state, attribute)
        positions = np.asarray(instances.position).copy()
        mask = np.zeros(np.asarray(instances.mask).shape, dtype=bool)
        for index, offset in enumerate(mobs.get(mob, [])):
            positions[index] = _absolute(state, offset)
            mask[index] = True
            mob_map[positions[index][0], positions[index][1]] = True
        updates[attribute] = instances.replace(
            position=jnp.asarray(positions, dtype=instances.position.dtype),
            mask=jnp.asarray(mask),
        )
    return state.replace(mob_map=jnp.asarray(mob_map), **updates)


def test_observe_keeps_only_resources_in_view() -> None:
    assert OBSERVATION_RADIUS == (3, 4)
    state = _grass_world(tree=[(1, 1), (3, -4), (4, 0), (0, 10)], water=[(-2, 3)])
    memory = EpisodeMemory().observe(state)

    assert memory.positions("tree") == tuple(
        sorted(_absolute(state, offset) for offset in ((1, 1), (3, -4)))
    )
    assert memory.positions("water") == (_absolute(state, (-2, 3)),)
    assert memory.positions("stone") == ()
    assert memory.positions("crafting_table") == ()


def test_a_mined_tree_is_dropped_only_once_back_in_view() -> None:
    state = _grass_world(tree=[(1, 1)])
    memory = EpisodeMemory().observe(state)
    tree = _absolute(state, (1, 1))

    mined = _grass_world()
    away = mined.player_position + jnp.array([0, 10])
    # Out of view, the tree is kept as last seen.
    assert memory.observe(mined.replace(player_position=away)).positions("tree") == (
        tree,
    )
    assert memory.observe(mined).positions("tree") == ()


def test_record_appends_readings_and_rules_out_after_repeated_misses() -> None:
    bare = _with_inventory(_grass_world(), wood=1)
    memory = EpisodeMemory()

    stocked = _with_inventory(bare, wood=2)
    after_success = memory.record(
        bare, SkillId.COLLECT_WOOD, SkillOutcome.SUCCESS, 7, stocked
    )
    (transition,) = after_success.transitions
    assert transition.decision_index == 0
    assert transition.outcome == "success"
    assert transition.steps == 7
    assert ("inventory.wood", 1) in transition.before
    assert ("inventory.wood", 2) in transition.after
    assert ("survival.drink", int(bare.player_drink)) in transition.before
    # Zero inventory counts are left out.
    assert all(value for field, value in transition.before if field != "survival.drink")

    once = after_success.record(bare, "collect_wood", "timeout", 64, bare)
    assert once.transitions[-1].decision_index == 1
    assert dict(once.misses)["tree"] == 1
    assert not once.ruled_out("tree")
    assert once.world_fact("world.tree_reachable") == 1

    twice = once.record(bare, SkillId.COLLECT_WOOD, SkillOutcome.TIMEOUT, 64, bare)
    assert twice.ruled_out("tree")
    assert twice.world_fact("world.tree_reachable") == 0
    # Only the searched resource is affected.
    assert twice.world_fact("world.stone_reachable") == 1

    # A sighting puts it back in play and resets the count.
    seen = twice.observe(_grass_world(tree=[(2, 2)]))
    assert dict(seen.misses)["tree"] == 0
    assert seen.world_fact("world.tree_reachable") == 1
    # So does a success.
    recovered = twice.record(bare, SkillId.COLLECT_WOOD, "success", 5, bare)
    assert not recovered.ruled_out("tree")


def test_a_failure_with_the_resource_known_is_not_a_miss() -> None:
    visible = _grass_world(stone=[(1, 0)])
    memory = EpisodeMemory()
    for _ in range(3):
        memory = memory.record(visible, SkillId.COLLECT_STONE, "timeout", 96, visible)
    assert dict(memory.misses)["stone"] == 0
    assert memory.world_fact("world.stone_reachable") == 1


def test_world_facts() -> None:
    memory = EpisodeMemory()
    assert memory.world_fact("world.tree_reachable") == 1
    assert memory.world_fact("world.stone_reachable") == 1
    assert memory.world_fact("world.water_reachable") == 1
    assert memory.world_fact("world.placement_space_reachable") == 1
    assert memory.world_fact("world.crafting_table_reachable") == 0
    assert memory.world_fact("inventory.wood") is None
    assert memory.world_fact("world.dragon_reachable") is None

    before = _with_inventory(_grass_world(), wood=2)
    placed = _with_inventory(_grass_world(crafting_table=[(0, 1)]), wood=0)
    memory = memory.record(before, SkillId.PLACE_TABLE, "success", 1, placed)
    assert memory.world_fact("world.crafting_table_reachable") == 1


def test_summary_is_json_ready_and_bounded() -> None:
    state = _with_inventory(_grass_world(tree=[(2, -1), (1, 3)]), wood=1)
    memory = EpisodeMemory()
    for index in range(10):
        memory = memory.record(
            _with_inventory(state, wood=index),
            SkillId.COLLECT_WOOD,
            "success",
            3,
            _with_inventory(state, wood=index + 1),
        )

    summary = memory.summary(state)
    json.dumps(summary)
    assert set(summary) == {"recent", "resources", "mobs"}
    assert len(summary["recent"]) == 6
    assert summary["recent"][-1] == "collect_wood success: wood 9->10"
    assert summary["resources"]["tree"] == {
        "known": 2,
        "nearest_offset": [2, -1],
        "ruled_out": False,
    }
    assert summary["resources"]["crafting_table"]["nearest_offset"] is None
    assert memory.summary()["resources"]["tree"]["nearest_offset"] is None
    unchanged = memory.record(state, SkillId.DRINK_WATER, "timeout", 64, state)
    assert unchanged.summary(max_transitions=1)["recent"] == [
        "drink_water timeout (target not seen): no change"
    ]


def test_updates_return_new_memories() -> None:
    state = _with_mobs(_grass_world(tree=[(1, 1)]), cow=[(1, 2)])
    memory = EpisodeMemory()

    observed = memory.observe(state)
    recorded = memory.record(state, SkillId.COLLECT_WOOD, "timeout", 64, state)

    assert observed is not memory and recorded is not memory
    assert memory.positions("tree") == ()
    assert memory.mob_sightings("cow") == ()
    assert memory.transitions == ()
    assert len(recorded.transitions) == 1


def test_every_tracked_block_is_remembered() -> None:
    state = _grass_world(
        coal=[(1, 0)],
        iron=[(-1, 0)],
        diamond=[(0, 2)],
        sand=[(2, 2)],
        lava=[(-3, 4)],
        furnace=[(0, -1)],
        plant=[(1, -2)],
        ripe_plant=[(-2, -2)],
    )
    memory = EpisodeMemory().observe(state)

    for block, offset in (
        ("coal", (1, 0)),
        ("iron", (-1, 0)),
        ("diamond", (0, 2)),
        ("sand", (2, 2)),
        ("lava", (-3, 4)),
        ("furnace", (0, -1)),
        ("plant", (1, -2)),
        ("ripe_plant", (-2, -2)),
    ):
        assert memory.positions(block) == (_absolute(state, offset),)


def test_world_facts_for_every_block() -> None:
    memory = EpisodeMemory()
    # Generated blocks are believed present until ruled out.
    for block in ("coal", "iron", "diamond", "sand", "lava"):
        assert memory.world_fact(f"world.{block}_reachable") == 1
    # Placed blocks count only once seen.
    for block in ("furnace", "plant", "ripe_plant"):
        assert memory.world_fact(f"world.{block}_reachable") == 0

    seen = memory.observe(_grass_world(furnace=[(1, 1)], ripe_plant=[(0, 2)]))
    assert seen.world_fact("world.furnace_reachable") == 1
    assert seen.world_fact("world.ripe_plant_reachable") == 1
    assert seen.world_fact("world.plant_reachable") == 0


def test_mobs_in_view_are_remembered_with_when() -> None:
    state = _with_mobs(
        _grass_world(),
        zombie=[(1, 2), (3, -4)],
        cow=[(-1, 0)],
        # Out of the 7x9 view, so never seen.
        skeleton=[(0, 6)],
    )
    state = state.replace(timestep=42)
    memory = EpisodeMemory().observe(state)

    zombies = memory.mob_sightings("zombie")
    assert [sighting.position for sighting in zombies] == sorted(
        _absolute(state, offset) for offset in ((1, 2), (3, -4))
    )
    assert all(sighting.boundary == 0 for sighting in zombies)
    assert all(sighting.timestep == 42 for sighting in zombies)
    assert [sighting.position for sighting in memory.mob_sightings("cow")] == [
        _absolute(state, (-1, 0))
    ]
    assert memory.mob_sightings("skeleton") == ()


def test_a_mob_gone_from_its_tile_is_dropped_once_back_in_view() -> None:
    state = _with_mobs(_grass_world(), zombie=[(1, 2)])
    memory = EpisodeMemory().observe(state)
    zombie = _absolute(state, (1, 2))

    gone = _with_mobs(_grass_world())
    away = gone.player_position + jnp.array([0, 10])
    kept = memory.observe(gone.replace(player_position=away))
    assert [sighting.position for sighting in kept.mob_sightings("zombie")] == [
        zombie
    ]
    assert memory.observe(gone).mob_sightings("zombie") == ()

    # A mob seen on another tile is a new sighting there.
    moved = memory.observe(_with_mobs(_grass_world(), zombie=[(2, 2)]))
    assert [sighting.position for sighting in moved.mob_sightings("zombie")] == [
        _absolute(state, (2, 2))
    ]


def test_an_unseen_mob_is_forgotten_after_a_few_boundaries() -> None:
    seen = _with_mobs(_grass_world(), cow=[(1, 1)])
    empty = _grass_world()
    away = empty.replace(player_position=empty.player_position + jnp.array([0, 10]))
    memory = EpisodeMemory(mob_forget_after=2).observe(seen)

    for boundaries in range(1, 4):
        memory = memory.record(away, SkillId.PLACE_TABLE, "timeout", 1, away)
        assert len(memory.transitions) == boundaries
        assert bool(memory.mob_sightings("cow")) is (boundaries <= 2)


def test_summary_reports_every_type_compactly() -> None:
    state = _with_mobs(
        _grass_world(tree=[(2, -1)], coal=[(1, 0), (3, 3)]),
        zombie=[(1, 2), (-3, 4)],
    )
    memory = EpisodeMemory().observe(state)
    bare = _grass_world()
    for _ in range(2):
        memory = memory.record(bare, SkillId.COLLECT_STONE, "timeout", 96, bare)
    # The zombies left the view at the first boundary, and are dropped.
    assert memory.mob_sightings("zombie") == ()

    memory = memory.observe(state)
    away = state.replace(player_position=state.player_position + jnp.array([0, 10]))
    memory = memory.record(away, SkillId.PLACE_TABLE, "timeout", 1, away)

    summary = memory.summary(state)
    json.dumps(summary)
    # The four skill blocks always; others only when seen or ruled out.
    assert list(summary["resources"]) == [
        "tree",
        "stone",
        "water",
        "crafting_table",
        "coal",
    ]
    assert summary["resources"]["coal"] == {
        "known": 2,
        "nearest_offset": [1, 0],
        "ruled_out": False,
    }
    assert summary["resources"]["stone"] == {
        "known": 0,
        "nearest_offset": None,
        "ruled_out": True,
    }
    assert summary["mobs"] == {
        "zombie": {"known": 2, "nearest_offset": [1, 2], "last_seen_ago": 1}
    }
    assert memory.summary()["mobs"]["zombie"]["nearest_offset"] is None
    assert EpisodeMemory().summary(state)["mobs"] == {}


def test_record_stores_the_cause_and_inventory_extremes() -> None:
    bare = _with_inventory(_grass_world(), wood=1)
    memory = EpisodeMemory().record(
        bare,
        SkillId.COLLECT_WOOD,
        "timeout",
        64,
        bare,
        inventory_extremes={"inventory.wood": (0, 2), "inventory.sapling": (0, 1)},
    )
    (transition,) = memory.transitions
    assert transition.cause == "target_not_seen"
    assert transition.inventory_extremes == (
        ("inventory.sapling", 0, 1),
        ("inventory.wood", 0, 2),
    )

    visible = _grass_world(tree=[(1, 0)])
    memory = memory.record(
        visible,
        SkillId.COLLECT_WOOD,
        SkillOutcome.SUCCESS,
        3,
        visible,
        inventory_extremes=[("inventory.wood", 0, 1)],
    )
    assert memory.transitions[-1].cause is None
    assert memory.transitions[-1].inventory_extremes == (("inventory.wood", 0, 1),)
    memory = memory.record(visible, SkillId.COLLECT_WOOD, "timeout", 64, visible)
    assert memory.transitions[-1].cause == "target_not_reached"
    assert memory.transitions[-1].inventory_extremes is None
    assert memory.summary(max_transitions=1)["recent"] == [
        "collect_wood timeout (target not reached): no change"
    ]
