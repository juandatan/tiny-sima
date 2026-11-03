import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import clear_mobs, make_skill_chain_environment
from tiny_sima.envs.skills import add_mob
from tiny_sima.evaluation.chains import (
    PROBE_OUTCOMES,
    BatchedProbeOutput,
    LearnedSkillExecutor,
    run_skill_probe,
)
from tiny_sima.evaluation.fights import (
    ARENA_RADIUS,
    ERRAND_TARGET_DISTANCE,
    MAX_DISTANCE,
    SECOND_MOB_MAX_DISTANCE,
    FightCell,
    FightSnapshot,
    benchmark_fights,
    chase_policy,
    fight_cells,
    fight_rollout_rows,
    fight_snapshot,
    fight_worlds,
    summarize_fights,
)
from tiny_sima.executor.model import (
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    MOB_SLOTS,
    SkillId,
    skill_index,
    target_mob_slots,
)
from tiny_sima.planner.saycan import SkillOutcome
from tiny_sima.planner.shelter import NIGHT_LIGHT_THRESHOLD


@pytest.fixture(scope="module")
def executor() -> LearnedSkillExecutor:
    """An untrained executor that knows every skill, the fights included."""

    config = SkillPPOConfig(
        num_skills=len(ALL_SKILL_IDS),
        hidden_size=16,
        skill_embedding_size=4,
        monitor_gpu=False,
    ).validate()
    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=config.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        environment.bundle.spec.observation_shape,
    )
    return LearnedSkillExecutor(params=params, config=config, model=model)


def _cell(**changes) -> FightCell:
    setting = {
        "skill_id": SkillId.DEFEAT_ZOMBIE,
        "distance": 2,
        "health": 9,
        "sword": "none",
        "mobs": 1,
        "night": False,
    }
    return FightCell(**{**setting, **changes})


def _entry(states, index: int):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], states)


def _stage(entries, seed: int):
    return fight_worlds(
        entries, jax.random.split(jax.random.PRNGKey(seed), len(entries))
    )


def _probe(executor, entries, states, *, horizon: int, policy=chase_policy):
    return run_skill_probe(
        executor,
        states,
        [entry.skill_id for entry in entries],
        jax.random.PRNGKey(0),
        skill_horizon=horizon,
        policy=policy,
        recorder=fight_snapshot,
    )


def test_cells_cover_the_grid_and_reject_settings_outside_it() -> None:
    cells = fight_cells(
        (SkillId.DEFEAT_ZOMBIE,), distances=(1, 3), healths=(2, 9), nights=(False,)
    )

    assert len(cells) == 2 * 2 * 3 * 2
    with pytest.raises(ValueError, match="neither a fight nor an errand"):
        fight_cells((SkillId.PLACE_TABLE,))
    assert {cell.skill_id for cell in fight_cells((SkillId.DRINK_WATER,))} == {
        SkillId.DRINK_WATER
    }
    with pytest.raises(ValueError, match="outside the arena"):
        fight_cells(distances=(MAX_DISTANCE + 1,))
    with pytest.raises(ValueError, match="sword"):
        fight_cells(swords=("iron",))
    with pytest.raises(ValueError, match="one creature or two"):
        fight_cells(mob_counts=(3,))


def test_target_slots_pad_every_kind_and_are_dead_for_other_skills() -> None:
    cells = (_cell(skill_id=SkillId.DEFEAT_SKELETON, mobs=2),)
    state = _entry(_stage(cells, 0), 0)

    positions, mask = target_mob_slots(
        jnp.asarray(skill_index(SkillId.DEFEAT_SKELETON)), state
    )
    assert positions.shape == (MOB_SLOTS, 2)
    # Craftax keeps two skeleton slots; the third is padding.
    assert mask.tolist() == [True, True, False]
    _, mask = target_mob_slots(jnp.asarray(skill_index(SkillId.COLLECT_WOOD)), state)
    assert not bool(jnp.any(mask))


def test_worlds_stage_the_cell_they_belong_to() -> None:
    cells = (
        _cell(distance=1, health=3, sword="stone", mobs=2, night=True),
        _cell(distance=3, health=7, sword="wood"),
    )
    entries = [cell for cell in cells for _ in range(4)]
    states = _stage(entries, 1)

    for index, cell in enumerate(entries):
        state = _entry(states, index)
        player = np.asarray(state.player_position)
        zombies = np.asarray(state.zombies.position)[np.asarray(state.zombies.mask)]
        assert len(zombies) == cell.mobs
        assert np.abs(zombies[0] - player).sum() == cell.distance
        apart = np.abs(zombies - player).sum(axis=1)
        assert np.all(apart >= cell.distance)
        assert np.all(apart <= max(cell.distance, SECOND_MOB_MAX_DISTANCE))
        assert len({tuple(zombie) for zombie in zombies}) == cell.mobs
        assert int(np.asarray(state.mob_map).sum()) == cell.mobs
        assert int(state.player_health) == cell.health
        assert int(state.inventory.wood_sword) == (cell.sword == "wood")
        assert int(state.inventory.stone_sword) == (cell.sword == "stone")
        assert (float(state.light_level) < NIGHT_LIGHT_THRESHOLD) == cell.night
        if not cell.night:
            # Training scenarios fight in full light.
            assert float(state.light_level) == 1.0
        arena = np.asarray(state.map)[
            player[0] - ARENA_RADIUS : player[0] + ARENA_RADIUS + 1,
            player[1] - ARENA_RADIUS : player[1] + ARENA_RADIUS + 1,
        ]
        assert np.all(arena == BlockType.GRASS.value)
    # Each rollout has its own map.
    assert not np.array_equal(
        np.asarray(_entry(states, 0).map), np.asarray(_entry(states, 1).map)
    )


def test_an_errand_is_staged_with_its_target_and_zombies_close() -> None:
    for cell, block in (
        (_cell(skill_id=SkillId.DRINK_WATER, distance=3, mobs=2), BlockType.WATER),
        (_cell(skill_id=SkillId.COLLECT_WOOD, distance=1), BlockType.TREE),
    ):
        entries = [cell] * 6
        states = _stage(entries, 2)
        for index in range(len(entries)):
            state = _entry(states, index)
            player = np.asarray(state.player_position)
            arena = np.asarray(state.map)[
                player[0] - ARENA_RADIUS : player[0] + ARENA_RADIUS + 1,
                player[1] - ARENA_RADIUS : player[1] + ARENA_RADIUS + 1,
            ]
            (target,) = np.argwhere(arena == block.value) - ARENA_RADIUS
            assert np.abs(target).sum() == ERRAND_TARGET_DISTANCE
            assert np.sum(arena != BlockType.GRASS.value) == 1
            zombies = np.asarray(state.zombies.position)[np.asarray(state.zombies.mask)]
            assert len(zombies) == cell.mobs
            assert not any(
                np.array_equal(zombie - player, target) for zombie in zombies
            )
            # The drink starts below full, so drinking can succeed.
            assert (int(state.player_drink) < 9) == (block is BlockType.WATER)
        # The probe reads an errand's zombies as the fight's targets.
        record = fight_snapshot(
            jnp.full(len(entries), skill_index(cell.skill_id)), states
        )
        assert np.asarray(record.mob_mask).sum(axis=1).tolist() == [cell.mobs] * 6


def test_probe_records_the_fight_before_and_after_every_step(
    executor: LearnedSkillExecutor,
) -> None:
    entries = [_cell(distance=1, health=4)] * 3
    states = _stage(entries, 2)

    output = _probe(executor, entries, states, horizon=24, policy=None)

    record = output.record
    assert record.health.shape == (25, 3)
    assert record.position.shape == (25, 3, 2)
    assert record.mob_positions.shape == (25, 3, MOB_SLOTS, 2)
    assert record.mob_mask.shape == (25, 3, MOB_SLOTS)
    for index in range(3):
        state = _entry(states, index)
        assert int(output.steps[index]) >= 1
        # Row 0 is the staged world, before the first step.
        assert int(record.health[0, index]) == 4
        assert record.mob_mask[0, index].tolist() == [True, False, False]
        assert np.array_equal(
            record.mob_positions[0, index, 0], np.asarray(state.zombies.position[0])
        )
        assert np.array_equal(
            record.position[0, index], np.asarray(state.player_position)
        )


def test_chase_reference_wins_an_easy_fight_without_missing(
    executor: LearnedSkillExecutor,
) -> None:
    cells = (_cell(distance=3, health=9, sword="stone"),)
    entries = list(cells) * 8
    states = _stage(entries, 3)

    output = _probe(executor, entries, states, horizon=24)
    rows = fight_rollout_rows(entries, output, policy="chase")

    assert len(rows) == 8
    for row in rows:
        assert row["outcome"] == "success"
        # A stone sword deals three, so a five-health zombie takes two hits.
        assert row["hits"] == 2
        assert row["misses"] == 0
        assert row["retreats"] == 0
        assert row["adjacent_steps"] >= row["facing_steps"] >= row["hits"]
        assert row["first_hit_step"] is not None
        assert row["health_lost"] == 9 - row["final_health"]
        assert not row["died"]


def test_a_hopeless_fight_is_counted_as_a_death(
    executor: LearnedSkillExecutor,
) -> None:
    # Two health beside a zombie whose cooldown is spent: the first step kills.
    cells = (_cell(distance=1, health=2),)
    entries = list(cells) * 2
    states = _stage(entries, 4)

    output = _probe(executor, entries, states, horizon=8)
    rows = fight_rollout_rows(entries, output, policy="chase")

    # The probe calls it a lost precondition; the row says why.
    assert [row["outcome"] for row in rows] == ["precondition_lost"] * 2
    assert all(row["died"] and row["damage_taken"] >= 2 for row in rows)


def _two_zombies_beside(state, direction: int):
    """Put zombies left and right of the player, facing ``direction``."""

    position = state.player_position
    state = clear_mobs(state)
    state = add_mob(state, "zombie", position + jnp.asarray([0, -1]))
    state = add_mob(state, "zombie", position + jnp.asarray([0, 1]), slot=1)
    return state.replace(player_direction=jnp.asarray(direction, dtype=jnp.int32))


def test_chase_hits_whichever_adjacent_creature_it_faces() -> None:
    entries = [_cell(distance=1)] * 2
    states = _stage(entries, 5)
    # Directions 1 and 2 face left and right, at columns -1 and +1.
    faced = [
        _two_zombies_beside(_entry(states, index), 1 + index) for index in range(2)
    ]
    stacked = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *faced)

    actions = chase_policy(jnp.full((2,), skill_index(SkillId.DEFEAT_ZOMBIE)), stacked)

    assert actions.tolist() == [Action.DO.value] * 2


def _snapshot(position, direction, health, mobs):
    mob_positions = np.zeros((1, MOB_SLOTS, 2), dtype=np.int32)
    mob_positions[0, : len(mobs)] = mobs
    return FightSnapshot(
        position=np.asarray([position], dtype=np.int32),
        direction=np.asarray([direction], dtype=np.int32),
        health=np.asarray([health], dtype=np.int32),
        mob_positions=mob_positions,
        mob_mask=np.arange(MOB_SLOTS)[None] < len(mobs),
    )


def _rows_for(snapshots, actions):
    record = jax.tree_util.tree_map(lambda *rows: np.stack(rows), *snapshots)
    output = BatchedProbeOutput(
        outcome=np.asarray([PROBE_OUTCOMES.index(SkillOutcome.TIMEOUT)]),
        steps=np.asarray([len(actions)]),
        positions=record.position[1:],
        directions=record.direction[1:],
        actions=np.asarray(actions, dtype=np.int32)[:, None],
        record=record,
    )
    return fight_rollout_rows([_cell()], output, policy="chase")


def test_stepping_from_one_creature_to_another_is_not_a_retreat() -> None:
    zombies = [(10, 10), (12, 11)]
    # Beside the first, step down beside the second, then step away.
    rows = _rows_for(
        [
            _snapshot((10, 11), 1, 9, zombies),
            _snapshot((11, 11), 4, 9, zombies),
            _snapshot((11, 12), 2, 9, zombies),
        ],
        [Action.DOWN.value, Action.RIGHT.value],
    )

    assert rows[0]["adjacent_steps"] == 2
    assert rows[0]["retreats"] == 1


def test_benchmark_rejects_empty_settings(executor: LearnedSkillExecutor) -> None:
    cells = fight_cells((SkillId.EAT_COW,), distances=(1,), healths=(9,))
    with pytest.raises(ValueError, match="rollouts_per_cell"):
        benchmark_fights(executor, cells, rollouts_per_cell=0)
    with pytest.raises(ValueError, match="batch_size"):
        benchmark_fights(executor, cells, batch_size=0)


def test_benchmark_pairs_both_policies_on_every_cell(
    executor: LearnedSkillExecutor,
) -> None:
    cells = fight_cells(
        (SkillId.DEFEAT_ZOMBIE, SkillId.EAT_COW),
        distances=(2,),
        healths=(9,),
        swords=("none",),
        mob_counts=(1,),
        nights=(False,),
    )

    rows = benchmark_fights(
        executor, cells, rollouts_per_cell=3, skill_horizon=16, batch_size=4
    )

    assert len(rows) == 2 * 2 * 3
    # Padding gets no row, whatever the chunking.
    unchunked = benchmark_fights(
        executor, cells, rollouts_per_cell=3, skill_horizon=16, batch_size=3
    )
    fields = ("skill_id", "distance", "health", "policy")
    assert [[row[name] for name in fields] for row in unchunked] == [
        [row[name] for name in fields] for row in rows
    ]
    summary = summarize_fights(rows)
    assert [(row["skill_id"], row["policy"]) for row in summary] == [
        ("defeat_zombie", "chase"),
        ("defeat_zombie", "executor"),
        ("eat_cow", "chase"),
        ("eat_cow", "executor"),
    ]
    # Eating needs food below full, which the staged vitals leave room for.
    assert all(
        row["outcome"] != "infeasible" for row in rows if row["skill_id"] == "eat_cow"
    )
    # An errand has no chase reference to set against.
    errand = fight_cells(
        (SkillId.DRINK_WATER,),
        distances=(2,),
        healths=(9,),
        swords=("none",),
        mob_counts=(1,),
        nights=(False,),
    )
    rows = benchmark_fights(executor, errand, rollouts_per_cell=2, skill_horizon=8)
    assert [(row["skill_id"], row["policy"]) for row in rows] == [
        ("drink_water", "executor")
    ] * 2


def test_summary_rates_and_per_win_costs() -> None:
    def row(outcome, health_lost, sword, died=False):
        return {
            "skill_id": "defeat_zombie",
            "policy": "chase",
            "sword": sword,
            "outcome": outcome,
            "died": died,
            "steps": 10,
            "health_lost": health_lost,
            "damage_taken": max(health_lost, 0),
            "adjacent_steps": 4,
            "facing_steps": 2,
            "hits": 1,
            "misses": 0,
            "retreats": 0,
            "first_hit_step": 3 if outcome == "success" else None,
        }

    rows = [
        row("success", 2, "none"),
        row("success", 4, "none"),
        row("precondition_lost", 9, "none", died=True),
        row("timeout", 0, "none"),
        row("success", 2, "stone"),
    ]

    by_sword = {
        entry["sword"]: entry for entry in summarize_fights(rows, by=("sword",))
    }

    none = by_sword["none"]
    assert none["count"] == 4
    assert none["win_rate"] == 0.5
    assert none["death_rate"] == 0.25
    assert none["win_health_lost"] == 3.0
    assert none["outcomes"] == {"success": 2, "precondition_lost": 1, "timeout": 1}
    assert none["adjacent_share"] == 0.4
    assert none["facing_when_adjacent"] == 0.5
    assert none["first_hit_step"] == 3.0
    assert by_sword["stone"]["win_rate"] == 1.0
