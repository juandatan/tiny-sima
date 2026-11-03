import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import BlockType

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    TRACED_MOB_SLOTS,
    LearnedSkillExecutor,
    run_skill_probe,
    target_mob_slots,
)
from tiny_sima.evaluation.fights import (
    ARENA_RADIUS,
    SECOND_MOB_MAX_DISTANCE,
    FightCell,
    benchmark_fights,
    chase_policy,
    fight_cells,
    fight_rollout_rows,
    fight_worlds,
    summarize_fights,
)
from tiny_sima.executor.model import (
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import ALL_SKILL_IDS, SkillId, skill_index
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


def test_cells_cover_the_grid_and_reject_settings_outside_it() -> None:
    cells = fight_cells(
        (SkillId.DEFEAT_ZOMBIE,), distances=(1, 3), healths=(2, 9), nights=(False,)
    )

    assert len(cells) == 2 * 2 * 3 * 2
    with pytest.raises(ValueError, match="attacks no creature"):
        fight_cells((SkillId.COLLECT_WOOD,))
    with pytest.raises(ValueError, match="outside the arena"):
        fight_cells(distances=(ARENA_RADIUS + 1,))
    with pytest.raises(ValueError, match="sword"):
        fight_cells(swords=("iron",))


def test_target_slots_pad_every_kind_and_are_dead_for_other_skills() -> None:
    cells = (_cell(skill_id=SkillId.DEFEAT_SKELETON, mobs=2),)
    state = _entry(fight_worlds(cells, jax.random.PRNGKey(0)), 0)

    positions, mask = target_mob_slots(
        jnp.asarray(skill_index(SkillId.DEFEAT_SKELETON)), state
    )
    assert positions.shape == (TRACED_MOB_SLOTS, 2)
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
    states = fight_worlds(entries, jax.random.PRNGKey(1))

    for index, cell in enumerate(entries):
        state = _entry(states, index)
        player = np.asarray(state.player_position)
        zombies = np.asarray(state.zombies.position)[np.asarray(state.zombies.mask)]
        assert len(zombies) == cell.mobs
        assert np.abs(zombies[0] - player).sum() == cell.distance
        assert np.all(np.abs(zombies - player).sum(axis=1) <= SECOND_MOB_MAX_DISTANCE)
        assert len({tuple(zombie) for zombie in zombies}) == cell.mobs
        assert int(np.asarray(state.mob_map).sum()) == cell.mobs
        assert int(state.player_health) == cell.health
        assert int(state.inventory.wood_sword) == (cell.sword == "wood")
        assert int(state.inventory.stone_sword) == (cell.sword == "stone")
        assert (float(state.light_level) < NIGHT_LIGHT_THRESHOLD) == cell.night
        arena = np.asarray(state.map)[
            player[0] - ARENA_RADIUS : player[0] + ARENA_RADIUS + 1,
            player[1] - ARENA_RADIUS : player[1] + ARENA_RADIUS + 1,
        ]
        assert np.all(arena == BlockType.GRASS.value)
    # Each rollout has its own map.
    assert not np.array_equal(
        np.asarray(_entry(states, 0).map), np.asarray(_entry(states, 1).map)
    )


def test_probe_traces_health_and_the_target_creatures(
    executor: LearnedSkillExecutor,
) -> None:
    states = fight_worlds([_cell(distance=1, health=4)] * 3, jax.random.PRNGKey(2))

    output = run_skill_probe(
        executor,
        states,
        [SkillId.DEFEAT_ZOMBIE] * 3,
        jax.random.PRNGKey(0),
        skill_horizon=24,
    )

    assert output.health.shape == (24, 3)
    assert output.mob_positions.shape == (24, 3, TRACED_MOB_SLOTS, 2)
    assert output.mob_mask.shape == (24, 3, TRACED_MOB_SLOTS)
    for index in range(3):
        state = _entry(states, index)
        assert int(output.steps[index]) >= 1
        assert 0 <= int(output.health[0, index]) <= 4
        # Row 0 is the world after the first step: the zombie is still in
        # slot 0, at most one move from where it was staged.
        assert output.mob_mask[0, index].tolist() == [True, False, False]
        start = np.asarray(state.zombies.position[0])
        assert np.abs(output.mob_positions[0, index, 0] - start).sum() <= 1


def test_chase_reference_wins_an_easy_fight_without_missing(
    executor: LearnedSkillExecutor,
) -> None:
    cells = (_cell(distance=3, health=9, sword="stone"),)
    entries = list(cells) * 8
    states = fight_worlds(entries, jax.random.PRNGKey(3))

    output = run_skill_probe(
        executor,
        states,
        [SkillId.DEFEAT_ZOMBIE] * 8,
        jax.random.PRNGKey(0),
        skill_horizon=24,
        policy=chase_policy,
    )
    rows = fight_rollout_rows(entries, states, output, policy="chase")

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
    states = fight_worlds(entries, jax.random.PRNGKey(4))

    output = run_skill_probe(
        executor,
        states,
        [SkillId.DEFEAT_ZOMBIE] * 2,
        jax.random.PRNGKey(0),
        skill_horizon=8,
        policy=chase_policy,
    )
    rows = fight_rollout_rows(entries, states, output, policy="chase")

    assert [row["outcome"] for row in rows] == ["died", "died"]
    assert all(row["died"] and row["damage_taken"] >= 2 for row in rows)


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


def test_summary_rates_and_per_win_costs() -> None:
    def row(outcome, health_lost, sword):
        return {
            "skill_id": "defeat_zombie",
            "policy": "chase",
            "sword": sword,
            "outcome": outcome,
            "died": outcome == "died",
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
        row("died", 9, "none"),
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
    assert none["outcomes"] == {"success": 2, "died": 1, "timeout": 1}
    assert none["adjacent_share"] == 0.4
    assert none["facing_when_adjacent"] == 0.5
    assert none["first_hit_step"] == 3.0
    assert by_sword["stone"]["win_rate"] == 1.0
