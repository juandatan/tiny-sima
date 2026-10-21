import dataclasses
import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tests.test_chains import _beside_a_table, _make_random_learned_executor
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    LearnedSkillExecutor,
    SkillProbe,
    SkillRolloutTrace,
    compare_paired_successes,
    evaluate_learned_skill_chain_batched,
    paired_success_summary,
    trace_skills_batched,
)
from tiny_sima.evaluation.executor_diagnostics import (
    COLLECT_FIRST_PLAN,
    COUNTERFACTUAL_EDITS,
    ChainBoundary,
    CounterfactualEdit,
    _apply_edit,
    benchmark_executor,
    chain_boundaries,
    compare_executor_benchmarks,
    distance_band,
    failed_rollout_features,
    json_ready_benchmark,
    probe_counterfactuals,
    rollout_features,
    success_table,
    summarize_rollout_features,
)
from tiny_sima.goals import OBSERVATION_RADIUS, SKILL_IDS, SkillId, view_mask
from tiny_sima.planner.saycan import SkillOutcome

_EDITS = {edit.name: edit for edit in COUNTERFACTUAL_EDITS}


@pytest.fixture(scope="module")
def executor() -> LearnedSkillExecutor:
    return _make_random_learned_executor()


@pytest.fixture(scope="module")
def chain_result(executor: LearnedSkillExecutor) -> dict[str, Any]:
    return evaluate_learned_skill_chain_batched(
        executor,
        plan=COLLECT_FIRST_PLAN,
        num_episodes=4,
        seed=3,
        skill_horizon=4,
        record_boundaries=True,
    )


def _reset_states(seed: int, num_episodes: int) -> list[Any]:
    """Reproduce the worlds a batched chain evaluation starts from."""

    environment = make_skill_chain_environment(distractor_count=3)
    reset_rng, _ = jax.random.split(jax.random.PRNGKey(seed))
    return [
        environment.reset(key).state
        for key in jax.random.split(reset_rng, num_episodes)
    ]


def _assert_same_state(actual: Any, expected: Any) -> None:
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf)
        )


def test_distance_bands_cover_training_and_chain_distances() -> None:
    assert distance_band(None) == "unreachable"
    assert distance_band(0) == "0"
    assert distance_band(1) == "1"
    assert [distance_band(value) for value in (2, 4)] == ["2-4", "2-4"]
    assert [distance_band(value) for value in (5, 7)] == ["5-7", "5-7"]
    assert distance_band(30) == "8+"


def test_paired_summary_counts_discordant_pairs() -> None:
    summary = paired_success_summary(
        [True, True, False, False],
        [True, False, True, True],
    )

    assert summary["num_pairs"] == 4
    assert (summary["both"], summary["neither"]) == (1, 0)
    assert (summary["reference_only"], summary["candidate_only"]) == (1, 2)
    assert summary["difference"] == pytest.approx(1 / 4)
    with pytest.raises(ValueError):
        paired_success_summary([], [])
    with pytest.raises(ValueError):
        paired_success_summary([True], [True, False])


def test_batched_chain_runs_the_requested_plan_episode_by_episode(
    chain_result: dict[str, Any],
) -> None:
    assert chain_result["plan"] == [skill.value for skill in COLLECT_FIRST_PLAN]
    assert chain_result["reference_skills"] == [
        skill.value for skill in WOOD_TO_STONE_CHAIN.reference_skills
    ]
    assert len(chain_result["episode_successes"]) == 4
    assert len(chain_result["episode_terminations"]) == 4
    assert chain_result["success_rate"] == pytest.approx(
        np.mean(chain_result["episode_successes"])
    )
    for outcomes, steps in zip(
        chain_result["first_attempt_outcomes"],
        chain_result["first_attempt_steps"],
        strict=True,
    ):
        assert len(outcomes) == len(steps) == len(COLLECT_FIRST_PLAN)
        # Every episode starts its first plan step; later steps run only in
        # order, so the reached steps form a prefix.
        assert outcomes[0] is not None
        reached = [outcome is not None for outcome in outcomes]
        assert reached == sorted(reached, reverse=True)
        for outcome, step_count in zip(outcomes, steps, strict=True):
            assert (outcome is None) == (step_count is None)
            if outcome is not None:
                assert 0 <= step_count <= 4
    # A batched result pairs against itself like any other.
    assert compare_paired_successes(chain_result, chain_result)["p_value"] == 1.0


def test_boundary_recording_does_not_change_the_rollout(
    executor: LearnedSkillExecutor,
    chain_result: dict[str, Any],
) -> None:
    unrecorded = evaluate_learned_skill_chain_batched(
        executor,
        plan=COLLECT_FIRST_PLAN,
        num_episodes=4,
        seed=3,
        skill_horizon=4,
    )

    assert "boundary_states" not in unrecorded
    for key in ("episode_successes", "first_attempt_outcomes", "first_attempt_steps"):
        assert unrecorded[key] == chain_result[key]


def test_first_boundary_is_the_reset_world(chain_result: dict[str, Any]) -> None:
    states = chain_result["boundary_states"]
    leaf = jax.tree_util.tree_leaves(states)[0]
    assert leaf.shape[:2] == (4, len(COLLECT_FIRST_PLAN))

    for episode, expected in enumerate(_reset_states(seed=3, num_episodes=4)):
        _assert_same_state(
            jax.tree_util.tree_map(lambda value, e=episode: value[e, 0], states),
            expected,
        )


def test_chain_boundaries_label_every_reached_first_attempt(
    chain_result: dict[str, Any],
) -> None:
    boundaries = chain_boundaries(chain_result, plan_name="collect_first")

    reached = sum(
        outcome is not None
        for outcomes in chain_result["first_attempt_outcomes"]
        for outcome in outcomes
    )
    assert len(boundaries) == reached
    occurrences = {(boundary.plan_step, boundary.occurrence) for boundary in boundaries}
    # The three collect_wood requests open the plan.
    assert occurrences <= {(0, 1), (1, 2), (2, 3), (3, 1), (4, 1), (5, 1)}
    for boundary in boundaries:
        assert boundary.plan == "collect_first"
        assert boundary.skill_id is COLLECT_FIRST_PLAN[boundary.plan_step]
        assert (
            boundary.outcome.value
            == (
                chain_result["first_attempt_outcomes"][boundary.episode][
                    boundary.plan_step
                ]
            )
        )
        row = boundary.as_row()
        assert "state" not in row
        assert row["band"] == distance_band(boundary.interaction_distance)
    first = [boundary for boundary in boundaries if boundary.plan_step == 0]
    assert len(first) == 4
    for boundary in first:
        # The chain world starts at the map centre with trees in view, at
        # dawn, with nothing collected yet.
        assert boundary.player_offset == (0, 0)
        assert boundary.targets_in_view >= 1
        assert boundary.interaction_distance is not None
        assert boundary.uncleared_in_view == 0
        assert boundary.wood == 0
        assert boundary.timestep == 0


def test_chain_boundaries_require_recorded_states(
    executor: LearnedSkillExecutor,
) -> None:
    result = evaluate_learned_skill_chain_batched(
        executor,
        num_episodes=2,
        seed=0,
        skill_horizon=2,
        max_decisions=2,
    )

    with pytest.raises(ValueError, match="record_boundaries"):
        chain_boundaries(result)


def test_success_table_groups_and_sorts() -> None:
    records = [
        {"skill": "b", "ok": True},
        {"skill": "a", "ok": False},
        {"skill": "b", "ok": False},
    ]

    table = success_table(
        records,
        lambda record: record["skill"],
        succeeded=lambda record: record["ok"],
    )

    assert list(table) == ["a", "b"]
    assert table["b"] == {"count": 2, "successes": 1, "success_rate": 0.5}


def test_traces_record_one_path_entry_per_step(
    executor: LearnedSkillExecutor,
) -> None:
    environment = make_skill_chain_environment()
    states = [environment.reset(jax.random.PRNGKey(seed)).state for seed in range(3)]

    traces = trace_skills_batched(
        executor,
        states,
        [SkillId.COLLECT_WOOD, SkillId.DRINK_WATER, SkillId.COLLECT_STONE],
        jax.random.PRNGKey(0),
        skill_horizon=5,
    )

    assert len(traces) == 3
    for trace in traces:
        assert trace.positions.shape == (5, 2)
        assert trace.directions.shape == (5,)
        assert trace.actions.shape == (5,)
    # Stone needs a pickaxe, so the option never starts.
    assert traces[2].probe.outcome is SkillOutcome.INFEASIBLE
    assert traces[2].probe.steps == 0
    # Positions follow from movement: each step moves at most one tile.
    for trace, state in zip(traces[:2], states[:2], strict=True):
        path = np.concatenate(
            [
                np.asarray(state.player_position)[None],
                trace.positions[: trace.probe.steps],
            ]
        )
        assert np.all(np.abs(np.diff(path, axis=0)).sum(axis=1) <= 1)


def test_remove_tables_edit_turns_tables_to_grass() -> None:
    state = _beside_a_table(make_skill_chain_environment(), seed=0)
    assert bool(jnp.any(state.map == BlockType.CRAFTING_TABLE.value))

    edited = _apply_edit(
        state, edit=_EDITS["remove_tables"], skill_id=SkillId.COLLECT_WOOD
    )

    assert not bool(jnp.any(edited.map == BlockType.CRAFTING_TABLE.value))
    assert SkillId.CRAFT_WOOD_PICKAXE not in _EDITS["remove_tables"].applies_to


def test_clear_view_edit_keeps_only_targets_in_view() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(1)).state
    view = np.asarray(view_mask(state, OBSERVATION_RADIUS))

    edited = _apply_edit(
        state, edit=_EDITS["clear_view"], skill_id=SkillId.COLLECT_WOOD
    )

    before = np.asarray(state.map)
    after = np.asarray(edited.map)
    np.testing.assert_array_equal(after[~view], before[~view])
    visible_trees = before[view] == BlockType.TREE.value
    np.testing.assert_array_equal(
        after[view][visible_trees], before[view][visible_trees]
    )
    assert np.all(after[view][~visible_trees] == BlockType.GRASS.value)


def test_training_state_edits_match_the_training_scenario() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(2)).state
    state = state.replace(
        inventory=state.inventory.replace(wood=5, sapling=1),
        player_drink=3,
        timestep=200,
        light_level=0.1,
    )

    inventory = _apply_edit(
        state, edit=_EDITS["training_inventory"], skill_id=SkillId.PLACE_TABLE
    ).inventory
    assert int(inventory.wood) == 2
    assert int(inventory.sapling) == 0
    assert inventory.wood.dtype == jnp.asarray(state.inventory.wood).dtype

    vitals = _apply_edit(
        state, edit=_EDITS["training_vitals"], skill_id=SkillId.DRINK_WATER
    )
    assert int(vitals.player_drink) == 5
    assert int(vitals.player_health) == 9
    assert (
        int(
            _apply_edit(
                state, edit=_EDITS["training_vitals"], skill_id=SkillId.COLLECT_WOOD
            ).player_drink
        )
        == 9
    )

    clock = _apply_edit(
        state, edit=_EDITS["training_clock"], skill_id=SkillId.COLLECT_WOOD
    )
    assert int(clock.timestep) == 0
    assert float(clock.light_level) == 1.0
    _assert_same_state(clock.map, state.map)


def test_unedited_probe_replays_the_chain_first_attempts(
    executor: LearnedSkillExecutor,
    chain_result: dict[str, Any],
) -> None:
    boundaries = chain_boundaries(chain_result)

    result = probe_counterfactuals(executor, boundaries, skill_horizon=4)

    # Greedy decoding and a world without mobs make an option a function of
    # its starting state, so the probe reproduces the chain exactly.
    assert result["replay_agreement"] == 1.0
    for boundary, trace in zip(boundaries, result["traces"]["none"], strict=True):
        assert trace.probe.outcome is boundary.outcome
        assert trace.probe.steps == boundary.steps
    assert set(result["comparisons"]) == {
        edit.name for edit in COUNTERFACTUAL_EDITS
    } | {"all"}
    for comparison in result["comparisons"].values():
        assert comparison["num_pairs"] == len(boundaries)


def test_identity_edit_has_no_discordant_pairs(
    executor: LearnedSkillExecutor,
    chain_result: dict[str, Any],
) -> None:
    identity = CounterfactualEdit("identity", "No change.", lambda state, _: state)

    result = probe_counterfactuals(
        executor,
        chain_boundaries(chain_result),
        edits=(identity,),
        include_all=False,
        skill_horizon=4,
    )

    comparison = result["comparisons"]["identity"]
    assert comparison["reference_only"] == comparison["candidate_only"] == 0


def test_edits_skip_boundaries_they_do_not_apply_to(
    executor: LearnedSkillExecutor,
    chain_result: dict[str, Any],
) -> None:
    wood = chain_boundaries(chain_result)[0]
    crafting = dataclasses.replace(
        wood,
        skill_id=SkillId.CRAFT_WOOD_PICKAXE,
        state=_beside_a_table(make_skill_chain_environment(), seed=0),
    )

    result = probe_counterfactuals(
        executor,
        [wood, crafting],
        edits=(_EDITS["remove_tables"],),
        include_all=False,
        skill_horizon=4,
    )

    assert result["comparisons"]["remove_tables"]["num_pairs"] == 1
    traces = result["traces"]["remove_tables"]
    assert traces[0] is not None
    assert traces[1] is None


def _boundary(state: Any, skill_id: SkillId) -> ChainBoundary:
    return ChainBoundary(
        plan="test",
        episode=0,
        plan_step=0,
        skill_id=skill_id,
        occurrence=1,
        outcome=SkillOutcome.TIMEOUT,
        steps=4,
        interaction_distance=2,
        targets_in_view=1,
        table_in_view=False,
        clutter_in_view=0,
        uncleared_in_view=0,
        player_offset=(0, 0),
        wood=0,
        drink=9,
        timestep=0,
        light_level=1.0,
        state=state,
    )


def test_rollout_features_describe_a_stalled_approach() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    row, column = (int(value) for value in np.asarray(state.player_position))
    world_map = jnp.full_like(state.map, BlockType.GRASS.value)
    state = state.replace(
        map=world_map.at[row, column + 2].set(BlockType.TREE.value),
        player_direction=jnp.asarray(
            Action.LEFT.value, dtype=state.player_direction.dtype
        ),
    )
    beside = (row, column + 1)
    # Step right, bump into the tree, hit it once, then idle. The two trailing
    # entries lie past the option's end and must be ignored.
    trace = SkillRolloutTrace(
        probe=SkillProbe(
            skill_id=SkillId.COLLECT_WOOD,
            outcome=SkillOutcome.TIMEOUT,
            steps=4,
        ),
        positions=np.asarray([beside] * 4 + [(0, 0)] * 2),
        directions=np.asarray([Action.RIGHT.value] * 4 + [Action.UP.value] * 2),
        actions=np.asarray(
            [
                Action.RIGHT.value,
                Action.RIGHT.value,
                Action.DO.value,
                Action.NOOP.value,
                Action.DO.value,
                Action.DO.value,
            ]
        ),
    )

    features = rollout_features(state, SkillId.COLLECT_WOOD, trace)

    assert features == {
        "skill_id": "collect_wood",
        "outcome": "timeout",
        "steps": 4,
        "started_near": False,
        "ever_near": True,
        "ever_facing": True,
        "steps_near": 4,
        "interact_while_facing": 1,
        "noop_steps": 1,
        "blocked_moves": 1,
        "distinct_positions": 2,
        "revisit_fraction": pytest.approx(0.6),
        "final_target_distance": 1,
    }

    rows = failed_rollout_features([_boundary(state, SkillId.COLLECT_WOOD)], [trace])
    assert len(rows) == 1
    assert rows[0]["plan"] == "test"
    assert rows[0]["band"] == "2-4"
    summary = summarize_rollout_features(rows)
    assert summary["count"] == 1
    assert summary["ever_near"] == 1.0
    assert summarize_rollout_features([]) == {"count": 0}


def test_failed_rollout_features_skip_successes_and_infeasible_starts() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    boundary = _boundary(state, SkillId.COLLECT_WOOD)
    empty = np.zeros((2, 2), dtype=np.int32)

    def trace(outcome: SkillOutcome) -> SkillRolloutTrace:
        return SkillRolloutTrace(
            probe=SkillProbe(SkillId.COLLECT_WOOD, outcome, 0),
            positions=empty,
            directions=empty[:, 0],
            actions=empty[:, 0],
        )

    rows = failed_rollout_features(
        [boundary] * 3,
        [trace(SkillOutcome.SUCCESS), trace(SkillOutcome.INFEASIBLE), None],
    )

    assert rows == []


def test_benchmark_pairs_plans_and_serializes(executor: LearnedSkillExecutor) -> None:
    benchmark = benchmark_executor(
        executor,
        num_episodes=2,
        seed=5,
        skill_horizon=4,
        local_episodes_per_skill=0,
    )

    assert set(benchmark["chains"]) == {"default", "collect_first"}
    assert set(benchmark["plan_comparisons"]) == {"default_vs_collect_first"}
    assert benchmark["chains"]["collect_first"]["plan"] == [
        skill.value for skill in COLLECT_FIRST_PLAN
    ]
    for name, boundaries in benchmark["boundaries"].items():
        assert all(boundary.plan == name for boundary in boundaries)

    serialized = json.loads(json.dumps(json_ready_benchmark(benchmark)))
    assert "boundary_states" not in serialized["chains"]["default"]
    assert all("state" not in row for row in serialized["boundaries"]["default"])

    comparison = compare_executor_benchmarks(benchmark, benchmark)
    for paired in comparison["plans"].values():
        assert paired["reference_only"] == paired["candidate_only"] == 0
    for bands in comparison["bands"].values():
        assert bands["reference"] == bands["candidate"]
    assert comparison["local_navigation"] == []


def test_every_edit_names_a_known_skill_set() -> None:
    for edit in COUNTERFACTUAL_EDITS:
        assert edit.applies_to <= frozenset(SKILL_IDS)
        assert edit.applies_to
