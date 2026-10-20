import dataclasses
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import SkillChainEnvironment, make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    _BATCHED_EXECUTIONS,
    WOOD_TO_STONE_CHAIN,
    ChainTermination,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    SkillChainSpec,
    SkillExecutionResult,
    _make_batched_skill_probe,
    _outcome_from_step,
    _record_skill_failures,
    _start_skill,
    compare_paired_successes,
    evaluate_chain_episode,
    evaluate_chain_episodes,
    evaluate_learned_skill_chain_batched,
    evaluate_skill_chain,
    probe_skills_batched,
)
from tiny_sima.executor.model import (
    apply_goal_conditioned_actor_critic,
    initial_hidden,
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
    select_action,
)
from tiny_sima.goals import SKILL_IDS, SkillId, skill_index, skill_precondition
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
    UtilityScoreBatch,
)


def _make_random_learned_executor() -> LearnedSkillExecutor:
    config = SkillPPOConfig(
        total_timesteps=12,
        num_envs=6,
        num_steps=2,
        num_minibatches=1,
        use_optimistic_resets=False,
        hidden_size=16,
        skill_embedding_size=4,
        monitor_gpu=False,
    ).validate()
    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        environment.bundle.spec.observation_shape,
    )
    return LearnedSkillExecutor(
        params=params,
        config=config,
        model=model,
    )


def _execute_stepwise(
    executor: LearnedSkillExecutor,
    environment: SkillChainEnvironment,
    state: Any,
    skill_id: SkillId,
    rng: jax.Array,
    *,
    horizon: int,
) -> SkillExecutionResult:
    """The executor as it was before compilation, kept as the reference.

    It steps the skill environment from Python and reads every outcome flag
    back to the host, which is slow but leaves no doubt about the semantics.
    """

    skill_environment, current = _start_skill(environment, state, skill_id, horizon)
    if not bool(np.asarray(skill_precondition(skill_environment.goal, state))):
        return SkillExecutionResult(
            state=state,
            outcome=SkillOutcome.INFEASIBLE,
            steps=0,
            skill_return=0.0,
            native_return=0.0,
        )
    hidden = initial_hidden(1, executor.config.hidden_size)
    categorical_skill_id = jnp.asarray(skill_index(skill_id), dtype=jnp.int32)
    skill_return = 0.0
    native_return = 0.0
    step_keys = jax.random.split(rng, horizon * 2)
    for step_index in range(horizon):
        output = apply_goal_conditioned_actor_critic(
            executor.model,
            executor.params,
            hidden,
            (
                environment.bundle.env.get_obs(current.env_state)[None, None],
                categorical_skill_id[None, None],
                jnp.asarray(step_index == 0)[None, None],
            ),
        )
        hidden = output.hidden
        action = select_action(
            output.logits[0],
            step_keys[2 * step_index],
            deterministic=executor.deterministic,
        )[0]
        result = skill_environment.step(
            step_keys[2 * step_index + 1],
            current,
            action,
        )
        current = result.state
        skill_return += float(np.asarray(result.reward))
        native_return += float(np.asarray(result.info["native_reward"]))
        outcome = _outcome_from_step(result)
        if outcome is not None:
            return SkillExecutionResult(
                state=current.env_state,
                outcome=outcome,
                steps=step_index + 1,
                skill_return=skill_return,
                native_return=native_return,
            )
    return SkillExecutionResult(
        state=current.env_state,
        outcome=SkillOutcome.TIMEOUT,
        steps=horizon,
        skill_return=skill_return,
        native_return=native_return,
    )


def _assert_same_execution(
    actual: SkillExecutionResult,
    expected: SkillExecutionResult,
) -> None:
    assert actual.outcome is expected.outcome
    assert actual.steps == expected.steps
    assert actual.skill_return == expected.skill_return
    assert actual.native_return == pytest.approx(expected.native_return, abs=1e-5)
    actual_leaves = jax.tree_util.tree_leaves(actual.state)
    expected_leaves = jax.tree_util.tree_leaves(expected.state)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=0,
            atol=1e-6,
        )


def _beside_a_table(environment: SkillChainEnvironment, seed: int) -> Any:
    """Return a fresh world with a table beside the player and one wood."""

    state = environment.reset(jax.random.PRNGKey(seed)).state
    row, column = (int(value) for value in np.asarray(state.player_position))
    return state.replace(
        map=jnp.asarray(state.map)
        .at[row, column + 1]
        .set(BlockType.CRAFTING_TABLE.value),
        inventory=state.inventory.replace(wood=1),
    )


def _craft_biased(executor: LearnedSkillExecutor) -> LearnedSkillExecutor:
    """Tilt a random policy toward crafting so a success is near certain."""

    params = jax.tree_util.tree_map(lambda leaf: leaf, executor.params)
    logits = params["params"]["actor_logits"]
    logits["bias"] = logits["bias"].at[Action.MAKE_WOOD_PICKAXE.value].add(4.0)
    return dataclasses.replace(executor, params=params, deterministic=False)


def _always_complete(_state: Any) -> bool:
    return True


def _always_complete_jax(_state: Any) -> jax.Array:
    return jnp.asarray(True)


def test_saycan_scores_utility_times_feasibility() -> None:
    state: Any = make_skill_chain_environment().reset(
        jax.random.PRNGKey(0)
    ).state
    context = PlannerContext(
        objective="Collect wood.",
    )
    decision = SayCanPlanner(
        utility_scorer=SequentialUtilityScorer((SkillId.COLLECT_WOOD,))
    ).select(state, context)

    assert decision is not None
    assert decision.skill_id is SkillId.COLLECT_WOOD
    selected = decision.scores[0]
    assert selected.utility == 1.0
    assert selected.feasibility == 1.0
    assert selected.combined == 1.0


def test_scripted_executor_completes_persistent_chain() -> None:
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        num_episodes=4,
        seed=3,
        skill_horizon=32,
    )

    assert result["chain"] == WOOD_TO_STONE_CHAIN.name
    assert result["success_rate"] == 1.0
    assert result["mean_completed_skills"] == len(
        WOOD_TO_STONE_CHAIN.reference_skills
    )
    assert result["final_failed_skill_counts"] == {}
    assert result["final_failure_outcome_counts"] == {}
    for episode in result["episodes"]:
        assert episode.succeeded
        assert episode.termination is ChainTermination.SUCCESS
        assert all(
            trace.outcome is SkillOutcome.SUCCESS for trace in episode.trace
        )


def test_infeasible_next_skill_stops_planner() -> None:
    environment = make_skill_chain_environment()
    state = environment.reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(
        objective="Craft before collecting resources.",
    )

    planner = SayCanPlanner(
        utility_scorer=SequentialUtilityScorer(
            (SkillId.CRAFT_WOOD_PICKAXE,)
        )
    )
    assert planner.select(state, context) is None


def test_alternative_valid_plan_satisfies_state_based_objective() -> None:
    alternative_plan = (
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.PLACE_TABLE,
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.COLLECT_STONE,
    )
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(alternative_plan)
        ),
        num_episodes=2,
        seed=7,
        skill_horizon=32,
    )

    assert result["success_rate"] == 1.0


def test_batched_learned_chain_evaluation_runs_in_jax() -> None:
    executor = _make_random_learned_executor()

    result = evaluate_learned_skill_chain_batched(
        executor,
        num_episodes=2,
        seed=0,
        skill_horizon=2,
        max_decisions=2,
    )

    assert result["batched"] is True
    assert result["num_episodes"] == 2
    assert 0.0 <= result["success_rate"] <= 1.0
    assert sum(result["termination_counts"].values()) == 2
    assert result["mean_environment_steps"] <= 4

    cached_result = evaluate_learned_skill_chain_batched(
        executor,
        num_episodes=2,
        seed=1,
        skill_horizon=2,
        max_decisions=2,
    )
    assert cached_result["compile_cache_hit"] is True


def test_batched_chain_uses_state_based_completion_and_stops_early() -> None:
    chain = SkillChainSpec(
        name="already_complete",
        objective="Objective already holds.",
        reference_skills=(SkillId.COLLECT_WOOD,),
        is_complete=_always_complete,
        is_complete_jax=_always_complete_jax,
    )

    result = evaluate_learned_skill_chain_batched(
        _make_random_learned_executor(),
        chain=chain,
        num_episodes=2,
        seed=0,
        skill_horizon=2,
        max_decisions=2,
    )

    assert result["success_rate"] == 1.0
    assert result["mean_completed_skills"] == 0.0
    assert result["mean_decisions"] == 0.0
    assert result["mean_environment_steps"] == 0.0
    assert result["termination_counts"]["success"] == 2


def test_batched_failure_counts_persist_across_successful_options() -> None:
    skill_ids = jnp.asarray(
        [SKILL_IDS.index(SkillId.COLLECT_WOOD)],
        dtype=jnp.int32,
    )
    failures = jnp.zeros((1, len(SKILL_IDS)), dtype=jnp.int32)

    failures, selected = _record_skill_failures(
        failures,
        skill_ids,
        jnp.asarray([True]),
    )
    assert int(selected[0]) == 1

    failures, selected = _record_skill_failures(
        failures,
        skill_ids,
        jnp.asarray([False]),
    )
    assert int(selected[0]) == 1

    _, selected = _record_skill_failures(
        failures,
        skill_ids,
        jnp.asarray([True]),
    )
    assert int(selected[0]) == 2


def _no_nearby_trees(state: Any, radius: int = 12) -> Any:
    """Clear trees around the player, leaving the precondition satisfied."""

    world_map = jnp.asarray(state.map)
    position = jnp.asarray(state.player_position)
    rows = jnp.arange(world_map.shape[0])[:, None]
    columns = jnp.arange(world_map.shape[1])[None, :]
    nearby = (jnp.abs(rows - position[0]) <= radius) & (
        jnp.abs(columns - position[1]) <= radius
    )
    cleared = jnp.where(
        nearby & (world_map == BlockType.TREE.value),
        BlockType.GRASS.value,
        world_map,
    )
    assert bool(jnp.any(cleared == BlockType.TREE.value))
    return state.replace(map=cleared)


def test_batched_probe_pairs_each_state_with_its_own_skill() -> None:
    executor = _make_random_learned_executor()
    environment = make_skill_chain_environment()
    states = [
        environment.reset(jax.random.PRNGKey(seed)).state for seed in range(2)
    ]
    pairs = [(state, skill_id) for state in states for skill_id in SKILL_IDS]

    probes = probe_skills_batched(
        executor,
        [state for state, _ in pairs],
        [skill_id for _, skill_id in pairs],
        jax.random.PRNGKey(0),
        skill_horizon=4,
    )

    assert len(probes) == len(pairs)
    assert [probe.skill_id for probe in probes] == [
        skill_id for _, skill_id in pairs
    ]
    for probe in probes:
        assert 0 <= probe.steps <= 4
        assert probe.succeeded == (probe.outcome is SkillOutcome.SUCCESS)


def test_batched_probe_reports_the_same_infeasibility_as_the_executor() -> None:
    executor = _make_random_learned_executor()
    environment = make_skill_chain_environment()
    state = environment.reset(jax.random.PRNGKey(0)).state

    probes = probe_skills_batched(
        executor,
        [state] * len(SKILL_IDS),
        list(SKILL_IDS),
        jax.random.PRNGKey(1),
        skill_horizon=4,
    )
    unbatched = [
        executor.execute(environment, state, skill_id, jax.random.PRNGKey(1), horizon=4)
        for skill_id in SKILL_IDS
    ]

    infeasible_batched = {
        probe.skill_id
        for probe in probes
        if probe.outcome is SkillOutcome.INFEASIBLE
    }
    infeasible_unbatched = {
        skill_id
        for skill_id, result in zip(SKILL_IDS, unbatched, strict=True)
        if result.outcome is SkillOutcome.INFEASIBLE
    }
    # Stone needs a pickaxe and crafting needs a table, neither of which a
    # fresh world provides, so the precondition fast path must agree exactly.
    assert infeasible_batched == infeasible_unbatched
    assert SkillId.COLLECT_STONE in infeasible_batched
    assert all(
        probe.steps == 0
        for probe in probes
        if probe.outcome is SkillOutcome.INFEASIBLE
    )


def test_batched_probe_times_out_when_no_target_is_in_reach() -> None:
    executor = _make_random_learned_executor()
    environment = make_skill_chain_environment()
    state = _no_nearby_trees(environment.reset(jax.random.PRNGKey(0)).state)

    probe = probe_skills_batched(
        executor,
        [state],
        [SkillId.COLLECT_WOOD],
        jax.random.PRNGKey(2),
        skill_horizon=4,
    )[0]

    # Wood exists somewhere, so the precondition holds and the option runs;
    # the nearest tree is far outside four steps, so it can only time out.
    assert probe.outcome is SkillOutcome.TIMEOUT
    assert probe.steps == 4


def test_batched_probe_compiles_once_per_batch_shape() -> None:
    executor = _make_random_learned_executor()
    environment = make_skill_chain_environment()
    state = environment.reset(jax.random.PRNGKey(0)).state
    other = environment.reset(jax.random.PRNGKey(3)).state

    _make_batched_skill_probe.cache_clear()
    for probe_state in (state, other):
        probe_skills_batched(
            executor,
            [probe_state] * len(SKILL_IDS),
            list(SKILL_IDS),
            jax.random.PRNGKey(0),
            skill_horizon=2,
        )

    info = _make_batched_skill_probe.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_compiled_executor_matches_the_stepwise_executor() -> None:
    executor = _craft_biased(_make_random_learned_executor())
    environment = make_skill_chain_environment()
    # A fresh world makes stone and crafting infeasible. Beside a table,
    # the tilted policy crafts within a few steps, which covers success.
    states = [
        environment.reset(jax.random.PRNGKey(0)).state,
        _beside_a_table(environment, seed=1),
    ]
    pairs = [(state, skill_id) for state in states for skill_id in SKILL_IDS]
    keys = list(jax.random.split(jax.random.PRNGKey(5), len(pairs)))

    expected = [
        _execute_stepwise(executor, environment, state, skill_id, key, horizon=24)
        for (state, skill_id), key in zip(pairs, keys, strict=True)
    ]
    single = [
        executor.execute(environment, state, skill_id, key, horizon=24)
        for (state, skill_id), key in zip(pairs, keys, strict=True)
    ]
    # Padding past the number of attempts must not leak into real entries.
    batched = executor.execute_many(
        environment,
        [state for state, _ in pairs],
        [skill_id for _, skill_id in pairs],
        keys,
        horizon=24,
        batch_size=len(pairs) + 3,
    )

    for actual_single, actual_batched, reference in zip(
        single,
        batched,
        expected,
        strict=True,
    ):
        _assert_same_execution(actual_single, reference)
        _assert_same_execution(actual_batched, reference)
    outcomes = {result.outcome for result in expected}
    assert {
        SkillOutcome.SUCCESS,
        SkillOutcome.TIMEOUT,
        SkillOutcome.INFEASIBLE,
    } <= outcomes


def test_compiled_executor_compiles_once_across_environments() -> None:
    executor = _make_random_learned_executor()
    first = make_skill_chain_environment()
    second = make_skill_chain_environment()
    fresh = [first.reset(jax.random.PRNGKey(seed)).state for seed in range(3)]

    _BATCHED_EXECUTIONS.clear()
    results = executor.execute_many(
        first,
        fresh,
        [SkillId.COLLECT_WOOD] * 3,
        list(jax.random.split(jax.random.PRNGKey(0), 3)),
        horizon=2,
        batch_size=4,
    )
    # A new but identical environment, fewer attempts, and states that came
    # back from the device all reuse the loop compiled for the first call.
    executor.execute_many(
        second,
        [result.state for result in results[:2]],
        [SkillId.DRINK_WATER] * 2,
        list(jax.random.split(jax.random.PRNGKey(1), 2)),
        horizon=2,
        batch_size=4,
    )

    assert len(_BATCHED_EXECUTIONS) == 1
    (compiled,) = _BATCHED_EXECUTIONS.values()
    assert compiled._cache_size() == 1  # type: ignore[attr-defined]


@dataclasses.dataclass
class _RecordingSequentialScorer:
    """The scripted reference plan, answered in batches it records."""

    reference_skills: tuple[SkillId, ...]
    batch_sizes: list[int] = dataclasses.field(default_factory=list)

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return SequentialUtilityScorer(self.reference_skills)(
            state,
            context,
            skill_id,
        )

    def score_all_batched(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
        candidate_skills: tuple[SkillId, ...],
    ) -> list[UtilityScoreBatch]:
        self.batch_sizes.append(len(states))
        return [
            UtilityScoreBatch(
                utilities={
                    skill_id: self(state, context, skill_id)
                    for skill_id in candidate_skills
                }
            )
            for state, context in zip(states, contexts, strict=True)
        ]


def _facing_left(state: Any) -> bool:
    return int(np.asarray(state.player_direction)) == Action.LEFT.value


def test_lockstep_episodes_match_running_them_one_by_one() -> None:
    executor = _craft_biased(_make_random_learned_executor())
    environment = make_skill_chain_environment()
    # Worlds that start facing left count as complete at once, so episodes
    # finish at different decisions and the lockstep loop must skip them.
    chain = WOOD_TO_STONE_CHAIN._replace(is_complete=_facing_left)
    keys = list(jax.random.split(jax.random.PRNGKey(4), 6))
    settings = {
        "chain": chain,
        "skill_horizon": 6,
        "max_decisions": 3,
        "max_failures_per_skill": 2,
    }

    def planner() -> SayCanPlanner:
        return SayCanPlanner(
            utility_scorer=_RecordingSequentialScorer(chain.reference_skills)
        )

    lockstep = evaluate_chain_episodes(
        planner(),
        executor,
        environment,
        keys,
        **settings,
    )
    one_by_one = [
        evaluate_chain_episode(planner(), executor, environment, key, **settings)
        for key in keys
    ]

    for actual, expected in zip(lockstep, one_by_one, strict=True):
        assert actual.succeeded == expected.succeeded
        assert actual.termination is expected.termination
        assert actual.decisions == expected.decisions
        assert actual.environment_steps == expected.environment_steps
        assert actual.native_return == pytest.approx(expected.native_return)
        assert [
            (trace.requested_skill, trace.outcome, trace.steps)
            for trace in actual.trace
        ] == [
            (trace.requested_skill, trace.outcome, trace.steps)
            for trace in expected.trace
        ]
    decisions = {episode.decisions for episode in lockstep}
    assert min(decisions) == 0
    assert max(decisions) > 0


def test_lockstep_plans_every_running_episode_in_one_call() -> None:
    scorer = _RecordingSequentialScorer(WOOD_TO_STONE_CHAIN.reference_skills)
    result = evaluate_skill_chain(
        _make_random_learned_executor(),
        planner=SayCanPlanner(utility_scorer=scorer),
        num_episodes=4,
        seed=0,
        skill_horizon=2,
        max_decisions=3,
    )

    # One planner call per decision index, never one per episode.
    decisions = [episode.decisions for episode in result["episodes"]]
    assert scorer.batch_sizes[0] == 4
    assert len(scorer.batch_sizes) == max(decisions)
    assert scorer.batch_sizes == sorted(scorer.batch_sizes, reverse=True)
    assert sum(scorer.batch_sizes) == sum(decisions)


def _paired_result(outcomes: Sequence[bool], seed: int = 0) -> dict[str, Any]:
    return {
        "seed": seed,
        "num_episodes": len(outcomes),
        "episodes": [SimpleNamespace(succeeded=outcome) for outcome in outcomes],
    }


def test_paired_comparison_reads_only_discordant_episodes() -> None:
    reference = _paired_result([True, True, False, False, False, False, False, False])
    candidate = _paired_result([True, False, True, True, True, True, True, False])

    comparison = compare_paired_successes(reference, candidate)

    assert comparison["both"] == 1
    assert comparison["neither"] == 1
    assert comparison["reference_only"] == 1
    assert comparison["candidate_only"] == 5
    assert comparison["difference"] == pytest.approx(4 / 8)
    # Two-sided exact binomial on six discordant pairs, one against five.
    assert comparison["p_value"] == pytest.approx(2 * (1 + 6) / 2**6)
    # Swapping the arms flips the sign and keeps the test.
    swapped = compare_paired_successes(candidate, reference)
    assert swapped["difference"] == pytest.approx(-4 / 8)
    assert swapped["p_value"] == comparison["p_value"]


def test_paired_comparison_without_discordant_pairs_is_uninformative() -> None:
    result = _paired_result([True, False, True])

    comparison = compare_paired_successes(result, result)

    assert comparison["p_value"] == 1.0
    assert comparison["difference"] == 0.0


def test_paired_comparison_rejects_results_from_different_worlds() -> None:
    with pytest.raises(ValueError, match="seed"):
        compare_paired_successes(
            _paired_result([True, False], seed=0),
            _paired_result([True, False], seed=1),
        )
    with pytest.raises(ValueError, match="num_episodes"):
        compare_paired_successes(
            _paired_result([True, False]),
            _paired_result([True, False, True]),
        )


def test_conditions_on_one_seed_pair_episode_by_episode() -> None:
    first = evaluate_skill_chain(ScriptedSkillExecutor(), num_episodes=3, seed=5)
    second = evaluate_skill_chain(ScriptedSkillExecutor(), num_episodes=3, seed=5)

    comparison = compare_paired_successes(first, second)

    assert comparison["num_pairs"] == 3
    assert comparison["reference_only"] == comparison["candidate_only"] == 0
