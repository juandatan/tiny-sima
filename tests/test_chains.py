from typing import Any

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import BlockType

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    ChainTermination,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    SkillChainSpec,
    _make_batched_skill_probe,
    _record_skill_failures,
    evaluate_learned_skill_chain_batched,
    evaluate_skill_chain,
    probe_skills_batched,
)
from tiny_sima.executor.model import (
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import SKILL_IDS, SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
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
