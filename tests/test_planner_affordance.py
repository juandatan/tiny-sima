"""Tests for grounding planner feasibility in reachability and the critic."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import BlockType

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.affordance import (
    collect_affordance_samples,
    evaluate_affordance_calibration,
    evaluate_competence_prior,
    fit_affordance_calibration,
    fit_affordance_competence_prior,
)
from tiny_sima.evaluation.chains import LearnedSkillExecutor
from tiny_sima.evaluation.planner import evaluate_planner_prefixes
from tiny_sima.executor.model import (
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import (
    SKILL_IDS,
    SkillGoal,
    SkillId,
    skill_feasible,
)
from tiny_sima.planner import (
    CriticAffordanceScorer,
    IsotonicCalibration,
    PlannerContext,
    ReachabilityFeasibilityScorer,
    SayCanPlanner,
    SkillOutcome,
    reachability_feasibility,
    symbolic_feasibility,
)


def _chain_state() -> Any:
    return make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state


def _state_with_distant_trees(radius: int = 12) -> Any:
    """Clear trees near the player while leaving the rest of the map alone.

    This reproduces the reported failure directly: wood is still obtainable
    somewhere, so the symbolic precondition holds, but nothing the planner can
    see affords it.
    """

    state = _chain_state()
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


def _constant_utility_for(*preferred: SkillId) -> Any:
    def scorer(
        _state: Any,
        _context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return 1.0 if skill_id in preferred else 0.4

    return scorer


def test_symbolic_feasibility_cannot_distinguish_reachable_resources() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")

    # Trees still exist somewhere on the 64x64 map, so the precondition holds
    # even though no tree is anywhere near the player. This is why the
    # precondition alone carries no information at a planner boundary.
    assert symbolic_feasibility(state, context, SkillId.COLLECT_WOOD) == 1.0
    assert symbolic_feasibility(state, context, SkillId.DRINK_WATER) == 1.0


def test_reachability_scorer_demotes_unreachable_targets() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")

    scores = reachability_feasibility.score_all_feasibility(
        state,
        context,
        SKILL_IDS,
    )

    assert scores[SkillId.COLLECT_WOOD] == pytest.approx(
        reachability_feasibility.config.unknown_feasibility
    )
    assert 0.0 < scores[SkillId.COLLECT_WOOD] < 1.0
    assert scores[SkillId.DRINK_WATER] == 1.0


def test_reachability_scorer_matches_the_strict_predicate_on_reachables() -> None:
    state = _chain_state()
    context = PlannerContext(objective="Collect wood.")

    scores = reachability_feasibility.score_all_feasibility(
        state,
        context,
        SKILL_IDS,
    )

    for skill_id in SKILL_IDS:
        reachable = bool(
            np.asarray(
                skill_feasible(
                    SkillGoal(skill_id),
                    state,
                    reachability_feasibility.config,
                )
            )
        )
        assert (scores[skill_id] == 1.0) is reachable


def test_reachability_scorer_never_zeroes_a_satisfied_precondition() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")

    scores = reachability_feasibility.score_all_feasibility(
        state,
        context,
        SKILL_IDS,
    )

    # Vetoing an out-of-view target would refuse plans that walk toward it,
    # which costs real episodes; only a failed precondition scores zero.
    for skill_id in SKILL_IDS:
        precondition = symbolic_feasibility(state, context, skill_id)
        assert (scores[skill_id] > 0.0) is (precondition > 0.0)


def test_single_skill_call_agrees_with_the_batched_path() -> None:
    state = _chain_state()
    context = PlannerContext(objective="Collect wood.")

    batched = reachability_feasibility.score_all_feasibility(
        state,
        context,
        SKILL_IDS,
    )

    for skill_id in SKILL_IDS:
        assert reachability_feasibility(state, context, skill_id) == pytest.approx(
            batched[skill_id]
        )


def test_planner_prefers_a_reachable_skill_over_a_preferred_distant_one() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")
    utility = _constant_utility_for(SkillId.COLLECT_WOOD)

    # With the precondition as the only grounding, the unreachable favourite
    # wins: this is the reported bug.
    symbolic_choice = SayCanPlanner(
        utility_scorer=utility,
        feasibility_scorer=symbolic_feasibility,
    ).select(state, context)
    assert symbolic_choice is not None
    assert symbolic_choice.skill_id is SkillId.COLLECT_WOOD

    # Grading by reachability, utility 1.0 x 0.25 loses to utility 0.4 x 1.0.
    grounded_choice = SayCanPlanner(
        utility_scorer=utility,
        feasibility_scorer=reachability_feasibility,
    ).select(state, context)
    assert grounded_choice is not None
    assert grounded_choice.skill_id is SkillId.DRINK_WATER


def test_planner_still_selects_an_unreachable_skill_when_nothing_else_helps() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")

    def only_wood(
        _state: Any,
        _context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return 1.0 if skill_id is SkillId.COLLECT_WOOD else 0.0

    planner = SayCanPlanner(
        utility_scorer=only_wood,
        feasibility_scorer=reachability_feasibility,
    )
    decision = planner.select(state, context)

    # Walking off to find a tree is the only plan on offer, so the planner must
    # still propose it rather than terminating with no viable skill.
    assert decision is not None
    assert decision.skill_id is SkillId.COLLECT_WOOD


def _fake_critic_scorer(
    values: dict[SkillId, float],
    *,
    calibration: IsotonicCalibration | None = None,
    reachability_gate: ReachabilityFeasibilityScorer | None = None,
) -> CriticAffordanceScorer:
    """Build a scorer whose critic head is replaced by fixed values."""

    environment = make_skill_chain_environment()
    scorer = CriticAffordanceScorer(
        params={},
        model=make_goal_conditioned_actor_critic(
            num_actions=environment.bundle.spec.num_actions,
            num_skills=len(SKILL_IDS),
            hidden_size=8,
            skill_embedding_size=4,
        ),
        hidden_size=8,
        observe=environment.bundle.env.get_obs,
        calibration=calibration,
        reachability_gate=reachability_gate,
    )
    object.__setattr__(
        scorer,
        "critic_values",
        lambda _state, skills=None: {
            skill: values[skill] for skill in (skills or SKILL_IDS)
        },
    )
    return scorer


def test_critic_affordance_clips_raw_values_without_a_calibration() -> None:
    state = _chain_state()
    context = PlannerContext(objective="Collect wood.")
    scorer = _fake_critic_scorer(
        {
            SkillId.COLLECT_WOOD: 0.19,
            SkillId.COLLECT_STONE: -0.5,
            SkillId.DRINK_WATER: 1.4,
            SkillId.PLACE_TABLE: 0.0,
            SkillId.CRAFT_WOOD_PICKAXE: 0.5,
            SkillId.CRAFT_WOOD_SWORD: 0.5,
        }
    )

    scores = scorer.score_all_feasibility(state, context, SKILL_IDS)

    assert scores[SkillId.COLLECT_WOOD] == pytest.approx(0.19)
    assert scores[SkillId.COLLECT_STONE] == 0.0
    assert scores[SkillId.DRINK_WATER] == 1.0


def test_calibration_lifts_discounted_critic_values() -> None:
    state = _chain_state()
    context = PlannerContext(objective="Collect wood.")
    # gamma ** 32 at gamma = 0.95: a certain but slow success looks unlikely.
    calibration = IsotonicCalibration(
        scores=(0.0, 0.19, 0.6),
        probabilities=(0.0, 0.95, 1.0),
    )
    scorer = _fake_critic_scorer(
        dict.fromkeys(SKILL_IDS, 0.19),
        calibration=calibration,
    )

    scores = scorer.score_all_feasibility(state, context, SKILL_IDS)

    assert scores[SkillId.COLLECT_WOOD] == pytest.approx(0.95)


def test_critic_affordance_composes_with_the_reachability_gate() -> None:
    state = _state_with_distant_trees()
    context = PlannerContext(objective="Collect wood.")
    scorer = _fake_critic_scorer(
        dict.fromkeys(SKILL_IDS, 0.8),
        reachability_gate=reachability_feasibility,
    )

    gated = scorer.score_all_feasibility(state, context, SKILL_IDS)
    ungated = _fake_critic_scorer(
        dict.fromkeys(SKILL_IDS, 0.8),
    ).score_all_feasibility(state, context, SKILL_IDS)

    assert ungated[SkillId.COLLECT_WOOD] == pytest.approx(0.8)
    assert gated[SkillId.COLLECT_WOOD] == pytest.approx(
        0.8 * reachability_feasibility.config.unknown_feasibility
    )
    assert gated[SkillId.DRINK_WATER] == pytest.approx(0.8)
    # Crafting has no table yet, so the gate zeroes it regardless of the critic.
    assert gated[SkillId.CRAFT_WOOD_PICKAXE] == 0.0


def test_critic_values_batch_over_skills_in_one_forward_pass() -> None:
    state = _chain_state()
    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=16,
        skill_embedding_size=4,
    )
    observation = environment.bundle.env.get_obs(state)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 16), dtype=jnp.float32),
        (
            observation[None, None, ...],
            jnp.zeros((1, 1), dtype=jnp.int32),
            jnp.ones((1, 1), dtype=jnp.bool_),
        ),
    )
    scorer = CriticAffordanceScorer(
        params=params,
        model=model,
        hidden_size=16,
        observe=environment.bundle.env.get_obs,
        reachability_gate=None,
    )

    values = scorer.critic_values(state)
    subset = scorer.critic_values(state, (SkillId.DRINK_WATER,))

    assert set(values) == set(SKILL_IDS)
    assert all(np.isfinite(value) for value in values.values())
    # Skill conditioning must actually reach the critic.
    assert len({round(value, 6) for value in values.values()}) > 1
    assert subset[SkillId.DRINK_WATER] == pytest.approx(
        values[SkillId.DRINK_WATER],
        abs=1e-5,
    )


def test_prefix_evaluation_reports_ungrounded_utility_rankings() -> None:
    # A utility model that always ranks crafting first, which is impossible
    # until a table exists, is exactly the failure mode being measured.
    def craft_first(
        _state: Any,
        _context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        return 1.0 if skill_id is SkillId.CRAFT_WOOD_PICKAXE else 0.3

    result = evaluate_planner_prefixes(
        SayCanPlanner(utility_scorer=craft_first),
        num_episodes=2,
        seed=0,
        skill_horizon=32,
    )

    assert result["num_boundaries"] > 0
    assert result["mean_feasible_skills"] > 0.0
    assert 0.0 <= result["ungrounded_top_choice_rate"] <= 1.0
    assert 0.0 <= result["feasibility_rescue_rate"] <= 1.0
    for trace in result["traces"]:
        assert trace.selected_feasible == (
            trace.selected_skill in trace.feasible_skills
        )


def _random_executor() -> LearnedSkillExecutor:
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
    return LearnedSkillExecutor(params=params, config=config, model=model)


def _random_scorer(executor: LearnedSkillExecutor) -> CriticAffordanceScorer:
    environment = make_skill_chain_environment()
    return CriticAffordanceScorer(
        params=executor.params,
        model=executor.model,
        hidden_size=executor.config.hidden_size,
        observe=environment.bundle.env.get_obs,
    )


def test_collected_samples_carry_every_feature_the_heads_need() -> None:
    executor = _random_executor()
    samples = collect_affordance_samples(
        _random_scorer(executor),
        executor,
        num_episodes=2,
        seed=0,
        skill_horizon=4,
    )

    assert samples
    assert {sample.episode for sample in samples} <= {0, 1}
    for sample in samples:
        assert sample.succeeded == (sample.outcome is SkillOutcome.SUCCESS)
        assert 0 <= sample.steps <= 4
        # Reachability is the strict predicate, so it implies the precondition
        # and a finite distance; the converse does not hold.
        assert sample.reachable == (
            sample.precondition and sample.steps_to_target is not None
        )
        if sample.outcome is SkillOutcome.INFEASIBLE:
            assert not sample.precondition
            assert sample.steps == 0
        if not sample.precondition:
            assert sample.reachability_gate == 0.0


def test_competence_prior_is_fitted_and_scored_from_probe_labels() -> None:
    executor = _random_executor()
    scorer = _random_scorer(executor)
    fit_samples = collect_affordance_samples(
        scorer,
        executor,
        num_episodes=2,
        seed=0,
        skill_horizon=4,
    )
    held_out = collect_affordance_samples(
        scorer,
        executor,
        num_episodes=2,
        seed=11,
        skill_horizon=4,
    )

    prior = fit_affordance_competence_prior(fit_samples)
    report = evaluate_competence_prior(held_out, prior)

    # Samples whose precondition fails are dropped: the precondition vetoes at
    # scoring time, so charging a skill for them would bias the head down.
    assert report["num_dropped_precondition"] == sum(
        not sample.precondition for sample in held_out
    )
    assert report["num_samples"] == len(held_out) - report[
        "num_dropped_precondition"
    ]
    assert 0.0 <= report["overall"]["raw_brier"] <= 1.0
    for cell in report["cells"]:
        assert 0.0 <= cell["probability"] <= 1.0
        assert cell["num_samples"] >= 1


def test_held_out_calibration_is_reported_as_such() -> None:
    executor = _random_executor()
    scorer = _random_scorer(executor)
    samples = collect_affordance_samples(
        scorer,
        executor,
        num_episodes=1,
        seed=0,
        skill_horizon=4,
    )
    calibration = fit_affordance_calibration(samples)

    in_sample = evaluate_affordance_calibration(samples)
    external = evaluate_affordance_calibration(samples, calibration)

    assert in_sample["held_out"] is False
    assert external["held_out"] is True
