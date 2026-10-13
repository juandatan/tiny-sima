from typing import Any

import jax

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    ChainTermination,
    ScriptedSkillExecutor,
    evaluate_skill_chain,
)
from tiny_sima.goals import SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
)


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
