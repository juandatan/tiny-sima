import json
from typing import Any

import jax
import pytest

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.envs.skills import make_vectorized_skill_environment
from tiny_sima.evaluation.affordance import AffordanceSample, evaluate_competence_prior
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    LearnedSkillExecutor,
    ScriptedSkillExecutor,
    chain_decision_limit,
    chain_environment,
    evaluate_skill_chain,
    json_ready_chain_result,
)
from tiny_sima.evaluation.sequencing import episode_sequence
from tiny_sima.executor.model import make_goal_conditioned_actor_critic
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    SKILL_IDS,
    STONE_PICKAXE_SKILL_IDS,
    SkillGoal,
    SkillId,
    executor_skill_ids,
    skill_precondition,
)
from tiny_sima.planner import SkipHeldCraftFeasibility, redundant_craft
from tiny_sima.planner.competence import CompetencePrior, fit_competence_prior
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
    reachability_feasibility,
)
from tiny_sima.planner.vlm import (
    PLANNER_SYSTEM_PROMPT,
    PlannerPromptStyle,
    build_planner_system_prompt,
)


def _with_inventory(state, **counts):
    return state.replace(inventory=state.inventory.replace(**counts))


def _chain_state(chain=WOOD_TO_STONE_PICKAXE_CHAIN, seed: int = 0):
    return chain_environment(chain).reset(jax.random.PRNGKey(seed)).state


def test_the_six_skill_catalog_is_unchanged() -> None:
    assert STONE_PICKAXE_SKILL_IDS[: len(SKILL_IDS)] == SKILL_IDS
    assert STONE_PICKAXE_SKILL_IDS[-1] is SkillId.CRAFT_STONE_PICKAXE
    assert executor_skill_ids(6) == SKILL_IDS
    assert executor_skill_ids(7) == STONE_PICKAXE_SKILL_IDS
    assert executor_skill_ids(16) == ALL_SKILL_IDS
    with pytest.raises(ValueError):
        executor_skill_ids(17)
    assert chain_decision_limit(WOOD_TO_STONE_CHAIN) == 12
    assert chain_decision_limit(WOOD_TO_STONE_PICKAXE_CHAIN) == 16


def test_the_stone_pickaxe_needs_wood_stone_and_a_table() -> None:
    state = _chain_state()
    goal = SkillGoal(SkillId.CRAFT_STONE_PICKAXE)
    # The fresh world has no table and an empty inventory.
    assert not bool(skill_precondition(goal, state))
    environment = make_skill_chain_environment()
    assert environment.tree_count == 3
    assert chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN).tree_count == 4


def test_scripted_executor_completes_the_stone_pickaxe_chain() -> None:
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        chain=WOOD_TO_STONE_PICKAXE_CHAIN,
        num_episodes=2,
        seed=7,
    )
    episodes = json.loads(json.dumps(json_ready_chain_result(result)))["episodes"]

    assert result["success_rate"] == 1.0
    for episode in episodes:
        assert episode_sequence(episode) == (
            "wood wood table wood pickaxe wood stone stone_pickaxe"
        )


def test_stone_pickaxe_chain_accepts_a_reordered_plan() -> None:
    # Four wood up front, then every craft: the same resources, another order.
    reordered = (
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.PLACE_TABLE,
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.COLLECT_STONE,
        SkillId.CRAFT_STONE_PICKAXE,
    )
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(reordered),
            candidate_skills=STONE_PICKAXE_SKILL_IDS,
        ),
        chain=WOOD_TO_STONE_PICKAXE_CHAIN,
        num_episodes=2,
        seed=7,
    )
    episodes = json.loads(json.dumps(json_ready_chain_result(result)))["episodes"]

    assert result["success_rate"] == 1.0
    for episode in episodes:
        assert "*" not in episode_sequence(episode)


def test_stone_pickaxe_valid_next_skills_follow_the_inventory() -> None:
    valid = WOOD_TO_STONE_PICKAXE_CHAIN.valid_next_skills
    assert valid is not None
    state = _chain_state()

    assert valid(state) == (SkillId.COLLECT_WOOD,)
    assert valid(_with_inventory(state, wood=4)) == (SkillId.PLACE_TABLE,)
    # Two wood can place the table, but the chain still needs more wood.
    assert valid(_with_inventory(state, wood=2)) == (
        SkillId.COLLECT_WOOD,
        SkillId.PLACE_TABLE,
    )
    # Stone already held means no wood pickaxe is needed for this objective.
    assert valid(_with_inventory(state, wood=3, stone=1)) == (SkillId.PLACE_TABLE,)
    assert valid(_with_inventory(state, stone_pickaxe=1)) == ()


def test_redundant_craft_reads_only_held_tools() -> None:
    state = _chain_state()
    holding = _with_inventory(state, wood=4, stone=1, wood_pickaxe=1)

    assert redundant_craft(holding, SkillId.CRAFT_WOOD_PICKAXE)
    assert not redundant_craft(state, SkillId.CRAFT_WOOD_PICKAXE)
    assert not redundant_craft(holding, SkillId.CRAFT_WOOD_SWORD)
    assert not redundant_craft(holding, SkillId.CRAFT_STONE_PICKAXE)
    assert redundant_craft(
        _with_inventory(holding, stone_pickaxe=1), SkillId.CRAFT_STONE_PICKAXE
    )
    # Consumables are never redundant, however many are held.
    plenty = _with_inventory(state, wood=9, stone=9)
    for skill_id in (
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_STONE,
        SkillId.DRINK_WATER,
        SkillId.PLACE_TABLE,
    ):
        assert not redundant_craft(plenty, skill_id)


def test_skip_held_craft_zeroes_only_redundant_crafts() -> None:
    state = _with_inventory(_chain_state(), wood=4, wood_pickaxe=1)
    context = PlannerContext(objective=WOOD_TO_STONE_PICKAXE_CHAIN.objective)
    scorer = SkipHeldCraftFeasibility()

    scores = scorer.score_all_feasibility(state, context, STONE_PICKAXE_SKILL_IDS)

    assert scores[SkillId.CRAFT_WOOD_PICKAXE] == 0.0
    for skill_id in STONE_PICKAXE_SKILL_IDS:
        if skill_id is not SkillId.CRAFT_WOOD_PICKAXE:
            assert scores[skill_id] == reachability_feasibility(
                state, context, skill_id
            )
    assert scorer(state, context, SkillId.CRAFT_WOOD_PICKAXE) == 0.0


def test_skip_held_craft_still_completes_the_stone_pickaxe_chain() -> None:
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(
                WOOD_TO_STONE_PICKAXE_CHAIN.reference_skills
            ),
            feasibility_scorer=SkipHeldCraftFeasibility(),
            candidate_skills=STONE_PICKAXE_SKILL_IDS,
        ),
        chain=WOOD_TO_STONE_PICKAXE_CHAIN,
        num_episodes=2,
        seed=7,
    )

    assert result["success_rate"] == 1.0


def test_the_prompt_lists_only_the_offered_skills() -> None:
    baseline = PlannerPromptStyle.BASELINE
    assert build_planner_system_prompt(baseline, SKILL_IDS) == PLANNER_SYSTEM_PROMPT
    assert "craft_stone_pickaxe" not in PLANNER_SYSTEM_PROMPT
    extended = build_planner_system_prompt(baseline, STONE_PICKAXE_SKILL_IDS)
    assert "craft_stone_pickaxe" in extended
    # Only the catalog line differs; the instructions are the measured ones.
    assert extended.splitlines()[:-1] == PLANNER_SYSTEM_PROMPT.splitlines()[:-1]


def test_config_bounds_the_executor_catalog() -> None:
    base: dict[str, Any] = dict(
        total_timesteps=28,
        num_envs=14,
        num_steps=2,
        num_minibatches=1,
        use_optimistic_resets=False,
        monitor_gpu=False,
    )
    assert SkillPPOConfig(**base).num_skills == 6
    assert SkillPPOConfig(**base, num_skills=7).validate().num_skills == 7
    assert SkillPPOConfig(**base, num_skills=16).validate().num_skills == 16
    for num_skills in (0, 17):
        with pytest.raises(ValueError, match="num_skills"):
            SkillPPOConfig(**base, num_skills=num_skills).validate()


def test_vectorized_environment_trains_the_seventh_skill() -> None:
    environment = make_vectorized_skill_environment(num_envs=14, num_skills=7)
    skills = [
        skill_environment.goal.skill_id
        for skill_environment in environment.environments
    ]

    assert environment.num_skills == 7
    assert set(skills) == set(STONE_PICKAXE_SKILL_IDS)
    with pytest.raises(ValueError, match="divisible"):
        make_vectorized_skill_environment(num_envs=12, num_skills=7)


def test_six_skill_executor_refuses_the_stone_pickaxe() -> None:
    config = SkillPPOConfig(
        total_timesteps=12,
        num_envs=6,
        num_steps=2,
        num_minibatches=1,
        use_optimistic_resets=False,
        monitor_gpu=False,
    ).validate()
    model = make_goal_conditioned_actor_critic(
        num_actions=make_skill_chain_environment().bundle.spec.num_actions,
        num_skills=config.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    # The catalog check runs before any parameter is read.
    executor = LearnedSkillExecutor(params={}, config=config, model=model)

    assert executor.skill_ids == SKILL_IDS
    executor.require_skills(SKILL_IDS)
    with pytest.raises(ValueError, match="craft_stone_pickaxe"):
        executor.require_skills((SkillId.CRAFT_STONE_PICKAXE,))
    with pytest.raises(ValueError, match="craft_stone_pickaxe"):
        evaluate_skill_chain(executor, chain=WOOD_TO_STONE_PICKAXE_CHAIN)


def test_competence_prior_keeps_its_own_catalog() -> None:
    skills = [SkillId.COLLECT_WOOD, SkillId.CRAFT_STONE_PICKAXE] * 4
    steps = [1, 2, None, 3] * 2
    labels = [True, True, False, True] * 2

    with pytest.raises(ValueError, match="outside the catalog"):
        fit_competence_prior(skills, steps, labels)
    prior = fit_competence_prior(skills, steps, labels, skills=STONE_PICKAXE_SKILL_IDS)
    restored = CompetencePrior.from_dict(json.loads(json.dumps(prior.to_dict())))

    assert restored.skills == STONE_PICKAXE_SKILL_IDS
    assert restored.rates == prior.rates
    # The held-out report reads the same catalog, so the seventh skill's
    # cells are printed rather than silently dropped.
    samples = [
        AffordanceSample(
            episode=0,
            boundary=index,
            skill_id=skill_id,
            critic_value=0.5,
            reachability_gate=1.0,
            reachable=step is not None,
            precondition=True,
            steps_to_target=step,
            succeeded=label,
            outcome=SkillOutcome.SUCCESS if label else SkillOutcome.TIMEOUT,
            steps=1,
        )
        for index, (skill_id, step, label) in enumerate(
            zip(skills, steps, labels, strict=True)
        )
    ]
    report = evaluate_competence_prior(samples, prior)
    assert {cell["skill"] for cell in report["cells"]} == {
        SkillId.COLLECT_WOOD.value,
        SkillId.CRAFT_STONE_PICKAXE.value,
    }
    # A six-skill table saved before priors recorded their catalog.
    six = fit_competence_prior(skills[::2], steps[::2], labels[::2])
    payload = six.to_dict()
    del payload["skills"]
    legacy = CompetencePrior.from_dict(payload)
    assert legacy.skills == SKILL_IDS
    # The six-skill prior cannot vouch for a skill its executor never learned.
    state = _with_inventory(_chain_state(), wood=1, stone=1)
    context = PlannerContext(objective="")
    assert (
        legacy.score_all_feasibility(
            state, context, (SkillId.CRAFT_STONE_PICKAXE,)
        )[SkillId.CRAFT_STONE_PICKAXE]
        == 0.0
    )
