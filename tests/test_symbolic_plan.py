import jax
import numpy as np
import pytest
from craftax.craftax_classic.constants import BlockType

from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    SkillChainSpec,
    chain_environment,
)
from tiny_sima.evaluation.planner import (
    PrefixTrajectory,
    collect_detour_trajectory,
    collect_prefix_trajectory,
    collect_search_trajectory,
    collect_surplus_trajectory,
    evaluate_planner_prefixes,
)
from tiny_sima.goals import ALL_SKILL_IDS, SkillId, StateRequirement
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    constant_feasibility,
    constant_utility,
    reachability_feasibility,
)
from tiny_sima.planner.symbolic import (
    PlanProgressFeasibility,
    plan_next_skills,
    read_field,
    shortest_plan,
)

CHAINS = (WOOD_TO_STONE_CHAIN, WOOD_TO_STONE_PICKAXE_CHAIN)


def _state(chain: SkillChainSpec = WOOD_TO_STONE_PICKAXE_CHAIN, seed: int = 0):
    return chain_environment(chain).reset(jax.random.PRNGKey(seed)).state


def _with_inventory(state, **counts):
    return state.replace(inventory=state.inventory.replace(**counts))


def _without_trees(state):
    world_map = np.asarray(state.map)
    cleared = np.where(
        world_map == BlockType.TREE.value, BlockType.GRASS.value, world_map
    )
    return state.replace(map=state.map.at[:].set(cleared))


def _goal(chain: SkillChainSpec):
    assert chain.goal is not None
    return chain.goal


def _points(trajectory: PrefixTrajectory):
    return [point for group in trajectory.boundaries for point in group]


def _derived_points(chain: SkillChainSpec):
    trajectory = collect_prefix_trajectory(chain=chain, num_episodes=4, seed=51)
    points = _points(trajectory)
    points += _points(collect_detour_trajectory(trajectory, chain=chain, seed=51))
    if chain is WOOD_TO_STONE_CHAIN:
        points += _points(collect_search_trajectory(trajectory, seed=51))
    else:
        points += _points(collect_surplus_trajectory(trajectory, chain=chain, seed=51))
    return points


@pytest.mark.parametrize("chain", CHAINS, ids=lambda chain: chain.name)
def test_plan_reproduces_the_hand_written_accepted_sets(chain) -> None:
    # The accepted sets were written by hand for each chain. Searching the
    # registry's mechanics from the formal goal alone gives the same sets at
    # every oracle, detour, search and surplus state.
    points = _derived_points(chain)

    assert len({point.label for point in points}) > 4
    for point in points:
        assert plan_next_skills(point.state, _goal(chain), chain.skills) == frozenset(
            point.accepted_skills
        ), point.label


@pytest.mark.parametrize("chain", CHAINS, ids=lambda chain: chain.name)
def test_goal_agrees_with_is_complete(chain) -> None:
    for point in _derived_points(chain):
        plan = shortest_plan(point.state, _goal(chain), chain.skills)
        assert plan is not None
        assert (plan[1] == 0) == chain.is_complete(point.state)
    complete = _with_inventory(_state(chain), wood_pickaxe=1, stone=1, stone_pickaxe=1)
    assert chain.is_complete(complete)
    assert shortest_plan(complete, _goal(chain), chain.skills) == (frozenset(), 0)


def test_reference_plan_length_is_the_shortest() -> None:
    for chain in CHAINS:
        plan = shortest_plan(_state(chain), _goal(chain), chain.skills)
        assert plan == (frozenset((SkillId.COLLECT_WOOD,)), len(chain.reference_skills))


def test_no_plan_once_the_trees_are_gone() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    state = _without_trees(_state(chain))

    assert read_field(state, "world.tree_reachable") == 0
    assert plan_next_skills(state, _goal(chain), chain.skills) is None
    # Wood already held is still enough for the steps that need no more.
    stocked = _with_inventory(state, wood=4)
    assert plan_next_skills(stocked, _goal(chain), chain.skills) == frozenset(
        (SkillId.PLACE_TABLE,)
    )


def test_waste_is_never_on_the_plan() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    goal = _goal(chain)
    # Wood pickaxe held and stone collected, wood spent: wood is the only step.
    surplus = _with_inventory(_state(chain), wood=0, stone=1, wood_pickaxe=1)
    assert plan_next_skills(surplus, goal) == frozenset((SkillId.COLLECT_WOOD,))
    # With wood in hand, neither the held pickaxe nor the sword is on the plan.
    stocked = _with_inventory(surplus, wood=3)
    plan = plan_next_skills(stocked, goal)
    assert plan is not None
    assert SkillId.CRAFT_WOOD_PICKAXE not in plan
    assert SkillId.CRAFT_WOOD_SWORD not in plan
    assert SkillId.DRINK_WATER not in plan
    assert SkillId.COLLECT_WOOD not in plan


def test_unknown_field_raises() -> None:
    goal = (StateRequirement("inventory.diamond_count", "at_least", 1),)
    with pytest.raises(AttributeError):
        plan_next_skills(_state(), goal, ())
    with pytest.raises(ValueError, match="world.lava_reachable"):
        read_field(_state(), "world.lava_reachable")


def test_plan_progress_vetoes_off_plan_skills_and_defers_without_a_plan() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    context = PlannerContext(objective=chain.objective)
    scorer = PlanProgressFeasibility(_goal(chain))
    state = _with_inventory(_state(chain), wood=0, stone=1, wood_pickaxe=1)

    scores = scorer.score_all_feasibility(state, context, ALL_SKILL_IDS)
    base = {
        skill_id: reachability_feasibility(state, context, skill_id)
        for skill_id in ALL_SKILL_IDS
    }
    assert scores[SkillId.COLLECT_WOOD] == base[SkillId.COLLECT_WOOD] > 0.0
    assert all(
        scores[skill_id] == 0.0
        for skill_id in ALL_SKILL_IDS
        if skill_id is not SkillId.COLLECT_WOOD
    )
    assert scorer(state, context, SkillId.COLLECT_STONE) == 0.0

    soft = PlanProgressFeasibility(
        _goal(chain), base=constant_feasibility, off_plan=0.5
    )
    assert soft(state, context, SkillId.COLLECT_STONE) == 0.5
    assert soft(state, context, SkillId.COLLECT_WOOD) == 1.0

    # No tree left and wood short: no plan, so the base scores stand.
    stranded = _without_trees(state)
    assert scorer.score_all_feasibility(stranded, context, ALL_SKILL_IDS) == {
        skill_id: reachability_feasibility(stranded, context, skill_id)
        for skill_id in ALL_SKILL_IDS
    }


@pytest.mark.parametrize("chain", CHAINS, ids=lambda chain: chain.name)
def test_plan_only_planner_passes_every_derived_state(chain) -> None:
    trajectory = collect_prefix_trajectory(chain=chain, num_episodes=4, seed=51)
    detours = collect_detour_trajectory(trajectory, chain=chain, seed=51)

    def evaluate(feasibility, derived):
        return evaluate_planner_prefixes(
            SayCanPlanner(
                candidate_skills=chain.skills,
                utility_scorer=constant_utility,
                feasibility_scorer=feasibility,
            ),
            chain=chain,
            num_episodes=4,
            seed=51,
            trajectory=derived,
        )

    planned = evaluate(PlanProgressFeasibility(_goal(chain)), detours)
    assert planned["accepted_next_skill_accuracy"] == 1.0
    assert set(planned["accuracy_by_label"].values()) == {1.0}
    assert evaluate(PlanProgressFeasibility(_goal(chain)), trajectory)[
        "accepted_next_skill_accuracy"
    ] == 1.0
    # Reachability alone falls for some of the same states.
    assert evaluate(reachability_feasibility, detours)[
        "accepted_next_skill_accuracy"
    ] < 1.0


def test_detour_states_follow_an_off_objective_request() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    trajectory = collect_prefix_trajectory(chain=chain, num_episodes=4, seed=51)
    detours = collect_detour_trajectory(trajectory, chain=chain, seed=51)
    points = _points(detours)

    assert len(detours.boundaries) == len(trajectory.boundaries)
    labels = {point.label for point in points}
    assert {"collect_stonex1", "collect_woodx1", "craft_wood_pickaxex1"} <= labels
    for point in points:
        skill, count = point.label.rsplit("x", 1)
        detour_skill = SkillId(skill)
        assert 1 <= int(count) <= 2
        assert chain.valid_next_skills is not None
        assert point.accepted_skills == chain.valid_next_skills(point.state)
        assert point.accepted_skills
        # The oracle's prefix, then only the detour skill.
        added = point.context.history[point.boundary :]
        assert added
        assert all(attempt.skill_id is detour_skill for attempt in added)
        assert added[-1].skill_id is detour_skill

    restricted = collect_detour_trajectory(
        trajectory,
        chain=chain,
        detour_skills=(SkillId.DRINK_WATER,),
        max_detours=1,
        seed=51,
    )
    assert {point.label for point in _points(restricted)} == {"drink_waterx1"}
    with pytest.raises(ValueError, match="wood_to_stone"):
        collect_detour_trajectory(trajectory, chain=WOOD_TO_STONE_CHAIN)
