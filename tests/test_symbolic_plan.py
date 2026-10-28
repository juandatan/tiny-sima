import jax
import numpy as np
import pytest
from craftax.craftax_classic.constants import Achievement, BlockType

from tiny_sima.evaluation.chains import (
    STONE_PICKAXE_AND_SWORD_CHAIN,
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    ScriptedSkillExecutor,
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
from tiny_sima.goals import (
    ACHIEVEMENT_FIELDS,
    ALL_SKILL_IDS,
    SKILL_ACHIEVEMENTS,
    SkillId,
    SkillMechanics,
    StateEffect,
    StateRequirement,
    achievement_field,
    get_skill_spec,
)
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SkillOutcome,
    constant_feasibility,
    constant_utility,
    reachability_feasibility,
)
from tiny_sima.planner.symbolic import (
    PLANNER_FIELDS,
    REGISTRY_MODEL,
    PlanProgressFeasibility,
    SkillModel,
    check_subgoal,
    plan_next_skills,
    read_field,
    registry_model,
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


def _without_tables(state):
    world_map = np.asarray(state.map)
    cleared = np.where(
        world_map == BlockType.CRAFTING_TABLE.value, BlockType.GRASS.value, world_map
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


class _Belief:
    def __init__(self, **facts: int) -> None:
        self.facts = {f"world.{name}": value for name, value in facts.items()}

    def world_fact(self, field: str) -> int | None:
        return self.facts.get(field)


def _cheap_table_model(model: SkillModel) -> SkillModel:
    cheap = SkillMechanics(
        requires=(
            StateRequirement("inventory.wood", "at_least", 1),
            StateRequirement("world.placement_space_reachable", "equals", True),
        ),
        effects=(
            StateEffect("inventory.wood", "decrease", 1),
            StateEffect("world.crafting_table_reachable", "set", True),
        ),
    )
    return tuple(
        (skill_id, cheap if skill_id is SkillId.PLACE_TABLE else mechanics)
        for skill_id, mechanics in model
    )


def test_registry_model_is_the_default() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    assert registry_model() == REGISTRY_MODEL
    assert [skill_id for skill_id, _ in REGISTRY_MODEL] == list(ALL_SKILL_IDS)
    for point in _points(collect_prefix_trajectory(chain=chain, num_episodes=2)):
        assert shortest_plan(
            point.state, _goal(chain), chain.skills, model=REGISTRY_MODEL
        ) == shortest_plan(point.state, _goal(chain), chain.skills)


def test_a_different_model_changes_the_plan() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    state = _with_inventory(_state(chain), wood=1)
    goal = (StateRequirement("world.crafting_table_reachable", "equals", True),)
    state = _without_tables(state)

    assert plan_next_skills(state, goal, chain.skills) == frozenset(
        (SkillId.COLLECT_WOOD,)
    )
    cheap = _cheap_table_model(REGISTRY_MODEL)
    assert shortest_plan(state, goal, chain.skills, model=cheap) == (
        frozenset((SkillId.PLACE_TABLE,)),
        1,
    )
    # The full chain is one wood shorter when the table costs one.
    start = _state(chain)
    full = shortest_plan(start, _goal(chain), chain.skills)
    shorter = shortest_plan(start, _goal(chain), chain.skills, model=cheap)
    assert full is not None and shorter is not None
    assert shorter[1] == full[1] - 1


def test_candidate_skills_restrict_a_model() -> None:
    goal = _goal(WOOD_TO_STONE_PICKAXE_CHAIN)
    state = _state()
    without_wood = tuple(
        skill_id for skill_id in ALL_SKILL_IDS if skill_id is not SkillId.COLLECT_WOOD
    )
    assert plan_next_skills(state, goal, without_wood, model=REGISTRY_MODEL) is None
    assert plan_next_skills(state, goal, model=REGISTRY_MODEL) == frozenset(
        (SkillId.COLLECT_WOOD,)
    )


def test_belief_overrides_world_facts_and_falls_back_to_the_map() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    goal = _goal(chain)
    state = _state(chain)

    no_trees = _Belief(tree_reachable=0)
    assert read_field(state, "world.tree_reachable") == 1
    assert read_field(state, "world.tree_reachable", no_trees) == 0
    assert plan_next_skills(state, goal, chain.skills, belief=no_trees) is None
    stocked = _with_inventory(state, wood=4)
    plan = plan_next_skills(stocked, goal, chain.skills, belief=no_trees)
    assert plan is not None and SkillId.COLLECT_WOOD not in plan

    # A belief without an opinion reads the map, and never touches inventory.
    silent = _Belief()
    assert plan_next_skills(state, goal, chain.skills, belief=silent) == (
        plan_next_skills(state, goal, chain.skills)
    )
    assert read_field(stocked, "inventory.wood", _Belief(wood=0)) == 4
    # A tree the map has lost can still be believed in.
    cleared = _without_trees(state)
    assert read_field(cleared, "world.tree_reachable", _Belief(tree_reachable=1)) == 1


def test_planner_fields_are_all_readable() -> None:
    state = _state()
    assert "inventory.stone_pickaxe" in PLANNER_FIELDS
    assert "inventory.iron" in PLANNER_FIELDS
    assert "survival.drink" in PLANNER_FIELDS
    for _, mechanics in REGISTRY_MODEL:
        for item in (*mechanics.requires, *mechanics.effects):
            assert item.field in PLANNER_FIELDS
    for field in PLANNER_FIELDS:
        assert isinstance(read_field(state, field), int), field


def test_check_subgoal_accepts_reachable_and_held_subgoals() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    state = _state(chain)

    pickaxe = (StateRequirement("inventory.wood_pickaxe", "at_least", 1),)
    check = check_subgoal(state, pickaxe, chain.skills)
    assert check.reachable and not check.already_holds
    assert check.first_steps == frozenset((SkillId.COLLECT_WOOD,))
    assert check.plan_length == 5
    assert check.reason == ""

    held = check_subgoal(_with_inventory(state, wood_pickaxe=1), pickaxe)
    assert held.reachable and held.already_holds
    assert held.plan_length == 0 and held.first_steps == frozenset()
    # The registry's own ``True`` values and integral floats both parse.
    table = (
        StateRequirement("world.crafting_table_reachable", "equals", 1.0),  # pyright: ignore[reportArgumentType]
    )
    assert check_subgoal(state, table, chain.skills).reachable


def test_check_subgoal_rejects_malformed_subgoals_without_raising() -> None:
    state = _state()
    cases = {
        (): "empty sub-goal",
        (StateRequirement("inventory.gold", "at_least", 1),): "unknown field",
        (StateRequirement("world.lava_reachable", "equals", 1),): "unknown field",
        (StateRequirement("inventory.wood", "most", 1),): "unknown comparison",  # pyright: ignore[reportArgumentType]
        (StateRequirement("inventory.wood", "at_least", 1.5),): "not an integer",  # pyright: ignore[reportArgumentType]
        (StateRequirement("inventory.wood", "at_least", "two"),): "not an integer",  # pyright: ignore[reportArgumentType]
    }
    for subgoal, reason in cases.items():
        check = check_subgoal(state, subgoal)
        assert not check.reachable, subgoal
        assert check.plan_length is None and check.first_steps == frozenset()
        assert reason in check.reason, check.reason


def test_check_subgoal_explains_an_unreachable_subgoal() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    state = _state(chain)

    iron = check_subgoal(
        state,
        (StateRequirement("inventory.iron_pickaxe", "at_least", 1),),
        chain.skills,
    )
    assert not iron.reachable
    assert iron.reason == (
        "inventory.iron_pickaxe needs 1 more: no available skill produces it"
    )

    wood = (StateRequirement("inventory.wood", "at_least", 2),)
    blocked = check_subgoal(state, wood, chain.skills, belief=_Belief(tree_reachable=0))
    assert not blocked.reachable
    assert blocked.reason == (
        "inventory.wood needs 2 more but world.tree_reachable is false for collect_wood"
    )

    # A producer blocked further down: stone needs the pickaxe the trees make.
    stone = (StateRequirement("inventory.stone", "at_least", 1),)
    deep = check_subgoal(state, stone, chain.skills, belief=_Belief(tree_reachable=0))
    assert not deep.reachable
    assert deep.reason == (
        "inventory.stone needs 1 more; collect_stone needs inventory.wood_pickaxe >= 1"
    )


def _with_achievements(state, *achievements: Achievement):
    unlocked = np.asarray(state.achievements).copy()
    for achievement in achievements:
        unlocked[achievement.value] = True
    return state.replace(achievements=state.achievements.at[:].set(unlocked))


def _unlocked(state) -> set[str]:
    return {field for field in ACHIEVEMENT_FIELDS if read_field(state, field)}


def test_achievements_are_planner_fields() -> None:
    assert len(ACHIEVEMENT_FIELDS) == len(Achievement) == 22
    assert set(ACHIEVEMENT_FIELDS) <= set(PLANNER_FIELDS)
    assert "achievement.make_wood_pickaxe" in PLANNER_FIELDS
    assert "achievement.collect_diamond" in PLANNER_FIELDS
    state = _state()
    assert _unlocked(state) == set()
    unlocked = _with_achievements(state, Achievement.MAKE_WOOD_PICKAXE)
    assert _unlocked(unlocked) == {"achievement.make_wood_pickaxe"}
    assert read_field(unlocked, "achievement.make_wood_pickaxe") == 1


def test_every_skill_unlocks_one_achievement_in_the_model_only() -> None:
    assert set(SKILL_ACHIEVEMENTS) == set(ALL_SKILL_IDS)
    for skill_id, mechanics in REGISTRY_MODEL:
        unlock = StateEffect(achievement_field(SKILL_ACHIEVEMENTS[skill_id]), "set", 1)
        registry = get_skill_spec(skill_id).mechanics
        # The prompts render the registry's mechanics, which stay as written.
        assert mechanics == SkillMechanics(
            requires=registry.requires,
            effects=(*registry.effects, unlock),
            repeatable=registry.repeatable,
        )
        assert not any(
            effect.field.startswith("achievement.") for effect in registry.effects
        )


def _run_reference(chain: SkillChainSpec, seed: int = 0):
    environment = chain_environment(chain)
    state = environment.reset(jax.random.PRNGKey(seed)).state
    executor = ScriptedSkillExecutor()
    for index, skill_id in enumerate(chain.reference_skills):
        result = executor.execute(
            environment, state, skill_id, jax.random.PRNGKey(index), horizon=96
        )
        assert result.outcome is SkillOutcome.SUCCESS, skill_id
        yield skill_id, state, result.state
        state = result.state


def test_skill_achievements_match_the_game() -> None:
    # Each scripted success unlocks the achievement the table names for it,
    # and nothing else.
    executed = set()
    for chain in (WOOD_TO_STONE_PICKAXE_CHAIN, STONE_PICKAXE_AND_SWORD_CHAIN):
        for skill_id, before, after in _run_reference(chain):
            field = achievement_field(SKILL_ACHIEVEMENTS[skill_id])
            assert read_field(after, field) == 1, skill_id
            assert _unlocked(after) - _unlocked(before) <= {field}, skill_id
            executed.add(skill_id)
    environment = chain_environment(WOOD_TO_STONE_CHAIN)
    thirsty = environment.reset(jax.random.PRNGKey(0)).state.replace(player_drink=5)
    drank = ScriptedSkillExecutor().execute(
        environment, thirsty, SkillId.DRINK_WATER, jax.random.PRNGKey(0), horizon=96
    )
    assert drank.outcome is SkillOutcome.SUCCESS
    assert _unlocked(drank.state) - _unlocked(thirsty) == {"achievement.collect_drink"}
    executed.add(SkillId.DRINK_WATER)
    assert executed == set(ALL_SKILL_IDS)


def test_achievement_subgoals_plan_end_to_end() -> None:
    chain = WOOD_TO_STONE_PICKAXE_CHAIN
    state = _state(chain)
    unlock = (StateRequirement("achievement.make_wood_pickaxe", "at_least", 1),)

    check = check_subgoal(state, unlock, chain.skills)
    assert check.reachable and not check.already_holds
    assert check.first_steps == frozenset((SkillId.COLLECT_WOOD,))
    # The same plan as holding the pickaxe: three wood, a table, the craft.
    assert check.plan_length == 5
    stone = (StateRequirement("achievement.make_stone_pickaxe", "at_least", 1),)
    assert shortest_plan(state, stone, chain.skills) == shortest_plan(
        state, _goal(chain), chain.skills
    )

    # An achievement stays unlocked once the tool it took is gone.
    unlocked = _with_achievements(state, Achievement.MAKE_WOOD_PICKAXE)
    held = check_subgoal(unlocked, unlock, chain.skills)
    assert held.reachable and held.already_holds
    assert held.plan_length == 0 and held.first_steps == frozenset()

    table = (StateRequirement("achievement.place_table", "at_least", 1),)
    assert plan_next_skills(_with_inventory(state, wood=2), table) == frozenset(
        (SkillId.PLACE_TABLE,)
    )

    diamond = check_subgoal(
        state,
        (StateRequirement("achievement.collect_diamond", "at_least", 1),),
        chain.skills,
    )
    assert not diamond.reachable
    assert diamond.reason == (
        "achievement.collect_diamond needs 1 more: no available skill produces it"
    )


def _without_achievements(model: SkillModel) -> SkillModel:
    return tuple(
        (skill_id, get_skill_spec(skill_id).mechanics) for skill_id, _ in model
    )


@pytest.mark.parametrize("chain", CHAINS, ids=lambda chain: chain.name)
def test_achievement_effects_leave_other_plans_unchanged(chain) -> None:
    # A goal that reads no achievement is planned exactly as over the
    # registry's own mechanics.
    bare = _without_achievements(REGISTRY_MODEL)
    goals = (
        _goal(chain),
        (StateRequirement("inventory.wood", "at_least", 3),),
        (StateRequirement("world.crafting_table_reachable", "equals", True),),
    )
    for point in _points(collect_prefix_trajectory(chain=chain, num_episodes=2)):
        for goal in goals:
            assert shortest_plan(point.state, goal, chain.skills) == shortest_plan(
                point.state, goal, chain.skills, model=bare
            ), point.label
