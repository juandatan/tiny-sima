import json
import math

import jax
import numpy as np
import pytest
from craftax.craftax_classic.constants import Achievement

from tiny_sima.envs import OpenEndedEnvironment, make_open_ended_environment
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.envs.craftax import CraftaxBundle
from tiny_sima.evaluation import (
    ACHIEVEMENT_NAMES,
    ACHIEVEMENT_TIERS,
    OPEN_ENDED_CHAIN,
    OPEN_ENDED_OBJECTIVE,
    ChainTermination,
    ScriptedSkillExecutor,
    crafter_score,
    evaluate_open_ended,
    evaluate_skill_chain,
    json_ready_open_ended_result,
    open_ended_planners,
    summarize_open_ended,
)
from tiny_sima.goals import ALL_SKILL_IDS
from tiny_sima.memory import EpisodeMemory
from tiny_sima.planner import (
    NOTHING_TO_DO,
    TECH_TREE_ORDER,
    PlannerContext,
    PlannerSelectionStatus,
    SayCanPlanner,
    ScriptedGoalSelector,
    SubgoalPlanner,
    achievement_subgoal,
)
from tiny_sima.planner.subgoals import (
    ALL_VITAL_THRESHOLDS,
    OPEN_ENDED_SUBGOAL_FIELDS,
    SUBGOAL_FIELDS,
    SubgoalEvent,
    SubgoalPlan,
    SubgoalRequest,
    SubgoalSource,
    open_ended_subgoal_fields,
    render_subgoal,
)


def _mob_counts(state) -> int:
    return sum(
        int(np.asarray(getattr(state, kind).mask).sum())
        for kind in ("zombies", "cows", "skeletons", "arrows")
    )


def _run_noops(environment: OpenEndedEnvironment, steps: int) -> int:
    state = environment.reset(jax.random.PRNGKey(3)).state
    bundle = environment.bundle
    step = jax.jit(lambda rng, state: bundle.env.step(rng, state, 0, bundle.params))
    most = 0
    for rng in jax.random.split(jax.random.PRNGKey(4), steps):
        _, state, _, _, _ = step(rng, state)
        most = max(most, _mob_counts(state))
    return most


def test_open_world_keeps_or_removes_creatures() -> None:
    with_mobs = make_open_ended_environment(mobs=True)
    without = make_open_ended_environment(mobs=False)
    assert isinstance(with_mobs, OpenEndedEnvironment)
    assert with_mobs.tree_count == OPEN_ENDED_CHAIN.tree_count
    assert with_mobs.bundle.params.spawn_zombie_night_chance > 0.0
    assert without.bundle.params.spawn_cow_chance == 0.0
    assert with_mobs.bundle.params.max_timesteps > 3000

    # The same key gives the same generated world, not a carved scenario.
    rng = jax.random.PRNGKey(0)
    on, off = with_mobs.reset(rng).state, without.reset(rng).state
    assert np.array_equal(np.asarray(on.map), np.asarray(off.map))
    assert _mob_counts(off) == 0
    chain_map = make_skill_chain_environment().reset(rng).state.map
    assert not np.array_equal(np.asarray(on.map), np.asarray(chain_map))

    assert _run_noops(with_mobs, 60) > 0
    assert _run_noops(without, 60) == 0


class _CowAtReset:
    """A Craftax env whose generated world starts with a cow."""

    def __init__(self, env) -> None:
        self.env = env

    def reset(self, rng, params):
        observation, state = self.env.reset(rng, params)
        cows = state.cows.replace(mask=state.cows.mask.at[0].set(True))
        return observation, state.replace(cows=cows)

    def get_obs(self, state):
        return self.env.get_obs(state)


def test_reset_without_mobs_clears_generated_creatures() -> None:
    bundle = make_open_ended_environment(mobs=False).bundle
    crowded = CraftaxBundle(_CowAtReset(bundle.env), bundle.params, bundle.spec)
    rng = jax.random.PRNGKey(0)
    assert _mob_counts(OpenEndedEnvironment(crowded, mobs=True).reset(rng).state)
    assert not _mob_counts(OpenEndedEnvironment(crowded, mobs=False).reset(rng).state)


def _chain_state(seed: int = 0, **inventory):
    state = make_skill_chain_environment(tree_count=5).reset(jax.random.PRNGKey(seed))
    state = state.state
    return state.replace(inventory=state.inventory.replace(**inventory))


def _unlock(state, *achievements: Achievement):
    unlocked = state.achievements
    for achievement in achievements:
        unlocked = unlocked.at[achievement.value].set(True)
    return state.replace(achievements=unlocked)


def _request(state, plan: SubgoalPlan | None = None, memory=None) -> SubgoalRequest:
    from tiny_sima.planner.subgoals import ReplanTrigger

    return SubgoalRequest(
        index=0,
        state=state,
        context=PlannerContext(OPEN_ENDED_OBJECTIVE, memory=memory),
        plan=plan or SubgoalPlan(),
        trigger=ReplanTrigger.START,
    )


def _proposed(selector: ScriptedGoalSelector, request: SubgoalRequest) -> str:
    payload = json.loads(selector.propose_subgoals([request])[0])
    (item,) = payload["subgoals"]
    return f"{item['field']} >= {item['at_least']}"


def test_tech_tree_order_lists_every_achievement_once() -> None:
    assert sorted(achievement.value for achievement in TECH_TREE_ORDER) == list(
        range(len(Achievement))
    )
    assert isinstance(ScriptedGoalSelector(), SubgoalSource)


def test_selector_proposes_the_next_reachable_locked_achievement() -> None:
    selector = ScriptedGoalSelector()
    state = _chain_state()
    assert _proposed(selector, _request(state)) == "achievement.collect_wood >= 1"

    state = _unlock(state, Achievement.COLLECT_WOOD)
    assert _proposed(selector, _request(state)) == "achievement.place_table >= 1"

    # Stone and iron tools need what the skills cannot make yet: with the
    # wood line done, the next reachable one is collect_stone, and after
    # the stone pickaxe it skips the stone sword, furnace, coal, iron and
    # diamond for collect_drink.
    state = _unlock(
        state,
        Achievement.PLACE_TABLE,
        Achievement.MAKE_WOOD_PICKAXE,
        Achievement.MAKE_WOOD_SWORD,
    )
    assert _proposed(selector, _request(state)) == "achievement.collect_stone >= 1"
    state = _unlock(state, Achievement.COLLECT_STONE, Achievement.MAKE_STONE_PICKAXE)
    assert _proposed(selector, _request(state)) == "achievement.collect_drink >= 1"

    # With nothing left, the answer always holds, so the planner turns it down.
    state = _unlock(state, Achievement.COLLECT_DRINK)
    assert _proposed(selector, _request(state)) == render_subgoal(NOTHING_TO_DO)


def test_low_drink_comes_before_achievements() -> None:
    selector = ScriptedGoalSelector()
    state = _chain_state().replace(player_drink=3)
    assert _proposed(selector, _request(state)) == "survival.drink >= 7"
    state = state.replace(player_drink=4)
    assert _proposed(selector, _request(state)) == "achievement.collect_wood >= 1"


def test_selector_skips_an_achievement_after_repeated_failures() -> None:
    selector = ScriptedGoalSelector()
    wood = render_subgoal(achievement_subgoal(Achievement.COLLECT_WOOD))
    once = SubgoalPlan(
        events=(SubgoalEvent(0, wood, "timed_out"),),
        decision_index=5,
    )
    assert _proposed(selector, _request(_chain_state(), once)) == wood

    twice = SubgoalPlan(
        events=(
            SubgoalEvent(0, wood, "timed_out"),
            SubgoalEvent(3, wood, "abandoned"),
            # Skill failures do not count against the sub-goal.
            SubgoalEvent(4, wood, "skill_failed"),
        ),
        decision_index=5,
    )
    # place_table is next and still reachable: it collects the wood itself.
    assert (
        _proposed(selector, _request(_chain_state(), twice))
        == "achievement.place_table >= 1"
    )


def test_a_rejection_in_this_decision_moves_on_to_the_next_candidate() -> None:
    selector = ScriptedGoalSelector()
    wood = render_subgoal(achievement_subgoal(Achievement.COLLECT_WOOD))
    earlier = SubgoalPlan(
        events=(SubgoalEvent(1, wood, "rejected", "made up"),), decision_index=4
    )
    assert _proposed(selector, _request(_chain_state(), earlier)) == wood
    now = SubgoalPlan(
        events=(SubgoalEvent(4, wood, "rejected", "made up"),), decision_index=4
    )
    assert (
        _proposed(selector, _request(_chain_state(), now))
        == "achievement.place_table >= 1"
    )


def test_selector_reads_world_facts_from_memory() -> None:
    # The memory has ruled water out, so drinking is not proposed even at
    # low drink, and the selector falls through to the achievements.
    state = _chain_state().replace(player_drink=2)
    memory = EpisodeMemory().observe(state)

    class NoWater:
        def world_fact(self, field: str) -> int | None:
            if field == "world.water_reachable":
                return 0
            return memory.world_fact(field)

    selector = ScriptedGoalSelector()
    assert _proposed(selector, _request(state, memory=memory)) == "survival.drink >= 7"
    assert (
        _proposed(selector, _request(state, memory=NoWater()))
        == "achievement.collect_wood >= 1"
    )
    blind = ScriptedGoalSelector(use_memory_belief=False)
    assert _proposed(blind, _request(state, memory=NoWater())) == "survival.drink >= 7"


def test_open_ended_fields_offer_only_the_achievements_skills_unlock() -> None:
    fields = open_ended_subgoal_fields(ALL_SKILL_IDS)

    assert "achievement.make_stone_pickaxe" in fields
    assert "achievement.collect_drink" in fields
    assert "achievement.place_stone" not in fields
    assert "achievement.eat_cow" not in fields
    assert set(fields) < set(OPEN_ENDED_SUBGOAL_FIELDS)
    assert open_ended_subgoal_fields(ALL_SKILL_IDS[:1]) == (
        *SUBGOAL_FIELDS,
        "achievement.collect_wood",
    )


def _scripted_design() -> SubgoalPlanner:
    return SubgoalPlanner(
        ScriptedGoalSelector(),
        fields=open_ended_subgoal_fields(),
        vital_thresholds=ALL_VITAL_THRESHOLDS,
        threat_radius=3,
        replan_at_night=True,
    )


def test_scripted_design_unlocks_achievements_in_the_closed_loop() -> None:
    result = evaluate_open_ended(
        _scripted_design(),
        ScriptedSkillExecutor(),
        make_open_ended_environment(mobs=False),
        num_episodes=2,
        seed=0,
        max_skill_calls=6,
        memory_factory=EpisodeMemory,
    )
    episodes = result["episodes"]
    assert len(episodes) == 2
    for episode in episodes:
        # The skill-call budget ends the episode before the skills run out.
        assert episode["termination"] == ChainTermination.DECISION_LIMIT.value
        assert episode["decisions"] == 6
        assert episode["achievements"][:3] == [
            "collect_wood",
            "place_table",
            "make_wood_pickaxe",
        ]
        unlocks = episode["achievement_unlocks"]
        decisions = [unlock["decision_index"] for unlock in unlocks]
        steps = [unlock["environment_step"] for unlock in unlocks]
        assert decisions == sorted(decisions) and steps == sorted(steps)
        assert 0 < steps[-1] <= episode["environment_steps"]
        assert episode["vlm_calls"] >= len(unlocks)
    summary = result["summary"]
    assert summary["num_episodes"] == 2
    assert summary["achievements_per_episode"]["mean"] >= 3
    assert summary["success_rates"]["collect_wood"] == 1.0
    assert summary["success_rates"]["collect_diamond"] == 0.0
    assert summary["termination_counts"]["decision_limit"] == 2
    assert summary["skill_calls"]["mean"] == 6
    json.dumps(json_ready_open_ended_result(result))


def test_scripted_design_stops_once_nothing_is_reachable() -> None:
    result = evaluate_open_ended(
        _scripted_design(),
        ScriptedSkillExecutor(),
        make_open_ended_environment(mobs=False),
        num_episodes=1,
        seed=1,
        memory_factory=EpisodeMemory,
        include_trace=False,
    )
    (episode,) = result["episodes"]
    assert "trace" not in episode
    assert episode["termination"] == ChainTermination.NO_VIABLE_SKILL.value
    assert episode["decisions"] < 80
    # Every achievement the seven skills unlock, and nothing else.
    assert sorted(episode["achievements"]) == sorted(
        [
            "collect_wood",
            "place_table",
            "make_wood_pickaxe",
            "make_wood_sword",
            "collect_stone",
            "make_stone_pickaxe",
            "collect_drink",
        ]
    )


def test_step_budget_ends_the_episode() -> None:
    result = evaluate_open_ended(
        _scripted_design(),
        ScriptedSkillExecutor(),
        make_open_ended_environment(mobs=False),
        num_episodes=1,
        seed=0,
        max_environment_steps=5,
        memory_factory=EpisodeMemory,
    )
    (episode,) = result["episodes"]
    assert episode["termination"] == ChainTermination.STEP_LIMIT.value
    # Checked after each skill, so the last skill may run past the budget.
    assert episode["environment_steps"] >= 5
    assert episode["environment_steps"] - episode["trace"][-1]["steps"] < 5
    assert result["summary"]["termination_counts"]["step_limit"] == 1


def test_death_ends_the_episode() -> None:
    class Starving(OpenEndedEnvironment):
        # One health left and every need empty, so the player dies within a
        # few skills whatever it does.
        def reset(self, rng):
            reset = super().reset(rng)
            state = reset.state.replace(
                player_health=1,
                player_food=0,
                player_drink=0,
                player_energy=0,
                player_recover=-0.9,
            )
            return reset._replace(state=state)

    environment = Starving(make_open_ended_environment(mobs=False).bundle, mobs=False)
    result = evaluate_open_ended(
        _scripted_design(),
        ScriptedSkillExecutor(),
        environment,
        num_episodes=1,
    )
    (episode,) = result["episodes"]
    assert episode["termination"] == ChainTermination.DIED.value
    assert episode["decisions"] < 80
    assert episode["trace"][0]["requested_skill"] == "drink_water"
    assert result["summary"]["termination_counts"]["died"] == 1


def test_chain_summaries_keep_their_termination_keys() -> None:
    result = evaluate_skill_chain(ScriptedSkillExecutor(), num_episodes=1)
    assert set(result["termination_counts"]) == {
        "success",
        "no_viable_skill",
        "failure_limit",
        "decision_limit",
        "invalid_planner_output",
    }


def test_crafter_score_on_known_rates() -> None:
    assert crafter_score({}) == 0.0
    assert crafter_score(dict.fromkeys(ACHIEVEMENT_NAMES, 1.0)) == pytest.approx(100.0)
    # One achievement always unlocked: exp(ln(101) / 22) - 1.
    assert crafter_score({"collect_wood": 1.0}) == pytest.approx(
        math.exp(math.log(101) / 22) - 1
    )
    half = crafter_score(dict.fromkeys(ACHIEVEMENT_NAMES, 0.5))
    assert half == pytest.approx(50.0)
    with pytest.raises(ValueError):
        crafter_score({"collect_gold": 1.0})


def test_tiers_cover_every_achievement_exactly_once() -> None:
    tiered = [name for names in ACHIEVEMENT_TIERS.values() for name in names]
    assert sorted(tiered) == sorted(ACHIEVEMENT_NAMES)
    assert len(tiered) == len(set(tiered)) == 22
    assert "wake_up" in ACHIEVEMENT_TIERS["survival"]


def test_summary_aggregates_records() -> None:
    def record(achievements, termination="decision_limit"):
        return {
            "achievements": achievements,
            "decisions": 10,
            "environment_steps": 100,
            "vlm_calls": 4,
            "termination": termination,
        }

    summary = summarize_open_ended(
        [
            record(["collect_wood", "place_table", "eat_cow"]),
            record(["collect_wood"], termination="died"),
        ]
    )
    assert summary["achievements_per_episode"] == {"mean": 2.0, "std": 1.0}
    assert summary["success_rates"]["place_table"] == 0.5
    assert summary["tiers"]["survival"]["success_rate"] == 0.25
    assert summary["tiers"]["basic"]["achievements_per_episode"]["mean"] == 1.5
    assert summary["termination_counts"]["died"] == 1
    expected = crafter_score({"collect_wood": 1.0, "place_table": 0.5, "eat_cow": 0.5})
    assert summary["crafter_score"] == pytest.approx(expected)


def test_design_factory_builds_the_three_designs() -> None:
    from tiny_sima.config import VLMPlannerConfig

    class Backend:
        def generate(self, request):
            raise AssertionError("not called")

    planners = open_ended_planners(
        ALL_SKILL_IDS, Backend(), VLMPlannerConfig(), None, None
    )
    assert list(planners) == [
        "saycan_recipe_aware",
        "subgoal_vlm",
        "subgoal_scripted",
    ]
    assert isinstance(planners["saycan_recipe_aware"], SayCanPlanner)
    vlm, scripted = planners["subgoal_vlm"], planners["subgoal_scripted"]
    assert isinstance(vlm, SubgoalPlanner) and isinstance(scripted, SubgoalPlanner)
    assert isinstance(scripted.backend, ScriptedGoalSelector)
    for planner in (vlm, scripted):
        assert planner.fields == open_ended_subgoal_fields(ALL_SKILL_IDS)
        assert planner.vital_thresholds == ALL_VITAL_THRESHOLDS
        assert planner.threat_radius is not None
        assert planner.replan_at_night

    # The scripted design answers without the VLM.
    selection = scripted.select_many(
        [_chain_state()], [PlannerContext(OPEN_ENDED_OBJECTIVE)]
    )[0]
    assert selection.status is PlannerSelectionStatus.SELECTED
