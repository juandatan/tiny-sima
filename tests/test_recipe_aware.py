import json
from dataclasses import dataclass, field

import jax
import numpy as np
import pytest

from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    chain_environment,
)
from tiny_sima.evaluation.planner import (
    COLLECTION_SKILLS,
    collect_prefix_trajectory,
    collect_surplus_trajectory,
    evaluate_planner_prefixes,
    is_surplus_boundary,
)
from tiny_sima.goals import ALL_SKILL_IDS, SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    constant_feasibility,
)
from tiny_sima.planner.vlm import (
    PlannerPromptStyle,
    VLMRequest,
    VLMUtilityScorer,
    build_planner_prompt,
    build_planner_system_prompt,
    inventory_shortfall,
    progress_assessment_limit,
    recipe_entries,
)

CHAIN = WOOD_TO_STONE_PICKAXE_CHAIN


@dataclass
class FakeBackend:
    response: str
    requests: list[VLMRequest] = field(default_factory=list)

    def generate(self, request: VLMRequest) -> str:
        self.requests.append(request)
        return self.response


def _state(seed: int = 0):
    return chain_environment(CHAIN).reset(jax.random.PRNGKey(seed)).state


def _with_inventory(state, **counts):
    return state.replace(inventory=state.inventory.replace(**counts))


def _surplus_state():
    # The diagnosis's commonest waste: wood pickaxe held, the last wood spent,
    # and one stone already collected. The derived fields ignore the table.
    return _with_inventory(_state(), wood=0, stone=1, wood_pickaxe=1)


def _candidates(user_prompt: str) -> list[dict]:
    prefix = "Candidate skills: "
    line = next(
        line for line in user_prompt.splitlines() if line.startswith(prefix)
    )
    return json.loads(line.removeprefix(prefix))


def _stone_first(state, context, skill_id: SkillId) -> float:
    # The ranking the objective-first prompt produced at these states: stone,
    # the resource the objective names, above wood.
    del state, context
    if skill_id is SkillId.COLLECT_STONE:
        return 1.0
    return 0.25 if skill_id is SkillId.COLLECT_WOOD else 0.0


def _shortfall_first(state, context, skill_id: SkillId) -> float:
    # A Say term that reads nothing but the derived fields: a collection skill
    # that fills some candidate's shortfall.
    del context
    filled = recipe_entries(state, ALL_SKILL_IDS)[skill_id]["fills_shortfall_of"]
    return float(skill_id in COLLECTION_SKILLS and bool(filled))


def test_shortfall_reads_the_declared_inventory_requirements() -> None:
    state = _state()

    assert inventory_shortfall(state, SkillId.COLLECT_WOOD) == {}
    assert inventory_shortfall(state, SkillId.PLACE_TABLE) == {"inventory.wood": 2}
    assert inventory_shortfall(state, SkillId.COLLECT_STONE) == {
        "inventory.wood_pickaxe": 1
    }
    # The recipe the objective-first prompt miscounted takes one of each.
    assert inventory_shortfall(state, SkillId.CRAFT_STONE_PICKAXE) == {
        "inventory.wood": 1,
        "inventory.stone": 1,
    }
    # Reach is the Can term's, so a craft with no table in view is not short
    # of anything once the wood is held.
    assert inventory_shortfall(
        _with_inventory(state, wood=1), SkillId.CRAFT_WOOD_PICKAXE
    ) == {}


def test_recipe_entries_say_which_resource_is_still_short() -> None:
    entries = recipe_entries(_surplus_state(), ALL_SKILL_IDS)

    assert entries[SkillId.CRAFT_STONE_PICKAXE]["inventory_shortfall"] == {
        "inventory.wood": 1
    }
    wood_fills = entries[SkillId.COLLECT_WOOD]["fills_shortfall_of"]
    assert "craft_stone_pickaxe" in wood_fills
    # Stone is no longer short anywhere, so collecting more fills nothing.
    assert entries[SkillId.COLLECT_STONE]["fills_shortfall_of"] == []
    # A held tool is reported, and its recipe never makes wood look needed.
    assert entries[SkillId.CRAFT_WOOD_PICKAXE]["result_held"] is True
    assert "craft_wood_pickaxe" not in wood_fills
    assert entries[SkillId.CRAFT_STONE_PICKAXE]["result_held"] is False

    # At reset, wood fills every craft and the wood pickaxe fills stone.
    reset = recipe_entries(_state(), ALL_SKILL_IDS)
    assert reset[SkillId.COLLECT_WOOD]["fills_shortfall_of"] == [
        "place_table",
        "craft_wood_pickaxe",
        "craft_wood_sword",
        "craft_stone_pickaxe",
    ]
    assert reset[SkillId.CRAFT_WOOD_PICKAXE]["fills_shortfall_of"] == [
        "collect_stone"
    ]


def test_recipe_aware_prompt_adds_only_the_derived_fields() -> None:
    style = PlannerPromptStyle.RECIPE_AWARE
    context = PlannerContext(objective=CHAIN.objective)
    state = _surplus_state()
    system_prompt = build_planner_system_prompt(style, CHAIN.skills)
    user_prompt = build_planner_prompt(state, context, CHAIN.skills, style=style)
    objective_first_system = build_planner_system_prompt(
        PlannerPromptStyle.OBJECTIVE_FIRST, CHAIN.skills
    )
    objective_first_user = build_planner_prompt(
        state, context, CHAIN.skills, style=PlannerPromptStyle.OBJECTIVE_FIRST
    )

    # The blanket repetition rule is replaced, not kept beside the new one.
    assert "demote redundant repetitions" not in system_prompt
    assert "demote redundant repetitions" in objective_first_system
    assert "do not recount them" in system_prompt
    assert "do not demote a skill because its target is out of view" in system_prompt
    assert "inventory_shortfall" in user_prompt
    assert "inventory_shortfall" not in objective_first_user
    # Same header, mechanics catalog and assessment limit as objective-first.
    assert user_prompt.splitlines()[:3] == objective_first_user.splitlines()[:3]
    assert system_prompt.splitlines()[-1] == objective_first_system.splitlines()[-1]
    assert progress_assessment_limit(style) == 96
    assert "at most 96 characters" in user_prompt

    recipe = _candidates(user_prompt)
    base = _candidates(objective_first_user)
    assert [entry["skill"] for entry in recipe] == [entry["skill"] for entry in base]
    for with_recipe, without in zip(recipe, base, strict=True):
        assert {key: with_recipe[key] for key in without} == without
        assert set(with_recipe) - set(without) == {
            "inventory_shortfall",
            "result_held",
            "fills_shortfall_of",
        }


def test_recipe_aware_uses_the_baseline_schema() -> None:
    ranking = [skill_id.value for skill_id in CHAIN.skills]
    backend = FakeBackend(
        response=json.dumps(
            {
                "progress_assessment": "Need one wood.",
                "selected_skill": ranking[0],
                "ranking": ranking,
            }
        )
    )
    scorer = VLMUtilityScorer(backend, prompt_style=PlannerPromptStyle.RECIPE_AWARE)

    batch = scorer.score_all(
        _surplus_state(),
        PlannerContext(objective=CHAIN.objective),
        CHAIN.skills,
    )

    assert batch.valid
    assert backend.requests[0].require_available is False
    assert backend.requests[0].assessment_max_length == 96


def test_surplus_boundary_needs_a_held_resource_that_is_not_accepted() -> None:
    state = _surplus_state()

    assert is_surplus_boundary(state, (SkillId.COLLECT_WOOD,))
    # Nothing held, or the held resource is itself the next step.
    assert not is_surplus_boundary(_state(), (SkillId.COLLECT_WOOD,))
    assert not is_surplus_boundary(
        state, (SkillId.COLLECT_WOOD, SkillId.COLLECT_STONE)
    )
    # A craft as the next step is the prefix ladder's question, not this one.
    assert not is_surplus_boundary(state, (SkillId.CRAFT_STONE_PICKAXE,))
    assert not is_surplus_boundary(state, ())
    # Stone held without a pickaxe could not be collected again.
    assert not is_surplus_boundary(
        _with_inventory(state, wood_pickaxe=0), (SkillId.COLLECT_WOOD,)
    )
    with pytest.raises(ValueError, match="drink_water"):
        is_surplus_boundary(state, (SkillId.COLLECT_WOOD,), SkillId.DRINK_WATER)


def test_surplus_trajectory_holds_stone_while_wood_is_short() -> None:
    trajectory = collect_prefix_trajectory(chain=CHAIN, num_episodes=8, seed=51)
    surplus = collect_surplus_trajectory(trajectory, chain=CHAIN, seed=51)
    points = [point for group in surplus.boundaries for point in group]

    assert points
    assert len(surplus.boundaries) == len(trajectory.boundaries)
    # Every surplus count is kept, so one oracle boundary can give several.
    assert len(points) > len({point.episode for point in points})
    for point in points:
        assert point.accepted_skills == (SkillId.COLLECT_WOOD,)
        assert is_surplus_boundary(point.state, point.accepted_skills)
        stone = int(np.asarray(point.state.inventory.stone))
        detours = point.context.history[point.boundary :]
        assert detours
        assert all(attempt.skill_id is SkillId.COLLECT_STONE for attempt in detours)
        assert stone == len(detours)


def test_surplus_trajectory_needs_the_matching_chain() -> None:
    trajectory = collect_prefix_trajectory(
        chain=WOOD_TO_STONE_CHAIN, num_episodes=1, seed=0
    )
    with pytest.raises(ValueError, match="wood_to_stone"):
        collect_surplus_trajectory(trajectory)


def test_surplus_states_separate_counting_from_association() -> None:
    trajectory = collect_prefix_trajectory(chain=CHAIN, num_episodes=8, seed=51)
    surplus = collect_surplus_trajectory(trajectory, chain=CHAIN, seed=51)

    def accuracy(utility) -> float:
        result = evaluate_planner_prefixes(
            SayCanPlanner(
                candidate_skills=CHAIN.skills,
                utility_scorer=utility,
                feasibility_scorer=constant_feasibility,
            ),
            chain=CHAIN,
            num_episodes=8,
            seed=51,
            trajectory=surplus,
        )
        return result["accepted_next_skill_accuracy"]

    # Ranking stone above wood because the objective names it fails every
    # surplus state; reading the shortfall passes every one.
    assert accuracy(_stone_first) == 0.0
    assert accuracy(_shortfall_first) == 1.0
