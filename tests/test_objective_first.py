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
    SEARCHABLE_SKILLS,
    collect_prefix_trajectory,
    collect_search_trajectory,
    evaluate_planner_prefixes,
    is_search_boundary,
)
from tiny_sima.goals import DEFAULT_REACHABILITY, SKILL_IDS, SkillGoal, SkillId
from tiny_sima.goals.reachability import skill_feasible
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    constant_feasibility,
    constant_utility,
    reachability_feasibility,
)
from tiny_sima.planner.vlm import (
    PlannerPromptStyle,
    VLMRequest,
    VLMUtilityScorer,
    build_planner_prompt,
    build_planner_system_prompt,
    progress_assessment_limit,
)


@dataclass
class FakeBackend:
    response: str
    requests: list[VLMRequest] = field(default_factory=list)

    def generate(self, request: VLMRequest) -> str:
        self.requests.append(request)
        return self.response


def _state(seed: int = 0):
    return chain_environment(WOOD_TO_STONE_CHAIN).reset(jax.random.PRNGKey(seed)).state


def _stone_first(state, context, skill_id: SkillId) -> float:
    # A Say term that already ranks stone first, as the objective-first prompt
    # is meant to, with wood second.
    del state, context
    if skill_id is SkillId.COLLECT_STONE:
        return 1.0
    return 0.25 if skill_id is SkillId.COLLECT_WOOD else 0.0


def test_objective_first_leaves_reachability_to_the_can_term() -> None:
    style = PlannerPromptStyle.OBJECTIVE_FIRST
    system_prompt = build_planner_system_prompt(style)
    user_prompt = build_planner_prompt(
        _state(),
        PlannerContext(objective=WOOD_TO_STONE_CHAIN.objective),
        SKILL_IDS,
        style=style,
    )
    baseline_user = build_planner_prompt(
        _state(),
        PlannerContext(objective=WOOD_TO_STONE_CHAIN.objective),
        SKILL_IDS,
        style=PlannerPromptStyle.BASELINE,
    )

    assert "do not demote a skill because its target is out of view" in system_prompt
    assert "tool already in the inventory" in system_prompt
    assert "only when nothing nearer" not in system_prompt
    assert "Rank such candidates below" in baseline_user
    assert "Rank such candidates below" not in user_prompt
    assert "not how useful the skill is" in user_prompt
    # Same state record and schema as the baseline; only the instructions move.
    assert user_prompt.splitlines()[:4] == baseline_user.splitlines()[:4]
    assert progress_assessment_limit(style) == 96
    assert "at most 96 characters" in user_prompt
    # The mechanics catalog is the shared one, for either chain's skills.
    assert system_prompt.splitlines()[-1] == (
        build_planner_system_prompt(PlannerPromptStyle.BASELINE).splitlines()[-1]
    )
    assert "craft_stone_pickaxe" in build_planner_system_prompt(
        style, WOOD_TO_STONE_PICKAXE_CHAIN.skills
    )


def test_objective_first_uses_the_baseline_schema() -> None:
    ranking = [skill_id.value for skill_id in SKILL_IDS]
    backend = FakeBackend(
        response=json.dumps(
            {
                "progress_assessment": "Need wood.",
                "selected_skill": ranking[0],
                "ranking": ranking,
            }
        )
    )
    scorer = VLMUtilityScorer(
        backend,
        prompt_style=PlannerPromptStyle.OBJECTIVE_FIRST,
    )

    batch = scorer.score_all(
        _state(),
        PlannerContext(objective=WOOD_TO_STONE_CHAIN.objective),
        SKILL_IDS,
    )

    assert batch.valid
    assert backend.requests[0].require_available is False
    assert backend.requests[0].assessment_max_length == 96
    with pytest.raises(ValueError, match="available_first"):
        build_planner_prompt(
            _state(),
            PlannerContext(objective=""),
            SKILL_IDS,
            style=PlannerPromptStyle.OBJECTIVE_FIRST,
            available_shuffle_seed=0,
        )


def test_search_trajectory_holds_only_out_of_view_resource_steps() -> None:
    trajectory = collect_prefix_trajectory(num_episodes=8, seed=51)
    search = collect_search_trajectory(trajectory, seed=51)
    points = [point for group in search.boundaries for point in group]

    assert points
    assert len(search.boundaries) == len(trajectory.boundaries)
    for point in points:
        assert point.accepted_skills == (SkillId.COLLECT_STONE,)
        assert SEARCHABLE_SKILLS.issuperset(point.accepted_skills)
        assert is_search_boundary(point.state, point.accepted_skills)
        assert not bool(
            np.asarray(
                skill_feasible(
                    SkillGoal(SkillId.COLLECT_STONE),
                    point.state,
                    DEFAULT_REACHABILITY,
                )
            )
        )
        # The oracle's prefix, then the detours the closed loop took, if the
        # oracle state was not already a search state.
        detours = point.context.history[point.boundary :]
        assert len(point.context.history) >= point.boundary
        assert all(attempt.skill_id is SkillId.COLLECT_WOOD for attempt in detours)


def test_search_boundary_needs_an_accepted_resource_out_of_view() -> None:
    state = _state()
    # Wood is in view at reset, and a craft is never a search.
    assert not is_search_boundary(state, (SkillId.COLLECT_WOOD,))
    assert not is_search_boundary(state, ())
    assert not is_search_boundary(state, (SkillId.CRAFT_WOOD_PICKAXE,))
    # Stone without a pickaxe fails its requirement rather than its reach.
    assert not is_search_boundary(state, (SkillId.COLLECT_STONE,))


def test_search_trajectory_needs_the_matching_chain() -> None:
    trajectory = collect_prefix_trajectory(num_episodes=1, seed=0)
    with pytest.raises(ValueError, match="wood_to_stone"):
        collect_search_trajectory(trajectory, chain=WOOD_TO_STONE_PICKAXE_CHAIN)


def test_search_states_separate_the_can_terms() -> None:
    trajectory = collect_prefix_trajectory(num_episodes=8, seed=51)
    search = collect_search_trajectory(trajectory, seed=51)

    points = [point for group in search.boundaries for point in group]
    wood_out_of_view = np.mean(
        [
            not bool(
                np.asarray(
                    skill_feasible(
                        SkillGoal(SkillId.COLLECT_WOOD),
                        point.state,
                        DEFAULT_REACHABILITY,
                    )
                )
            )
            for point in points
        ]
    )

    def accuracy(utility, feasibility) -> float:
        result = evaluate_planner_prefixes(
            SayCanPlanner(utility_scorer=utility, feasibility_scorer=feasibility),
            num_episodes=8,
            seed=51,
            trajectory=search,
        )
        return result["accepted_next_skill_accuracy"]

    # Reachability alone never walks off for stone.
    assert accuracy(constant_utility, reachability_feasibility) == 0.0
    # Even a Say term that ranks stone first only ties a reachable wood under
    # the 0.25 out-of-view discount, and the tie goes to catalog order, so
    # stone wins only where the trees are out of view too: the reason the
    # objective-first prompt is also measured with the competence prior.
    assert accuracy(_stone_first, reachability_feasibility) == pytest.approx(
        wood_out_of_view
    )
    assert wood_out_of_view < 0.5
    assert accuracy(_stone_first, constant_feasibility) == 1.0
