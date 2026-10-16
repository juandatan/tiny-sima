import json
from dataclasses import dataclass, field
from typing import Any

import jax
import pytest

import tiny_sima.planner.vlm as vlm_module
from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    ChainTermination,
    ScriptedSkillExecutor,
    evaluate_skill_chain,
)
from tiny_sima.evaluation.planner import evaluate_planner_prefixes
from tiny_sima.evaluation.vlm_smoke import run_vlm_smoke_test
from tiny_sima.goals import SKILL_IDS, SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillAttempt,
    SkillOutcome,
    constant_feasibility,
)
from tiny_sima.planner.vlm import (
    TransformersVLMBackend,
    VLMRequest,
    VLMUtilityScorer,
    build_planner_prompt,
    build_planner_system_prompt,
    parse_vlm_response,
    summarize_state,
)


@dataclass
class FakeBackend:
    response: str
    requests: list[VLMRequest] = field(default_factory=list)

    def generate(self, request: VLMRequest) -> str:
        self.requests.append(request)
        return self.response


def _ranking(*leading_skills: SkillId) -> list[str]:
    remaining = [
        skill_id
        for skill_id in SKILL_IDS
        if skill_id not in leading_skills
    ]
    return [
        skill_id.value
        for skill_id in (*leading_skills, *remaining)
    ]


def _response(
    *leading_skills: SkillId,
    progress_assessment: str = "Ready for the next skill.",
) -> str:
    ranking = _ranking(*leading_skills)
    return json.dumps(
        {
            "progress_assessment": progress_assessment,
            "selected_skill": ranking[0],
            "ranking": ranking,
        }
    )


def test_planner_prompt_requests_state_dependent_next_skill() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(
        objective="Craft a wood pickaxe and collect stone.",
    )

    system_prompt = build_planner_system_prompt()
    user_prompt = build_planner_prompt(state, context, SKILL_IDS)
    state_summary = json.loads(summarize_state(state))
    catalog_line = next(
        line
        for line in user_prompt.splitlines()
        if line.startswith("Candidate skills: ")
    )
    catalog = json.loads(catalog_line.removeprefix("Candidate skills: "))

    assert "next requested skill" in system_prompt
    assert "recomputed after every completed or failed skill" in system_prompt
    assert "game-mechanics catalog" in system_prompt
    assert "Do not treat current infeasibility" not in system_prompt
    assert "progress already made" in user_prompt
    assert "at most 96 characters" in user_prompt
    assert state_summary["world"] == {
        "crafting_table_reachable": False,
        "crafting_table_steps": None,
        "placement_space_reachable": True,
        "placement_space_steps": 1,
        "stone_reachable": True,
        "stone_steps": 2,
        "tree_reachable": True,
        "tree_steps": 3,
        "water_reachable": True,
        "water_steps": 2,
    }
    assert next(
        item["current_value"]
        for item in catalog
        if item["skill"] == SkillId.CRAFT_WOOD_PICKAXE.value
    ) == 0
    mechanics_json = system_prompt.split("Known game mechanics: ", maxsplit=1)[1]
    mechanics_catalog = json.loads(mechanics_json)
    pickaxe = next(
        item
        for item in mechanics_catalog
        if item["skill"] == SkillId.CRAFT_WOOD_PICKAXE.value
    )
    assert {
        requirement["field"]
        for requirement in pickaxe["requires"]
    } == {"inventory.wood", "world.crafting_table_reachable"}
    assert "description" not in catalog[0]


def test_planner_prompt_exposes_updated_skill_progress() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    updated_state = state.replace(
        inventory=state.inventory.replace(
            wood_pickaxe=state.inventory.wood_pickaxe + 1,
        )
    )
    prompt = build_planner_prompt(
        updated_state,
        PlannerContext(objective="Collect stone with a wood pickaxe."),
        SKILL_IDS,
    )
    catalog_line = next(
        line
        for line in prompt.splitlines()
        if line.startswith("Candidate skills: ")
    )
    catalog = json.loads(catalog_line.removeprefix("Candidate skills: "))

    assert next(
        item["current_value"]
        for item in catalog
        if item["skill"] == SkillId.CRAFT_WOOD_PICKAXE.value
    ) == 1


def test_parse_vlm_response_derives_utilities_from_complete_ranking() -> None:
    parsed = parse_vlm_response(
        "```json\n"
        + _response(
            SkillId.COLLECT_WOOD,
            SkillId.PLACE_TABLE,
            progress_assessment="Wood is the unmet prerequisite.",
        )
        + "\n```",
        SKILL_IDS,
    )

    assert parsed.valid
    assert parsed.selected_skill is SkillId.COLLECT_WOOD
    assert parsed.utilities[SkillId.COLLECT_WOOD] == 1.0
    assert parsed.utilities[SkillId.PLACE_TABLE] == 0.25
    assert set(parsed.utilities) == set(SKILL_IDS)


def test_rank_utilities_keep_the_top_skill_after_one_failure() -> None:
    backend = FakeBackend(
        response=_response(
            SkillId.COLLECT_WOOD,
            SkillId.COLLECT_STONE,
            progress_assessment="Wood remains the required prerequisite.",
        )
    )
    planner = SayCanPlanner(
        utility_scorer=VLMUtilityScorer(backend),
        feasibility_scorer=constant_feasibility,
    )
    decision = planner.select(
        make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state,
        PlannerContext(
            objective="Collect wood.",
            history=(
                SkillAttempt(
                    skill_id=SkillId.COLLECT_WOOD,
                    outcome=SkillOutcome.TIMEOUT,
                    steps=64,
                    native_return=0.0,
                ),
            ),
        ),
    )

    assert decision is not None
    assert decision.skill_id is SkillId.COLLECT_WOOD


def test_parse_vlm_response_rejects_incomplete_rankings() -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "progress_assessment": "Wood is still required.",
                "selected_skill": "collect_wood",
                "ranking": ["collect_wood", "place_table"],
            }
        ),
        SKILL_IDS,
    )

    assert not parsed.valid
    assert parsed.utilities == {}
    assert parsed.error == "ranking_wrong_length"


def test_parse_vlm_response_fails_closed_for_malformed_output() -> None:
    parsed = parse_vlm_response("collect wood", SKILL_IDS)

    assert not parsed.valid
    assert parsed.utilities == {}
    assert parsed.selected_skill is None


def test_parse_vlm_response_accepts_an_empty_candidate_catalog() -> None:
    parsed = parse_vlm_response(
        (
            '{"progress_assessment":"No skills remain.",'
            '"selected_skill":null,"ranking":[]}'
        ),
        (),
    )

    assert parsed.valid
    assert parsed.utilities == {}
    assert parsed.ranking == ()
    assert parsed.selected_skill is None


def test_parse_vlm_response_rejects_inconsistent_selection_and_ranking() -> None:
    parsed = parse_vlm_response(
        json.dumps(
            {
                "progress_assessment": "Stone is the next unmet step.",
                "selected_skill": "collect_stone",
                "ranking": _ranking(SkillId.COLLECT_WOOD),
            }
        ),
        SKILL_IDS,
    )

    assert not parsed.valid
    assert parsed.error == "selected_skill_ranking_mismatch"


def test_empty_candidate_catalog_returns_no_selection_without_generation() -> None:
    backend = FakeBackend(response="unused")
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    scorer = VLMUtilityScorer(backend)
    planner = SayCanPlanner(
        utility_scorer=scorer,
        candidate_skills=(),
    )

    selection = planner.select_with_diagnostics(
        state,
        PlannerContext(objective="Nothing remains."),
    )

    assert selection.decision is None
    assert selection.utility_error is None
    assert backend.requests == []
    assert scorer.last_response is not None
    assert scorer.last_response.valid
    assert scorer.last_response.selected_skill is None


def test_transformers_backend_caches_constrained_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersVLMBackend(VLMPlannerConfig())
    created: list[tuple[Any, Any]] = []

    def fake_output_type(candidate_skills: tuple[SkillId, ...]) -> object:
        return ("schema", candidate_skills)

    def fake_generator_factory(model: Any, output_type: Any) -> object:
        created.append((model, output_type))
        return object()

    monkeypatch.setattr(vlm_module, "_ranking_output_type", fake_output_type)
    outlines_model = object()

    first = backend._generator_for(
        SKILL_IDS,
        outlines_model,
        fake_generator_factory,
    )
    second = backend._generator_for(
        SKILL_IDS,
        outlines_model,
        fake_generator_factory,
    )

    assert first is second
    assert len(created) == 1


def test_vlm_smoke_test_validates_one_generation_and_grounded_selection() -> None:
    backend = FakeBackend(
        response=_response(
            SkillId.CRAFT_WOOD_PICKAXE,
            SkillId.COLLECT_WOOD,
            progress_assessment="Wood must be collected before crafting.",
        )
    )

    result = run_vlm_smoke_test(
        VLMPlannerConfig(model_id="fake/model"),
        backend=backend,
        seed=3,
    )

    assert len(backend.requests) == 1
    assert result.utility_selected_skill == "craft_wood_pickaxe"
    assert result.grounded_selected_skill == "collect_wood"
    assert result.grounded_feasibility == 1.0
    assert result.ranking[0] == "craft_wood_pickaxe"
    assert result.progress_assessment == "Wood must be collected before crafting."


def test_vlm_scores_once_and_symbolic_feasibility_grounds_selection() -> None:
    state = make_skill_chain_environment().reset(
        jax.random.PRNGKey(0)
    ).state
    context = PlannerContext(
        objective="Craft a wood pickaxe and collect stone.",
    )
    backend = FakeBackend(
        response=_response(
            SkillId.CRAFT_WOOD_PICKAXE,
            SkillId.COLLECT_WOOD,
            progress_assessment="The pickaxe is useful after collecting wood.",
        )
    )
    scorer = VLMUtilityScorer(backend)

    utility_only = SayCanPlanner(
        utility_scorer=scorer,
        feasibility_scorer=constant_feasibility,
    ).select(state, context)
    grounded = SayCanPlanner(utility_scorer=scorer).select(state, context)

    assert utility_only is not None
    assert utility_only.skill_id is SkillId.CRAFT_WOOD_PICKAXE
    assert grounded is not None
    assert grounded.skill_id is SkillId.COLLECT_WOOD
    assert len(backend.requests) == 1
    assert "reference_skills" not in backend.requests[0].prompt
    assert "high-level planner" in backend.requests[0].system_prompt
    assert "Objective:" in backend.requests[0].user_prompt


def test_oracle_planner_matches_every_controlled_prefix() -> None:
    result = evaluate_planner_prefixes(
        SayCanPlanner(
            utility_scorer=SequentialUtilityScorer(
                (
                    SkillId.COLLECT_WOOD,
                    SkillId.COLLECT_WOOD,
                    SkillId.PLACE_TABLE,
                    SkillId.COLLECT_WOOD,
                    SkillId.CRAFT_WOOD_PICKAXE,
                    SkillId.COLLECT_STONE,
                )
            )
        ),
        num_episodes=1,
        seed=2,
        skill_horizon=32,
    )

    assert result["oracle_progress_failures"] == 0
    assert result["accepted_next_skill_accuracy"] == 1.0
    assert result["grounded_selection_rate"] == 1.0


def test_collect_first_is_accepted_at_the_reordering_boundary() -> None:
    result = evaluate_planner_prefixes(
        SayCanPlanner(
            utility_scorer=lambda state, _context, skill_id: float(
                skill_id
                in (
                    WOOD_TO_STONE_CHAIN.valid_next_skills(state)
                    if WOOD_TO_STONE_CHAIN.valid_next_skills is not None
                    else ()
                )
            )
        ),
        num_episodes=1,
        seed=2,
        skill_horizon=32,
    )

    third_boundary = result["traces"][2]
    assert SkillId.COLLECT_WOOD in third_boundary.accepted_skills
    assert SkillId.PLACE_TABLE in third_boundary.accepted_skills
    assert third_boundary.matched
    assert result["accepted_next_skill_accuracy"] == 1.0


def test_invalid_vlm_output_has_distinct_chain_termination() -> None:
    scorer = VLMUtilityScorer(FakeBackend("not json"))
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(utility_scorer=scorer),
        num_episodes=1,
        seed=0,
    )

    assert result["termination_counts"][
        ChainTermination.INVALID_PLANNER_OUTPUT.value
    ] == 1
    assert result["planner_error_counts"] == {"malformed_json": 1}
    assert result["episodes"][0].planner_raw_response == "not json"
