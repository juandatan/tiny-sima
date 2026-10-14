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
    parse_vlm_response,
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


def test_parse_vlm_response_derives_utilities_from_complete_ranking() -> None:
    parsed = parse_vlm_response(
        "```json\n"
        + json.dumps(
            {
                "ranking": _ranking(
                    SkillId.COLLECT_WOOD,
                    SkillId.PLACE_TABLE,
                ),
                "rationale": "Wood is required first.",
            }
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
        response=json.dumps(
            {
                "ranking": _ranking(
                    SkillId.COLLECT_WOOD,
                    SkillId.COLLECT_STONE,
                ),
                "rationale": "Wood remains the required prerequisite.",
            }
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
                "ranking": ["collect_wood", "place_table"],
                "rationale": "Wood first.",
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
        '{"ranking":[],"rationale":"No skills remain."}',
        (),
    )

    assert parsed.valid
    assert parsed.utilities == {}
    assert parsed.ranking == ()
    assert parsed.selected_skill is None


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
        response=json.dumps(
            {
                "ranking": _ranking(
                    SkillId.CRAFT_WOOD_PICKAXE,
                    SkillId.COLLECT_WOOD,
                ),
                "rationale": "Collect wood before crafting.",
            }
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


def test_vlm_scores_once_and_symbolic_feasibility_grounds_selection() -> None:
    state = make_skill_chain_environment().reset(
        jax.random.PRNGKey(0)
    ).state
    context = PlannerContext(
        objective="Craft a wood pickaxe and collect stone.",
    )
    backend = FakeBackend(
        response=json.dumps(
            {
                "ranking": _ranking(
                    SkillId.CRAFT_WOOD_PICKAXE,
                    SkillId.COLLECT_WOOD,
                ),
                "rationale": "Craft first.",
            }
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
