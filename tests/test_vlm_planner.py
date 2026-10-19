import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
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
    PlannerPromptStyle,
    TransformersVLMBackend,
    VLMRequest,
    VLMUtilityScorer,
    _ranking_output_type,
    build_planner_prompt,
    build_planner_system_prompt,
    generate_all,
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


@dataclass
class FakeBatchedBackend:
    """Backend that records how its requests were grouped into batches."""

    response: str
    batches: list[tuple[VLMRequest, ...]] = field(default_factory=list)

    def generate(self, request: VLMRequest) -> str:
        self.batches.append((request,))
        return self.response

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
        self.batches.append(tuple(requests))
        return [self.response for _ in requests]

    @property
    def requests(self) -> list[VLMRequest]:
        return [request for batch in self.batches for request in batch]


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
    """Pin the baseline prompt that measured 74.5% ungrounded top choices.

    The default style has moved on, but this prompt is the measurement's
    reference point, so its wording must not drift underneath it.
    """

    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(
        objective="Craft a wood pickaxe and collect stone.",
    )
    style = PlannerPromptStyle.BASELINE

    system_prompt = build_planner_system_prompt(style)
    user_prompt = build_planner_prompt(state, context, SKILL_IDS, style=style)
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
    assert "available_now" not in catalog[0]


def test_available_first_prompt_puts_availability_before_the_choice() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(
        objective="Craft a wood pickaxe and collect stone.",
    )

    system_prompt = build_planner_system_prompt()
    user_prompt = build_planner_prompt(state, context, SKILL_IDS)
    catalog_line = next(
        line
        for line in user_prompt.splitlines()
        if line.startswith("Candidate skills: ")
    )
    catalog = json.loads(catalog_line.removeprefix("Candidate skills: "))
    available_line = next(
        line
        for line in user_prompt.splitlines()
        if line.startswith("Skills available now: ")
    )
    available = json.loads(
        available_line.removeprefix("Skills available now: ")
    )

    # The baseline deferred executability to "a separate affordance model",
    # which is exactly the instruction that let an unreachable skill rank first.
    assert "A separate affordance model" in build_planner_system_prompt(
        PlannerPromptStyle.BASELINE
    )
    assert "A separate affordance model" not in system_prompt
    assert "includes being doable now" in system_prompt
    assert "copy the available-now list" in user_prompt

    # Availability is computed for the planner rather than asked of it, and the
    # available candidates lead the list so position agrees with the answer.
    assert all("available_now" in entry for entry in catalog)
    assert available == [
        entry["skill"] for entry in catalog if entry["available_now"]
    ]
    assert [entry["available_now"] for entry in catalog] == sorted(
        (entry["available_now"] for entry in catalog),
        reverse=True,
    )
    blocked = {
        entry["skill"]: entry["blocked_reason"]
        for entry in catalog
        if not entry["available_now"]
    }
    # No table has been placed yet, so crafting fails on its requirement
    # rather than on distance.
    assert blocked[SkillId.CRAFT_WOOD_PICKAXE.value] == "missing_requirement"
    assert SkillId.COLLECT_WOOD.value in available


def test_available_now_is_declared_before_the_selected_skill() -> None:
    # Constrained decoding emits fields in declaration order, so the position
    # of available_now is what forces the readout to precede the choice.
    fields = list(
        _ranking_output_type(SKILL_IDS, True).model_fields  # type: ignore[attr-defined]
    )
    baseline = list(
        _ranking_output_type(SKILL_IDS).model_fields  # type: ignore[attr-defined]
    )

    assert fields.index("available_now") < fields.index("selected_skill")
    assert fields.index("progress_assessment") < fields.index("available_now")
    assert "available_now" not in baseline


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


def test_parse_vlm_response_records_the_claimed_available_skills() -> None:
    ranking = _ranking(SkillId.COLLECT_WOOD, SkillId.DRINK_WATER)
    parsed = parse_vlm_response(
        json.dumps(
            {
                "progress_assessment": "Wood is the unmet prerequisite.",
                "available_now": [
                    SkillId.COLLECT_WOOD.value,
                    SkillId.DRINK_WATER.value,
                ],
                "selected_skill": ranking[0],
                "ranking": ranking,
            }
        ),
        SKILL_IDS,
    )

    assert parsed.valid
    # Comparing this against the true reachable set is what separates a
    # perception failure from an instruction-following failure.
    assert parsed.claimed_available == (
        SkillId.COLLECT_WOOD,
        SkillId.DRINK_WATER,
    )


def test_parse_vlm_response_leaves_claimed_availability_empty_for_baseline() -> None:
    parsed = parse_vlm_response(_response(SkillId.COLLECT_WOOD), SKILL_IDS)

    assert parsed.valid
    assert parsed.claimed_available == ()


@pytest.mark.parametrize(
    ("available_now", "error"),
    [
        ("collect_wood", "available_now_not_an_array"),
        (["chop_tree"], "unknown_available_skill"),
        (["collect_wood", "collect_wood"], "duplicate_available_skill"),
    ],
)
def test_parse_vlm_response_rejects_malformed_availability(
    available_now: Any,
    error: str,
) -> None:
    ranking = _ranking(SkillId.COLLECT_WOOD)
    parsed = parse_vlm_response(
        json.dumps(
            {
                "progress_assessment": "Wood is the unmet prerequisite.",
                "available_now": available_now,
                "selected_skill": ranking[0],
                "ranking": ranking,
            }
        ),
        SKILL_IDS,
    )

    assert not parsed.valid
    assert parsed.error == error
    assert parsed.claimed_available == ()


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


def test_prompt_style_reaches_the_backend_as_a_schema_requirement() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(objective="Collect wood.")
    available_first = FakeBackend(response=_response(SkillId.COLLECT_WOOD))
    baseline = FakeBackend(response=_response(SkillId.COLLECT_WOOD))

    VLMUtilityScorer(available_first).score_all(state, context, SKILL_IDS)
    VLMUtilityScorer(
        baseline,
        prompt_style=PlannerPromptStyle.BASELINE,
    ).score_all(state, context, SKILL_IDS)

    # The prompt asks for available_now, so the constrained schema has to make
    # room for it; otherwise the request is rejected as extra output.
    assert available_first.requests[0].require_available is True
    assert baseline.requests[0].require_available is False


def test_generate_all_falls_back_to_one_request_at_a_time() -> None:
    # Batching is an optimisation, not part of the backend contract, so a
    # backend that only implements generate must keep working.
    single = FakeBackend(response="one")
    batched = FakeBatchedBackend(response="many")
    requests = tuple(
        VLMRequest(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            system_prompt="system",
            user_prompt=f"user {index}",
            candidate_skills=SKILL_IDS,
        )
        for index in range(3)
    )

    assert generate_all(single, requests) == ["one"] * 3
    assert len(single.requests) == 3
    assert generate_all(batched, requests) == ["many"] * 3
    assert batched.batches == [requests]
    assert generate_all(batched, ()) == []


def test_generate_all_rejects_a_batch_of_the_wrong_length() -> None:
    @dataclass
    class ShortBatchBackend:
        def generate(self, request: VLMRequest) -> str:
            return "unused"

        def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
            return ["only one"]

    request = VLMRequest(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        system_prompt="system",
        user_prompt="user",
        candidate_skills=SKILL_IDS,
    )

    with pytest.raises(ValueError, match="1 responses for 2 requests"):
        generate_all(ShortBatchBackend(), (request, request))


def test_batched_scoring_matches_scoring_one_state_at_a_time() -> None:
    environment = make_skill_chain_environment()
    states = [
        environment.reset(jax.random.PRNGKey(seed)).state for seed in (0, 1, 2)
    ]
    contexts = [PlannerContext(objective="Collect stone.")] * len(states)
    batched_backend = FakeBatchedBackend(
        response=_response(SkillId.COLLECT_WOOD)
    )
    sequential_backend = FakeBackend(response=_response(SkillId.COLLECT_WOOD))
    batched_scorer = VLMUtilityScorer(batched_backend)
    sequential_scorer = VLMUtilityScorer(sequential_backend)

    batched = batched_scorer.score_all_batched(states, contexts, SKILL_IDS)
    sequential = [
        sequential_scorer.score_all(state, context, SKILL_IDS)
        for state, context in zip(states, contexts, strict=True)
    ]

    assert batched == sequential
    # One padded forward pass instead of one generation per state, with the
    # same prompts in the same order.
    assert len(batched_backend.batches) == 1
    assert [request.user_prompt for request in batched_backend.requests] == [
        request.user_prompt for request in sequential_backend.requests
    ]
    assert len(batched_scorer.last_responses) == len(states)
    assert batched_scorer.last_response is batched_scorer.last_responses[-1]


def test_batched_scoring_generates_once_for_repeated_boundaries() -> None:
    state = make_skill_chain_environment().reset(jax.random.PRNGKey(0)).state
    context = PlannerContext(objective="Collect stone.")
    backend = FakeBatchedBackend(response=_response(SkillId.COLLECT_WOOD))

    scores = VLMUtilityScorer(backend).score_all_batched(
        (state, state, state),
        (context, context, context),
        SKILL_IDS,
    )

    assert len(scores) == 3
    assert len(backend.requests) == 1


def test_select_many_matches_selecting_one_state_at_a_time() -> None:
    environment = make_skill_chain_environment()
    states = [
        environment.reset(jax.random.PRNGKey(seed)).state for seed in (0, 1)
    ]
    context = PlannerContext(objective="Collect stone.")
    planner = SayCanPlanner(
        utility_scorer=VLMUtilityScorer(
            FakeBatchedBackend(response=_response(SkillId.COLLECT_WOOD))
        )
    )

    assert planner.select_many(states, [context] * len(states)) == [
        planner.select_with_diagnostics(state, context) for state in states
    ]

    with pytest.raises(ValueError, match="equal length"):
        planner.select_many(states, [context])


def test_planner_prefixes_decode_one_batch_per_boundary() -> None:
    backend = FakeBatchedBackend(response=_response(SkillId.COLLECT_WOOD))
    result = evaluate_planner_prefixes(
        SayCanPlanner(utility_scorer=VLMUtilityScorer(backend)),
        num_episodes=3,
        seed=2,
        skill_horizon=32,
    )

    # Episodes advance in lockstep, so a boundary costs one generation call
    # for all surviving episodes rather than one per episode.
    assert len(backend.batches) <= len(WOOD_TO_STONE_CHAIN.reference_skills)
    assert len(backend.batches[0]) == 3
    assert len(backend.requests) == len(result["traces"])
    # Traces still read as per-episode transcripts.
    assert [
        (trace.episode, trace.boundary) for trace in result["traces"]
    ] == sorted(
        (trace.episode, trace.boundary) for trace in result["traces"]
    )


def test_transformers_backend_caches_constrained_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersVLMBackend(VLMPlannerConfig())
    created: list[tuple[Any, Any]] = []

    def fake_output_type(
        candidate_skills: tuple[SkillId, ...],
        require_available: bool = False,
    ) -> object:
        return ("schema", candidate_skills, require_available)

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
    available_first = backend._generator_for(
        SKILL_IDS,
        outlines_model,
        fake_generator_factory,
        require_available=True,
    )

    assert first is second
    # The two prompt styles need different schemas, so they must not share one
    # compiled generator.
    assert available_first is not first
    assert [output_type for _, output_type in created] == [
        ("schema", SKILL_IDS, False),
        ("schema", SKILL_IDS, True),
    ]


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
