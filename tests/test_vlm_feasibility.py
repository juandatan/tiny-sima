"""Tests for the VLM feasibility rater and its study against probe outcomes."""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import numpy as np
import pytest

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.affordance import (
    AffordanceSample,
    collect_affordance_samples,
    fit_affordance_competence_prior,
)
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    LearnedSkillExecutor,
    chain_environment,
)
from tiny_sima.evaluation.vlm_feasibility import (
    PREDICTORS,
    VLMFeasibilityRater,
    affordance_boundary_states,
    check_boundary_states,
    evaluate_vlm_feasibility,
    feasibility_output_type,
    fit_logistic_blend,
    parse_feasibility_response,
)
from tiny_sima.executor.model import (
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
)
from tiny_sima.goals import SKILL_IDS, SkillId
from tiny_sima.planner import CriticAffordanceScorer, SkillOutcome
from tiny_sima.planner.calibration import roc_auc
from tiny_sima.planner.observation import TEXT_ONLY
from tiny_sima.planner.vlm import VLMRequest

TWO_SKILLS = (SkillId.COLLECT_WOOD, SkillId.PLACE_TABLE)


def _response(ratings: Mapping[SkillId, Any], assessment: str = "ok") -> str:
    return json.dumps(
        {
            "assessment": assessment,
            "ratings": {skill.value: value for skill, value in ratings.items()},
        }
    )


class _LookupBackend:
    """Answer each request from a table keyed by its user prompt.

    Batched, so the study's single ``generate_all`` call is what gets counted.
    """

    def __init__(self, responses: Mapping[str, str], default: str = "") -> None:
        self.responses = dict(responses)
        self.default = default
        self.requests: list[VLMRequest] = []
        self.batches = 0

    def generate(self, request: VLMRequest) -> str:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
        self.batches += 1
        self.requests.extend(requests)
        return [
            self.responses.get(request.user_prompt, self.default)
            for request in requests
        ]


def _states(count: int, offset: int = 0) -> list[Any]:
    environment = chain_environment(WOOD_TO_STONE_CHAIN)
    return [
        environment.reset(jax.random.PRNGKey(offset + index)).state
        for index in range(count)
    ]


def _rule_backend(
    rater: VLMFeasibilityRater,
    states: Mapping[tuple[int, int], Any],
    samples: Sequence[AffordanceSample],
    rule: Callable[[AffordanceSample], int],
) -> _LookupBackend:
    """Rate each state's skills by ``rule`` applied to that state's samples."""

    by_key: dict[tuple[int, int], dict[SkillId, int]] = {}
    for sample in samples:
        by_key.setdefault((sample.episode, sample.boundary), {})[sample.skill_id] = (
            rule(sample)
        )
    table = {}
    for key, ratings in by_key.items():
        _, request = rater.request(states[key])
        full = {skill: ratings.get(skill, 0) for skill in rater.candidate_skills}
        table[request.user_prompt] = _response(full)
    return _LookupBackend(table)


def _sample(
    episode: int,
    skill_id: SkillId,
    succeeded: bool,
    *,
    steps: int | None = 3,
    precondition: bool = True,
) -> AffordanceSample:
    return AffordanceSample(
        episode=episode,
        boundary=0,
        skill_id=skill_id,
        critic_value=0.0,
        reachability_gate=1.0,
        reachable=precondition and steps is not None,
        precondition=precondition,
        steps_to_target=steps,
        succeeded=succeeded,
        outcome=SkillOutcome.SUCCESS if succeeded else SkillOutcome.TIMEOUT,
        steps=4,
    )


# ---------------------------------------------------------------------------
# Schema and parsing
# ---------------------------------------------------------------------------


def test_output_schema_requires_one_bounded_rating_per_candidate() -> None:
    output_type = feasibility_output_type(TWO_SKILLS, 40)
    schema = output_type.model_json_schema()

    assert list(schema["properties"]) == ["assessment", "ratings"]
    assert schema["properties"]["assessment"]["maxLength"] == 40
    assert schema["additionalProperties"] is False
    (ratings_schema,) = schema["$defs"].values()
    assert list(ratings_schema["properties"]) == [skill.value for skill in TWO_SKILLS]
    assert sorted(ratings_schema["required"]) == sorted(
        skill.value for skill in TWO_SKILLS
    )
    assert ratings_schema["additionalProperties"] is False
    for entry in ratings_schema["properties"].values():
        assert entry["enum"] == list(range(11))

    output_type.model_validate_json(
        _response({SkillId.COLLECT_WOOD: 0, SkillId.PLACE_TABLE: 10})
    )
    for invalid in (
        _response({SkillId.COLLECT_WOOD: 11, SkillId.PLACE_TABLE: 1}),
        _response({SkillId.COLLECT_WOOD: 3}),
        _response(
            {SkillId.COLLECT_WOOD: 1, SkillId.PLACE_TABLE: 1, SkillId.DRINK_WATER: 1}
        ),
    ):
        with pytest.raises(ValueError):
            output_type.model_validate_json(invalid)
    # Cached per catalog, so the backend compiles one generator per schema.
    assert feasibility_output_type(TWO_SKILLS, 40) is output_type


def test_parse_reads_every_rating_as_a_tenth() -> None:
    parsed = parse_feasibility_response(
        "```json\n"
        + _response({SkillId.COLLECT_WOOD: 7, SkillId.PLACE_TABLE: 0}, "wood near")
        + "\n```",
        TWO_SKILLS,
    )

    assert parsed.valid
    assert parsed.error is None
    assert parsed.assessment == "wood near"
    assert parsed.ratings == {SkillId.COLLECT_WOOD: 7.0, SkillId.PLACE_TABLE: 0.0}
    assert parsed.probability(SkillId.COLLECT_WOOD) == pytest.approx(0.7)
    assert parsed.probability(SkillId.PLACE_TABLE) == 0.0
    assert math.isnan(parsed.probability(SkillId.DRINK_WATER))


@pytest.mark.parametrize(
    "bad_value",
    [11, -1, True, "7", None, float("inf")],
)
def test_parse_fails_one_skill_without_discarding_the_rest(bad_value: Any) -> None:
    text = json.dumps(
        {
            "assessment": "x",
            "ratings": {"collect_wood": 4, "place_table": bad_value, "extra": 3},
        }
    )
    parsed = parse_feasibility_response(text, TWO_SKILLS)

    assert not parsed.valid
    assert parsed.error is not None and "place_table" in parsed.error
    assert parsed.ratings[SkillId.COLLECT_WOOD] == 4.0
    assert math.isnan(parsed.ratings[SkillId.PLACE_TABLE])


def test_parse_accepts_fractional_ratings_and_reports_missing_skills() -> None:
    parsed = parse_feasibility_response(
        json.dumps({"ratings": {"collect_wood": 2.5}}),
        TWO_SKILLS,
    )

    assert parsed.ratings[SkillId.COLLECT_WOOD] == 2.5
    assert math.isnan(parsed.ratings[SkillId.PLACE_TABLE])
    assert parsed.assessment == ""
    assert not parsed.valid


@pytest.mark.parametrize(
    "text",
    ["", "no json here", '{"assessment": "x", "ratings": [1, 2]}', "[1, 2]"],
)
def test_parse_without_a_ratings_object_rates_nothing(text: str) -> None:
    parsed = parse_feasibility_response(text, TWO_SKILLS)

    assert not parsed.valid
    assert set(parsed.ratings) == set(TWO_SKILLS)
    assert all(math.isnan(value) for value in parsed.ratings.values())


# ---------------------------------------------------------------------------
# Rater
# ---------------------------------------------------------------------------


def test_rater_batches_and_caches_identical_states() -> None:
    (state,) = _states(1)
    backend = _LookupBackend(
        {}, default=_response({SkillId.COLLECT_WOOD: 9, SkillId.PLACE_TABLE: 2})
    )
    rater = VLMFeasibilityRater(
        backend,
        candidate_skills=TWO_SKILLS,
        observation=TEXT_ONLY,
    )

    first = rater.rate_all([state, state])
    second = rater.rate_all([state])

    assert backend.batches == 1
    assert len(backend.requests) == 1
    assert [ratings.ratings for ratings in (*first, *second)] == [
        {SkillId.COLLECT_WOOD: 9.0, SkillId.PLACE_TABLE: 2.0}
    ] * 3
    rater.clear_cache()
    rater.rate_all([state])
    assert len(backend.requests) == 2


def test_text_only_request_points_at_the_local_map() -> None:
    (state,) = _states(1)
    rater = VLMFeasibilityRater(
        _LookupBackend({}),
        candidate_skills=TWO_SKILLS,
        observation=TEXT_ONLY,
        skill_horizon=32,
    )
    _, request = rater.request(state)

    assert request.image is None
    assert "screenshot" not in request.system_prompt + request.user_prompt
    assert "local_map" in request.user_prompt
    assert "within 32 environment steps" in request.system_prompt
    assert "Known game mechanics" in request.system_prompt
    assert '"place_table"' in request.system_prompt
    # Only the offered skills are described.
    assert '"drink_water"' not in request.system_prompt
    assert request.output_type is feasibility_output_type(TWO_SKILLS, 96)
    assert '"world"' in request.user_prompt
    _, without_world = VLMFeasibilityRater(
        _LookupBackend({}),
        candidate_skills=TWO_SKILLS,
        observation=TEXT_ONLY,
        include_world=False,
    ).request(state)
    assert '"world"' not in without_world.user_prompt
    assert '"inventory"' in without_world.user_prompt


def test_image_request_sends_the_screenshot_and_keys_on_it() -> None:
    first, second = _states(2)
    rater = VLMFeasibilityRater(_LookupBackend({}), candidate_skills=TWO_SKILLS)
    first_key, request = rater.request(first)
    second_key, _ = rater.request(second)

    assert request.image is not None and request.image.ndim == 3
    assert "screenshot" in request.user_prompt
    assert "Text observation" not in request.user_prompt
    assert first_key != second_key
    assert rater.request(first)[0] == first_key


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------


def test_blend_recovers_the_informative_input() -> None:
    rng = np.random.default_rng(0)
    informative = rng.uniform(0.05, 0.95, size=4000)
    noise = rng.uniform(0.05, 0.95, size=4000)
    labels = (rng.uniform(size=4000) < informative).astype(np.float64)

    as_prior = fit_logistic_blend(informative, noise, labels)
    as_vlm = fit_logistic_blend(noise, informative, labels)

    # The labels are drawn at exactly the informative probability, so its
    # weight is one, the noise input's zero and the intercept zero.
    assert as_prior.prior_weight == pytest.approx(1.0, abs=0.15)
    assert abs(as_prior.vlm_weight) < 0.15
    assert abs(as_prior.intercept) < 0.15
    assert as_vlm.vlm_weight == pytest.approx(as_prior.prior_weight, abs=1e-6)
    blended = as_prior.apply(informative, noise)
    assert np.max(np.abs(blended - informative)) < 0.1


def test_blend_stays_finite_on_separable_or_constant_labels() -> None:
    scores = np.asarray([0.0, 0.0, 1.0, 1.0])
    separable = fit_logistic_blend(scores, scores, scores)
    constant = fit_logistic_blend(scores, scores, np.ones(4))

    for blend in (separable, constant):
        assert all(
            math.isfinite(value)
            for value in (blend.prior_weight, blend.vlm_weight, blend.intercept)
        )
    assert separable.apply(np.ones(1), np.ones(1))[0] > 0.9
    assert np.all(constant.apply(scores, scores) > 0.9)
    with pytest.raises(ValueError):
        fit_logistic_blend(scores, scores, scores, clip=0.5)


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def _synthetic_split(
    episodes: range,
) -> tuple[list[AffordanceSample], dict[tuple[int, int], Any]]:
    """Two skills per state; wood succeeds on even episodes, the table never."""

    states = dict(
        zip(
            ((episode, 0) for episode in episodes),
            _states(len(episodes), offset=episodes.start),
            strict=True,
        )
    )
    samples = [
        sample
        for episode in episodes
        for sample in (
            _sample(episode, SkillId.COLLECT_WOOD, episode % 2 == 0),
            _sample(episode, SkillId.PLACE_TABLE, False, steps=None),
            # Dropped by the precondition filter, but rated all the same.
            _sample(episode, SkillId.PLACE_TABLE, False, precondition=False),
        )
    ]
    return samples, states


def test_study_scores_an_oracle_rater_perfectly() -> None:
    fit_samples, fit_states = _synthetic_split(range(0, 6))
    test_samples, test_states = _synthetic_split(range(6, 12))
    prior = fit_affordance_competence_prior(fit_samples)
    rater = VLMFeasibilityRater(
        _LookupBackend({}),
        candidate_skills=TWO_SKILLS,
        observation=TEXT_ONLY,
    )
    rater.backend = _rule_backend(
        rater,
        {**fit_states, **test_states},
        [*fit_samples, *test_samples],
        lambda sample: 10 if sample.succeeded else 0,
    )

    study = evaluate_vlm_feasibility(
        rater, prior, fit_samples, fit_states, test_samples, test_states
    )

    assert isinstance(rater.backend, _LookupBackend)
    assert rater.backend.batches == 1
    assert study.num_states == 12
    assert study.num_invalid_responses == 0
    assert study.num_missing_ratings == 0
    assert study.num_test_samples == 12
    assert study.num_fit_samples == 12
    assert set(study.overall) == set(PREDICTORS)
    assert study.overall["vlm"]["auc"] == 1.0
    assert study.overall["vlm"]["ece"] == 0.0
    # The prior cannot tell an even episode from an odd one.
    assert study.overall["prior"]["auc"] < 1.0
    # The product keeps the VLM's ordering, and the blend learns to trust it.
    assert study.overall["product"]["auc"] == 1.0
    assert study.overall["blend"]["auc"] == 1.0
    assert study.blend.vlm_weight > abs(study.blend.prior_weight)

    wood = study.per_skill[SkillId.COLLECT_WOOD.value]
    table = study.per_skill[SkillId.PLACE_TABLE.value]
    assert wood["num_samples"] == 6
    assert wood["success_rate"] == 0.5
    assert wood["mean_rating"] == 5.0
    assert wood["vlm"]["auc"] == 1.0
    assert table["success_rate"] == 0.0
    assert table["mean_rating"] == 0.0
    assert math.isnan(table["vlm"]["auc"])
    # Every probe, the precondition failures included.
    assert study.vlm_all_samples["num_samples"] == 18.0

    payload = study.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["per_skill"]["place_table"]["vlm"]["auc"] is None
    assert len(payload["test_records"]) == 12
    assert len(payload["state_records"]) == 12
    assert payload["observation"]["image"] is False


def test_study_counts_invalid_responses_and_uses_the_fallback() -> None:
    fit_samples, fit_states = _synthetic_split(range(0, 4))
    test_samples, test_states = _synthetic_split(range(4, 8))
    prior = fit_affordance_competence_prior(fit_samples)
    rater = VLMFeasibilityRater(
        _LookupBackend({}),
        candidate_skills=TWO_SKILLS,
        observation=TEXT_ONLY,
    )
    backend = _rule_backend(
        rater,
        {**fit_states, **test_states},
        [*fit_samples, *test_samples],
        lambda sample: 10 if sample.succeeded else 0,
    )
    _, broken = rater.request(test_states[(4, 0)])
    backend.responses[broken.user_prompt] = "I cannot rate this."
    rater.backend = backend

    study = evaluate_vlm_feasibility(
        rater,
        prior,
        fit_samples,
        fit_states,
        test_samples,
        test_states,
        missing_probability=0.3,
    )

    assert study.num_invalid_responses == 1
    # Episode 4's two precondition-held samples took the fallback.
    assert study.num_missing_ratings == 2
    records = [record for record in study.test_records if record["episode"] == 4]
    assert all(record["vlm"] == pytest.approx(0.3) for record in records)
    assert all(math.isnan(record["rating"]) for record in records)
    assert study.per_skill["collect_wood"]["num_missing_ratings"] == 1
    json.dumps(study.to_dict(), allow_nan=False)


def test_study_rejects_unpaired_samples_and_unoffered_skills() -> None:
    samples, states = _synthetic_split(range(0, 2))
    prior = fit_affordance_competence_prior(samples)
    rater = VLMFeasibilityRater(
        _LookupBackend({}), candidate_skills=TWO_SKILLS, observation=TEXT_ONLY
    )

    with pytest.raises(ValueError, match="No test state"):
        evaluate_vlm_feasibility(rater, prior, samples, states, samples, {})
    narrow = VLMFeasibilityRater(
        _LookupBackend({}),
        candidate_skills=(SkillId.COLLECT_WOOD,),
        observation=TEXT_ONLY,
    )
    with pytest.raises(ValueError, match="not a candidate"):
        evaluate_vlm_feasibility(narrow, prior, samples, states, samples, states)


# ---------------------------------------------------------------------------
# Real probes
# ---------------------------------------------------------------------------


def _random_executor() -> LearnedSkillExecutor:
    config = SkillPPOConfig(
        total_timesteps=12,
        num_envs=6,
        num_steps=2,
        num_minibatches=1,
        use_optimistic_resets=False,
        hidden_size=16,
        skill_embedding_size=4,
        monitor_gpu=False,
    ).validate()
    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        environment.bundle.spec.observation_shape,
    )
    return LearnedSkillExecutor(params=params, config=config, model=model)


def test_replayed_states_pair_with_real_probes() -> None:
    executor = _random_executor()
    scorer = CriticAffordanceScorer(
        params=executor.params,
        model=executor.model,
        hidden_size=executor.config.hidden_size,
        observe=make_skill_chain_environment().bundle.env.get_obs,
    )
    samples = collect_affordance_samples(
        scorer, executor, num_episodes=1, seed=0, skill_horizon=4
    )
    states = affordance_boundary_states(num_episodes=1, seed=0, skill_horizon=4)

    assert {(sample.episode, sample.boundary) for sample in samples} == set(states)
    check_boundary_states(samples, states)
    # Another seed's states do not pass for these samples.
    other = affordance_boundary_states(num_episodes=1, seed=5, skill_horizon=4)
    with pytest.raises(ValueError, match="does not match"):
        check_boundary_states(samples, {(0, 0): other[(0, 0)]})
    with pytest.raises(ValueError, match="No state"):
        check_boundary_states(samples, {})

    # A rater that reads the symbolic precondition scores exactly as the
    # precondition does, on every probe.
    rater = VLMFeasibilityRater(_LookupBackend({}), observation=TEXT_ONLY)
    rater.backend = _rule_backend(
        rater, states, samples, lambda sample: 10 * sample.precondition
    )
    study = evaluate_vlm_feasibility(
        rater,
        fit_affordance_competence_prior(samples),
        samples,
        states,
        samples,
        states,
    )

    labels = np.asarray([sample.succeeded for sample in samples], dtype=np.float64)
    preconditions = np.asarray(
        [sample.precondition for sample in samples], dtype=np.float64
    )
    expected = roc_auc(preconditions, labels)
    observed = study.vlm_all_samples["auc"]
    assert (math.isnan(expected) and math.isnan(observed)) or observed == expected
    assert study.num_invalid_responses == 0
    assert study.num_states == 2 * len(states)
    json.dumps(study.to_dict(), allow_nan=False)
