"""Ask the frozen VLM how likely each skill is to succeed, and score its answer.

Feasibility has so far been measured from the executor's side: the competence
prior tabulates probe outcomes by ``(skill, reachable, distance band)``, and
the critic was tested and retired. This module asks the opposite question --
whether the VLM itself can judge when the low-level controller will complete a
skill -- and scores its judgement against the probe outcomes recorded at the
same boundary states by :func:`collect_affordance_samples`.

One request per state rates every candidate skill from 0 to 10, read as a
probability ``r / 10``. The backend exposes no logprobs, so the integer rating
is the whole signal. The study compares four predictors on a held-out seed:
the competence prior, the VLM, their product, and a logistic blend of the two
fitted on the seed the prior was fitted on.

:func:`collect_affordance_samples` does not return its states, so
:func:`affordance_boundary_states` replays the same oracle walk -- the same
random-key splits, without the probes -- and :func:`check_boundary_states`
confirms the replay against the features the samples recorded.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from hashlib import sha256
from typing import Any, Literal

import jax
import numpy as np

from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.craftax import render_state
from tiny_sima.evaluation.affordance import AffordanceSample
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_CHAIN,
    ScriptedSkillExecutor,
    SkillChainSpec,
    chain_environment,
)
from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_precondition,
    skill_reachability_report,
)
from tiny_sima.planner.calibration import expected_calibration_error, roc_auc
from tiny_sima.planner.competence import CompetencePrior
from tiny_sima.planner.observation import (
    ObservationConfig,
    observation_block,
    observation_system_text,
)
from tiny_sima.planner.saycan import SkillOutcome
from tiny_sima.planner.vlm import (
    VisionLanguageBackend,
    VLMRequest,
    _json_object,
    _mechanics_line,
    _without_screenshot,
    generate_all,
    summarize_state,
)

MAX_RATING = 10
# Long enough for one sentence about what is held and what is in reach, short
# enough that the ratings, not the prose, take the token budget.
DEFAULT_ASSESSMENT_MAX_LENGTH = 96
PREDICTORS: tuple[str, ...] = ("prior", "vlm", "product", "blend")

# Keyed by ``(episode, boundary)``, the pair every affordance sample carries.
BoundaryKey = tuple[int, int]


# ---------------------------------------------------------------------------
# Boundary states
# ---------------------------------------------------------------------------


def affordance_boundary_states(
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 8,
    seed: int = 0,
    skill_horizon: int = 64,
    distractor_count: int = 3,
) -> dict[BoundaryKey, Any]:
    """Replay the oracle walk of :func:`collect_affordance_samples`.

    The arguments must match the ones the samples were collected with. The
    random keys are split in the same order, the probe key included, so the
    oracle advances through the same states; only the probes are skipped. Note
    that :func:`tiny_sima.evaluation.collect_prefix_trajectory` splits its
    keys per episode and therefore reaches different states for the same
    seed. Run :func:`check_boundary_states` on the result before using it.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    environment = chain_environment(chain, distractor_count=distractor_count)
    oracle = ScriptedSkillExecutor()

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, num_episodes)
    states = [environment.reset(reset_key).state for reset_key in reset_keys]
    alive = [True] * num_episodes
    boundary_states: dict[BoundaryKey, Any] = {}

    for boundary, reference_skill in enumerate(chain.reference_skills):
        if not any(alive):
            break
        for index in range(num_episodes):
            if alive[index]:
                boundary_states[(index, boundary)] = states[index]
        # The probe key is drawn and discarded so the advance keys line up.
        rng, _probe_rng = jax.random.split(rng)
        for index in range(num_episodes):
            if not alive[index]:
                continue
            rng, advance_key = jax.random.split(rng)
            result = oracle.execute(
                environment,
                states[index],
                reference_skill,
                advance_key,
                horizon=skill_horizon,
            )
            if result.outcome is not SkillOutcome.SUCCESS:
                alive[index] = False
                continue
            states[index] = result.state

    return boundary_states


def check_boundary_states(
    samples: Sequence[AffordanceSample],
    states: Mapping[BoundaryKey, Any],
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
) -> None:
    """Raise unless every sample's recorded features match its paired state.

    Each sample recorded its skill's precondition and steps to target; both
    are recomputed from the state paired with it. A replay with different
    arguments, or one that drifted from the collection walk, fails here
    rather than silently rating the wrong states. ``reachability`` must be
    the configuration the samples were collected with.
    """

    by_key: dict[BoundaryKey, list[AffordanceSample]] = {}
    for sample in samples:
        by_key.setdefault((sample.episode, sample.boundary), []).append(sample)
    missing = sorted(key for key in by_key if key not in states)
    if missing:
        raise ValueError(f"No state for samples at (episode, boundary) {missing}.")
    for key, group in sorted(by_key.items()):
        state = states[key]
        skills = tuple(dict.fromkeys(sample.skill_id for sample in group))
        reports = skill_reachability_report(state, reachability, skills)
        for sample in group:
            precondition = bool(
                np.asarray(skill_precondition(SkillGoal(sample.skill_id), state))
            )
            if (
                precondition != sample.precondition
                or reports[sample.skill_id] != sample.steps_to_target
            ):
                raise ValueError(
                    f"State at (episode, boundary) {key} does not match the "
                    f"{sample.skill_id.value} sample: precondition "
                    f"{precondition} against {sample.precondition}, steps "
                    f"{reports[sample.skill_id]} against "
                    f"{sample.steps_to_target}."
                )


# ---------------------------------------------------------------------------
# Rater
# ---------------------------------------------------------------------------


_Rating = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@lru_cache(maxsize=16)
def feasibility_output_type(
    candidate_skills: tuple[SkillId, ...],
    assessment_max_length: int = DEFAULT_ASSESSMENT_MAX_LENGTH,
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines.

    ``assessment`` comes first so the model states the situation before it
    rates anything; see ``_ranking_output_type`` in the VLM module. Ratings
    are a nested object with one required field per candidate, in catalog
    order, so the schema forces a rating for every skill and no other. The
    rating is an integer enum rather than a bounded integer, because an enum
    is the constraint every JSON-schema-to-grammar compiler supports.
    """

    if not candidate_skills:
        raise ValueError("candidate_skills cannot be empty.")
    try:
        from pydantic import (  # pyright: ignore[reportMissingImports]
            BaseModel,
            ConfigDict,
            Field,
            create_model,
        )
    except ImportError as error:
        raise ImportError(
            'Install the VLM dependencies with pip install -e ".[vlm]".'
        ) from error

    rating_fields: dict[str, Any] = {
        skill_id.value: (_Rating, ...) for skill_id in candidate_skills
    }
    ratings_type: type[BaseModel] = create_model(
        "SkillRatings_" + "_".join(skill_id.name for skill_id in candidate_skills),
        __config__=ConfigDict(extra="forbid"),
        **rating_fields,
    )
    return create_model(
        "FeasibilityOutput",
        __config__=ConfigDict(extra="forbid"),
        assessment=(str, Field(max_length=assessment_max_length)),
        ratings=(ratings_type, ...),
    )


_FEASIBILITY_INSTRUCTIONS = (
    "You judge skills for a Craftax survival agent. You do not choose what to do next.",
    "Each skill is carried out by a learned low-level controller that walks "
    "to the skill's target and interacts with it. The skill succeeds when its "
    "effect happens within {skill_horizon} environment steps, and fails when "
    "the steps run out or its requirements are not met.",
    "Rate every candidate skill from 0 to 10 for how likely the controller "
    "is to complete it from the current state: 0 means it cannot succeed, 10 "
    "means it is certain to. Rate feasibility, not usefulness: a skill that "
    "does nothing for any objective can still deserve a 10.",
    "Weigh the requirements in the mechanics catalog, what the screenshot "
    "shows, and how far the player would have to walk. The catalog's "
    "reachability requirements describe targets the controller can use "
    "without searching: a target out of view is not impossible, but the "
    "controller must find it within the same step budget.",
    "You may rely on the supplied game-mechanics catalog as authoritative "
    "knowledge of what each skill needs and produces.",
    "Return only JSON with assessment (at most {assessment_max_length} "
    "characters) and ratings, an object with one integer from 0 to 10 per "
    "candidate skill.",
)


def build_feasibility_system_prompt(
    candidate_skills: Sequence[SkillId] = SKILL_IDS,
    *,
    skill_horizon: int = 64,
    assessment_max_length: int = DEFAULT_ASSESSMENT_MAX_LENGTH,
    observation: ObservationConfig | None = None,
) -> str:
    """Return the rater's role, the mechanics catalog and the output format."""

    prompt = "\n".join(
        (
            *(
                line.format(
                    skill_horizon=skill_horizon,
                    assessment_max_length=assessment_max_length,
                )
                for line in _FEASIBILITY_INSTRUCTIONS
            ),
            _mechanics_line(candidate_skills),
        )
    )
    if observation is not None:
        observation_text = observation_system_text(observation)
        if observation_text:
            prompt = f"{prompt}\n{observation_text}"
    return prompt


def build_feasibility_prompt(
    state: Any,
    candidate_skills: Sequence[SkillId],
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
    *,
    assessment_max_length: int = DEFAULT_ASSESSMENT_MAX_LENGTH,
    observation: ObservationConfig | None = None,
    include_world: bool = True,
) -> str:
    """Build the per-state rating prompt.

    The structured state is the per-turn planner's: inventory, survival and
    the ``world`` reachability readout. ``include_world=False`` drops the
    readout, which is the ablation for whether the VLM judges feasibility
    from the scene or merely reads the distances it is handed -- the same
    distances the competence prior conditions on.
    """

    summary = summarize_state(state, reachability)
    if not include_world:
        values = json.loads(summary)
        values.pop("world", None)
        summary = json.dumps(values, sort_keys=True)
    lines = [f"Current structured state: {summary}"]
    if observation is not None and observation.has_text:
        lines.append(
            "Text observation: " + json.dumps(observation_block(state, observation))
        )
    lines.extend(
        (
            "Candidate skills: "
            + json.dumps([skill_id.value for skill_id in candidate_skills]),
            "Inspect the screenshot and structured state. First give an "
            f"assessment of at most {assessment_max_length} characters of what "
            "the player holds and which targets are in reach. Then rate every "
            "candidate skill exactly once from 0 to 10.",
        )
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class FeasibilityRatings:
    """One parsed rating response.

    ``ratings`` holds every candidate; a rating that was missing or invalid is
    NaN, and the response is then not ``valid``. Ratings are kept on the 0-10
    scale the model answered on; :meth:`probability` maps them to ``r / 10``.
    """

    ratings: dict[SkillId, float]
    assessment: str
    raw_response: str
    valid: bool
    error: str | None = None

    def probability(self, skill_id: SkillId) -> float:
        """Return ``rating / 10``, or NaN when the skill has no valid rating."""

        return self.ratings.get(skill_id, math.nan) / MAX_RATING


def _rating_value(value: Any) -> float | None:
    """Return a usable rating, or None for anything outside ``[0, 10]``.

    A fractional rating is accepted, since an unconstrained backend may give
    one and the probability does not need an integer. Booleans and numeric
    strings are rejected rather than guessed at.
    """

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    rating = float(value)
    if not math.isfinite(rating) or not 0.0 <= rating <= MAX_RATING:
        return None
    return rating


def parse_feasibility_response(
    text: str,
    candidate_skills: Sequence[SkillId],
) -> FeasibilityRatings:
    """Parse one response, failing per skill rather than per response.

    A response with no JSON object, or no ``ratings`` object, rates nothing.
    Otherwise each candidate keeps its rating if valid and gets NaN if not,
    so one malformed entry does not discard the other skills' ratings. Keys
    for skills outside the candidates are ignored.
    """

    skills = tuple(candidate_skills)
    unrated = dict.fromkeys(skills, math.nan)
    payload = _json_object(text)
    if payload is None:
        return FeasibilityRatings(unrated, "", text, False, "no JSON object")
    raw_assessment = payload.get("assessment", "")
    assessment = raw_assessment if isinstance(raw_assessment, str) else ""
    raw_ratings = payload.get("ratings")
    if not isinstance(raw_ratings, dict):
        return FeasibilityRatings(
            unrated, assessment, text, False, "ratings was not an object"
        )
    ratings: dict[SkillId, float] = {}
    failed: list[str] = []
    for skill_id in skills:
        rating = _rating_value(raw_ratings.get(skill_id.value))
        if rating is None:
            failed.append(skill_id.value)
            ratings[skill_id] = math.nan
        else:
            ratings[skill_id] = rating
    return FeasibilityRatings(
        ratings,
        assessment,
        text,
        not failed,
        f"missing or invalid ratings: {failed}" if failed else None,
    )


@dataclass
class VLMFeasibilityRater:
    """Rate every candidate skill's chance of success with one VLM call.

    ``observation`` chooses what the model sees, as for the planner: None
    sends the screenshot alone, and a configuration without ``image`` sends
    the text observation and points the prompts at it instead.
    """

    backend: VisionLanguageBackend
    config: VLMPlannerConfig = VLMPlannerConfig()
    candidate_skills: tuple[SkillId, ...] = SKILL_IDS
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY
    observation: ObservationConfig | None = None
    # The budget the controller is told it has; match the probes' horizon.
    skill_horizon: int = 64
    assessment_max_length: int = DEFAULT_ASSESSMENT_MAX_LENGTH
    include_world: bool = True
    # Raw response text, so a parser change does not invalidate generations.
    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.candidate_skills:
            raise ValueError("candidate_skills cannot be empty.")

    def request(self, state: Any) -> tuple[str, VLMRequest]:
        """Render one state into a request and its response cache key."""

        send_image = self.observation is None or self.observation.image
        image = (
            render_state(
                state,
                self.config.environment_name,
                block_pixel_size=self.config.block_pixel_size,
            )
            if send_image
            else None
        )
        system_prompt = build_feasibility_system_prompt(
            self.candidate_skills,
            skill_horizon=self.skill_horizon,
            assessment_max_length=self.assessment_max_length,
            observation=self.observation,
        )
        user_prompt = build_feasibility_prompt(
            state,
            self.candidate_skills,
            self.reachability,
            assessment_max_length=self.assessment_max_length,
            observation=self.observation,
            include_world=self.include_world,
        )
        if not send_image:
            system_prompt = _without_screenshot(system_prompt)
            user_prompt = _without_screenshot(user_prompt)
        cache_key = sha256(
            system_prompt.encode("utf-8")
            + user_prompt.encode("utf-8")
            + (b"" if image is None else np.ascontiguousarray(image).tobytes())
        ).hexdigest()
        return cache_key, VLMRequest(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            candidate_skills=self.candidate_skills,
            assessment_max_length=self.assessment_max_length,
            output_type=feasibility_output_type(
                self.candidate_skills,
                self.assessment_max_length,
            ),
        )

    def rate_all(self, states: Sequence[Any]) -> list[FeasibilityRatings]:
        """Rate several states with one batched backend invocation.

        Identical states share one generation, within a call and across calls.
        Returned ratings stay aligned with ``states``.
        """

        prepared = [self.request(state) for state in states]
        pending = {key: request for key, request in prepared if key not in self._cache}
        if pending:
            responses = generate_all(self.backend, list(pending.values()))
            self._cache.update(zip(pending, responses, strict=True))
        return [
            parse_feasibility_response(self._cache[key], self.candidate_skills)
            for key, _ in prepared
        ]

    def clear_cache(self) -> None:
        """Discard cached responses."""

        self._cache.clear()


# ---------------------------------------------------------------------------
# Blend
# ---------------------------------------------------------------------------


def _logit(probabilities: np.ndarray, clip: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), clip, 1.0 - clip)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * values))


@dataclass(frozen=True)
class LogisticBlend:
    """``sigmoid(a * logit(prior) + b * logit(vlm) + c)``.

    Inputs are clipped to ``[clip, 1 - clip]`` before the logit, so a rating
    of 0 or 10 is a strong vote rather than an infinite one.
    """

    prior_weight: float
    vlm_weight: float
    intercept: float
    clip: float

    def apply(self, prior: np.ndarray, vlm: np.ndarray) -> np.ndarray:
        """Return the blended probabilities."""

        return _sigmoid(
            self.prior_weight * _logit(prior, self.clip)
            + self.vlm_weight * _logit(vlm, self.clip)
            + self.intercept
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_logistic_blend(
    prior: np.ndarray,
    vlm: np.ndarray,
    labels: np.ndarray,
    *,
    clip: float = 0.01,
    l2: float = 1e-3,
    max_iterations: int = 100,
) -> LogisticBlend:
    """Fit the blend by Newton's method on the mean log loss.

    ``l2`` penalises the two weights, not the intercept, so that a perfectly
    separating input yields a large finite weight instead of diverging. The
    problem is convex with three parameters, which is why it is solved here
    rather than with a library the project does not depend on.
    """

    if not 0.0 < clip < 0.5:
        raise ValueError("clip must lie in (0, 0.5).")
    targets = np.asarray(labels, dtype=np.float64).ravel()
    if targets.size == 0:
        raise ValueError("Cannot fit a blend from zero samples.")
    features = np.column_stack(
        (
            _logit(prior, clip).ravel(),
            _logit(vlm, clip).ravel(),
            np.ones_like(targets),
        )
    )
    if features.shape[0] != targets.size:
        raise ValueError("prior, vlm and labels must align.")
    # A vanishing ridge on the intercept keeps the Hessian invertible when
    # every label is the same.
    penalty = np.diag([l2, l2, 1e-8])
    weights = np.zeros(3)
    for _ in range(max_iterations):
        predicted = _sigmoid(features @ weights)
        gradient = features.T @ (predicted - targets) / targets.size
        gradient += penalty @ weights
        curvature = predicted * (1.0 - predicted)
        hessian = (features * curvature[:, None]).T @ features / targets.size
        step = np.linalg.solve(hessian + penalty, gradient)
        weights -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return LogisticBlend(
        prior_weight=float(weights[0]),
        vlm_weight=float(weights[1]),
        intercept=float(weights[2]),
        clip=clip,
    )


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


def _metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int,
) -> dict[str, float]:
    return {
        "num_samples": float(labels.size),
        "mean_probability": float(probabilities.mean()) if labels.size else math.nan,
        "auc": roc_auc(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels, num_bins=num_bins),
        "brier": float(np.mean((probabilities - labels) ** 2))
        if labels.size
        else math.nan,
    }


def _json_float(value: Any) -> Any:
    """Return JSON-safe values: NaN becomes None, containers are walked."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_float(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_float(item) for item in value]
    return value


@dataclass(frozen=True)
class _Split:
    """One split's samples with every predictor's input aligned to them."""

    samples: tuple[AffordanceSample, ...]
    prior: np.ndarray
    vlm: np.ndarray
    ratings: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class VLMFeasibilityStudy:
    """The VLM's success judgements scored against probe outcomes.

    ``overall`` and ``per_skill`` are on the held-out split. ``vlm_all_samples``
    scores the VLM alone on every held-out probe, the ones whose precondition
    failed included, which shows whether it knows the requirements as well as
    the odds. Missing ratings took ``missing_probability`` and are counted in
    ``num_missing_ratings``.
    """

    candidate_skills: tuple[SkillId, ...]
    observation: ObservationConfig | None
    include_world: bool
    require_precondition: bool
    missing_probability: float
    num_fit_samples: int
    num_test_samples: int
    num_states: int
    num_invalid_responses: int
    num_missing_ratings: int
    blend: LogisticBlend
    overall: dict[str, dict[str, float]]
    per_skill: dict[str, dict[str, Any]]
    vlm_all_samples: dict[str, float]
    # One row per held-out sample and one per rated state, for inspection.
    test_records: tuple[dict[str, Any], ...]
    state_records: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary; NaN metrics become None."""

        return _json_float(
            {
                "candidate_skills": [skill.value for skill in self.candidate_skills],
                "observation": None
                if self.observation is None
                else asdict(self.observation),
                "include_world": self.include_world,
                "require_precondition": self.require_precondition,
                "missing_probability": self.missing_probability,
                "num_fit_samples": self.num_fit_samples,
                "num_test_samples": self.num_test_samples,
                "num_states": self.num_states,
                "num_invalid_responses": self.num_invalid_responses,
                "num_missing_ratings": self.num_missing_ratings,
                "blend": self.blend.to_dict(),
                "overall": self.overall,
                "per_skill": self.per_skill,
                "vlm_all_samples": self.vlm_all_samples,
                "test_records": list(self.test_records),
                "state_records": list(self.state_records),
            }
        )


def evaluate_vlm_feasibility(
    rater: VLMFeasibilityRater,
    prior: CompetencePrior,
    fit_samples: Sequence[AffordanceSample],
    fit_states: Mapping[BoundaryKey, Any],
    test_samples: Sequence[AffordanceSample],
    test_states: Mapping[BoundaryKey, Any],
    *,
    require_precondition: bool = True,
    missing_probability: float = 0.5,
    clip: float = 0.01,
    l2: float = 1e-3,
    num_bins: int = 10,
) -> VLMFeasibilityStudy:
    """Score the prior, the VLM, their product and their blend on held-out probes.

    The prior should be the one fitted on ``fit_samples``; the blend is fitted
    on the same split, so its prior input is in-sample there and the weights
    lean towards the prior if anything. Every state of both splits is rated in
    one batched call. ``require_precondition`` drops probes whose symbolic
    precondition failed, as :func:`evaluate_competence_prior` does, since the
    planner never consults a Can term for them. Samples must be for skills in
    ``rater.candidate_skills``.
    """

    if not 0.0 <= missing_probability <= 1.0:
        raise ValueError("missing_probability must lie in [0, 1].")
    offered = set(rater.candidate_skills)
    for sample in (*fit_samples, *test_samples):
        if sample.skill_id not in offered:
            raise ValueError(
                f"{sample.skill_id.value} is not a candidate of the rater."
            )

    # Rate each split's states once, in one backend call.
    keyed: list[tuple[str, BoundaryKey, Any]] = []
    for split, samples, states in (
        ("fit", fit_samples, fit_states),
        ("test", test_samples, test_states),
    ):
        keys = sorted({(sample.episode, sample.boundary) for sample in samples})
        missing = [key for key in keys if key not in states]
        if missing:
            raise ValueError(f"No {split} state for (episode, boundary) {missing}.")
        keyed.extend((split, key, states[key]) for key in keys)
    responses = rater.rate_all([state for _, _, state in keyed])
    rated = {
        (split, key): response
        for (split, key, _), response in zip(keyed, responses, strict=True)
    }

    def split_arrays(split: str, samples: Sequence[AffordanceSample]) -> _Split:
        rows = tuple(s for s in samples if s.precondition or not require_precondition)
        ratings = np.asarray(
            [rated[(split, (s.episode, s.boundary))].ratings[s.skill_id] for s in rows],
            dtype=np.float64,
        )
        vlm = np.where(np.isnan(ratings), missing_probability, ratings / MAX_RATING)
        return _Split(
            samples=rows,
            prior=np.asarray(
                [
                    prior.success_probability(s.skill_id, s.steps_to_target)
                    for s in rows
                ],
                dtype=np.float64,
            ),
            vlm=vlm,
            ratings=ratings,
            labels=np.asarray([float(s.succeeded) for s in rows], dtype=np.float64),
        )

    fit = split_arrays("fit", fit_samples)
    test = split_arrays("test", test_samples)
    if not fit.samples or not test.samples:
        raise ValueError("Both splits need at least one usable sample.")
    blend = fit_logistic_blend(fit.prior, fit.vlm, fit.labels, clip=clip, l2=l2)
    predictions = {
        "prior": test.prior,
        "vlm": test.vlm,
        "product": test.prior * test.vlm,
        "blend": blend.apply(test.prior, test.vlm),
    }

    overall = {
        name: _metrics(values, test.labels, num_bins)
        for name, values in predictions.items()
    }
    per_skill: dict[str, dict[str, Any]] = {}
    for skill_id in rater.candidate_skills:
        selected = np.asarray([s.skill_id is skill_id for s in test.samples])
        if not np.any(selected):
            continue
        skill_ratings = test.ratings[selected]
        valid_ratings = skill_ratings[~np.isnan(skill_ratings)]
        per_skill[skill_id.value] = {
            "num_samples": int(np.count_nonzero(selected)),
            "success_rate": float(test.labels[selected].mean()),
            "mean_rating": float(valid_ratings.mean())
            if valid_ratings.size
            else math.nan,
            "num_missing_ratings": int(skill_ratings.size - valid_ratings.size),
            **{
                name: _metrics(values[selected], test.labels[selected], num_bins)
                for name, values in predictions.items()
            },
        }

    # The VLM on every held-out probe, before the precondition filter.
    all_ratings = np.asarray(
        [
            rated[("test", (s.episode, s.boundary))].ratings[s.skill_id]
            for s in test_samples
        ],
        dtype=np.float64,
    )
    vlm_all_samples = _metrics(
        np.where(np.isnan(all_ratings), missing_probability, all_ratings / MAX_RATING),
        np.asarray([float(s.succeeded) for s in test_samples], dtype=np.float64),
        num_bins,
    )

    test_records = tuple(
        {
            "episode": sample.episode,
            "boundary": sample.boundary,
            "skill": sample.skill_id.value,
            "steps_to_target": sample.steps_to_target,
            "succeeded": sample.succeeded,
            "rating": float(test.ratings[index]),
            **{name: float(values[index]) for name, values in predictions.items()},
        }
        for index, sample in enumerate(test.samples)
    )
    state_records = tuple(
        {
            "split": split,
            "episode": key[0],
            "boundary": key[1],
            "valid": response.valid,
            "error": response.error,
            "assessment": response.assessment,
            "ratings": {
                skill_id.value: rating for skill_id, rating in response.ratings.items()
            },
        }
        for (split, key), response in rated.items()
    )
    return VLMFeasibilityStudy(
        candidate_skills=rater.candidate_skills,
        observation=rater.observation,
        include_world=rater.include_world,
        require_precondition=require_precondition,
        missing_probability=missing_probability,
        num_fit_samples=len(fit.samples),
        num_test_samples=len(test.samples),
        num_states=len(responses),
        num_invalid_responses=sum(not response.valid for response in responses),
        num_missing_ratings=int(np.count_nonzero(np.isnan(test.ratings))),
        blend=blend,
        overall=overall,
        per_skill=per_skill,
        vlm_all_samples=vlm_all_samples,
        test_records=test_records,
        state_records=state_records,
    )
