"""Frozen vision-language utility scoring for the hierarchical planner."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from craftax.craftax_classic.constants import BlockType

from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.craftax import render_state
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    block_mask,
    interaction_distance,
    planner_skill_catalog,
    skill_interaction_distance,
    skill_metric,
    skill_precondition,
    travel_distance_map,
    walkable_mask,
)
from tiny_sima.planner.saycan import PlannerContext, UtilityScoreBatch


class PlannerPromptStyle(str, Enum):
    """Which planner prompt and response schema to use.

    ``BASELINE`` is preserved verbatim as the prompt that measured the problem:
    on the Phase 4 prefix ablation it ranked an unreachable skill first at 74.5%
    of boundaries where a reachable one existed, and no Can term can repair
    that, because feasibility cannot change what the VLM ranked first.

    ``AVAILABLE_FIRST`` attacks the cause rather than the wording. Generation is
    schema-constrained and left-to-right, so the old field order let the model
    commit to ``selected_skill`` after 96 characters of free text and before it
    had said anything about the state. This style inserts an ``available_now``
    field ahead of the choice, which the prompt supplies verbatim as structured
    input -- turning "consider what is reachable" from an instruction a 2B model
    may skip into a copy step it must emit before it can choose.
    """

    BASELINE = "baseline"
    AVAILABLE_FIRST = "available_first"


DEFAULT_PROMPT_STYLE = PlannerPromptStyle.AVAILABLE_FIRST

# The baseline keeps the limit it was measured with. At 96 characters the
# available-first assessments were cut off mid-sentence and drifted out of
# English near the cutoff, so that style gets more room.
_PROGRESS_ASSESSMENT_LIMITS = {
    PlannerPromptStyle.BASELINE: 96,
    PlannerPromptStyle.AVAILABLE_FIRST: 160,
}


def progress_assessment_limit(
    style: PlannerPromptStyle = DEFAULT_PROMPT_STYLE,
) -> int:
    """Return the character limit the prompt and schema place on the assessment."""

    return _PROGRESS_ASSESSMENT_LIMITS[style]


class VLMRequest(NamedTuple):
    """One image-text generation request at an option boundary."""

    image: np.ndarray
    system_prompt: str
    user_prompt: str
    candidate_skills: tuple[SkillId, ...]
    require_available: bool = False
    assessment_max_length: int = 96

    @property
    def prompt(self) -> str:
        """Return both prompt roles for diagnostics and cache inspection."""

        return f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{self.user_prompt}"


class VisionLanguageBackend(Protocol):
    """Minimal backend boundary used by real and test VLMs."""

    def generate(self, request: VLMRequest) -> str:
        """Generate one planner response."""

        ...


@runtime_checkable
class BatchedVisionLanguageBackend(Protocol):
    """Optional backend extension that decodes several requests together."""

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
        """Generate one response per request, in the order given."""

        ...


def generate_all(
    backend: VisionLanguageBackend,
    requests: Sequence[VLMRequest],
) -> list[str]:
    """Generate every response, batching when the backend supports it.

    Batching is an optimisation rather than part of the backend contract, so
    a backend written against :class:`VisionLanguageBackend` alone still
    works -- one request at a time.
    """

    if not requests:
        return []
    if isinstance(backend, BatchedVisionLanguageBackend):
        responses = list(backend.generate_batch(requests))
        if len(responses) != len(requests):
            raise ValueError(
                f"generate_batch returned {len(responses)} responses for "
                f"{len(requests)} requests."
            )
        return responses
    return [backend.generate(request) for request in requests]


@dataclass(frozen=True)
class ParsedVLMResponse:
    """Validated skill utilities extracted from model text."""

    utilities: dict[SkillId, float]
    ranking: tuple[SkillId, ...]
    selected_skill: SkillId | None
    progress_assessment: str
    raw_response: str
    valid: bool
    error: str | None = None
    # What the model itself claimed was available before it chose. Comparing it
    # against the true reachable set separates a perception failure from an
    # instruction-following one, which point at different fixes.
    claimed_available: tuple[SkillId, ...] = ()
    # True when the ranking only became a permutation after repair; see
    # :func:`parse_vlm_response`.
    repaired: bool = False


def _rank_utility(index: int) -> float:
    """Map a zero-based rank to a failure-tolerant utility value."""

    return 1.0 / float((index + 1) ** 2)


def _json_object(text: str) -> dict[str, Any] | None:
    stripped = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _invalid_vlm_response(
    response: str,
    error: str,
) -> ParsedVLMResponse:
    """Return one fail-closed response with stable empty values."""

    return ParsedVLMResponse(
        utilities={},
        ranking=(),
        selected_skill=None,
        progress_assessment="",
        raw_response=response,
        valid=False,
        error=error,
    )


def parse_vlm_response(
    response: str,
    candidate_skills: tuple[SkillId, ...],
    *,
    repair_ranking: bool = False,
) -> ParsedVLMResponse:
    """Parse a planner response without trusting unknown skill names.

    ``repair_ranking`` accepts a ranking that names only known skills but
    repeats some and so omits others. The constrained schema fixes the
    ranking's length but cannot make its entries distinct, and under the
    available-first prompt the model fills the slots by cycling through the
    skills it just reported as available: 10 of 48 prefix boundaries, every
    one with a correct ``selected_skill``. Repair keeps each skill's first
    position and appends the omitted skills in catalog order, which leaves
    the choice and the order the model did express untouched.

    Repair also resolves a ``selected_skill`` that the ranking does not put
    first, by moving the selection to the top. The schema decodes the
    selection before the ranking, so it is the earlier commitment, and in the
    closed loop an invalid response ends the episode: one such response in 48
    boundaries would end roughly one 12-decision episode in six. Either repair
    flags the response ``repaired`` so the strict result is still recoverable.
    """

    payload = _json_object(response)
    if payload is None:
        return _invalid_vlm_response(
            response,
            "malformed_json",
        )

    progress_assessment = payload.get("progress_assessment")
    if not isinstance(progress_assessment, str):
        return _invalid_vlm_response(
            response,
            "progress_assessment_not_a_string",
        )

    # Optional so that a ``BASELINE`` response, which has no such field, still
    # parses; the ``AVAILABLE_FIRST`` schema makes the model emit it.
    claimed_available: tuple[SkillId, ...] = ()
    raw_available = payload.get("available_now")
    if raw_available is not None:
        if not isinstance(raw_available, list):
            return _invalid_vlm_response(response, "available_now_not_an_array")
        claimed: list[SkillId] = []
        for name in raw_available:
            try:
                skill_id = SkillId(str(name))
            except ValueError:
                return _invalid_vlm_response(response, "unknown_available_skill")
            if skill_id not in candidate_skills:
                return _invalid_vlm_response(response, "unknown_available_skill")
            if skill_id in claimed:
                return _invalid_vlm_response(response, "duplicate_available_skill")
            claimed.append(skill_id)
        claimed_available = tuple(claimed)

    raw_selected = payload.get("selected_skill")
    selected_skill: SkillId | None = None
    if raw_selected is not None:
        try:
            selected_skill = SkillId(str(raw_selected))
        except ValueError:
            return _invalid_vlm_response(
                response,
                "unknown_selected_skill",
            )
        if selected_skill not in candidate_skills:
            return _invalid_vlm_response(
                response,
                "unknown_selected_skill",
            )
    elif candidate_skills:
        return _invalid_vlm_response(
            response,
            "selected_skill_missing",
        )

    raw_ranking = payload.get("ranking")
    if not isinstance(raw_ranking, list):
        return _invalid_vlm_response(
            response,
            "ranking_not_an_array",
        )

    ranking: list[SkillId] = []
    invalid_skills: list[str] = []
    candidate_set = set(candidate_skills)
    for name in raw_ranking:
        try:
            skill_id = SkillId(str(name))
        except ValueError:
            invalid_skills.append(str(name))
            continue
        if skill_id not in candidate_set:
            invalid_skills.append(str(name))
            continue
        ranking.append(skill_id)

    if invalid_skills:
        names = ",".join(sorted(invalid_skills))
        return _invalid_vlm_response(
            response,
            f"unknown_ranking_skills:{names}",
        )

    is_permutation = (
        len(ranking) == len(candidate_skills) and set(ranking) == candidate_set
    )
    repaired = False
    if not is_permutation and repair_ranking and ranking:
        kept = list(dict.fromkeys(ranking))
        ranking = [
            *kept,
            *(skill_id for skill_id in candidate_skills if skill_id not in kept),
        ]
        repaired = True
    elif len(ranking) != len(candidate_skills):
        return _invalid_vlm_response(
            response,
            "ranking_wrong_length",
        )
    elif not is_permutation:
        return _invalid_vlm_response(
            response,
            "ranking_not_a_permutation",
        )

    if ranking and selected_skill is not None and selected_skill != ranking[0]:
        if not repair_ranking:
            return _invalid_vlm_response(
                response,
                "selected_skill_ranking_mismatch",
            )
        # Both fields are validated against the candidates by now, so the
        # selection is in the ranking.
        ranking = [
            selected_skill,
            *(skill_id for skill_id in ranking if skill_id != selected_skill),
        ]
        repaired = True
    ranking_tuple = tuple(ranking)
    utilities = {
        skill_id: _rank_utility(index)
        for index, skill_id in enumerate(ranking_tuple)
    }
    return ParsedVLMResponse(
        utilities=utilities,
        ranking=ranking_tuple,
        selected_skill=selected_skill,
        progress_assessment=progress_assessment,
        raw_response=response,
        valid=True,
        claimed_available=claimed_available,
        repaired=repaired,
    )


def _reachability_values(
    state: Any,
    config: ReachabilityConfig,
) -> dict[str, Any]:
    """Return reachability facts keyed exactly as the mechanics contract.

    Each ``*_reachable`` flag answers whether the player can stand next to the
    resource and interact with it inside the skill horizon, searching only the
    visible window. The matching ``*_steps`` value is a lower bound on how many
    environment steps that takes, or ``None`` when it is out of range. The
    previous implementation reported whether a block existed anywhere on the
    64x64 map, which was true in essentially every state and told the planner
    nothing.
    """

    distances = travel_distance_map(state, config)
    targets: dict[str, Any] = {
        "tree": block_mask(state, BlockType.TREE),
        "stone": block_mask(state, BlockType.STONE),
        "water": block_mask(state, BlockType.WATER),
        "placement_space": walkable_mask(state),
    }
    values: dict[str, Any] = {}
    for name, target_mask in targets.items():
        bound = int(
            np.asarray(
                interaction_distance(
                    state,
                    target_mask,
                    config,
                    distances=distances,
                )
            ).item()
        )
        reachable = bound < config.unreachable
        values[f"{name}_reachable"] = reachable
        values[f"{name}_steps"] = bound if reachable else None

    # Crafting uses the eight surrounding tiles rather than the faced tile, so
    # it is derived from a skill goal instead of a raw block mask.
    table_bound = int(
        np.asarray(
            skill_interaction_distance(
                SkillGoal(SkillId.CRAFT_WOOD_PICKAXE),
                state,
                config,
                distances=distances,
            )
        ).item()
    )
    table_reachable = table_bound < config.unreachable
    values["crafting_table_reachable"] = table_reachable
    values["crafting_table_steps"] = table_bound if table_reachable else None
    return values


def summarize_state(
    state: Any,
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
) -> str:
    """Return planner-visible inventory, survival, and reachability values."""

    inventory = state.inventory
    inventory_fields = (
        "wood",
        "stone",
        "coal",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    )
    inventory_values = {
        name: int(np.asarray(getattr(inventory, name)).item())
        for name in inventory_fields
        if hasattr(inventory, name)
    }
    survival_values = {
        "health": int(np.asarray(state.player_health).item()),
        "food": int(np.asarray(state.player_food).item()),
        "drink": int(np.asarray(state.player_drink).item()),
        "energy": int(np.asarray(state.player_energy).item()),
    }
    world_values = _reachability_values(state, reachability)
    return json.dumps(
        {
            "inventory": inventory_values,
            "survival": survival_values,
            "world": world_values,
        },
        sort_keys=True,
    )


def _mechanics_line(candidate_skills: Sequence[SkillId]) -> str:
    """Describe the candidates' mechanics, in registry order.

    Registry order makes the line depend only on which skills are offered,
    and the six-skill catalog renders the line every earlier prompt had.
    """

    offered = set(candidate_skills)
    return "Known game mechanics: " + json.dumps(
        planner_skill_catalog(
            [skill_id for skill_id in ALL_SKILL_IDS if skill_id in offered]
        ),
        separators=(",", ":"),
        sort_keys=True,
    )


_BASELINE_INSTRUCTIONS = (
    "You are the high-level planner for a Craftax survival agent.",
    "You may rely on the supplied game-mechanics catalog as authoritative "
    "knowledge of skill requirements, resource costs, and successful "
    "effects.",
    "At each option boundary, rank only the supplied skills by how useful "
    "they are as the next requested skill from the current state.",
    "The ranking is recomputed after every completed or failed skill.",
    "Prioritize unmet prerequisites in execution order. Once a prerequisite "
    "is achieved, promote the next unmet objective step and demote redundant "
    "repetitions unless another repetition is necessary.",
    "A separate affordance model checks whether each skill is executable, "
    "but state progress and recent outcomes must still affect usefulness.",
    "A requirement named world.<resource>_reachable means the player can "
    "walk to that resource and interact with it within the skill horizon, "
    "searching only the area currently visible. The matching "
    "world.<resource>_steps value is the fewest steps that needs, or null "
    "when the resource is out of range. A resource that is not reachable "
    "cannot be collected from where the player stands now, so prefer a "
    "skill that makes progress immediately; choose an out-of-range resource "
    "only when nothing nearer advances the objective.",
    "First assess current objective progress, then choose the best next "
    "skill, then rank the complete catalog consistently with that choice.",
)


_AVAILABLE_FIRST_INSTRUCTIONS = (
    "You are the high-level planner for a Craftax survival agent.",
    "You may rely on the supplied game-mechanics catalog as authoritative "
    "knowledge of skill requirements, resource costs, and successful "
    "effects.",
    "At each option boundary, rank only the supplied skills by how useful "
    "they are as the next requested skill from the current state.",
    "The ranking is recomputed after every completed or failed skill, so a "
    "skill you cannot do now costs nothing to defer.",
    # The baseline told the planner that a separate affordance model would
    # handle executability. It then ranked an unreachable skill first at
    # three boundaries in four, and no downstream factor can undo that, so
    # availability is the planner's responsibility here.
    "Usefulness means useful as the very next requested skill, which "
    "includes being doable now. A skill that cannot start from where the "
    "player stands is not a useful next step however central it is to the "
    "objective.",
    "Each candidate arrives with available_now and blocked_reason already "
    "computed for you. Treat them as ground truth and do not re-derive "
    "them from the screenshot.",
    "Among the available skills, prioritize unmet prerequisites in "
    "execution order; once a prerequisite is achieved, promote the next "
    "unmet objective step and demote redundant repetitions unless another "
    "repetition is necessary.",
    "Report the available skills before choosing one, then choose from "
    "what you reported, then rank the complete catalog starting with that "
    "choice.",
)


def build_planner_system_prompt(
    style: PlannerPromptStyle = DEFAULT_PROMPT_STYLE,
    candidate_skills: Sequence[SkillId] = SKILL_IDS,
) -> str:
    """Return stable planner instructions sent with the system role.

    The mechanics catalog covers only the offered skills, so a planner never
    reads about a skill it cannot request.
    """

    instructions = (
        _BASELINE_INSTRUCTIONS
        if style is PlannerPromptStyle.BASELINE
        else _AVAILABLE_FIRST_INSTRUCTIONS
    )
    return "\n".join((*instructions, _mechanics_line(candidate_skills)))


PLANNER_SYSTEM_PROMPT = build_planner_system_prompt(PlannerPromptStyle.BASELINE)
AVAILABLE_FIRST_SYSTEM_PROMPT = build_planner_system_prompt(
    PlannerPromptStyle.AVAILABLE_FIRST
)


def _candidate_entries(
    state: Any,
    candidate_skills: tuple[SkillId, ...],
    reachability: ReachabilityConfig,
    *,
    include_availability: bool,
) -> list[dict[str, Any]]:
    """Describe each candidate from the shared reachability search."""

    distances = travel_distance_map(state, reachability)
    entries: list[dict[str, Any]] = []
    for skill_id in candidate_skills:
        goal = SkillGoal(skill_id)
        bound = int(
            np.asarray(
                skill_interaction_distance(
                    goal,
                    state,
                    reachability,
                    distances=distances,
                )
            ).item()
        )
        reachable = bound < reachability.unreachable
        entry: dict[str, Any] = {
            "skill": skill_id.value,
            "current_value": int(
                np.asarray(skill_metric(skill_id, state)).item()
            ),
            "steps_to_target": bound if reachable else None,
        }
        if include_availability:
            satisfied = bool(np.asarray(skill_precondition(goal, state)))
            if not satisfied:
                blocked_reason = "missing_requirement"
            elif not reachable:
                blocked_reason = "no_target_in_view"
            else:
                blocked_reason = None
            entry["available_now"] = blocked_reason is None
            entry["blocked_reason"] = blocked_reason
        entries.append(entry)
    return entries


def build_planner_prompt(
    state: Any,
    context: PlannerContext,
    candidate_skills: tuple[SkillId, ...],
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY,
    style: PlannerPromptStyle = DEFAULT_PROMPT_STYLE,
    *,
    available_shuffle_seed: int | None = None,
) -> str:
    """Build a reference-plan-free prompt for one skill boundary.

    ``available_shuffle_seed`` lists the available skills in a seeded order
    instead of catalog order. It is the control for position bias: with the
    catalog order collect_wood is always listed first, and the model picked it
    over an available, claimed collect_stone at six of eight final boundaries.
    The order is a pure function of the seed and the boundary, so the response
    cache still hits on a repeated boundary.
    """

    if (
        available_shuffle_seed is not None
        and style is not PlannerPromptStyle.AVAILABLE_FIRST
    ):
        raise ValueError("available_shuffle_seed needs the available_first style.")

    history = [
        {
            "skill": attempt.skill_id.value,
            "outcome": attempt.outcome.value,
            "steps": attempt.steps,
            "native_return": round(attempt.native_return, 3),
        }
        for attempt in context.history[-8:]
    ]
    catalog = _candidate_entries(
        state,
        candidate_skills,
        reachability,
        include_availability=style is PlannerPromptStyle.AVAILABLE_FIRST,
    )
    header = (
        f"Objective: {context.objective}",
        "Current structured state: " + summarize_state(state, reachability),
        f"Recent skill outcomes: {json.dumps(history)}",
    )
    if style is PlannerPromptStyle.BASELINE:
        return "\n".join(
            (
                *header,
                f"Candidate skills: {json.dumps(catalog)}",
                "Inspect the screenshot and structured state. Rank every "
                "candidate exactly once from most useful to least useful as "
                "the next skill.",
                "A candidate whose steps_to_target is null has no reachable "
                "target in view, so requesting it means walking off to search. "
                "Rank such candidates below any skill that can make progress "
                "now, and prefer a smaller steps_to_target when two skills are "
                "equally useful.",
                "The first skill must be the best next step given progress "
                "already made, not merely the skill most associated with the "
                "overall goal.",
                "Rank completed or redundant prerequisites below the next unmet "
                "objective step unless repeating them is necessary.",
                "First give a progress assessment of at most 96 characters. "
                "Then select the best next skill and rank it first in the full "
                "ordering.",
            )
        )

    # Available candidates are listed first so that position agrees with the
    # answer being asked for; the ranking is still over the whole catalog.
    available_entries = [entry for entry in catalog if entry["available_now"]]
    if available_shuffle_seed is not None:
        boundary_key = sha256("\n".join(header).encode("utf-8")).digest()
        rng = np.random.default_rng(
            (available_shuffle_seed, int.from_bytes(boundary_key[:8], "big"))
        )
        available_entries = [
            available_entries[index]
            for index in rng.permutation(len(available_entries))
        ]
    ordered = [
        *available_entries,
        *(entry for entry in catalog if not entry["available_now"]),
    ]
    available = [entry["skill"] for entry in available_entries]
    assessment_limit = progress_assessment_limit(style)
    return "\n".join(
        (
            *header,
            f"Candidate skills: {json.dumps(ordered)}",
            "A candidate is available_now only when its requirements are met "
            "and its target is close enough to reach and use from where the "
            "player stands. blocked_reason says which half failed: "
            "missing_requirement means an item or structure is absent, and "
            "no_target_in_view means requesting it would just walk off to "
            "search.",
            "Skills available now: " + (json.dumps(available) or "[]"),
            "Answer in this order. First a progress assessment in English of "
            f"at most {assessment_limit} characters. Second, copy the "
            "available-now list into the "
            "available_now field. Third, choose selected_skill from that list; "
            "if the list is empty, choose the candidate with the smallest "
            "steps_to_target. Fourth, rank every candidate exactly once, "
            "starting with selected_skill.",
            "Among available skills prefer the next unmet objective step over "
            "a completed or redundant one, and a smaller steps_to_target when "
            "two are equally useful. Do not choose the skill most associated "
            "with the overall goal if it is not available now.",
        )
    )


class _PreparedRequest(NamedTuple):
    """One rendered boundary and the cache key that identifies it."""

    cache_key: str
    request: VLMRequest


_EMPTY_CATALOG_RESPONSE = ParsedVLMResponse(
    utilities={},
    ranking=(),
    selected_skill=None,
    progress_assessment="No candidate skills were supplied.",
    raw_response="",
    valid=True,
)


@dataclass
class VLMUtilityScorer:
    """Use one VLM call to score all candidate skills at a boundary."""

    backend: VisionLanguageBackend
    config: VLMPlannerConfig = VLMPlannerConfig()
    reachability: ReachabilityConfig = DEFAULT_REACHABILITY
    prompt_style: PlannerPromptStyle = DEFAULT_PROMPT_STYLE
    # Accept rankings that repeat skills; see :func:`parse_vlm_response`.
    # False is the strict-parse ablation.
    repair_ranking: bool = True
    # Seeded order for the available-skill list, the position-bias control;
    # see :func:`build_planner_prompt`.
    available_shuffle_seed: int | None = None
    last_response: ParsedVLMResponse | None = None
    # Every response from the most recent call, in request order, so a batched
    # evaluation can still attribute a parse failure to its own boundary.
    last_responses: tuple[ParsedVLMResponse, ...] = ()
    # Raw response text rather than parsed results, so the parse settings can
    # change without invalidating generations.
    _cache: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def _prepare(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> _PreparedRequest:
        """Render one boundary into a request and its response cache key."""

        image = render_state(
            state,
            self.config.environment_name,
            block_pixel_size=self.config.block_pixel_size,
        )
        system_prompt = build_planner_system_prompt(
            self.prompt_style,
            candidate_skills,
        )
        user_prompt = build_planner_prompt(
            state,
            context,
            candidate_skills,
            self.reachability,
            self.prompt_style,
            available_shuffle_seed=self.available_shuffle_seed,
        )
        cache_key = sha256(
            system_prompt.encode("utf-8")
            + user_prompt.encode("utf-8")
            + image.tobytes()
        ).hexdigest()
        return _PreparedRequest(
            cache_key=cache_key,
            request=VLMRequest(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                candidate_skills=candidate_skills,
                require_available=(
                    self.prompt_style is PlannerPromptStyle.AVAILABLE_FIRST
                ),
                assessment_max_length=progress_assessment_limit(
                    self.prompt_style
                ),
            ),
        )

    def score_all(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> UtilityScoreBatch:
        """Render the state, invoke the VLM once, and return safe utilities."""

        return self.score_all_batched((state,), (context,), candidate_skills)[0]

    def score_all_batched(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
        candidate_skills: tuple[SkillId, ...],
    ) -> list[UtilityScoreBatch]:
        """Score several boundaries with one batched backend invocation.

        Decoding dominates the wall clock of a planner evaluation and the
        boundaries of independent episodes are independent, so scoring them
        together turns one generation per episode into one padded forward
        pass. Returned batches stay aligned with ``states``.
        """

        if len(states) != len(contexts):
            raise ValueError("states and contexts must have equal length.")
        if not candidate_skills:
            self.last_responses = tuple(
                _EMPTY_CATALOG_RESPONSE for _ in states
            )
            self.last_response = _EMPTY_CATALOG_RESPONSE
            return [UtilityScoreBatch(utilities={}) for _ in states]

        prepared = [
            self._prepare(state, context, candidate_skills)
            for state, context in zip(states, contexts, strict=True)
        ]
        # Identical boundaries within one batch share a single generation, the
        # same way repeated boundaries across ablations share the cache.
        pending: dict[str, VLMRequest] = {
            cache_key: request
            for cache_key, request in prepared
            if cache_key not in self._cache
        }
        if pending:
            responses = generate_all(self.backend, tuple(pending.values()))
            self._cache.update(zip(pending, responses, strict=True))
        parsed_responses = [
            parse_vlm_response(
                self._cache[cache_key],
                candidate_skills,
                repair_ranking=self.repair_ranking,
            )
            for cache_key, _ in prepared
        ]
        self.last_responses = tuple(parsed_responses)
        if parsed_responses:
            self.last_response = parsed_responses[-1]
        return [
            UtilityScoreBatch(
                utilities=parsed.utilities,
                valid=parsed.valid,
                error=parsed.error,
                ranking=parsed.ranking,
                raw_response=parsed.raw_response,
                repaired=parsed.repaired,
            )
            for parsed in parsed_responses
        ]

    def clear_cache(self) -> None:
        """Discard responses cached for matched planner ablations."""

        self._cache.clear()


@lru_cache(maxsize=16)
def _ranking_output_type(
    candidate_skills: tuple[SkillId, ...],
    require_available: bool = False,
    assessment_max_length: int = 96,
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines.

    Field declaration order is the point, not a detail. Outlines constrains
    generation left to right over the schema's properties, so a field placed
    before ``selected_skill`` is one the model must produce before it can
    commit to a choice. ``require_available`` inserts the availability readout
    there; see :class:`PlannerPromptStyle`.
    """

    if not candidate_skills:
        raise ValueError("candidate_skills cannot be empty.")

    try:
        from pydantic import (  # pyright: ignore[reportMissingImports]
            ConfigDict,
            Field,
            create_model,
        )
    except ImportError as error:
        raise ImportError(
            'Install the VLM dependencies with pip install -e ".[vlm]".'
        ) from error

    enum_name = "CandidateSkill_" + "_".join(
        skill_id.name for skill_id in candidate_skills
    )
    candidate_enum = Enum(
        enum_name,
        {skill_id.name: skill_id.value for skill_id in candidate_skills},
        type=str,
    )
    ranking_type = list[candidate_enum]  # pyright: ignore[reportInvalidTypeForm]
    fields: dict[str, Any] = {
        "progress_assessment": (str, Field(max_length=assessment_max_length)),
    }
    if require_available:
        fields["available_now"] = (
            ranking_type,
            Field(min_length=0, max_length=len(candidate_skills)),
        )
    fields["selected_skill"] = (
        candidate_enum,  # pyright: ignore[reportArgumentType]
        ...,
    )
    fields["ranking"] = (
        ranking_type,
        Field(
            min_length=len(candidate_skills),
            max_length=len(candidate_skills),
        ),
    )
    return create_model(
        "SkillRankingOutput",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _planner_chat(request: VLMRequest) -> Any:
    """Build the one-turn multimodal chat for a planner request."""

    try:
        from outlines.inputs import (  # pyright: ignore[reportMissingImports]
            Chat,
        )
        from outlines.inputs import (  # pyright: ignore[reportMissingImports]
            Image as OutlinesImage,
        )
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            'Install the VLM dependencies with pip install -e ".[vlm]".'
        ) from error

    image = Image.fromarray(request.image)
    image.format = "PNG"
    return Chat(
        [
            {
                "role": "system",
                "content": request.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    request.user_prompt,
                    OutlinesImage(image),
                ],
            },
        ]
    )


def _response_text(output: Any) -> str:
    """Normalise an Outlines result to the JSON text the parser expects."""

    if hasattr(output, "model_dump_json"):
        return str(output.model_dump_json())
    return str(output)


class TransformersVLMBackend:
    """Outlines-constrained backend around a frozen Transformers VLM."""

    def __init__(self, config: VLMPlannerConfig) -> None:
        self.config = config.validate()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._outlines_model: Any | None = None
        self._generator_factory: Any | None = None
        self._generators: dict[tuple[tuple[SkillId, ...], bool, int], Any] = {}

    def _load(self) -> tuple[Any, Any, Any]:
        if (
            self._processor is not None
            and self._outlines_model is not None
            and self._generator_factory is not None
        ):
            return (
                self._processor,
                self._outlines_model,
                self._generator_factory,
            )
        try:
            import outlines  # pyright: ignore[reportMissingImports]
            import torch  # pyright: ignore[reportMissingImports]
            import torchvision  # pyright: ignore[reportMissingImports]
            import transformers  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModelForImageTextToText,
                AutoProcessor,
            )
            from transformers.utils import (  # pyright: ignore[reportMissingImports]
                is_torchvision_available,
            )
        except ImportError as error:
            raise ImportError(
                'Install the VLM dependencies with pip install -e ".[vlm]".'
            ) from error

        if not is_torchvision_available():
            raise RuntimeError(
                "Transformers does not see Torchvision in this kernel even "
                f"though torchvision {torchvision.__version__} imports. "
                f"Transformers {transformers.__version__} was likely imported "
                "before Torchvision was installed. Restart the kernel and run "
                "the installation cell before importing Tiny-SIMA."
            )

        try:
            processor = AutoProcessor.from_pretrained(self.config.model_id)
        except ImportError as error:
            if "Torchvision" in str(error) or "torchvision" in str(error):
                raise ImportError(
                    "The VLM processor requires torchvision. Reinstall with "
                    'pip install -e ".[vlm]" and restart the notebook kernel.'
                ) from error
            raise
        dtype = (
            "auto"
            if self.config.dtype == "auto"
            else getattr(torch, self.config.dtype)
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            dtype=dtype,
            device_map=self.config.device_map,
        )
        model.eval()
        self._processor = processor
        self._model = model
        self._outlines_model = outlines.from_transformers(model, processor)
        self._generator_factory = outlines.Generator
        return processor, self._outlines_model, self._generator_factory

    def _generator_for(
        self,
        candidate_skills: tuple[SkillId, ...],
        outlines_model: Any,
        generator_factory: Any,
        *,
        require_available: bool = False,
        assessment_max_length: int = 96,
    ) -> Any:
        """Return one cached constrained generator per catalog and schema."""

        key = (candidate_skills, require_available, assessment_max_length)
        generator = self._generators.get(key)
        if generator is None:
            generator = generator_factory(
                outlines_model,
                _ranking_output_type(*key),
            )
            self._generators[key] = generator
        return generator

    def generate(self, request: VLMRequest) -> str:
        """Generate one schema-constrained ranking at an option boundary."""

        _, outlines_model, generator_factory = self._load()
        generator = self._generator_for(
            request.candidate_skills,
            outlines_model,
            generator_factory,
            require_available=request.require_available,
            assessment_max_length=request.assessment_max_length,
        )
        output = generator(
            _planner_chat(request),
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
        )
        return _response_text(output)

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
        """Generate one ranking per request with batched constrained decoding.

        A constrained generator owns one logits processor, which allocates a
        guide per sequence once the batch arrives, so requests are grouped by
        catalog and schema and each group decodes in chunks of
        ``config.max_batch_size``. Responses keep the request order.
        """

        if not requests:
            return []
        _, outlines_model, generator_factory = self._load()
        groups: dict[tuple[tuple[SkillId, ...], bool, int], list[int]] = {}
        for index, request in enumerate(requests):
            key = (
                request.candidate_skills,
                request.require_available,
                request.assessment_max_length,
            )
            groups.setdefault(key, []).append(index)

        # The groups partition the request indices, so every slot is written.
        responses = [""] * len(requests)
        for key, indices in groups.items():
            candidate_skills, require_available, assessment_max_length = key
            generator = self._generator_for(
                candidate_skills,
                outlines_model,
                generator_factory,
                require_available=require_available,
                assessment_max_length=assessment_max_length,
            )
            for start in range(0, len(indices), self.config.max_batch_size):
                chunk = indices[start : start + self.config.max_batch_size]
                outputs = generator.batch(
                    [_planner_chat(requests[index]) for index in chunk],
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                )
                for index, output in zip(chunk, outputs, strict=True):
                    responses[index] = _response_text(output)
        return responses
