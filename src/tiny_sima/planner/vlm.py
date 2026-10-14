"""Frozen vision-language utility scoring for the hierarchical planner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from typing import Any, NamedTuple, Protocol

import numpy as np

from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.craftax import render_state
from tiny_sima.goals import SKILL_REGISTRY, SkillId
from tiny_sima.planner.saycan import PlannerContext, UtilityScoreBatch


class VLMRequest(NamedTuple):
    """One image-text generation request at an option boundary."""

    image: np.ndarray
    system_prompt: str
    user_prompt: str
    candidate_skills: tuple[SkillId, ...]

    @property
    def prompt(self) -> str:
        """Return both prompt roles for diagnostics and cache inspection."""

        return f"SYSTEM:\n{self.system_prompt}\n\nUSER:\n{self.user_prompt}"


class VisionLanguageBackend(Protocol):
    """Minimal backend boundary used by real and test VLMs."""

    def generate(self, request: VLMRequest) -> str:
        """Generate one planner response."""

        ...


@dataclass(frozen=True)
class ParsedVLMResponse:
    """Validated skill utilities extracted from model text."""

    utilities: dict[SkillId, float]
    ranking: tuple[SkillId, ...]
    selected_skill: SkillId | None
    rationale: str
    raw_response: str
    valid: bool
    error: str | None = None


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


def parse_vlm_response(
    response: str,
    candidate_skills: tuple[SkillId, ...],
) -> ParsedVLMResponse:
    """Parse a strict planner response without trusting unknown skill names."""

    payload = _json_object(response)
    if payload is None:
        return ParsedVLMResponse(
            {},
            (),
            None,
            "",
            response,
            False,
            "malformed_json",
        )

    raw_ranking = payload.get("ranking")
    if not isinstance(raw_ranking, list):
        return ParsedVLMResponse(
            {},
            (),
            None,
            "",
            response,
            False,
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
        return ParsedVLMResponse(
            {},
            (),
            None,
            "",
            response,
            False,
            f"unknown_ranking_skills:{names}",
        )

    if len(ranking) != len(candidate_skills):
        return ParsedVLMResponse(
            {},
            (),
            None,
            "",
            response,
            False,
            "ranking_wrong_length",
        )

    if len(set(ranking)) != len(ranking) or set(ranking) != candidate_set:
        return ParsedVLMResponse(
            {},
            (),
            None,
            "",
            response,
            False,
            "ranking_not_a_permutation",
        )

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    ranking_tuple = tuple(ranking)
    utilities = {
        skill_id: _rank_utility(index)
        for index, skill_id in enumerate(ranking_tuple)
    }
    return ParsedVLMResponse(
        utilities=utilities,
        ranking=ranking_tuple,
        selected_skill=ranking_tuple[0] if ranking_tuple else None,
        rationale=rationale,
        raw_response=response,
        valid=True,
    )


def summarize_state(state: Any) -> str:
    """Return planner-visible inventory and survival values."""

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
    return json.dumps(
        {"inventory": inventory_values, "survival": survival_values},
        sort_keys=True,
    )


PLANNER_SYSTEM_PROMPT = "\n".join(
    (
        "You are the high-level planner for a Craftax survival agent.",
        "Rank only the supplied skills by how useful they are for the objective.",
        "A separate affordance model checks whether each skill is executable.",
        "Do not treat current infeasibility as low long-term usefulness.",
        "Prefer prerequisite actions that make later objective steps possible.",
    )
)


def build_planner_system_prompt() -> str:
    """Return stable planner instructions sent with the system role."""

    return PLANNER_SYSTEM_PROMPT


def build_planner_prompt(
    state: Any,
    context: PlannerContext,
    candidate_skills: tuple[SkillId, ...],
) -> str:
    """Build a reference-plan-free prompt for one skill boundary."""

    history = [
        {
            "skill": attempt.skill_id.value,
            "outcome": attempt.outcome.value,
            "steps": attempt.steps,
            "native_return": round(attempt.native_return, 3),
        }
        for attempt in context.history[-8:]
    ]
    catalog = [
        {
            "skill": skill_id.value,
            "description": SKILL_REGISTRY[skill_id].description,
        }
        for skill_id in candidate_skills
    ]
    return "\n".join(
        (
            f"Objective: {context.objective}",
            f"Current structured state: {summarize_state(state)}",
            f"Recent skill outcomes: {json.dumps(history)}",
            f"Candidate skills: {json.dumps(catalog)}",
            "Inspect the screenshot. Rank every candidate exactly once from "
            "most useful to least useful for completing the objective.",
            "Give a brief rationale for the ordering.",
        )
    )


@dataclass
class VLMUtilityScorer:
    """Use one VLM call to score all candidate skills at a boundary."""

    backend: VisionLanguageBackend
    config: VLMPlannerConfig = VLMPlannerConfig()
    last_response: ParsedVLMResponse | None = None
    _cache: dict[str, ParsedVLMResponse] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def score_all(
        self,
        state: Any,
        context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> UtilityScoreBatch:
        """Render the state, invoke the VLM once, and return safe utilities."""

        if not candidate_skills:
            parsed = ParsedVLMResponse(
                utilities={},
                ranking=(),
                selected_skill=None,
                rationale="No candidate skills were supplied.",
                raw_response="",
                valid=True,
            )
            self.last_response = parsed
            return UtilityScoreBatch(utilities={})

        image = render_state(
            state,
            self.config.environment_name,
            block_pixel_size=self.config.block_pixel_size,
        )
        system_prompt = build_planner_system_prompt()
        user_prompt = build_planner_prompt(state, context, candidate_skills)
        cache_key = sha256(
            system_prompt.encode("utf-8")
            + user_prompt.encode("utf-8")
            + image.tobytes()
        ).hexdigest()
        parsed = self._cache.get(cache_key)
        if parsed is None:
            request = VLMRequest(
                image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                candidate_skills=candidate_skills,
            )
            parsed = parse_vlm_response(
                self.backend.generate(request),
                candidate_skills,
            )
            self._cache[cache_key] = parsed
        self.last_response = parsed
        return UtilityScoreBatch(
            utilities=parsed.utilities,
            valid=parsed.valid,
            error=parsed.error,
            ranking=parsed.ranking,
            raw_response=parsed.raw_response,
        )

    def clear_cache(self) -> None:
        """Discard responses cached for matched planner ablations."""

        self._cache.clear()


@lru_cache(maxsize=16)
def _ranking_output_type(
    candidate_skills: tuple[SkillId, ...],
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines."""

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
    return create_model(
        "SkillRankingOutput",
        __config__=ConfigDict(extra="forbid"),
        ranking=(
            ranking_type,
            Field(
                min_length=len(candidate_skills),
                max_length=len(candidate_skills),
            ),
        ),
        rationale=(str, Field(max_length=256)),
    )


class TransformersVLMBackend:
    """Outlines-constrained backend around a frozen Transformers VLM."""

    def __init__(self, config: VLMPlannerConfig) -> None:
        self.config = config.validate()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._outlines_model: Any | None = None
        self._generator_factory: Any | None = None
        self._generators: dict[tuple[SkillId, ...], Any] = {}

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
    ) -> Any:
        """Return one cached constrained generator per candidate catalog."""

        generator = self._generators.get(candidate_skills)
        if generator is None:
            generator = generator_factory(
                outlines_model,
                _ranking_output_type(candidate_skills),
            )
            self._generators[candidate_skills] = generator
        return generator

    def generate(self, request: VLMRequest) -> str:
        """Generate one schema-constrained ranking at an option boundary."""

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

        _, outlines_model, generator_factory = self._load()
        image = Image.fromarray(request.image)
        image.format = "PNG"
        chat = Chat(
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
        generator = self._generator_for(
            request.candidate_skills,
            outlines_model,
            generator_factory,
        )
        output = generator(
            chat,
            max_new_tokens=self.config.max_new_tokens,
        )
        if hasattr(output, "model_dump_json"):
            return str(output.model_dump_json())
        return str(output)
