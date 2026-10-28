"""Induce a skill model from the manual, validate it, and correct it from runs.

The symbolic planner searches whatever :data:`~tiny_sima.planner.symbolic.
SkillModel` it is given. This module builds that model the way SCALAR
(Zabounidis et al., arXiv 2603.09036) does, so we can see whether a model our
VLM writes, then corrects from executor runs, reaches the hand-written
registry:

1. **Induction.** The VLM reads a qualitative game manual, which names every
   recipe's ingredients but none of their amounts, and writes each skill's
   requirements and effects over the planner's fields under a constrained
   schema.
2. **Validation.** SCALAR's feasibility and novelty checks: every skill's
   requirements must be reachable from the episode start by the other skills,
   every skill must change something, and no two skills may change the same
   things. A violating model goes back to the VLM once whole, then as
   narrow questions with fixed answers, and skills still violating are
   dropped.
3. **Correction.** SCALAR's pivotal trajectory analysis, made deterministic:
   successful transitions fix each skill's inventory effects and
   requirements. The VLM's amounts are the part SCALAR found most wrong.
4. **Failure analysis.** A failure the model predicted to succeed, for a
   reason other than a missing target, goes back to the VLM with successful
   pre-states beside it, and the VLM names at most one missing requirement.

The experiment that runs these over closed-loop rounds is in
:mod:`tiny_sima.planner.model_rounds`.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from tiny_sima.goals import (
    ALL_SKILL_IDS,
    SkillId,
    SkillMechanics,
    StateEffect,
    StateRequirement,
    get_skill_spec,
)
from tiny_sima.memory import TransitionRecord
from tiny_sima.planner.symbolic import (
    DEFAULT_MAX_PLAN_LENGTH,
    PLANNER_FIELDS,
    SkillModel,
    _blocked_reason,
    _first_steps,
    _holds,
    _moves_toward,
    _planning_problem,
    _requirement_text,
    _with_achievement,
    read_field,
)
from tiny_sima.planner.vlm import (
    VisionLanguageBackend,
    VLMRequest,
    _json_object,
    generate_all,
)

# The fields a skill may read or change: every planner field except the
# achievements, which each skill's unlock effect covers on its own.
INDUCTION_FIELDS: tuple[str, ...] = tuple(
    field for field in PLANNER_FIELDS if not field.startswith("achievement.")
)
INVENTORY_FIELDS: tuple[str, ...] = tuple(
    field for field in PLANNER_FIELDS if field.startswith("inventory.")
)
# Fields a transition record reads: inventory counts and the drink level.
OBSERVED_FIELDS: tuple[str, ...] = (*INVENTORY_FIELDS, "survival.drink")

MAX_INDUCED_VALUE = 9
MAX_INDUCED_ITEMS = 4
# Successes a requirement the prior model did not have needs before
# correction adds it; see :func:`correct_model`.
DEFAULT_MIN_SUPPORT = 3
# Failure causes that say the model was wrong rather than that the target was
# hard to find or the executor ran out of time.
DEFAULT_FAILURE_CAUSES: tuple[str, ...] = ("infeasible", "precondition_lost")
DEFAULT_MAX_EXAMPLES = 4
DEFAULT_REPAIR_QUESTIONS = 4
_ANALYSIS_MAX_LENGTH = 160

MANUAL_PATH = Path(__file__).resolve().parents[3] / "docs" / "craftax_classic_manual.md"

# The state at a Craftax-Classic reset, as the planner reads it: nothing held,
# drink full, trees, stone, water and free ground about, and no table yet.
EPISODE_START: tuple[tuple[str, int], ...] = tuple(
    (
        field,
        9
        if field == "survival.drink"
        else int(
            field
            in {
                "world.tree_reachable",
                "world.stone_reachable",
                "world.water_reachable",
                "world.placement_space_reachable",
            }
        ),
    )
    for field in PLANNER_FIELDS
)


def load_manual(path: Path | str | None = None) -> str:
    """Return the qualitative game manual the induction prompt quotes."""

    manual_path = Path(path) if path is not None else MANUAL_PATH
    if not manual_path.is_file():
        raise FileNotFoundError(
            f"No game manual at {manual_path}; pass its text as manual=."
        )
    return manual_path.read_text()


def start_values(state: Any) -> dict[str, int]:
    """Return every planner field of ``state``, for :func:`validate_model`."""

    return {field: read_field(state, field) for field in PLANNER_FIELDS}


# ---------------------------------------------------------------------------
# Model rendering and comparison


def _is_achievement(field: str) -> bool:
    return field.startswith("achievement.")


def _without_achievements(mechanics: SkillMechanics) -> SkillMechanics:
    return dataclasses.replace(
        mechanics,
        effects=tuple(
            effect for effect in mechanics.effects if not _is_achievement(effect.field)
        ),
    )


def _finish(skill_id: SkillId, mechanics: SkillMechanics) -> SkillMechanics:
    """Add the achievement unlock the same way :func:`registry_model` does."""

    return _with_achievement(skill_id, _without_achievements(mechanics))


def model_to_dict(model: SkillModel) -> dict[str, Any]:
    """Return ``model`` as JSON-ready mechanics per skill, without unlocks.

    It is the format the induction schema asks for, so the registry rendered
    this way is a valid induction answer.
    """

    return {
        skill_id.value: {
            "requires": [
                {
                    "field": requirement.field,
                    "comparison": requirement.comparison,
                    "value": int(requirement.value),
                }
                for requirement in mechanics.requires
            ],
            "effects": [
                {
                    "field": effect.field,
                    "operation": effect.operation,
                    "value": int(effect.value),
                }
                for effect in mechanics.effects
                if not _is_achievement(effect.field)
            ],
        }
        for skill_id, mechanics in model
    }


def model_from_dict(data: Mapping[str, Any]) -> SkillModel:
    """Rebuild a model saved by :func:`model_to_dict`, unlocks included.

    Unknown skills are skipped; an invalid entry raises ``ValueError``.
    """

    parsed = parse_induced_model(
        json.dumps(dict(data)),
        tuple(skill_id for skill_id in ALL_SKILL_IDS if skill_id.value in data),
    )
    if not parsed.valid or parsed.problems:
        raise ValueError(
            "; ".join(parsed.problems) or parsed.error or "invalid skill model"
        )
    return parsed.model


def _render_requirement(requirement: StateRequirement) -> str:
    return f"{requirement.comparison} {int(requirement.value)}"


def _render_effect(effect: StateEffect) -> str:
    return f"{effect.operation} {int(effect.value)}"


@dataclass(frozen=True)
class ModelChange:
    """One field-level difference between two skill models.

    ``before`` and ``after`` render the condition, such as ``"at_least 2"`` or
    ``"decrease 2"``; ``None`` means the side has no such condition.
    Requirements are matched by field and comparison, effects by field.
    """

    skill_id: SkillId
    part: Literal["requires", "effects"]
    field: str
    before: str | None
    after: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_id.value,
            "part": self.part,
            "field": self.field,
            "before": self.before,
            "after": self.after,
        }

    def __str__(self) -> str:
        return (
            f"{self.skill_id.value}.{self.part}: {self.field} "
            f"{self.before or '-'} -> {self.after or '-'}"
        )


def _conditions(
    mechanics: SkillMechanics | None,
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    if mechanics is None:
        return {}, {}
    requires = {
        (requirement.field, requirement.comparison): _render_requirement(requirement)
        for requirement in mechanics.requires
    }
    effects: dict[str, str] = {}
    for effect in mechanics.effects:
        if _is_achievement(effect.field):
            continue
        rendered = _render_effect(effect)
        effects[effect.field] = (
            f"{effects[effect.field]}, {rendered}"
            if effect.field in effects
            else rendered
        )
    return requires, effects


def model_diff(reference: SkillModel, model: SkillModel) -> tuple[ModelChange, ...]:
    """Return every field-level change from ``reference`` to ``model``.

    Skills are listed in catalog order, then requirements before effects.
    Achievement unlocks are left out: both sides add them the same way.
    """

    before_model, after_model = dict(reference), dict(model)
    changes: list[ModelChange] = []
    for skill_id in ALL_SKILL_IDS:
        if skill_id not in before_model and skill_id not in after_model:
            continue
        old_requires, old_effects = _conditions(before_model.get(skill_id))
        new_requires, new_effects = _conditions(after_model.get(skill_id))
        for key in sorted(old_requires.keys() | new_requires.keys()):
            old, new = old_requires.get(key), new_requires.get(key)
            if old != new:
                changes.append(ModelChange(skill_id, "requires", key[0], old, new))
        for field in sorted(old_effects.keys() | new_effects.keys()):
            old, new = old_effects.get(field), new_effects.get(field)
            if old != new:
                changes.append(ModelChange(skill_id, "effects", field, old, new))
    return tuple(changes)


# ---------------------------------------------------------------------------
# Validation


@dataclass(frozen=True)
class ModelViolation:
    """One skill breaking a validation rule, with a reason for the VLM.

    ``rule`` is ``unreachable_requirements``, ``no_effect`` or
    ``duplicate_effects``. For the first, ``blocking`` holds the
    requirements no plan reaches even alone.
    """

    skill_id: SkillId
    rule: str
    detail: str
    blocking: tuple[StateRequirement, ...] = ()

    def to_dict(self) -> dict[str, str]:
        return {"skill": self.skill_id.value, "rule": self.rule, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.skill_id.value}: {self.rule.replace('_', ' ')} ({self.detail})"


@dataclass(frozen=True)
class ValidationReport:
    """Every violation :func:`validate_model` found, in catalog order."""

    violations: tuple[ModelViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations

    @property
    def invalid_skills(self) -> frozenset[SkillId]:
        return frozenset(violation.skill_id for violation in self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _changes_something(effect: StateEffect) -> bool:
    return effect.operation == "set" or int(effect.value) != 0


def _effect_key(mechanics: SkillMechanics) -> frozenset[tuple[str, str, int]]:
    return frozenset(
        (effect.field, effect.operation, int(effect.value))
        for effect in mechanics.effects
        if not _is_achievement(effect.field) and _changes_something(effect)
    )


def validate_model(
    model: SkillModel,
    *,
    start: Mapping[str, int] | None = None,
    exempt_prefixes: tuple[str, ...] = ("survival.",),
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> ValidationReport:
    """Check SCALAR's feasibility and novelty rules over ``model``.

    - **Feasibility:** a skill's requirements must be reachable from
      ``start`` by some sequence of the *other* skills under the model, so a
      skill that is the only source of its own precondition is caught. The
      start defaults to :data:`EPISODE_START`; :func:`start_values` reads one
      from a reset state. Requirements on ``exempt_prefixes`` are skipped:
      vitals drain with time, which no skill models, so ``drink < 9`` holds
      soon after any start.
    - **Effect:** a skill must change at least one field besides its
      achievement unlock.
    - **Novelty:** no two skills may have the same effects. The later skill in
      catalog order is the one flagged.
    """

    values = dict(EPISODE_START)
    values.update(start or {})
    violations: list[ModelViolation] = []
    seen: dict[frozenset[tuple[str, str, int]], SkillId] = {}
    for skill_id, mechanics in model:
        goal = tuple(
            requirement
            for requirement in mechanics.requires
            if not requirement.field.startswith(exempt_prefixes)
        )
        if goal:
            others = tuple(entry for entry in model if entry[0] is not skill_id)
            fields, planned = _planning_problem(goal, others)
            start_values_ = tuple((field, values.get(field, 0)) for field in fields)
            if _first_steps(start_values_, goal, planned, max_plan_length) is None:
                current = dict(start_values_)
                unmet = [
                    requirement
                    for requirement in goal
                    if not _holds(requirement, current)
                ]
                # Name the requirements no plan reaches even alone: the
                # others are only blocked by them, and listing those too
                # sends a repair after the wrong fields.
                alone = [
                    requirement
                    for requirement in unmet
                    if _first_steps(
                        start_values_, (requirement,), planned, max_plan_length
                    )
                    is None
                ]
                reasons = [
                    _blocked_reason(requirement, current, planned)
                    for requirement in alone or unmet
                ][:3] or [f"no plan of at most {max_plan_length} skills"]
                violations.append(
                    ModelViolation(
                        skill_id,
                        "unreachable_requirements",
                        "; ".join(reasons),
                        tuple(alone),
                    )
                )
        key = _effect_key(mechanics)
        if not key:
            violations.append(
                ModelViolation(skill_id, "no_effect", "it changes no field")
            )
        elif key in seen:
            violations.append(
                ModelViolation(
                    skill_id,
                    "duplicate_effects",
                    f"the same effects as {seen[key].value}",
                )
            )
        else:
            seen[key] = skill_id
    return ValidationReport(tuple(violations))


def drop_invalid_skills(
    model: SkillModel,
    *,
    start: Mapping[str, int] | None = None,
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH,
) -> tuple[SkillModel, tuple[SkillId, ...]]:
    """Drop invalid skills until the rest validate; return them and the drops.

    Dropping a skill can strand another that needed it, so validation is
    repeated until nothing more is dropped.
    """

    dropped: list[SkillId] = []
    while True:
        report = validate_model(model, start=start, max_plan_length=max_plan_length)
        if report.valid:
            return model, tuple(dropped)
        invalid = report.invalid_skills
        dropped.extend(skill_id for skill_id, _ in model if skill_id in invalid)
        model = tuple(entry for entry in model if entry[0] not in invalid)


# ---------------------------------------------------------------------------
# Induction


def _field_description(field: str) -> str:
    if field.startswith("inventory."):
        item = field.removeprefix("inventory.").replace("_", " ")
        return f"how many {item} the player holds"
    if field == "survival.drink":
        return "the player's drink level, which drains over time"
    if field == "world.placement_space_reachable":
        return "1 while a free tile to build on is within reach, else 0"
    block = field.removeprefix("world.").removesuffix("_reachable")
    return f"1 while a {block.replace('_', ' ')} is within reach, else 0"


_INDUCTION_INSTRUCTIONS = (
    "You are writing the action model a symbolic planner uses to sequence the "
    "skills of a Craftax-Classic agent. Each skill is a short behaviour the "
    "agent can be asked to perform. For every skill, list what must hold "
    "before it can succeed (requires) and how one successful use changes the "
    "state (effects), over the fields listed and no others.\n"
    "A requirement is {field, comparison, value}: at_least holds when the "
    "field is at least the value, less_than when it is below the value, and "
    "equals when it is equal. An effect is {field, operation, value}: increase "
    "and decrease change a count by the value, and set makes the field equal "
    "to the value. World facts are 1 when true and 0 when false.\n"
    "Read the amounts off the manual where it gives them and estimate them "
    "where it does not. A requirement a skill does not need stops the planner "
    "from using it, and one it does need but is missing makes the plan fail. "
    "Use at_least for an item a skill needs or uses up: holding more than "
    "enough never stops it.\n"
    "Each field starts every episode at the value listed for it. A "
    "requirement the start does not meet can only be met through another "
    "skill's effects, so the skill that brings it about must list that "
    "effect. Every skill must change at least one field."
)


def build_induction_system_prompt(manual: str) -> str:
    """Return the instructions followed by the game manual."""

    return f"{_INDUCTION_INSTRUCTIONS}\n\nGame manual:\n{manual.strip()}"


def _skill_lines(
    skills: Sequence[SkillId],
    descriptions: Mapping[SkillId, str] | None,
) -> str:
    # The registry's instruction, not its description: the descriptions
    # state the amounts the model is meant to work out.
    return "\n".join(
        f"- {skill_id.value}: "
        f"{(descriptions or {}).get(skill_id) or get_skill_spec(skill_id).instruction}"
        for skill_id in skills
    )


def build_induction_prompt(
    skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    fields: Sequence[str] = INDUCTION_FIELDS,
    descriptions: Mapping[SkillId, str] | None = None,
    start: Mapping[str, int] | None = None,
) -> str:
    """Return the user prompt: the skills, the fields and the answer format.

    Each field is listed with its value at ``start``, which defaults to
    :data:`EPISODE_START` as in :func:`validate_model`.
    """

    values = dict(EPISODE_START)
    values.update(start or {})
    field_lines = "\n".join(
        f"- {field}: {_field_description(field)}; starts at {values.get(field, 0)}"
        for field in fields
    )
    return (
        f"Skills:\n{_skill_lines(skills, descriptions)}\n\n"
        f"Fields (values are whole numbers from 0 to {MAX_INDUCED_VALUE}):\n"
        f"{field_lines}\n\n"
        "Answer with a JSON object holding one entry per skill, keyed by its "
        "name, each with a requires list and an effects list."
    )


def _pydantic() -> Any:
    try:
        import pydantic  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise ImportError(
            'Install the VLM dependencies with pip install -e ".[vlm]".'
        ) from error
    return pydantic


def _item_types(fields: tuple[str, ...], max_value: int) -> tuple[Any, Any]:
    pydantic = _pydantic()
    field_enum = Enum(
        "ModelField",
        {name.replace(".", "__"): name for name in fields},
        type=str,
    )
    comparison_enum = Enum(
        "ModelComparison",
        {name: name for name in ("at_least", "less_than", "equals")},
        type=str,
    )
    operation_enum = Enum(
        "ModelOperation",
        {name: name for name in ("increase", "decrease", "set")},
        type=str,
    )
    config = pydantic.ConfigDict(extra="forbid")
    value = (int, pydantic.Field(ge=0, le=max_value))
    requirement = pydantic.create_model(
        "ModelRequirement",
        __config__=config,
        field=(field_enum, ...),
        comparison=(comparison_enum, ...),
        value=value,
    )
    effect = pydantic.create_model(
        "ModelEffect",
        __config__=config,
        field=(field_enum, ...),
        operation=(operation_enum, ...),
        value=value,
    )
    return requirement, effect


@lru_cache(maxsize=8)
def induction_output_type(
    skills: tuple[SkillId, ...] = ALL_SKILL_IDS,
    fields: tuple[str, ...] = INDUCTION_FIELDS,
    max_items: int = MAX_INDUCED_ITEMS,
    max_value: int = MAX_INDUCED_VALUE,
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines.

    One property per skill, in catalog order, so every skill is answered
    exactly once; fields come from ``fields`` and values are bounded. Each
    skill needs at least one effect.
    """

    if not skills:
        raise ValueError("skills cannot be empty.")
    pydantic = _pydantic()
    requirement, effect = _item_types(fields, max_value)
    config = pydantic.ConfigDict(extra="forbid")
    mechanics = pydantic.create_model(
        "ModelMechanics",
        __config__=config,
        requires=(list[requirement], pydantic.Field(max_length=max_items)),
        effects=(
            list[effect],
            pydantic.Field(min_length=1, max_length=max_items),
        ),
    )
    return pydantic.create_model(
        "InducedSkillModel",
        __config__=config,
        **{skill_id.value: (mechanics, ...) for skill_id in skills},
    )


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _parse_condition(
    item: Any,
    fields: Sequence[str],
    key: str,
    allowed: tuple[str, ...],
) -> tuple[str, str, int] | str:
    """Return ``(field, kind, value)`` from one item, or why it is invalid."""

    if not isinstance(item, dict):
        return "an entry was not an object"
    field = item.get("field")
    if not isinstance(field, str) or field not in fields:
        return f"unknown field {field!r}"
    kind = item.get(key)
    if kind not in allowed:
        return f"{field}: unknown {key} {kind!r}"
    value = _integer(item.get("value"))
    if value is None or value < 0:
        return f"{field}: value {item.get('value')!r} is not a whole number"
    return field, str(kind), value


def _parse_mechanics(
    entry: Any,
    fields: Sequence[str],
) -> tuple[SkillMechanics, list[str]]:
    problems: list[str] = []
    requires: list[StateRequirement] = []
    effects: list[StateEffect] = []
    if not isinstance(entry, dict):
        return SkillMechanics((), ()), ["entry was not an object"]
    for item in entry.get("requires") or []:
        parsed = _parse_condition(
            item, fields, "comparison", ("at_least", "less_than", "equals")
        )
        if isinstance(parsed, str):
            problems.append(parsed)
            continue
        field, comparison, value = parsed
        if any(
            (existing.field, existing.comparison) == (field, comparison)
            for existing in requires
        ):
            problems.append(f"{field}: repeated {comparison} requirement")
            continue
        requires.append(StateRequirement(field, comparison, value))  # pyright: ignore[reportArgumentType]
    for item in entry.get("effects") or []:
        parsed = _parse_condition(
            item, fields, "operation", ("increase", "decrease", "set")
        )
        if isinstance(parsed, str):
            problems.append(parsed)
            continue
        field, operation, value = parsed
        if any(existing.field == field for existing in effects):
            problems.append(f"{field}: repeated effect")
            continue
        effects.append(StateEffect(field, operation, value))  # pyright: ignore[reportArgumentType]
    return SkillMechanics(tuple(requires), tuple(effects)), problems


@dataclass(frozen=True)
class InducedModel:
    """A skill model parsed from a VLM answer, with what was wrong with it.

    ``problems`` lists the entries the parser dropped, ``report`` the
    validation result, ``dropped`` the skills removed for failing it,
    ``repaired`` whether the answer came from a repair round, and
    ``questions`` the focused repairs asked after it.
    """

    model: SkillModel
    raw_response: str
    valid: bool
    error: str | None = None
    problems: tuple[str, ...] = ()
    report: ValidationReport = ValidationReport()
    dropped: tuple[SkillId, ...] = ()
    repaired: bool = False
    questions: tuple[RepairQuestion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": model_to_dict(self.model),
            "raw_response": self.raw_response,
            "valid": self.valid,
            "error": self.error,
            "problems": list(self.problems),
            "validation": self.report.to_dict(),
            "dropped": [skill_id.value for skill_id in self.dropped],
            "repaired": self.repaired,
            "repair_questions": [question.to_dict() for question in self.questions],
        }


def parse_induced_model(
    text: str,
    skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    fields: Sequence[str] = INDUCTION_FIELDS,
) -> InducedModel:
    """Parse an induction answer into a model, without trusting its entries.

    An entry with an unknown field, comparison or operation, a value that is
    not a whole number, or a repeat of an earlier one is dropped and listed
    in ``problems``; so is a missing skill. Each parsed skill gets its
    achievement unlock as in :func:`~tiny_sima.planner.symbolic.
    registry_model`. The answer is invalid only when it is not a JSON object
    or answers no skill.
    """

    payload = _json_object(text)
    if payload is None:
        return InducedModel((), text, valid=False, error="response was not JSON")
    entries: list[tuple[SkillId, SkillMechanics]] = []
    problems: list[str] = []
    for skill_id in skills:
        if skill_id.value not in payload:
            problems.append(f"{skill_id.value}: missing")
            continue
        mechanics, skill_problems = _parse_mechanics(payload[skill_id.value], fields)
        problems.extend(f"{skill_id.value}: {problem}" for problem in skill_problems)
        entries.append((skill_id, _finish(skill_id, mechanics)))
    if not entries:
        return InducedModel(
            (),
            text,
            valid=False,
            error="no skill was answered",
            problems=tuple(problems),
        )
    return InducedModel(tuple(entries), text, valid=True, problems=tuple(problems))


def _induction_request(
    system_prompt: str,
    user_prompt: str,
    skills: tuple[SkillId, ...],
    fields: tuple[str, ...],
) -> VLMRequest:
    return VLMRequest(
        image=None,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        candidate_skills=skills,
        output_type=induction_output_type(skills, fields),
    )


def build_repair_prompt(
    induced: InducedModel,
    report: ValidationReport,
    skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    fields: Sequence[str] = INDUCTION_FIELDS,
    descriptions: Mapping[SkillId, str] | None = None,
    start: Mapping[str, int] | None = None,
) -> str:
    """Return the induction prompt with the previous model and its violations."""

    problems = [str(violation) for violation in report.violations]
    problems.extend(induced.problems)
    prompt = build_induction_prompt(
        skills, fields=fields, descriptions=descriptions, start=start
    )
    return (
        f"{prompt}"
        "\n\nYour previous model was:\n"
        f"{json.dumps(model_to_dict(induced.model), separators=(',', ':'))}\n"
        "A planner check found these problems:\n"
        + "\n".join(f"- {problem}" for problem in problems)
        + "\nWhere no skill changes a field a requirement needs, add that "
        "effect to the skill that brings it about, or drop the requirement if "
        "the skill does not need it. Answer with the whole model again, with "
        "the problems fixed."
    )


def _validated(
    induced: InducedModel,
    start: Mapping[str, int] | None,
) -> InducedModel:
    return dataclasses.replace(
        induced, report=validate_model(induced.model, start=start)
    )


def induce_models(
    backend: VisionLanguageBackend,
    skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    manual: str | None = None,
    fields: Sequence[str] = INDUCTION_FIELDS,
    descriptions: Mapping[SkillId, str] | None = None,
    repeats: int = 1,
    start: Mapping[str, int] | None = None,
) -> list[InducedModel]:
    """Ask the VLM for the model ``repeats`` times and validate each answer.

    The request is text only. Repeats differ only when the backend samples;
    under greedy decoding one is enough.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive.")
    skill_tuple, field_tuple = tuple(skills), tuple(fields)
    request = _induction_request(
        build_induction_system_prompt(manual if manual is not None else load_manual()),
        build_induction_prompt(
            skill_tuple, fields=field_tuple, descriptions=descriptions, start=start
        ),
        skill_tuple,
        field_tuple,
    )
    return [
        _validated(parse_induced_model(text, skill_tuple, fields=field_tuple), start)
        for text in generate_all(backend, [request] * repeats)
    ]


def _rank(induced: InducedModel) -> tuple[int, int, int]:
    return (
        int(not induced.valid),
        len(induced.report.violations),
        len(induced.problems),
    )


# ---------------------------------------------------------------------------
# Focused repair

_QUESTION_INSTRUCTIONS = (
    "You are fixing the action model a symbolic planner uses to sequence the "
    "skills of a Craftax-Classic agent, one question at a time. Answer from "
    "the game manual with one of the answers offered."
)
_NONE = "none"
_TOOLS = ("pickaxe", "sword")


@dataclass(frozen=True)
class RepairQuestion:
    """One focused repair: the question, the VLM's answer and the change made.

    ``kind`` is ``producer`` (which skill brings about a requirement no skill
    produces) or ``needless`` (whether a skill needs a requirement that keeps
    others blocked). ``change`` is empty when nothing was applied.
    """

    kind: str
    question: str
    options: tuple[str, ...]
    answer: str | None
    change: str
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "question": self.question,
            "options": list(self.options),
            "answer": self.answer,
            "change": self.change,
            "raw_response": self.raw_response,
        }


@lru_cache(maxsize=32)
def question_output_type(options: tuple[str, ...]) -> Any:
    """Create the constrained type for one answer: one of ``options``.

    No reason comes first: on the 2B model a reason changed no answer, and a
    cut-off one could lead it to the wrong one.
    """

    if not options:
        raise ValueError("options cannot be empty.")
    pydantic = _pydantic()
    answer = Enum(
        "RepairOption",
        {f"option_{index}": option for index, option in enumerate(options)},
        type=str,
    )
    return pydantic.create_model(
        "RepairAnswer",
        __config__=pydantic.ConfigDict(extra="forbid"),
        answer=(answer, ...),
    )


def _requirement_phrase(requirement: StateRequirement) -> str:
    """Say what holding ``requirement`` means, as a player would put it."""

    field, value = requirement.field, int(requirement.value)
    if requirement.comparison != "less_than" and value >= 1:
        if field in INVENTORY_FIELDS:
            item = field.split(".", 1)[1].replace("_", " ")
            if value > 1 and requirement.comparison == "at_least":
                return f"hold at least {value} {item}"
            if value == 1:
                return f"hold {'a ' if item.endswith(_TOOLS) else ''}{item}"
        if field.startswith("world.") and field.endswith("_reachable"):
            block = field[len("world.") : -len("_reachable")].replace("_", " ")
            thing = (
                "a free tile to build on" if block == "placement space" else f"a {block}"
            )
            return f"have {thing} within reach"
    return f"meet {_requirement_text(requirement)}"


def _with_producer(
    mechanics: SkillMechanics,
    requirement: StateRequirement,
) -> SkillMechanics:
    # A count is gained a unit at a time, which correction then fixes; a
    # world fact is made to hold. What a skill brings about it cannot need.
    field = requirement.field
    effect = (
        StateEffect(field, "increase", 1)
        if field in INVENTORY_FIELDS and requirement.comparison == "at_least"
        else StateEffect(field, "set", int(requirement.value))
    )
    return dataclasses.replace(
        mechanics,
        requires=tuple(item for item in mechanics.requires if item.field != field),
        effects=(
            *(item for item in mechanics.effects if item.field != field),
            effect,
        ),
    )


def _without_requirement(
    mechanics: SkillMechanics,
    requirement: StateRequirement,
) -> SkillMechanics:
    return dataclasses.replace(
        mechanics,
        requires=tuple(item for item in mechanics.requires if item != requirement),
    )


def _replaced(
    model: SkillModel, skill_id: SkillId, mechanics: SkillMechanics
) -> SkillModel:
    return tuple(
        (entry_id, mechanics if entry_id is skill_id else entry)
        for entry_id, entry in model
    )


# kind, question, options, and the model and change each applying answer gives.
_Question = tuple[str, str, tuple[str, ...], dict[str, tuple[SkillModel, str]]]


def _producer_questions(
    model: SkillModel,
    values: Mapping[str, int],
    exempt_prefixes: tuple[str, ...],
) -> Iterable[_Question]:
    """Yield a question per requirement unmet at the start that nothing produces."""

    seen: set[StateRequirement] = set()
    for _, mechanics in model:
        for requirement in mechanics.requires:
            field = requirement.field
            value = values.get(field, 0)
            if (
                requirement in seen
                or requirement.comparison == "less_than"
                or field.startswith(exempt_prefixes)
                or _holds(requirement, {field: value})
                or any(
                    _moves_toward(effect, requirement, value)
                    for _, other in model
                    for effect in other.effects
                )
            ):
                continue
            seen.add(requirement)
            outcomes = {
                skill_id.value: (
                    _replaced(
                        model,
                        skill_id,
                        _finish(
                            skill_id,
                            _without_achievements(_with_producer(entry, requirement)),
                        ),
                    ),
                    f"{skill_id.value} now brings about "
                    f"{_requirement_text(requirement)}",
                )
                for skill_id, entry in model
            }
            # "Which skill brings it about?" drew none from the 2B model, with
            # place_table named in its own reason; this form drew place_table.
            yield (
                "producer",
                f"{field} is {_field_description(field)}. It is {value} at the "
                "start of every episode. Which skill makes it "
                f"{int(requirement.value)}?",
                (*outcomes, _NONE),
                outcomes,
            )


def _needless_questions(
    model: SkillModel,
    report: ValidationReport,
    *,
    start: Mapping[str, int] | None,
    exempt_prefixes: tuple[str, ...],
    descriptions: Mapping[SkillId, str] | None,
) -> list[_Question]:
    """Ask, per requirement that keeps skills blocked, whether it is needed.

    The candidates are the requirements no plan reaches even alone whose
    removal clears a violation, and of those, the ones that clear another
    skill's too when there are any: they break a deadlock rather than free
    one skill from it. Asked to pick which of two deadlocked requirements is
    wrong, the 2B model picked the same wrong one in either order; asked of
    each, as here, it answered both right.
    """

    violations = len(report.violations)
    found: list[tuple[int, _Question]] = []
    asked: set[tuple[SkillId, StateRequirement]] = set()
    for violation in report.violations:
        skill_id = violation.skill_id
        mechanics = dict(model)[skill_id]
        for requirement in violation.blocking:
            if (skill_id, requirement) in asked:
                continue
            asked.add((skill_id, requirement))
            relaxed = _replaced(
                model, skill_id, _without_requirement(mechanics, requirement)
            )
            cleared = violations - len(
                validate_model(
                    relaxed, start=start, exempt_prefixes=exempt_prefixes
                ).violations
            )
            if cleared <= 0:
                continue
            behaviour = (
                (descriptions or {}).get(skill_id)
                or get_skill_spec(skill_id).instruction
            ).rstrip(".")
            change = f"dropped: {skill_id.value} needs {_requirement_text(requirement)}"
            found.append(
                (
                    cleared,
                    (
                        "needless",
                        f"{behaviour}: must the player "
                        f"{_requirement_phrase(requirement)} first?",
                        ("yes", "no"),
                        {"no": (relaxed, change)},
                    ),
                )
            )
    least = 2 if any(cleared > 1 for cleared, _ in found) else 1
    return [question for cleared, question in found if cleared >= least]


def repair_model(
    backend: VisionLanguageBackend,
    model: SkillModel,
    *,
    manual: str | None = None,
    descriptions: Mapping[SkillId, str] | None = None,
    start: Mapping[str, int] | None = None,
    max_questions: int = DEFAULT_REPAIR_QUESTIONS,
    exempt_prefixes: tuple[str, ...] = ("survival.",),
) -> tuple[SkillModel, tuple[RepairQuestion, ...]]:
    """Repair unreachable requirements with narrow, fixed-answer questions.

    The 2B model asked for the whole model again repeats it. Each question
    here names one problem, and its answer is applied directly:

    - **Producer:** a requirement unmet at the start that no skill moves
      toward, such as a crafting table every craft needs and none places.
      The chosen skill gains the effect and loses its own requirement on
      that field.
    - **Needless:** otherwise, whether a skill needs a requirement that
      keeps skills blocked, such as a wood pickaxe needing stone while stone
      needs that pickaxe. One answered no is dropped; correction can add an
      inventory requirement back from runs.

    A question answered none or yes is not asked again. At most
    ``max_questions`` go out, one at a time, each against the model the
    last one left.
    """

    values = dict(EPISODE_START)
    values.update(start or {})
    manual_text = manual if manual is not None else load_manual()
    system_prompt = f"{_QUESTION_INSTRUCTIONS}\n\nGame manual:\n{manual_text.strip()}"
    declined: set[str] = set()
    asked: list[RepairQuestion] = []
    while len(asked) < max_questions:
        report = validate_model(model, start=start, exempt_prefixes=exempt_prefixes)
        if report.valid:
            break
        candidates = [
            *_producer_questions(model, values, exempt_prefixes),
            *_needless_questions(
                model,
                report,
                start=start,
                exempt_prefixes=exempt_prefixes,
                descriptions=descriptions,
            ),
        ]
        question = next(
            (candidate for candidate in candidates if candidate[1] not in declined),
            None,
        )
        if question is None:
            break
        kind, text, options, outcomes = question
        (raw,) = generate_all(
            backend,
            [
                VLMRequest(
                    image=None,
                    system_prompt=system_prompt,
                    user_prompt=text,
                    candidate_skills=tuple(skill_id for skill_id, _ in model),
                    output_type=question_output_type(options),
                )
            ],
        )
        payload = _json_object(raw)
        answer = payload.get("answer") if payload is not None else None
        answer = answer if answer in options else None
        change = ""
        if answer in outcomes:
            model, change = outcomes[answer]
        else:
            declined.add(text)
        asked.append(RepairQuestion(kind, text, options, answer, change, raw))
    return model, tuple(asked)


def induce_model(
    backend: VisionLanguageBackend,
    skills: Sequence[SkillId] = ALL_SKILL_IDS,
    *,
    manual: str | None = None,
    fields: Sequence[str] = INDUCTION_FIELDS,
    descriptions: Mapping[SkillId, str] | None = None,
    repeats: int = 1,
    repair: bool = False,
    drop_invalid: bool = False,
    start: Mapping[str, int] | None = None,
) -> InducedModel:
    """Induce one model, then optionally repair it and drop what fails.

    Of ``repeats`` answers, the valid one with the fewest violations and then
    the fewest parse problems is kept, the earliest on a tie. With ``repair``,
    a model that fails validation goes back to the VLM once with its
    violations, and the answer replaces it if it ranks no worse; whatever
    still fails then goes through :func:`repair_model`'s questions. With
    ``drop_invalid``, skills still failing are removed, see
    :func:`drop_invalid_skills`.
    """

    manual_text = manual if manual is not None else load_manual()
    candidates = induce_models(
        backend,
        skills,
        manual=manual_text,
        fields=fields,
        descriptions=descriptions,
        repeats=repeats,
        start=start,
    )
    best = min(candidates, key=_rank)
    skill_tuple, field_tuple = tuple(skills), tuple(fields)
    if repair and best.valid and not best.report.valid:
        request = _induction_request(
            build_induction_system_prompt(manual_text),
            build_repair_prompt(
                best,
                best.report,
                skill_tuple,
                fields=field_tuple,
                descriptions=descriptions,
                start=start,
            ),
            skill_tuple,
            field_tuple,
        )
        (text,) = generate_all(backend, [request])
        repaired = dataclasses.replace(
            _validated(
                parse_induced_model(text, skill_tuple, fields=field_tuple), start
            ),
            repaired=True,
        )
        if _rank(repaired) <= _rank(best):
            best = repaired
        if not best.report.valid:
            model, questions = repair_model(
                backend,
                best.model,
                manual=manual_text,
                descriptions=descriptions,
                start=start,
            )
            best = dataclasses.replace(
                best,
                model=model,
                questions=questions,
                report=validate_model(model, start=start),
            )
    if drop_invalid and best.valid and not best.report.valid:
        model, dropped = drop_invalid_skills(best.model, start=start)
        best = dataclasses.replace(
            best,
            model=model,
            dropped=dropped,
            report=validate_model(model, start=start),
        )
    return best


# ---------------------------------------------------------------------------
# Correction from successful transitions


def _values(readings: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Readings as a dict; an inventory count left out of them is zero."""

    values = dict.fromkeys(INVENTORY_FIELDS, 0)
    values.update(readings)
    return values


def observed_effects(record: TransitionRecord) -> dict[str, tuple[str, int]]:
    """Return what one transition shows the skill did to each observed field.

    A field is ``("decrease", c)`` when the skill spent ``c`` of it, which
    the extremes reveal even when the count ends where it started: the
    amount spent is at least the fall from the highest count to the end, and
    at least the fall from the start to the lowest. A gain on the way, such
    as a table skill gathering the wood it then spends, is part of affording
    the skill, not its effect. Otherwise a rise is ``("increase", g)``.
    ``survival.drink`` only ever counts a rise, since it drains with time.
    Unchanged fields are left out.
    """

    before, after = _values(record.before), _values(record.after)
    extremes = {
        field: (low, high) for field, low, high in record.inventory_extremes or ()
    }
    effects: dict[str, tuple[str, int]] = {}
    for field in INVENTORY_FIELDS:
        start, end = before[field], after[field]
        low, high = extremes.get(field, (start, end))
        low, high = min(low, start, end), max(high, start, end)
        spent = max(high - end, start - low)
        if spent > 0:
            effects[field] = ("decrease", spent)
        elif end > start:
            effects[field] = ("increase", end - start)
    drink_gain = after.get("survival.drink", 0) - before.get("survival.drink", 0)
    if drink_gain > 0:
        effects["survival.drink"] = ("increase", drink_gain)
    return effects


def _mode(
    observations: Sequence[tuple[str, int] | None],
    prior: tuple[str, int] | None,
) -> tuple[str, int] | None:
    """The most common observation; ties go to the prior, then to no change."""

    counts = Counter(observations)
    top = max(counts.values())
    tied = [value for value, count in counts.items() if count == top]
    if len(tied) == 1:
        return tied[0]
    if prior in tied:
        return prior
    if None in tied:
        return None
    return min(value for value in tied if value is not None)


@dataclass(frozen=True)
class ModelCorrection:
    """A corrected model, its changes from the input, and its evidence.

    ``support`` counts the successful transitions behind each skill; a skill
    without any is left as it was.
    """

    model: SkillModel
    changes: tuple[ModelChange, ...]
    support: tuple[tuple[SkillId, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": model_to_dict(self.model),
            "changes": [change.to_dict() for change in self.changes],
            "support": {skill_id.value: count for skill_id, count in self.support},
        }


def _correct_skill(
    skill_id: SkillId,
    mechanics: SkillMechanics,
    successes: Sequence[TransitionRecord],
    min_support: int,
) -> SkillMechanics:
    prior_effects = {
        effect.field: (effect.operation, int(effect.value))
        for effect in mechanics.effects
        if effect.field in OBSERVED_FIELDS
    }
    observations = [observed_effects(record) for record in successes]
    observed_fields = set(prior_effects).union(*observations)
    learned: dict[str, tuple[str, int]] = {}
    for field in OBSERVED_FIELDS:
        if field not in observed_fields:
            continue
        mode = _mode(
            [observation.get(field) for observation in observations],
            prior_effects.get(field),
        )
        if mode is not None:
            learned[field] = mode

    # Effects: the prior's unobservable ones stay, observed ones are replaced
    # in place, and newly observed ones follow, in field order.
    effects: list[StateEffect] = []
    for effect in mechanics.effects:
        if _is_achievement(effect.field):
            continue
        if effect.field not in OBSERVED_FIELDS:
            effects.append(effect)
        elif effect.field in learned:
            operation, value = learned.pop(effect.field)
            effects.append(StateEffect(effect.field, operation, value))  # pyright: ignore[reportArgumentType]
    consumed = {
        field: value
        for field, (operation, value) in (
            *(
                (effect.field, (effect.operation, int(effect.value)))
                for effect in effects
            ),
            *learned.items(),
        )
        if operation == "decrease" and field in INVENTORY_FIELDS
    }
    effects.extend(
        StateEffect(field, operation, value)  # pyright: ignore[reportArgumentType]
        for field, (operation, value) in learned.items()
    )

    # Requirements: a consumed amount must be held; any other count held in
    # every success is required at its minimum.
    pre_states = [_values(record.before) for record in successes]
    minimum = {
        field: min(values[field] for values in pre_states) for field in INVENTORY_FIELDS
    }

    def inferred(field: str, prior: int | None) -> int | None:
        if field in consumed:
            return consumed[field]
        if minimum[field] <= 0:
            return None
        if prior is not None:
            return min(prior, minimum[field])
        return minimum[field] if len(successes) >= min_support else None

    requires: list[StateRequirement] = []
    handled: set[str] = set()
    for requirement in mechanics.requires:
        field = requirement.field
        if field not in INVENTORY_FIELDS:
            requires.append(requirement)
        elif requirement.comparison != "at_least":
            # A bound a success broke is refuted; one none broke stays.
            if all(_holds(requirement, values) for values in pre_states):
                requires.append(requirement)
        elif field not in handled:
            handled.add(field)
            value = inferred(field, int(requirement.value))
            if value is not None:
                requires.append(StateRequirement(field, "at_least", value))
    for field in INVENTORY_FIELDS:
        if field in handled:
            continue
        value = inferred(field, None)
        if value is not None:
            requires.append(StateRequirement(field, "at_least", value))
    return _finish(
        skill_id,
        SkillMechanics(tuple(requires), tuple(effects), mechanics.repeatable),
    )


def correct_model(
    model: SkillModel,
    transitions: Iterable[TransitionRecord],
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> ModelCorrection:
    """Correct ``model`` from successful transitions, SCALAR-style.

    For each skill with at least one success:

    - **Effects** on inventory counts and drink are each field's most common
      :func:`observed_effects` value across the successes, ties going to the
      model's own effect, then to no change. Effects on fields a transition
      does not record, such as a table appearing, are kept from the model.
    - **Requirements:** a field the skill decreases by ``c`` requires
      ``at_least c``. Any other inventory count that was positive before
      every success requires ``at_least`` its minimum; the model's own
      ``at_least`` is kept at the smaller of the two, and a new one needs
      ``min_support`` successes. An ``at_least`` some success went without is
      dropped, and a ``less_than`` or ``equals`` some success broke. World
      and drink requirements are kept, since inventory readings cannot
      confirm or refute them.

    Known failure modes. A count that is positive before every success by
    the plan's structure rather than by need becomes a requirement: on the
    stone-pickaxe chain the stone pickaxe is always crafted with the wood
    pickaxe held, and nothing short of a run that crafts it without one will
    remove the requirement. When a count is positive by chance with
    probability ``p`` per success, it survives ``K`` successes with
    probability ``p**K``, so ``K >= log(0.05) / log(p)`` (six at ``p = 0.6``)
    keeps the chance below five percent; ``min_support`` is that ``K`` for
    requirements the model did not already have. A skill the executor never
    completes is never corrected, and a model that vetoes the skill (too high
    a cost) or sends it in too early (too low a cost) yields successes only if
    the planner still reaches it; :func:`analyse_failures` covers the second
    case. Mid-skill extremes give a lower bound on what was spent, so an
    executor that spends and regathers twice within one skill is
    under-counted.
    """

    records = list(transitions)
    corrected: list[tuple[SkillId, SkillMechanics]] = []
    support: list[tuple[SkillId, int]] = []
    for skill_id, mechanics in model:
        successes = [
            record
            for record in records
            if record.skill_id is skill_id and record.outcome == "success"
        ]
        support.append((skill_id, len(successes)))
        corrected.append(
            (
                skill_id,
                _correct_skill(skill_id, mechanics, successes, min_support)
                if successes
                else mechanics,
            )
        )
    new_model = tuple(corrected)
    return ModelCorrection(new_model, model_diff(model, new_model), tuple(support))


# ---------------------------------------------------------------------------
# Failure analysis


def _observable(requirement: StateRequirement) -> bool:
    return requirement.field in OBSERVED_FIELDS


def predicted_success(mechanics: SkillMechanics, record: TransitionRecord) -> bool:
    """Return whether the pre-state met every requirement it records.

    World requirements are not in a transition's readings and count as met.
    """

    values = _values(record.before)
    return all(
        _holds(requirement, values)
        for requirement in mechanics.requires
        if _observable(requirement)
    )


def unexplained_failures(
    model: SkillModel,
    transitions: Iterable[TransitionRecord],
    *,
    causes: Sequence[str] = DEFAULT_FAILURE_CAUSES,
) -> dict[SkillId, tuple[TransitionRecord, ...]]:
    """Return the failures the model predicted to succeed, per skill.

    Only failures with one of ``causes`` count: a target never seen or a
    timeout says nothing about a missing requirement.
    """

    mechanics = dict(model)
    found: dict[SkillId, list[TransitionRecord]] = {}
    for record in transitions:
        if (
            record.outcome != "success"
            and record.cause in causes
            and record.skill_id in mechanics
            and predicted_success(mechanics[record.skill_id], record)
        ):
            found.setdefault(record.skill_id, []).append(record)
    return {skill_id: tuple(records) for skill_id, records in found.items()}


def gain_fields(model: SkillModel) -> tuple[str, ...]:
    """Return the fields some skill increases or sets, in planner order."""

    gained = {
        effect.field
        for _, mechanics in model
        for effect in mechanics.effects
        if effect.operation in ("increase", "set") and not _is_achievement(effect.field)
    }
    return tuple(field for field in INDUCTION_FIELDS if field in gained)


@lru_cache(maxsize=16)
def failure_output_type(
    fields: tuple[str, ...],
    max_value: int = MAX_INDUCED_VALUE,
) -> Any:
    """Create the constrained type for one failure-analysis answer.

    ``analysis`` comes first, so the model compares the states before it
    commits; ``missing_requirements`` holds at most one requirement.
    """

    if not fields:
        raise ValueError("fields cannot be empty.")
    pydantic = _pydantic()
    requirement, _ = _item_types(fields, max_value)
    return pydantic.create_model(
        "FailureAnalysisOutput",
        __config__=pydantic.ConfigDict(extra="forbid"),
        analysis=(str, pydantic.Field(max_length=_ANALYSIS_MAX_LENGTH)),
        missing_requirements=(
            list[requirement],
            pydantic.Field(min_length=0, max_length=1),
        ),
    )


_FAILURE_INSTRUCTIONS = (
    "You are correcting the action model a symbolic planner uses for a "
    "Craftax-Classic agent. A skill failed in states where the model said it "
    "would succeed, so the model is probably missing a requirement. Compare "
    "the states before the failures with the states before successes, and "
    "name at most one missing requirement over the fields listed, as {field, "
    "comparison, value}. Leave the list empty if no single requirement "
    "explains the failures."
)


def _render_values(record: TransitionRecord) -> str:
    return (
        ", ".join(f"{field.split('.', 1)[1]}={value}" for field, value in record.before)
        or "nothing held"
    )


def build_failure_prompt(
    skill_id: SkillId,
    mechanics: SkillMechanics,
    failures: Sequence[TransitionRecord],
    successes: Sequence[TransitionRecord],
    fields: Sequence[str],
    *,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> str:
    """Return the user prompt for one skill's unexplained failures."""

    requires = (
        ", ".join(
            f"{requirement.field} {_render_requirement(requirement)}"
            for requirement in mechanics.requires
        )
        or "none"
    )
    effects = (
        ", ".join(
            f"{effect.field} {_render_effect(effect)}"
            for effect in mechanics.effects
            if not _is_achievement(effect.field)
        )
        or "none"
    )
    failed_lines = "\n".join(
        f"- {_render_values(record)} "
        f"(failed: {(record.cause or record.outcome).replace('_', ' ')})"
        for record in failures[:max_examples]
    )
    success_lines = (
        "\n".join(f"- {_render_values(record)}" for record in successes[:max_examples])
        or "- none recorded"
    )
    return (
        f"Skill: {skill_id.value} ({get_skill_spec(skill_id).instruction})\n"
        f"Current requirements: {requires}\n"
        f"Current effects: {effects}\n"
        f"Inventory before failures the model did not expect:\n{failed_lines}\n"
        f"Inventory before successes:\n{success_lines}\n"
        f"Fields: {', '.join(fields)}"
    )


def parse_failure_response(
    text: str,
    fields: Sequence[str],
) -> tuple[StateRequirement | None, str | None]:
    """Return the one proposed requirement, or ``None`` and an error if any."""

    payload = _json_object(text)
    if payload is None:
        return None, "response was not JSON"
    items = payload.get("missing_requirements")
    if not isinstance(items, list):
        return None, "response had no missing_requirements list"
    if not items:
        return None, None
    parsed = _parse_condition(
        items[0], fields, "comparison", ("at_least", "less_than", "equals")
    )
    if isinstance(parsed, str):
        return None, parsed
    field, comparison, value = parsed
    return StateRequirement(field, comparison, value), None  # pyright: ignore[reportArgumentType]


@dataclass(frozen=True)
class FailureAnalysis:
    """One skill's failure analysis: the proposal, and whether it was applied.

    ``reason`` says why a proposal was rejected, or is empty when applied.
    """

    skill_id: SkillId
    failures: int
    requirement: StateRequirement | None
    applied: bool
    reason: str
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill_id.value,
            "failures": self.failures,
            "requirement": (
                None if self.requirement is None else self.requirement.to_dict()
            ),
            "applied": self.applied,
            "reason": self.reason,
            "raw_response": self.raw_response,
        }


def _with_requirement(
    mechanics: SkillMechanics,
    requirement: StateRequirement,
) -> SkillMechanics:
    kept = tuple(
        existing
        for existing in mechanics.requires
        if (existing.field, existing.comparison)
        != (requirement.field, requirement.comparison)
    )
    return dataclasses.replace(mechanics, requires=(*kept, requirement))


def _judge(
    requirement: StateRequirement | None,
    error: str | None,
    mechanics: SkillMechanics,
    failures: Sequence[TransitionRecord],
    successes: Sequence[TransitionRecord],
) -> str:
    """Return why a proposal is rejected, or an empty string to apply it."""

    if requirement is None:
        return error or "no requirement proposed"
    if requirement in mechanics.requires:
        return "already required"
    if not _observable(requirement):
        # A world fact is not in the readings, so the evidence cannot judge it.
        return ""
    if any(not _holds(requirement, _values(record.before)) for record in successes):
        return "a success went without it"
    if all(_holds(requirement, _values(record.before)) for record in failures):
        return "it holds before every failure, so it explains none"
    return ""


def analyse_failures(
    backend: VisionLanguageBackend,
    model: SkillModel,
    transitions: Iterable[TransitionRecord],
    *,
    causes: Sequence[str] = DEFAULT_FAILURE_CAUSES,
    fields: Sequence[str] | None = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> tuple[SkillModel, tuple[FailureAnalysis, ...]]:
    """Ask the VLM for one missing requirement per skill with unexplained failures.

    The candidates are :func:`unexplained_failures`. Fields default to
    :func:`gain_fields`, since a requirement no skill can produce would make
    the skill unusable. A proposal is applied unless a successful pre-state
    breaks it or every failed pre-state already met it; a world requirement
    is not in the readings and is applied unchecked. An applied requirement
    replaces the skill's own on the same field and comparison. Every skill's
    request goes out in one batch.
    """

    records = list(transitions)
    field_tuple = tuple(fields) if fields is not None else gain_fields(model)
    unexplained = unexplained_failures(model, records, causes=causes)
    if not unexplained or not field_tuple:
        return model, ()
    mechanics_by_skill = dict(model)
    system_prompt = _FAILURE_INSTRUCTIONS
    skills = [skill_id for skill_id, _ in model if skill_id in unexplained]
    successes = {
        skill_id: [
            record
            for record in records
            if record.skill_id is skill_id and record.outcome == "success"
        ]
        for skill_id in skills
    }
    requests = [
        VLMRequest(
            image=None,
            system_prompt=system_prompt,
            user_prompt=build_failure_prompt(
                skill_id,
                mechanics_by_skill[skill_id],
                unexplained[skill_id],
                successes[skill_id],
                field_tuple,
                max_examples=max_examples,
            ),
            candidate_skills=(skill_id,),
            output_type=failure_output_type(field_tuple),
        )
        for skill_id in skills
    ]
    analyses: list[FailureAnalysis] = []
    for skill_id, text in zip(skills, generate_all(backend, requests), strict=True):
        requirement, error = parse_failure_response(text, field_tuple)
        mechanics = mechanics_by_skill[skill_id]
        reason = _judge(
            requirement, error, mechanics, unexplained[skill_id], successes[skill_id]
        )
        if not reason and requirement is not None:
            mechanics_by_skill[skill_id] = _with_requirement(mechanics, requirement)
        analyses.append(
            FailureAnalysis(
                skill_id=skill_id,
                failures=len(unexplained[skill_id]),
                requirement=requirement,
                applied=not reason,
                reason=reason,
                raw_response=text,
            )
        )
    new_model = tuple((skill_id, mechanics_by_skill[skill_id]) for skill_id, _ in model)
    return new_model, tuple(analyses)
