"""Frozen-VLM sub-goal planning over a symbolic skill planner.

The VLM reads the objective, the frame, the inventory and the episode memory
and proposes an ordered list of formal sub-goals. A symbolic planner checks
each one against the skill model and chooses every skill as a first step of
the shortest plan to the current sub-goal. A sub-goal the planner cannot
reach goes back to the VLM with the reason, in the LLM-Modulo pattern, so
the VLM never has to sequence skills itself.

The VLM is called only when the plan needs it: at the first decision, once
every sub-goal is reached but the objective is not, after a failed skill,
when a sub-goal uses up its skill budget, when a vital falls to its
threshold, when a hostile creature comes close, at nightfall, and when the
current sub-goal stops being reachable. Each alert (a low vital, a threat,
the night) asks once per spell rather than at every decision. Between calls
the planner follows the plan alone.

The goal source is pluggable: a :class:`SubgoalSource` such as a scripted
goal selector can stand in for the VLM, and its answers go through the same
parsing, checking and rejection feedback.

The planner never sees the chain's formal goal. Turning the objective into
sub-goals is the VLM's whole job, and the closed loop still judges success
with the chain's own completion test.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from hashlib import sha256
from typing import Any, Protocol, runtime_checkable

import numpy as np

from tiny_sima.config import VLMPlannerConfig
from tiny_sima.envs.craftax import render_state
from tiny_sima.goals import (
    ACHIEVEMENT_FIELDS,
    ALL_SKILL_IDS,
    SKILL_ACHIEVEMENTS,
    SkillId,
    StateRequirement,
    achievement_field,
)
from tiny_sima.planner.observation import (
    ObservationConfig,
    mobs_in_view,
    observation_block,
    observation_system_text,
    vitals,
)
from tiny_sima.planner.saycan import (
    BatchedFeasibilityScorer,
    FeasibilityScorer,
    PlannerContext,
    PlannerDecision,
    PlannerSelection,
    PlannerSelectionStatus,
    SkillScore,
    catalog_feasibility,
    reachability_feasibility,
)
from tiny_sima.planner.symbolic import (
    DEFAULT_MAX_PLAN_LENGTH,
    GoalCondition,
    SkillModel,
    check_subgoal,
    read_field,
)
from tiny_sima.planner.vlm import (
    VisionLanguageBackend,
    VLMRequest,
    _json_object,
    _mechanics_line,
    generate_all,
)

# Outcomes the agent can see in its own inventory and status bar. Tools it
# cannot make with the registry stay in the list: proposing one is a mistake
# the symbolic planner should catch and explain.
SUBGOAL_FIELDS: tuple[str, ...] = (
    "inventory.wood",
    "inventory.stone",
    "inventory.coal",
    "inventory.wood_pickaxe",
    "inventory.stone_pickaxe",
    "inventory.iron_pickaxe",
    "inventory.wood_sword",
    "inventory.stone_sword",
    "inventory.iron_sword",
    "survival.drink",
)
# Every achievement as a sub-goal field, including ones no current skill
# unlocks. The open-ended designs use :func:`open_ended_subgoal_fields`.
OPEN_ENDED_SUBGOAL_FIELDS: tuple[str, ...] = SUBGOAL_FIELDS + ACHIEVEMENT_FIELDS
DEFAULT_MAX_SUBGOALS = 4
DEFAULT_MAX_REJECTIONS = 2
DEFAULT_MAX_EVENTS = 8
# Skills a sub-goal may take beyond its plan length before it times out.
DEFAULT_SUBGOAL_SLACK = 4
# The vital readings, as :func:`vitals` names them, that raise an alert at or
# below a threshold. Only drink has a skill that restores it so far.
DEFAULT_VITAL_THRESHOLDS: tuple[tuple[str, int], ...] = (("drink", 3),)
ALL_VITAL_THRESHOLDS: tuple[tuple[str, int], ...] = (
    ("health", 3),
    ("food", 3),
    ("drink", 3),
    ("energy", 3),
)
HOSTILE_MOBS: tuple[str, ...] = ("zombie", "skeleton")
_ASSESSMENT_MAX_LENGTH = 120
_MAX_FIELD_VALUE = 9


class ReplanTrigger(str, Enum):
    """Why the planner asked the VLM for sub-goals at a decision."""

    START = "start"
    SUBGOALS_DONE = "subgoals_done"
    SKILL_FAILED = "skill_failed"
    TIMED_OUT = "timed_out"
    LOW_VITALS = "low_vitals"
    THREAT = "threat"
    NIGHT = "night"
    UNREACHABLE = "unreachable"


_TRIGGER_TEXT = {
    ReplanTrigger.START: "The episode has just started.",
    ReplanTrigger.SUBGOALS_DONE: (
        "Every sub-goal so far has been reached, but the objective is not complete yet."
    ),
    ReplanTrigger.SKILL_FAILED: "The last skill failed",
    ReplanTrigger.TIMED_OUT: (
        "The last sub-goal was not reached within its skill budget"
    ),
    ReplanTrigger.LOW_VITALS: "{vitals} {verb} low.",
    ReplanTrigger.THREAT: "A hostile creature is close",
    ReplanTrigger.NIGHT: "Night has fallen.",
    ReplanTrigger.UNREACHABLE: "The current sub-goal can no longer be reached.",
}


def open_ended_subgoal_fields(
    candidate_skills: Sequence[SkillId] = ALL_SKILL_IDS,
) -> tuple[str, ...]:
    """Return the sub-goal fields plus the achievements the skills unlock.

    Offered every achievement, the 2B model spent its rejection rounds on
    ones no skill produces, such as place_stone and the iron tools, and
    stopped short of ones it could reach. The agent knows its own skills, so
    leaving the rest out reads nothing it does not have.
    """

    unlocked = {
        achievement_field(SKILL_ACHIEVEMENTS[skill])
        for skill in candidate_skills
        if skill in SKILL_ACHIEVEMENTS
    }
    return SUBGOAL_FIELDS + tuple(
        name for name in ACHIEVEMENT_FIELDS if name in unlocked
    )


def _trigger_text(trigger: ReplanTrigger, detail: str = "") -> str:
    """Return why the VLM is asked, with the trigger's detail when it has one."""

    text = _TRIGGER_TEXT[trigger]
    if trigger is ReplanTrigger.LOW_VITALS:
        # "Drink is low." or "Food and drink are low."
        verb = "are" if " and " in detail else "is"
        return text.format(vitals=detail.capitalize(), verb=verb)
    if text.endswith("."):
        return text
    return f"{text}: {detail}." if detail else f"{text}."


def render_subgoal(subgoal: GoalCondition) -> str:
    """Render a sub-goal as the text the VLM and the traces read."""

    symbols = {"at_least": ">=", "less_than": "<", "equals": "=="}
    return " and ".join(
        f"{requirement.field} {symbols.get(requirement.comparison, '?')} "
        f"{requirement.value}"
        for requirement in subgoal
    )


@dataclass(frozen=True)
class SubgoalEvent:
    """One change to the sub-goal plan, for the prompt and the traces."""

    decision_index: int
    subgoal: str
    # "proposed", "achieved", "already_held", "rejected", "abandoned",
    # "timed_out" or "skill_failed".
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "subgoal": self.subgoal,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SubgoalPlan:
    """Per-episode planner state, carried between decisions by the loop."""

    pending: tuple[GoalCondition, ...] = ()
    assessment: str = ""
    events: tuple[SubgoalEvent, ...] = ()
    decision_index: int = 0
    trigger: ReplanTrigger | None = None
    trigger_detail: str = ""
    vlm_calls: int = 0
    rejections: int = 0
    invalid_responses: int = 0
    # Alerts ("drink", "threat", "night", ...) already raised with the VLM.
    # One leaves when its condition clears, so an alert asks once per spell
    # rather than at every decision.
    alerts: tuple[str, ...] = ()
    # Skills spent on the current sub-goal, and how many it may take: its
    # plan length when it became current plus the planner's slack.
    skills_on_current: int = 0
    current_budget: int | None = None
    raw_response: str | None = None

    @property
    def current(self) -> GoalCondition | None:
        return self.pending[0] if self.pending else None

    def advanced(self, status: str, reason: str = "") -> SubgoalPlan:
        """Drop the current sub-goal with an event and reset its budget."""

        if self.current is None:
            return self
        return replace(
            self.with_event(render_subgoal(self.current), status, reason),
            pending=self.pending[1:],
            skills_on_current=0,
            current_budget=None,
        )

    def with_event(self, subgoal: str, status: str, reason: str = "") -> SubgoalPlan:
        return replace(
            self,
            events=(
                *self.events,
                SubgoalEvent(self.decision_index, subgoal, status, reason),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return this decision's view of the plan.

        Events are limited to the ones from this decision, so a trace that
        serialises the plan at every step stays linear in episode length.
        """

        return {
            "decision_index": self.decision_index,
            "pending": [render_subgoal(subgoal) for subgoal in self.pending],
            "assessment": self.assessment,
            "trigger": None if self.trigger is None else self.trigger.value,
            "trigger_detail": self.trigger_detail,
            "events": [
                event.to_dict()
                for event in self.events
                if event.decision_index == self.decision_index
            ],
            "vlm_calls": self.vlm_calls,
            "rejections": self.rejections,
            "invalid_responses": self.invalid_responses,
            "alerts": list(self.alerts),
            "skills_on_current": self.skills_on_current,
            "current_budget": self.current_budget,
        }


@dataclass(frozen=True)
class ParsedSubgoals:
    """Sub-goals extracted from one VLM response."""

    subgoals: tuple[GoalCondition, ...]
    assessment: str
    raw_response: str
    valid: bool
    error: str | None = None


def parse_subgoal_response(
    text: str,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
) -> ParsedSubgoals:
    """Parse a sub-goal response without trusting its fields.

    Only the shape is checked here. Whether a field exists or a value can be
    reached is the symbolic planner's call, so that the VLM hears why.
    """

    def invalid(error: str) -> ParsedSubgoals:
        return ParsedSubgoals((), "", text, valid=False, error=error)

    payload = _json_object(text)
    if payload is None:
        return invalid("response was not a JSON object")
    items = payload.get("subgoals")
    if not isinstance(items, list) or not items:
        return invalid("response had no subgoals list")
    subgoals: list[GoalCondition] = []
    for item in items[:max_subgoals]:
        if not isinstance(item, dict):
            return invalid("a sub-goal was not an object")
        field_name = item.get("field")
        value = item.get("at_least")
        if not isinstance(field_name, str):
            return invalid("a sub-goal had no field")
        if isinstance(value, bool) or not isinstance(value, int):
            return invalid(f"{field_name}: at_least was not an integer")
        subgoals.append((StateRequirement(field_name, "at_least", value),))
    assessment = payload.get("assessment", "")
    return ParsedSubgoals(
        subgoals=tuple(subgoals),
        assessment=assessment if isinstance(assessment, str) else "",
        raw_response=text,
        valid=True,
    )


@lru_cache(maxsize=8)
def subgoal_output_type(
    fields: tuple[str, ...] = SUBGOAL_FIELDS,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines.

    ``assessment`` comes first so the model states the situation before it
    commits to sub-goals; see ``_ranking_output_type`` in the VLM module.
    """

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

    field_enum = Enum(
        "SubgoalField",
        {name.replace(".", "__"): name for name in fields},
        type=str,
    )
    item_type: type[BaseModel] = create_model(
        "Subgoal",
        __config__=ConfigDict(extra="forbid"),
        field=(field_enum, ...),  # pyright: ignore[reportArgumentType]
        at_least=(int, Field(ge=1, le=_MAX_FIELD_VALUE)),
    )
    return create_model(
        "SubgoalOutput",
        __config__=ConfigDict(extra="forbid"),
        assessment=(str, Field(max_length=_ASSESSMENT_MAX_LENGTH)),
        subgoals=(
            list[item_type],  # pyright: ignore[reportInvalidTypeForm]
            Field(min_length=1, max_length=max_subgoals),
        ),
    )


_ACHIEVEMENT_NOTE = (
    "An achievement field is 1 once that achievement is unlocked, so a "
    "sub-goal of at least 1 asks for it to be unlocked. "
)

_SUBGOAL_INSTRUCTIONS = (
    "You are the high-level planner for a Craftax survival agent.",
    "You do not choose skills or actions. You set sub-goals: a count that "
    "an inventory item or drink should reach. A separate planner reaches "
    "each sub-goal in turn with the skills below, working out the steps "
    "itself, so choose what to get rather than how to get it.",
    "You may rely on the supplied game-mechanics catalog as authoritative "
    "knowledge of what each skill needs and produces.",
    "Return 1 to {max_subgoals} sub-goals in the order they should be "
    "reached. Finish with what the objective asks for: the plan is done "
    "when the last sub-goal is reached.",
    "Keep the plan short. Add an earlier sub-goal only when the order "
    "matters, for example drinking first when drink is low.",
    "If the planner rejected a sub-goal, its reason is given; do not "
    "propose that sub-goal again unless the state has changed.",
    "Sub-goal fields: {fields}.",
    "{achievement_note}"
    "Return only JSON with assessment (at most "
    f"{_ASSESSMENT_MAX_LENGTH} characters) and subgoals, a list of "
    '{{"field": ..., "at_least": ...}} objects.',
)


def build_subgoal_system_prompt(
    candidate_skills: Sequence[SkillId] = ALL_SKILL_IDS,
    fields: Sequence[str] = SUBGOAL_FIELDS,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
    observation: ObservationConfig | None = None,
) -> str:
    """Return the sub-goal planner's role, documentation and output format.

    The achievement note and the observation legend appear only when the
    fields include achievements or the observation has text, so the default
    prompt is unchanged.
    """

    has_achievements = any(name.startswith("achievement.") for name in fields)
    instructions = " ".join(_SUBGOAL_INSTRUCTIONS).format(
        max_subgoals=max_subgoals,
        fields=", ".join(fields),
        achievement_note=_ACHIEVEMENT_NOTE if has_achievements else "",
    )
    prompt = instructions + "\n" + _mechanics_line(candidate_skills)
    if observation is not None:
        observation_text = observation_system_text(observation)
        if observation_text:
            prompt = f"{prompt}\n{observation_text}"
    return prompt


def _visible_state(state: Any, fields: Sequence[str]) -> dict[str, int]:
    """The fields the agent can read off its own status bar, when nonzero."""

    values = {name: read_field(state, name) for name in fields}
    return {
        name: value
        for name, value in values.items()
        if value or name == "survival.drink"
    }


def build_subgoal_prompt(
    state: Any,
    context: PlannerContext,
    plan: SubgoalPlan,
    trigger: ReplanTrigger,
    feedback: Sequence[str] = (),
    *,
    fields: Sequence[str] = SUBGOAL_FIELDS,
    max_events: int = DEFAULT_MAX_EVENTS,
    observation: ObservationConfig | None = None,
) -> str:
    """Return the user prompt for one sub-goal request."""

    lines = [
        f"Objective: {context.objective}",
        f"Why you are asked now: {_trigger_text(trigger, plan.trigger_detail)}",
        "Current state: " + json.dumps(_visible_state(state, fields), sort_keys=True),
    ]
    if observation is not None and observation.has_text:
        lines.append(
            "Text observation: " + json.dumps(observation_block(state, observation))
        )
    summary = getattr(context.memory, "summary", None)
    if callable(summary):
        lines.append(
            "Memory: " + json.dumps(summary(state), sort_keys=True, default=str)
        )
    if plan.pending:
        lines.append(
            "Current sub-goals: "
            + json.dumps([render_subgoal(subgoal) for subgoal in plan.pending])
        )
    if plan.events:
        lines.append(
            "Recent sub-goal events: "
            + json.dumps(
                [
                    f"{event.subgoal}: {event.status}"
                    + (f" ({event.reason})" if event.reason else "")
                    for event in plan.events[-max_events:]
                ]
            )
        )
    if feedback:
        lines.append("Planner rejected your last answer: " + " ".join(feedback))
    if observation is None or observation.image:
        lines.append("The image shows the agent's surroundings.")
    else:
        lines.append("The local map shows the agent's surroundings.")
    return "\n".join(lines)


def _failure_text(context: PlannerContext) -> str:
    """Describe the last skill's failure, with the memory's cause if it has one.

    The cause comes from the last transition an episode memory recorded, such
    as :class:`tiny_sima.memory.EpisodeMemory`, when it is for that skill.
    """

    attempt = context.history[-1]
    text = f"{attempt.skill_id.value} {attempt.outcome.value}"
    transitions = getattr(context.memory, "transitions", ())
    if transitions:
        last = transitions[-1]
        cause = getattr(last, "cause", None)
        if (
            getattr(last, "decision_index", None) == len(context.history) - 1
            and cause
            and cause != attempt.outcome.value
        ):
            text += f", {cause.replace('_', ' ')}"
    return text


def _holds(state: Any, subgoal: GoalCondition) -> bool:
    return check_subgoal(state, subgoal, ()).already_holds


@dataclass
class SubgoalRequest:
    """One episode waiting for sub-goals within a single decision.

    ``plan`` carries the trigger's detail and the events so far, and
    ``feedback`` the planner's reasons for rejecting the previous answer.
    """

    index: int
    state: Any
    context: PlannerContext
    plan: SubgoalPlan
    trigger: ReplanTrigger
    feedback: tuple[str, ...] = ()


@runtime_checkable
class SubgoalSource(Protocol):
    """A goal source that answers sub-goal requests without the VLM.

    It returns one JSON answer per request in the VLM's output format, so
    a scripted selector's answers are parsed, checked and rejected exactly
    as the VLM's are.
    """

    def propose_subgoals(self, requests: Sequence[SubgoalRequest]) -> list[str]: ...


@dataclass
class SubgoalPlanner:
    """Follow VLM sub-goals with a symbolic planner, re-asking on triggers.

    ``model`` is the skill model the symbolic planner searches; ``None`` is
    the registry. With ``use_memory_belief`` the world facts come from the
    context's memory when it has any, rather than from a scan of the whole
    map, which only the plan-veto upper bound may use.

    ``backend`` is either a VLM backend, prompted with the frame and the
    state, or a :class:`SubgoalSource` that answers the requests itself.
    ``observation`` chooses what the VLM sees (see
    :mod:`tiny_sima.planner.observation`); ``None`` is the image alone.
    ``subgoal_slack`` gives each sub-goal its plan length plus that many
    skills before it times out; ``None`` never times out.
    ``vital_thresholds`` raise an alert when a vital falls to its threshold,
    ``threat_radius`` when a hostile creature comes within that many tiles
    (Chebyshev), and ``replan_at_night`` at nightfall.
    """

    backend: VisionLanguageBackend | SubgoalSource
    candidate_skills: tuple[SkillId, ...] = ALL_SKILL_IDS
    config: VLMPlannerConfig = VLMPlannerConfig()
    model: SkillModel | None = None
    use_memory_belief: bool = True
    # Breaks ties between the plan's first steps, for example between a
    # wood pickaxe and a sword when the objective needs both.
    tie_break: FeasibilityScorer | BatchedFeasibilityScorer = reachability_feasibility
    fields: tuple[str, ...] = SUBGOAL_FIELDS
    max_subgoals: int = DEFAULT_MAX_SUBGOALS
    max_rejections: int = DEFAULT_MAX_REJECTIONS
    replan_on_failure: bool = True
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH
    observation: ObservationConfig | None = None
    max_events: int = DEFAULT_MAX_EVENTS
    subgoal_slack: int | None = DEFAULT_SUBGOAL_SLACK
    vital_thresholds: tuple[tuple[str, int], ...] = DEFAULT_VITAL_THRESHOLDS
    threat_radius: int | None = None
    replan_at_night: bool = False
    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _belief(self, context: PlannerContext) -> Any:
        memory = context.memory
        if self.use_memory_belief and hasattr(memory, "world_fact"):
            return memory
        return None

    def _check(self, state: Any, context: PlannerContext, subgoal: GoalCondition):
        return check_subgoal(
            state,
            subgoal,
            self.candidate_skills,
            model=self.model,
            belief=self._belief(context),
            max_plan_length=self.max_plan_length,
        )

    def _values(
        self, state: Any, context: PlannerContext, subgoal: GoalCondition
    ) -> str:
        """The current value of each field a sub-goal names, for feedback."""

        belief = self._belief(context)
        fields = dict.fromkeys(requirement.field for requirement in subgoal)
        return ", ".join(
            f"{field} is {read_field(state, field, belief)}" for field in fields
        )

    def _alerts(self, state: Any) -> dict[str, str]:
        """Return the alert conditions that hold now, with their details.

        Keys are vital names, ``"threat"`` and ``"night"``, in the order
        their triggers take precedence.
        """

        readings = vitals(state)
        active = {
            vital: f"{vital} {readings[vital]}"
            for vital, threshold in self.vital_thresholds
            if readings[vital] <= threshold
        }
        if self.threat_radius is not None:
            threats = [
                mob
                for mob in mobs_in_view(state)
                if mob["type"] in HOSTILE_MOBS
                and max(abs(offset) for offset in mob["offset"]) <= self.threat_radius
            ]
            if threats:
                active["threat"] = ", ".join(
                    f"{mob['type']} at offset {mob['offset']}" for mob in threats
                )
        if self.replan_at_night and readings["night"]:
            active["night"] = ""
        return active

    def _prepare(
        self,
        state: Any,
        context: PlannerContext,
        plan: SubgoalPlan,
    ) -> tuple[SubgoalPlan, ReplanTrigger | None]:
        """Advance past reached sub-goals and decide whether to ask the VLM."""

        decision_index = len(context.history)
        spent = max(decision_index - plan.decision_index, 0)
        trigger: ReplanTrigger | None = None
        detail = ""
        plan = replace(
            plan,
            decision_index=decision_index,
            trigger=None,
            trigger_detail="",
            raw_response=None,
            skills_on_current=plan.skills_on_current + spent,
        )
        if plan.vlm_calls == 0:
            trigger = ReplanTrigger.START
        while plan.current is not None and _holds(state, plan.current):
            plan = plan.advanced("achieved")
        active = self._alerts(state)
        plan = replace(plan, alerts=tuple(a for a in plan.alerts if a in active))
        fresh = {key: text for key, text in active.items() if key not in plan.alerts}
        fresh_vitals = [key for key in fresh if key not in ("threat", "night")]
        failed = (
            self.replan_on_failure
            and bool(context.history)
            and not context.history[-1].succeeded
        )
        if trigger is not None:
            pass
        elif plan.current is None:
            trigger = ReplanTrigger.SUBGOALS_DONE
        elif (
            plan.current_budget is not None
            and plan.skills_on_current >= plan.current_budget
        ):
            detail = (
                f"{render_subgoal(plan.current)} after {plan.skills_on_current} skills"
            )
            plan = plan.advanced(
                "timed_out",
                f"not reached in {plan.skills_on_current} skills "
                f"(budget {plan.current_budget})",
            )
            trigger = ReplanTrigger.TIMED_OUT
        elif failed:
            detail = _failure_text(context)
            plan = plan.with_event(render_subgoal(plan.current), "skill_failed", detail)
            trigger = ReplanTrigger.SKILL_FAILED
        elif fresh_vitals:
            detail = " and ".join(fresh_vitals)
            trigger = ReplanTrigger.LOW_VITALS
        elif "threat" in fresh:
            detail = fresh["threat"]
            trigger = ReplanTrigger.THREAT
        elif "night" in fresh:
            trigger = ReplanTrigger.NIGHT
        else:
            check = self._check(state, context, plan.current)
            if not check.reachable:
                plan = plan.advanced("abandoned", check.reason)
                trigger = ReplanTrigger.UNREACHABLE
        if trigger is not None:
            # Whatever the trigger, the VLM sees every alert that holds now.
            plan = replace(plan, alerts=(*plan.alerts, *fresh), trigger_detail=detail)
        return plan, trigger

    def _request(self, pending: SubgoalRequest) -> tuple[str, VLMRequest]:
        send_image = self.observation is None or self.observation.image
        image = (
            render_state(
                pending.state,
                self.config.environment_name,
                block_pixel_size=self.config.block_pixel_size,
            )
            if send_image
            else None
        )
        system_prompt = build_subgoal_system_prompt(
            self.candidate_skills,
            self.fields,
            self.max_subgoals,
            observation=self.observation,
        )
        user_prompt = build_subgoal_prompt(
            pending.state,
            pending.context,
            pending.plan,
            pending.trigger,
            pending.feedback,
            fields=self.fields,
            max_events=self.max_events,
            observation=self.observation,
        )
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
            output_type=subgoal_output_type(self.fields, self.max_subgoals),
        )

    def _generate(self, waiting: Sequence[SubgoalRequest]) -> list[str]:
        """Return one response per waiting episode, reusing cached text."""

        if isinstance(self.backend, SubgoalSource):
            return self.backend.propose_subgoals(waiting)
        prepared = [self._request(pending) for pending in waiting]
        missing = {key: request for key, request in prepared if key not in self._cache}
        responses = generate_all(self.backend, list(missing.values()))
        self._cache.update(zip(missing, responses, strict=True))
        return [self._cache[key] for key, _ in prepared]

    def _accept(self, pending: SubgoalRequest, response: str) -> SubgoalPlan | None:
        """Return the plan a response gives, or record why it was rejected.

        On rejection the episode's ``plan`` keeps the counts and events and
        its ``feedback`` holds the reasons for the next round.
        """

        plan = replace(
            pending.plan,
            vlm_calls=pending.plan.vlm_calls + 1,
            trigger=pending.trigger,
            raw_response=response,
        )
        parsed = parse_subgoal_response(response, self.max_subgoals)
        if not parsed.valid:
            pending.plan = replace(plan, invalid_responses=plan.invalid_responses + 1)
            pending.feedback = (f"Your answer was malformed: {parsed.error}.",)
            return None
        accepted: list[GoalCondition] = []
        reasons: list[str] = []
        held: list[str] = []
        # Sub-goals an earlier round of this decision already reported as
        # held; the 2B model repeats them until its rounds run out.
        reported = {
            event.subgoal
            for event in plan.events
            if event.decision_index == plan.decision_index
            and event.status == "already_held"
        }
        for subgoal in parsed.subgoals:
            text = render_subgoal(subgoal)
            # The schema asks for at_least >= 1, but decoding does not always
            # enforce it, and the 2B model writes the current count of 0.
            if any(requirement.value < 1 for requirement in subgoal):
                reason = "always true; at_least is the count to reach, at least 1"
                reasons.append(f"{text}: {reason}")
                plan = plan.with_event(text, "rejected", reason)
                continue
            check = self._check(pending.state, pending.context, subgoal)
            if check.already_holds:
                values = self._values(pending.state, pending.context, subgoal)
                plan = plan.with_event(text, "already_held", values)
                if text in reported:
                    held.append(
                        f"{text} already holds ({values}) and was reported "
                        "before; do not propose it again."
                    )
                else:
                    held.append(f"{text} already holds ({values}).")
            elif check.reachable:
                accepted.append(subgoal)
            else:
                reasons.append(f"{text}: {check.reason}")
                plan = plan.with_event(text, "rejected", check.reason)
        if reasons or not accepted:
            pending.plan = replace(plan, rejections=plan.rejections + 1)
            pending.feedback = (*reasons, *held)
            if not reasons:
                pending.feedback += (
                    "The objective is not complete. Give sub-goals that are "
                    "not yet true: what the objective still needs, not what "
                    "the player already has.",
                )
            return None
        for subgoal in accepted:
            plan = plan.with_event(render_subgoal(subgoal), "proposed")
        return replace(
            plan,
            pending=tuple(accepted),
            assessment=parsed.assessment,
            skills_on_current=0,
            current_budget=None,
        )

    def _selection(
        self,
        state: Any,
        context: PlannerContext,
        plan: SubgoalPlan,
    ) -> PlannerSelection:
        """Choose the best first step towards the current sub-goal."""

        if plan.current is None:
            last = plan.events[-1] if plan.events else None
            return PlannerSelection(
                decision=None,
                status=PlannerSelectionStatus.NO_VIABLE_SKILL,
                utility_error=(
                    "no reachable sub-goal"
                    + (f": {last.subgoal}: {last.reason}" if last else "")
                ),
                utility_raw_response=plan.raw_response,
                plan_state=plan,
            )
        check = self._check(state, context, plan.current)
        if (
            plan.current_budget is None
            and self.subgoal_slack is not None
            and check.plan_length is not None
        ):
            # The budget is fixed by the plan from where the sub-goal starts.
            plan = replace(plan, current_budget=check.plan_length + self.subgoal_slack)
        tie_break = catalog_feasibility(
            self.tie_break, state, context, self.candidate_skills
        )
        scores = sorted(
            (
                SkillScore(
                    skill_id=skill_id,
                    utility=float(skill_id in check.first_steps),
                    feasibility=tie_break[skill_id],
                    failure_penalty=1.0,
                    combined=float(skill_id in check.first_steps)
                    * max(tie_break[skill_id], 1e-6),
                )
                for skill_id in self.candidate_skills
            ),
            key=lambda score: -score.combined,
        )
        if not check.first_steps:
            return PlannerSelection(
                decision=None,
                status=PlannerSelectionStatus.NO_VIABLE_SKILL,
                utility_error=check.reason or "no first step",
                utility_raw_response=plan.raw_response,
                plan_state=plan,
            )
        return PlannerSelection(
            decision=PlannerDecision(skill_id=scores[0].skill_id, scores=tuple(scores)),
            status=PlannerSelectionStatus.SELECTED,
            utility_ranking=tuple(
                score.skill_id for score in scores if score.utility > 0.0
            ),
            utility_raw_response=plan.raw_response,
            plan_state=plan,
        )

    def select_many(
        self,
        states: Sequence[Any],
        contexts: Sequence[PlannerContext],
    ) -> list[PlannerSelection]:
        """Select at several boundaries, batching the VLM calls they need.

        Each round asks the VLM once for every episode still without an
        accepted plan, so rejections cost a round rather than a serial call.
        An episode whose answers are all rejected keeps the plan it had, if
        that plan is still reachable, and otherwise stops.
        """

        if len(states) != len(contexts):
            raise ValueError("states and contexts must have equal length.")
        plans: dict[int, SubgoalPlan] = {}
        waiting: list[SubgoalRequest] = []
        previous: dict[int, SubgoalPlan] = {}
        for index, (state, context) in enumerate(zip(states, contexts, strict=True)):
            plan = context.plan_state
            if not isinstance(plan, SubgoalPlan):
                plan = SubgoalPlan()
            plan, trigger = self._prepare(state, context, plan)
            if trigger is None:
                plans[index] = plan
            else:
                previous[index] = plan
                waiting.append(SubgoalRequest(index, state, context, plan, trigger))
        for _ in range(1 + self.max_rejections):
            if not waiting:
                break
            responses = self._generate(waiting)
            rejected: list[SubgoalRequest] = []
            for pending, response in zip(waiting, responses, strict=True):
                accepted = self._accept(pending, response)
                if accepted is None:
                    rejected.append(pending)
                else:
                    plans[pending.index] = accepted
            waiting = rejected
        for pending in waiting:
            # Keep the counts and events from the failed rounds, but fall
            # back to the sub-goals the episode already had.
            kept = previous[pending.index].pending
            if (
                kept
                and not self._check(pending.state, pending.context, kept[0]).reachable
            ):
                kept = ()
            plans[pending.index] = replace(pending.plan, pending=kept)
        return [
            self._selection(state, context, plans[index])
            for index, (state, context) in enumerate(zip(states, contexts, strict=True))
        ]

    def clear_cache(self) -> None:
        """Discard cached responses between matched ablations."""

        self._cache.clear()
