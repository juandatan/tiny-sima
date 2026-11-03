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
parsing, checking and rejection feedback. A source can also back the VLM up:
when every answer of a decision is rejected and no earlier sub-goal is still
reachable, the fallback answers once instead of the episode ending.

The planner never sees the chain's formal goal. Turning the objective into
sub-goals is the VLM's whole job, and the closed loop still judges success
with the chain's own completion test.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
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
    SKILL_ACHIEVEMENTS,
    SKILL_MOBS,
    STONE_PICKAXE_SKILL_IDS,
    SkillId,
    StateRequirement,
    achievement_field,
    get_skill_spec,
)
from tiny_sima.planner.fight_odds import FightOdds, FightRate
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
    SkillOutcome,
    SkillScore,
    catalog_feasibility,
    reachability_feasibility,
)
from tiny_sima.planner.shelter import (
    SHELTER_NEED_FLOOR,
    is_night,
    room_has_water,
    shelter_blocker,
    shelter_status,
    steps_until_morning,
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
# Every vital. Health, food and drink alert early, while there is time to
# restock before night; energy alerts late, since sleeping in the open is
# when zombies hit hardest.
ALL_VITAL_THRESHOLDS: tuple[tuple[str, int], ...] = (
    ("health", 5),
    ("food", 5),
    ("drink", 5),
    ("energy", 3),
)
HOSTILE_MOBS: tuple[str, ...] = ("zombie", "skeleton")
# The goal source's night choices: shelter at once, once the answer's
# sub-goals are reached, or not at all.
SHELTER_CHOICES: tuple[str, ...] = ("now", "after", "no")
_HUNTING_SKILLS: dict[str, SkillId] = {mob: skill for skill, mob in SKILL_MOBS.items()}
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
    candidate_skills: Sequence[SkillId] = STONE_PICKAXE_SKILL_IDS,
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
    # Counts and vitals the skills change beyond the fixed list, such as
    # saplings, food and energy; the stone-pickaxe catalog adds none.
    changed = tuple(
        dict.fromkeys(
            effect.field
            for skill in candidate_skills
            for effect in get_skill_spec(skill).mechanics.effects
            if not effect.field.startswith("world.")
            and effect.field not in SUBGOAL_FIELDS
        )
    )
    return (
        SUBGOAL_FIELDS
        + changed
        + tuple(name for name in ACHIEVEMENT_FIELDS if name in unlocked)
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
    # "timed_out" or "skill_failed". A sub-goal the fallback source proposed
    # has reason "fallback".
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
class SubgoalChoice:
    """One answer's shelter or fight choice, for the reports.

    ``possible`` says whether the shelter could start when the answer came;
    a fight is only offered while one could start. With it a report can set
    the answer against the curriculum's rule in the same state.
    ``win_rate`` is the measured chance of winning the fight the prompt
    gave, when it gave one.
    """

    decision_index: int
    # "shelter" or "fight".
    kind: str
    # One of SHELTER_CHOICES, or "true" or "false".
    answer: str
    possible: bool
    fallback: bool = False
    win_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_index": self.decision_index,
            "kind": self.kind,
            "answer": self.answer,
            "possible": self.possible,
            "fallback": self.fallback,
            "win_rate": self.win_rate,
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
    # Answers from the planner's fallback source, counted apart from the
    # goal source's own calls.
    fallback_calls: int = 0
    rejections: int = 0
    invalid_responses: int = 0
    # Alerts ("drink", "threat:zombie", "night", ...) already raised with the VLM.
    # One leaves when its condition clears, so an alert asks once per spell
    # rather than at every decision.
    alerts: tuple[str, ...] = ()
    # Skills spent on the current sub-goal, and how many it may take: its
    # plan length when it became current plus the planner's slack.
    skills_on_current: int = 0
    current_budget: int | None = None
    raw_response: str | None = None
    # Whether this decision fights off a nearby mob rather than following the
    # current sub-goal, so it neither spends the sub-goal's budget nor counts
    # as the sub-goal's failure.
    defending: bool = False
    # The goal source's shelter for this night, one of SHELTER_CHOICES:
    # sheltering "now", "after" the pending sub-goals are reached, or "no".
    shelter: str = "no"
    # Whether the goal source chose to fight off the zombie that is close.
    fighting: bool = False
    # Every valid answer's shelter and fight choices, where they were offered.
    choices: tuple[SubgoalChoice, ...] = ()

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
            "fallback_calls": self.fallback_calls,
            "rejections": self.rejections,
            "invalid_responses": self.invalid_responses,
            "alerts": list(self.alerts),
            "skills_on_current": self.skills_on_current,
            "current_budget": self.current_budget,
            "defending": self.defending,
            "shelter": self.shelter,
            "fighting": self.fighting,
        }


@dataclass(frozen=True)
class ParsedSubgoals:
    """Sub-goals extracted from one VLM response."""

    subgoals: tuple[GoalCondition, ...]
    assessment: str
    raw_response: str
    valid: bool
    error: str | None = None
    shelter: str = "no"
    fight: bool = False


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
    shelter = payload.get("shelter")
    return ParsedSubgoals(
        subgoals=tuple(subgoals),
        assessment=assessment if isinstance(assessment, str) else "",
        raw_response=text,
        valid=True,
        shelter=shelter if shelter in SHELTER_CHOICES else "no",
        fight=payload.get("fight") is True,
    )


def subgoal_output_type(
    fields: tuple[str, ...] = SUBGOAL_FIELDS,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
    shelter: bool = False,
    fight: bool = False,
) -> Any:
    """Create the constrained Pydantic type consumed by Outlines.

    ``assessment`` comes first so the model states the situation before it
    commits to sub-goals; see ``_ranking_output_type`` in the VLM module.
    With ``shelter`` the model also says whether to shelter for the night,
    and with ``fight`` whether to fight a close zombie. The same arguments,
    however passed, give the same type.
    """

    return _subgoal_output_type(tuple(fields), max_subgoals, shelter, fight)


@lru_cache(maxsize=8)
def _subgoal_output_type(
    fields: tuple[str, ...], max_subgoals: int, shelter: bool, fight: bool
) -> Any:

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
    choice: dict[str, Any] = {}
    if shelter:
        shelter_enum = Enum(
            "ShelterChoice", {name: name for name in SHELTER_CHOICES}, type=str
        )
        choice["shelter"] = (shelter_enum, ...)
    if fight:
        choice["fight"] = (bool, ...)
    return create_model(
        "SubgoalOutput",
        __config__=ConfigDict(extra="forbid"),
        assessment=(str, Field(max_length=_ASSESSMENT_MAX_LENGTH)),
        **choice,
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


_SHELTER_NOTE = (
    "Zombies come out at night and hit hard, hardest while the player "
    "sleeps. Also return shelter: now, after or no. now walls the player "
    "into a small stone room until morning, where it sleeps and drinks from "
    "any water beside the room; the sub-goals wait until it is out. after "
    "reaches the sub-goals first and then walls it in. no keeps working "
    "through the night. The shelter needs a wood pickaxe and stone or stone "
    "nearby. The player leaves early once food, or drink with no water "
    "beside the room, falls to {floor}; asleep, each falls by 1 about every "
    "40 steps. Shelter only counts at night."
)

_FIGHT_NOTE = (
    "Also return fight, true or false. True fights the close zombie before "
    "the sub-goals go on. A zombie has 5 health; the player hits for 1, or "
    "more with a sword, and the zombie hits for 2 every few steps. False "
    "keeps to the sub-goals, and the zombie may follow. Health comes back "
    "slowly while food, drink and energy are above 0. Fight only counts "
    "while a zombie is close."
)

_FIGHT_ODDS_NOTE = (
    "The close threats line gives the odds: how often this player won, and "
    "how often it died, in staged fights at that health, with that sword and "
    "that many zombies. Walking away was not measured."
)


def build_subgoal_system_prompt(
    candidate_skills: Sequence[SkillId] = STONE_PICKAXE_SKILL_IDS,
    fields: Sequence[str] = SUBGOAL_FIELDS,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
    observation: ObservationConfig | None = None,
    shelter: bool = False,
    fight: bool = False,
    fight_odds: bool = False,
) -> str:
    """Return the sub-goal planner's role, documentation and output format.

    The achievement note, the shelter and fight choices and the observation
    legend appear only when the fields include achievements, ``shelter`` or
    ``fight`` is set or the observation has text, so the default prompt is
    unchanged. With ``fight`` and ``fight_odds`` it says what the fight odds
    in the user prompt mean.
    """

    has_achievements = any(name.startswith("achievement.") for name in fields)
    instructions = " ".join(_SUBGOAL_INSTRUCTIONS).format(
        max_subgoals=max_subgoals,
        fields=", ".join(fields),
        achievement_note=_ACHIEVEMENT_NOTE if has_achievements else "",
    )
    if shelter:
        instructions += " " + _SHELTER_NOTE.format(floor=SHELTER_NEED_FLOOR)
    if fight:
        instructions += " " + _FIGHT_NOTE
        if fight_odds:
            instructions += " " + _FIGHT_ODDS_NOTE
    prompt = instructions + "\n" + _mechanics_line(candidate_skills)
    if observation is not None:
        observation_text = observation_system_text(observation)
        if observation_text:
            prompt = f"{prompt}\n{observation_text}"
    return prompt


def _visible_state(state: Any, fields: Sequence[str]) -> dict[str, int]:
    """The fields the agent can read off its own status bar, when nonzero.

    Vitals are shown even at zero, since an empty one is what matters.
    """

    values = {name: read_field(state, name) for name in fields}
    return {
        name: value
        for name, value in values.items()
        if value or name.startswith("survival.")
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
    shelter: bool = False,
    threats: Sequence[dict[str, Any]] | None = None,
    fight_rate: FightRate | None = None,
    fight_mob: str = "zombie",
) -> str:
    """Return the user prompt for one sub-goal request.

    With ``shelter`` it says whether the shelter could start now, and why
    not, with whether its room has water and how long the night has left.
    With ``threats``, the close hostile mobs, it lists them with the
    player's health for the fight choice, and with ``fight_rate`` the
    measured odds of fighting the close ``fight_mob``.
    """

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
    if shelter:
        lines.append(_shelter_text(state))
    if threats is not None:
        line = (
            f"Close threats: {_threat_text(threats) or 'none'}; "
            f"health is {vitals(state)['health']}."
        )
        if fight_rate is not None:
            line += " " + _fight_rate_text(fight_rate, fight_mob)
        lines.append(line)
    if feedback:
        lines.append("Planner rejected your last answer: " + " ".join(feedback))
    if observation is None or observation.image:
        lines.append("The image shows the agent's surroundings.")
    else:
        lines.append("The local map shows the agent's surroundings.")
    return "\n".join(lines)


def _shelter_text(state: Any) -> str:
    """Whether the shelter could start now, and what the night holds."""

    if not is_night(state):
        return "Shelter: not possible now: it is day."
    blocker, room = shelter_status(state)
    text = "possible now" if blocker is None else f"not possible now: {blocker}"
    lines = [f"Shelter: {text}."]
    if room is not None:
        water = "has" if room_has_water(state, room) else "has no"
        lines.append(f"The nearest room {water} water beside it.")
    lines.append(f"Morning is in {steps_until_morning(state)} steps.")
    return " ".join(lines)


def _fight_rate_text(rate: FightRate, mob: str) -> str:
    sword = "no sword" if rate.sword == "none" else f"a {rate.sword} sword"
    mobs = f"{rate.mobs} {mob}{'s' if rate.mobs > 1 else ''}"
    return (
        f"Fight odds at health {rate.health}, {sword} and {mobs}: "
        f"won {rate.win_rate:.0%}, died {rate.death_rate:.0%}."
    )


def _is_threat(alert: str) -> bool:
    return alert.startswith("threat:")


def _threat_text(threats: Sequence[dict[str, Any]]) -> str:
    return ", ".join(f"{mob['type']} at offset {mob['offset']}" for mob in threats)


def _start_shelter(
    plan: SubgoalPlan, state: Any, status: str, reason: str = ""
) -> tuple[SubgoalPlan, str | None]:
    """Record whether the shelter starts now; return why not, if it cannot."""

    blocker = shelter_blocker(state)
    if blocker is None:
        return plan.with_event("shelter", status, reason), None
    return plan.with_event("shelter", "rejected", blocker), blocker


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
    (Chebyshev), and ``replan_at_night`` at nightfall. A mob in
    ``defend_against`` within ``threat_radius`` is fought with its hunting
    skill, when that is a candidate, before the plan goes on: at once, or
    with ``choose_fights`` only once the goal source's answer says to fight,
    which holds while the mob stays close. The prompt then lists the close
    threats with the player's health, and with ``fight_odds`` the measured
    odds of the fight: the executor's win and death rates in staged fights
    at the player's health, best sword and number of close mobs of that
    kind.
    With ``shelter_at_night`` the goal source is offered the scripted
    ``shelter`` skill (see :mod:`tiny_sima.planner.shelter`): its answer says
    whether to wall the player in now, after reaching its sub-goals, such as
    drinking first, or not at all. Once accepted at night the player
    shelters while it can until morning. Like a defence fight, that spends
    no sub-goal budget. The prompt says whether the shelter could start,
    whether its room has water and when morning comes; the choices are the
    goal source's.
    ``fallback`` answers once for an episode whose answers were all rejected
    and whose earlier sub-goals are no longer reachable, so it keeps playing
    rather than stopping; its sub-goals are proposed with reason
    ``"fallback"``.
    """

    backend: VisionLanguageBackend | SubgoalSource
    candidate_skills: tuple[SkillId, ...] = STONE_PICKAXE_SKILL_IDS
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
    defend_against: tuple[str, ...] = ()
    shelter_at_night: bool = False
    choose_fights: bool = False
    fight_odds: FightOdds | None = None
    fallback: SubgoalSource | None = None
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

    def _threats(self, state: Any) -> list[dict[str, Any]]:
        """Return the hostile mobs within ``threat_radius``, nearest first."""

        if self.threat_radius is None:
            return []
        return [
            mob
            for mob in mobs_in_view(state)
            if mob["type"] in HOSTILE_MOBS
            and max(abs(offset) for offset in mob["offset"]) <= self.threat_radius
        ]

    def _defended_mob(self, state: Any) -> str | None:
        """Return the kind of the nearest mob to fight off, if any."""

        for mob in self._threats(state):
            skill_id = _HUNTING_SKILLS.get(mob["type"])
            if mob["type"] in self.defend_against and skill_id in self.candidate_skills:
                return mob["type"]
        return None

    def _defence(self, state: Any) -> SkillId | None:
        """Return the hunting skill for the nearest mob to fight off, if any."""

        mob = self._defended_mob(state)
        return None if mob is None else _HUNTING_SKILLS[mob]

    def _fight_rate(self, state: Any) -> FightRate | None:
        """Return the measured odds of the fight on offer, if any."""

        mob = self._defended_mob(state)
        if self.fight_odds is None or not self.choose_fights or mob is None:
            return None
        close = sum(threat["type"] == mob for threat in self._threats(state))
        return self.fight_odds.for_state(state, mob, close)

    def _alerts(self, state: Any) -> dict[str, str]:
        """Return the alert conditions that hold now, with their details.

        Keys are vital names, ``"threat:<mob>"`` for each kind of close
        hostile mob and ``"night"``, in the order their triggers take
        precedence. Keyed by kind, a zombie arriving while a skeleton is
        close is still a fresh threat.
        """

        readings = vitals(state)
        active = {
            vital: f"{vital} {readings[vital]}"
            for vital, threshold in self.vital_thresholds
            if readings[vital] <= threshold
        }
        threats = self._threats(state)
        for kind in dict.fromkeys(mob["type"] for mob in threats):
            active[f"threat:{kind}"] = _threat_text(
                [mob for mob in threats if mob["type"] == kind]
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
        # A defence fight or an interrupted skill spends no budget, and
        # neither failure is the plan's.
        defended = plan.defending or (
            bool(context.history)
            and context.history[-1].outcome is SkillOutcome.INTERRUPTED
        )
        spent = max(decision_index - plan.decision_index - int(defended), 0)
        trigger: ReplanTrigger | None = None
        detail = ""
        plan = replace(
            plan,
            decision_index=decision_index,
            trigger=None,
            trigger_detail="",
            raw_response=None,
            skills_on_current=plan.skills_on_current + spent,
            defending=False,
        )
        # A fight lasts while its mob stays close, and a shelter until it
        # cannot go on, by dawn at the latest; the goal source is then asked
        # again if nothing else is pending.
        if plan.fighting and self._defence(state) is None:
            plan = replace(plan, fighting=False).with_event("fight", "ended")
        if plan.shelter == "now":
            blocker = shelter_blocker(state)
            if blocker is not None:
                plan = replace(plan, shelter="no").with_event(
                    "shelter", "ended", blocker
                )
        elif plan.shelter == "after" and not is_night(state):
            plan = replace(plan, shelter="no")
        if plan.vlm_calls == 0:
            trigger = ReplanTrigger.START
        while plan.current is not None and _holds(state, plan.current):
            plan = plan.advanced("achieved")
        if plan.shelter == "after" and plan.current is None:
            # The goal source asked for the shelter once these were reached.
            plan, blocker = _start_shelter(plan, state, "started")
            plan = replace(plan, shelter="now" if blocker is None else "no")
        active = self._alerts(state)
        plan = replace(plan, alerts=tuple(a for a in plan.alerts if a in active))
        fresh = {key: text for key, text in active.items() if key not in plan.alerts}
        fresh_threats = [text for key, text in fresh.items() if _is_threat(key)]
        fresh_vitals = [key for key in fresh if not _is_threat(key) and key != "night"]
        failed = (
            self.replan_on_failure
            and bool(context.history)
            and not context.history[-1].succeeded
            and not defended
        )
        # A shelter or a fight with no sub-goals after it is a plan of its
        # own, which the alerts below may still interrupt.
        standing = plan.shelter == "now" or plan.fighting
        if trigger is not None:
            pass
        elif plan.current is None and not standing:
            trigger = ReplanTrigger.SUBGOALS_DONE
        elif (
            plan.current is not None
            and plan.current_budget is not None
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
        elif failed and plan.current is not None:
            detail = _failure_text(context)
            plan = plan.with_event(render_subgoal(plan.current), "skill_failed", detail)
            trigger = ReplanTrigger.SKILL_FAILED
        elif fresh_vitals:
            detail = " and ".join(fresh_vitals)
            trigger = ReplanTrigger.LOW_VITALS
        elif fresh_threats:
            detail = ", ".join(fresh_threats)
            trigger = ReplanTrigger.THREAT
        elif "night" in fresh:
            trigger = ReplanTrigger.NIGHT
        elif plan.current is not None:
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
            shelter=self.shelter_at_night,
            fight=self.choose_fights,
            fight_odds=self.fight_odds is not None,
        )
        fight_mob = self._defended_mob(pending.state)
        user_prompt = build_subgoal_prompt(
            pending.state,
            pending.context,
            pending.plan,
            pending.trigger,
            pending.feedback,
            fields=self.fields,
            max_events=self.max_events,
            observation=self.observation,
            shelter=self.shelter_at_night,
            threats=self._threats(pending.state) if self.choose_fights else None,
            fight_rate=self._fight_rate(pending.state),
            fight_mob=fight_mob or "zombie",
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
            output_type=subgoal_output_type(
                self.fields,
                self.max_subgoals,
                self.shelter_at_night,
                self.choose_fights,
            ),
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

    def _accept(
        self, pending: SubgoalRequest, response: str, *, fallback: bool = False
    ) -> SubgoalPlan | None:
        """Return the plan a response gives, or record why it was rejected.

        On rejection the episode's ``plan`` keeps the counts and events and
        its ``feedback`` holds the reasons for the next round. ``fallback``
        counts the response as the fallback source's rather than a call.
        """

        plan = replace(
            pending.plan,
            vlm_calls=pending.plan.vlm_calls + (not fallback),
            fallback_calls=pending.plan.fallback_calls + fallback,
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
        source = "fallback" if fallback else ""
        night = self.shelter_at_night and is_night(pending.state)
        # The latest answer while a mob is close decides the fight.
        fight_offered = self.choose_fights and self._defence(pending.state) is not None
        fighting = fight_offered and parsed.fight
        choices = []
        if night:
            choices.append(
                SubgoalChoice(
                    plan.decision_index,
                    "shelter",
                    parsed.shelter,
                    shelter_blocker(pending.state) is None,
                    fallback,
                )
            )
        if fight_offered:
            rate = self._fight_rate(pending.state)
            choices.append(
                SubgoalChoice(
                    plan.decision_index,
                    "fight",
                    "true" if parsed.fight else "false",
                    True,
                    fallback,
                    None if rate is None else rate.win_rate,
                )
            )
        plan = replace(plan, choices=(*plan.choices, *choices))
        if fighting:
            plan = plan.with_event("fight", "proposed", source)
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
        # The reachable sub-goals of an answer are kept; the rest stay in the
        # events with their reasons, which the next request shows. Turning
        # the whole answer down for one unreachable sub-goal ended episodes
        # where the VLM kept asking for food with no cow left.
        # Likewise the latest answer at night decides the shelter; by day the
        # choice is moot. A shelter after the sub-goals is checked once they
        # are reached, since they may be what it lacks now, such as drink,
        # and with none accepted it starts now.
        shelter = parsed.shelter if night else "no"
        if shelter == "after" and not accepted:
            shelter = "now"
        if shelter == "now":
            plan, blocker = _start_shelter(plan, pending.state, "proposed", source)
            if blocker is not None:
                shelter = "no"
                reasons.append(f"shelter: {blocker}")
        elif shelter == "after":
            plan = plan.with_event("shelter", "proposed", source or "after sub-goals")
        if (shelter == "now" or fighting) and not accepted:
            # The night's shelter or the fight alone is a plan; the
            # sub-goals wait.
            return replace(
                plan,
                pending=pending.plan.pending,
                assessment=parsed.assessment,
                shelter=shelter,
                fighting=fighting,
            )
        if not accepted:
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
            plan = plan.with_event(render_subgoal(subgoal), "proposed", source)
        return replace(
            plan,
            pending=tuple(accepted),
            assessment=parsed.assessment,
            skills_on_current=0,
            current_budget=None,
            shelter=shelter,
            fighting=fighting,
        )

    def _selection(
        self,
        state: Any,
        context: PlannerContext,
        plan: SubgoalPlan,
    ) -> PlannerSelection:
        """Choose the best first step towards the current sub-goal."""

        defence = self._defence(state)
        if defence is not None and (plan.fighting or not self.choose_fights):
            return self._selected(
                state, context, replace(plan, defending=True), {defence}
            )
        if plan.shelter == "now":
            return self._selected(
                state, context, replace(plan, defending=True), {SkillId.SHELTER}
            )
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
        if not check.first_steps:
            return PlannerSelection(
                decision=None,
                status=PlannerSelectionStatus.NO_VIABLE_SKILL,
                utility_error=check.reason or "no first step",
                utility_raw_response=plan.raw_response,
                plan_state=plan,
            )
        return self._selected(state, context, plan, check.first_steps)

    def _selected(
        self,
        state: Any,
        context: PlannerContext,
        plan: SubgoalPlan,
        first_steps: Collection[SkillId],
    ) -> PlannerSelection:
        """Select the best of ``first_steps`` by tie-break, scoring the catalog."""

        tie_break = catalog_feasibility(
            self.tie_break, state, context, self.candidate_skills
        )
        # A macro such as the shelter is scored too, though not a candidate.
        skill_ids = dict.fromkeys((*self.candidate_skills, *sorted(first_steps)))
        scores = sorted(
            (
                SkillScore(
                    skill_id=skill_id,
                    utility=float(skill_id in first_steps),
                    feasibility=tie_break.get(skill_id, 1.0),
                    failure_penalty=1.0,
                    combined=float(skill_id in first_steps)
                    * max(tie_break.get(skill_id, 1.0), 1e-6),
                )
                for skill_id in skill_ids
            ),
            key=lambda score: -score.combined,
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
        that plan is still reachable. Otherwise the fallback source answers
        once, and without one, or when that answer is rejected too, the
        episode stops.
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
        fallback = self.fallback
        stranded: list[SubgoalRequest] = []
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
            if not kept and fallback is not None:
                stranded.append(pending)
        if fallback is not None and stranded:
            responses = fallback.propose_subgoals(stranded)
            for pending, response in zip(stranded, responses, strict=True):
                accepted = self._accept(pending, response, fallback=True)
                plans[pending.index] = (
                    replace(pending.plan, pending=()) if accepted is None else accepted
                )
        return [
            self._selection(state, context, plans[index])
            for index, (state, context) in enumerate(zip(states, contexts, strict=True))
        ]

    def clear_cache(self) -> None:
        """Discard cached responses between matched ablations."""

        self._cache.clear()
