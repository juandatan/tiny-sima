"""Where a closed-loop planner spends its decisions.

Once the executor rarely fails a skill, a chain episode that does not succeed
has run out of decisions, so the question becomes which decisions bought
nothing. The functions here read episodes serialized by
:func:`~tiny_sima.evaluation.chains.chain_episode_record`, so a saved closed
loop can be diagnosed without rerunning it.

Each decision is judged against the skills its chain accepted from the state it
was made in (``accepted_skills``, read from the chain's ``valid_next_skills``,
the set the prefix ladder calls accepted). Requesting one of them is progress;
requesting anything else is off the objective -- a drink, a sword, wood the
pickaxe no longer needs. Comparing the chosen skill with the one the Say term
ranked first then says which planner term an off-objective choice belongs to.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from tiny_sima.goals import SKILL_IDS

SKILL_ABBREVIATIONS = {
    "collect_wood": "wood",
    "collect_stone": "stone",
    "drink_water": "water",
    "place_table": "table",
    "craft_wood_pickaxe": "pickaxe",
    "craft_wood_sword": "sword",
    "craft_stone_pickaxe": "stone_pickaxe",
}

# Say against Can for one choice, keyed by whether the Say term's first choice
# and the chosen skill were accepted. A tie for the top utility counts as one
# first choice when every tied skill falls on the same side of the accepted
# set; ``no_say_top`` is a tie across it, as under constant_utility, where the
# Say term expressed no preference between progress and waste.
ATTRIBUTIONS = (
    "both_accepted",
    "can_rescued",
    "can_misled",
    "say_led",
    "no_say_top",
)


class DecisionKind(str, Enum):
    """What one decision bought."""

    PROGRESS = "progress"
    OFF_OBJECTIVE = "off_objective"
    FAILED = "failed"
    # Succeeded, but the chain recorded no accepted set to judge it against.
    UNKNOWN = "unknown"


def decision_kind(step: Mapping[str, Any]) -> DecisionKind:
    """Classify a serialized decision; a failed skill is failed whatever it was."""

    if step["outcome"] != "success":
        return DecisionKind.FAILED
    accepted = step.get("accepted_skills") or ()
    if not accepted:
        return DecisionKind.UNKNOWN
    if step["requested_skill"] in accepted:
        return DecisionKind.PROGRESS
    return DecisionKind.OFF_OBJECTIVE


def say_top_skill(step: Mapping[str, Any]) -> str | None:
    """Return the skill the Say term alone ranked first, or None on a tie."""

    scores = step.get("candidate_scores") or ()
    if not scores:
        return None
    best = max(score["utility"] for score in scores)
    top = [score["skill_id"] for score in scores if score["utility"] == best]
    return top[0] if len(top) == 1 else None


def choice_attribution(step: Mapping[str, Any]) -> str | None:
    """Name the planner term a choice belongs to, or None without an accepted set.

    This reads the planner's choice alone, not the executor's outcome: a skill
    the objective accepted is a correct choice even if the executor then failed
    it.
    """

    accepted = step.get("accepted_skills") or ()
    if not accepted:
        return None
    say_accepted = _say_top_accepted(step, accepted)
    if say_accepted is None:
        return "no_say_top"
    chosen_accepted = step["requested_skill"] in accepted
    if say_accepted:
        return "both_accepted" if chosen_accepted else "can_misled"
    return "can_rescued" if chosen_accepted else "say_led"


def _say_top_accepted(step: Mapping[str, Any], accepted: Sequence[str]) -> bool | None:
    # Whether the skills tied for the top utility were all accepted, all
    # rejected, or split -- None, like a step with no scores.
    scores = step.get("candidate_scores") or ()
    if not scores:
        return None
    best = max(score["utility"] for score in scores)
    top = {score["skill_id"] for score in scores if score["utility"] == best}
    if top <= set(accepted):
        return True
    if top.isdisjoint(accepted):
        return False
    return None


def episode_sequence(
    episode: Mapping[str, Any],
    *,
    abbreviations: Mapping[str, str] = SKILL_ABBREVIATIONS,
) -> str:
    """Render an episode's requests on one line.

    ``*`` marks a request off the objective, ``!outcome`` one the executor did
    not finish, and ``(skill)`` the Say term's first choice wherever the Can
    term chose something else.
    """

    tokens = []
    for step in episode["trace"]:
        token = abbreviations.get(step["requested_skill"], step["requested_skill"])
        kind = decision_kind(step)
        if kind is DecisionKind.FAILED:
            token += f"!{step['outcome']}"
        elif kind is DecisionKind.OFF_OBJECTIVE:
            token += "*"
        say_top = say_top_skill(step)
        if say_top is not None and say_top != step["requested_skill"]:
            token += f"({abbreviations.get(say_top, say_top)})"
        tokens.append(token)
    return " ".join(tokens) or "-"


def sequencing_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count where a set of serialized episodes spent their decisions.

    ``repeats`` counts requests of the skill requested at the decision just
    before that bought nothing, which is where a planner stuck on one skill
    shows up. Repeats that progress are left out: the reference plan itself
    opens with two wood.
    """

    skills = [skill_id.value for skill_id in SKILL_IDS]
    for episode in episodes:
        for step in episode["trace"]:
            if step["requested_skill"] not in skills:
                skills.append(step["requested_skill"])
    per_skill = {
        skill: {"requests": 0, "repeats": 0, **{kind.value: 0 for kind in DecisionKind}}
        for skill in skills
    }
    kinds: Counter[str] = Counter()
    attributions: Counter[str] = Counter()
    for episode in episodes:
        previous = None
        for step in episode["trace"]:
            row = per_skill[step["requested_skill"]]
            kind = decision_kind(step).value
            row["requests"] += 1
            row[kind] += 1
            row["repeats"] += (
                step["requested_skill"] == previous
                and kind != DecisionKind.PROGRESS.value
            )
            previous = step["requested_skill"]
            kinds[kind] += 1
            attribution = choice_attribution(step)
            if attribution is not None:
                attributions[attribution] += 1
    decisions = [len(episode["trace"]) for episode in episodes]
    return {
        "num_episodes": len(episodes),
        "mean_decisions": sum(decisions) / len(decisions)
        if decisions
        else float("nan"),
        "termination_counts": dict(
            Counter(episode["termination"] for episode in episodes)
        ),
        "decision_kinds": {kind.value: kinds[kind.value] for kind in DecisionKind},
        "attributions": {name: attributions[name] for name in ATTRIBUTIONS},
        "per_skill": per_skill,
    }


def off_objective_choices(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[tuple[tuple[str, ...], str | None, str], list[Mapping[str, Any]]]:
    """Group the off-objective decisions by ``(accepted, say_top, chosen)``.

    These are the decisions :func:`decision_kind` calls off-objective, so the
    totals match :func:`sequencing_summary`; an unaccepted skill the executor
    then failed is counted as failed there and left out here.

    Groups come most frequent first, each with its decisions so an example's
    raw response can be read. They are the planner's characteristic mistakes:
    a group whose Say term top is the chosen skill is the VLM's, one whose Say
    term top was accepted is the Can term's.
    """

    groups: dict[tuple[tuple[str, ...], str | None, str], list[Mapping[str, Any]]]
    groups = {}
    for episode in episodes:
        for step in episode["trace"]:
            if decision_kind(step) is DecisionKind.OFF_OBJECTIVE:
                accepted = tuple(step["accepted_skills"])
                key = (accepted, say_top_skill(step), step["requested_skill"])
                groups.setdefault(key, []).append(step)
    return dict(sorted(groups.items(), key=lambda item: -len(item[1])))


def planner_assessment(step: Mapping[str, Any]) -> str | None:
    """Return the ``progress_assessment`` a VLM gave for this decision, if any."""

    raw = step.get("planner_raw_response")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    assessment = (
        payload.get("progress_assessment") if isinstance(payload, dict) else None
    )
    return assessment if isinstance(assessment, str) else None
