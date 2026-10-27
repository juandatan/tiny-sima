"""A measured SayCan competence head over the features that carry signal.

:mod:`tiny_sima.planner.affordance` tried to read ``P(skill succeeds)`` out of
the executor's value function. Measured against real rollouts that failed: the
critic separates success at AUC 0.43 -- indistinguishable from chance -- because
it was trained on scenarios that always place a reachable target, so it encodes
which skill it was asked for rather than whether this state admits it.

The same 432 labelled probes show where the signal actually is. Reachability
splits success 0.70 against 0.03, and the per-skill success rates run from 0.17
to 0.63, so this module fits the three features that are left:

* which skill was requested,
* whether its target is reachable inside the observation window,
* how many steps away that target is.

The result is a lookup table, not a learned model. Cells are thin at a few
hundred samples, so every cell is shrunk toward its parent -- distance toward
the skill's reachable rate, the skill toward the reachability rate, and that
toward the base rate -- which keeps an unobserved cell at its parent's estimate
instead of at zero or one.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tiny_sima.goals import (
    DEFAULT_REACHABILITY,
    SKILL_IDS,
    ReachabilityConfig,
    SkillGoal,
    SkillId,
    skill_interaction_distance,
    skill_precondition,
    travel_distance_map,
)
from tiny_sima.planner.saycan import PlannerContext

# Three distance bands rather than a regression: the probe set only supports a
# handful of samples per skill once it is split by reachability at all.
DEFAULT_DISTANCE_EDGES = (2, 6)

# Pseudo-count used when shrinking a cell toward its parent. Four is roughly a
# tenth of the per-skill support in the Phase 4 probe set, so a well-observed
# cell keeps its own rate while a cell with one or two samples does not.
DEFAULT_PRIOR_STRENGTH = 4.0

# An unreachable target succeeded 3% of the time, which is small enough that a
# naive estimate would behave like the hard gate that cost whole episodes.
# Walking toward a target the planner cannot yet see stays selectable.
DEFAULT_FLOOR = 0.02

UNREACHABLE_CELL = 0


def distance_cell(
    steps: int | None,
    distance_edges: Sequence[int] = DEFAULT_DISTANCE_EDGES,
) -> int:
    """Map an interaction distance to a table column.

    Column zero is reserved for an unreachable target, so ``None`` -- which is
    how :func:`tiny_sima.goals.skill_reachability_report` reports "no target in
    reach" -- lands there and real distances start at one. Edges are inclusive
    upper bounds: with edges ``(2, 6)`` the bands are ``steps <= 2``,
    ``3 <= steps <= 6`` and ``steps > 6``.
    """

    if steps is None:
        return UNREACHABLE_CELL
    return 1 + bisect_left(tuple(distance_edges), int(steps))


def num_distance_cells(
    distance_edges: Sequence[int] = DEFAULT_DISTANCE_EDGES,
) -> int:
    """Return the table width implied by a set of distance band edges."""

    return 2 + len(tuple(distance_edges))


def _shrink(successes: float, count: float, parent: float, strength: float) -> float:
    """Blend an observed rate toward a parent rate by pseudo-counts."""

    return float((successes + strength * parent) / (count + strength))


@dataclass(frozen=True)
class CompetencePrior:
    """Per-skill success rates conditioned on reachability and distance.

    ``rates[skills.index(skill)][cell]`` is ``P(success)`` for that skill in
    that distance band, with column zero meaning the target is out of reach.
    ``skills`` is the catalog of the executor the table was fitted on; a skill
    outside it scores zero, since that executor cannot run it. The
    symbolic precondition still vetoes outright -- a crafting skill with no
    table nearby cannot succeed by any distance -- so this fills in the graded
    part that :class:`~tiny_sima.planner.saycan.ReachabilityFeasibilityScorer`
    had to guess with a single hand-set constant.
    """

    rates: tuple[tuple[float, ...], ...]
    counts: tuple[tuple[int, ...], ...]
    base_rate: float
    distance_edges: tuple[int, ...] = DEFAULT_DISTANCE_EDGES
    floor: float = DEFAULT_FLOOR
    config: ReachabilityConfig = DEFAULT_REACHABILITY
    skills: tuple[SkillId, ...] = SKILL_IDS

    def __post_init__(self) -> None:
        width = num_distance_cells(self.distance_edges)
        if len(self.rates) != len(self.skills):
            raise ValueError("A competence prior needs one row per catalog skill.")
        if any(len(row) != width for row in self.rates):
            raise ValueError("Rate rows must match the distance band count.")
        if len(self.counts) != len(self.rates):
            raise ValueError("Counts must have the same shape as rates.")

    def success_probability(self, skill_id: SkillId, steps: int | None) -> float:
        """Return the fitted success rate for one skill at one distance."""

        row = self.rates[self._row(skill_id)]
        return max(row[distance_cell(steps, self.distance_edges)], self.floor)

    def support(self, skill_id: SkillId, steps: int | None) -> int:
        """Return how many probes landed in the cell that answered a query."""

        return self.support_by_cell(
            skill_id,
            distance_cell(steps, self.distance_edges),
        )

    def rate_by_cell(self, skill_id: SkillId, cell: int) -> float:
        """Return the fitted rate for one table column."""

        return max(self.rates[self._row(skill_id)][cell], self.floor)

    def support_by_cell(self, skill_id: SkillId, cell: int) -> int:
        """Return how many probes were observed in one table column."""

        return self.counts[self._row(skill_id)][cell]

    def _row(self, skill_id: SkillId) -> int:
        try:
            return self.skills.index(SkillId(skill_id))
        except ValueError as error:
            raise ValueError(
                f"The competence prior has no row for {SkillId(skill_id).value}."
            ) from error

    def score_all_feasibility(
        self,
        state: Any,
        _context: PlannerContext,
        candidate_skills: tuple[SkillId, ...],
    ) -> dict[SkillId, float]:
        """Score every candidate from one shared breadth-first distance map."""

        if not candidate_skills:
            return {}
        distances = travel_distance_map(state, self.config)
        scores: dict[SkillId, float] = {}
        for skill_id in candidate_skills:
            goal = SkillGoal(skill_id)
            if skill_id not in self.skills or not bool(
                np.asarray(skill_precondition(goal, state))
            ):
                scores[skill_id] = 0.0
                continue
            bound = int(
                np.asarray(
                    skill_interaction_distance(
                        goal,
                        state,
                        self.config,
                        distances=distances,
                    )
                )
            )
            steps = None if bound >= self.config.unreachable else bound
            scores[skill_id] = self.success_probability(skill_id, steps)
        return scores

    def __call__(
        self,
        state: Any,
        context: PlannerContext,
        skill_id: SkillId,
    ) -> float:
        """Score one candidate, recomputing the distance map on demand."""

        return self.score_all_feasibility(state, context, (skill_id,))[skill_id]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the fitted table."""

        return {
            "rates": [list(row) for row in self.rates],
            "counts": [list(row) for row in self.counts],
            "base_rate": self.base_rate,
            "distance_edges": list(self.distance_edges),
            "floor": self.floor,
            "skills": [skill_id.value for skill_id in self.skills],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        config: ReachabilityConfig = DEFAULT_REACHABILITY,
    ) -> CompetencePrior:
        """Rebuild a prior from :meth:`to_dict` output.

        A table saved before priors recorded their skills is a six-skill one.
        """

        skills = tuple(SkillId(value) for value in payload.get("skills", SKILL_IDS))
        return cls(
            rates=tuple(
                tuple(float(value) for value in row) for row in payload["rates"]
            ),
            counts=tuple(
                tuple(int(value) for value in row) for row in payload["counts"]
            ),
            base_rate=float(payload["base_rate"]),
            distance_edges=tuple(int(edge) for edge in payload["distance_edges"]),
            floor=float(payload.get("floor", DEFAULT_FLOOR)),
            config=config,
            skills=skills,
        )

    def save(self, path: str | Path) -> None:
        """Write the fitted table so a later session can skip re-probing."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: ReachabilityConfig = DEFAULT_REACHABILITY,
    ) -> CompetencePrior:
        """Read a table written by :meth:`save`."""

        return cls.from_dict(json.loads(Path(path).read_text()), config=config)


def fit_competence_prior(
    skill_ids: Sequence[SkillId],
    steps_to_target: Sequence[int | None],
    succeeded: Sequence[bool],
    *,
    distance_edges: Sequence[int] = DEFAULT_DISTANCE_EDGES,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    floor: float = DEFAULT_FLOOR,
    config: ReachabilityConfig = DEFAULT_REACHABILITY,
    skills: Sequence[SkillId] = SKILL_IDS,
) -> CompetencePrior:
    """Fit the competence table from labelled probes by shrinkage.

    ``steps_to_target`` is ``None`` where the skill's target was not reachable
    inside the observation window, which is the single most informative split in
    the probe set. Samples whose symbolic precondition failed should be dropped
    before calling: the precondition vetoes at scoring time, so including them
    would charge a skill for states it is never asked about.
    """

    if not (len(skill_ids) == len(steps_to_target) == len(succeeded)):
        raise ValueError("Skills, distances and labels must describe the same probes.")
    if not skill_ids:
        raise ValueError("Cannot fit a competence prior from zero samples.")
    if prior_strength <= 0.0:
        raise ValueError("prior_strength must be positive.")

    edges = tuple(int(edge) for edge in distance_edges)
    if any(
        later <= earlier for earlier, later in zip(edges, edges[1:], strict=False)
    ):
        raise ValueError("distance_edges must be strictly increasing.")
    width = num_distance_cells(edges)

    labels = np.asarray([float(value) for value in succeeded], dtype=np.float64)
    cells = np.asarray(
        [distance_cell(steps, edges) for steps in steps_to_target],
        dtype=np.int64,
    )
    catalog = tuple(SkillId(skill_id) for skill_id in skills)
    unknown = {SkillId(skill_id) for skill_id in skill_ids} - set(catalog)
    if unknown:
        raise ValueError(
            "Probes name skills outside the catalog: "
            + ", ".join(sorted(skill_id.value for skill_id in unknown))
        )
    rows = np.asarray(
        [catalog.index(SkillId(skill_id)) for skill_id in skill_ids],
        dtype=np.int64,
    )
    reachable = cells != UNREACHABLE_CELL

    base_rate = float(labels.mean())
    # Level one: reachability alone, shrunk toward the base rate.
    reachability_rates: dict[bool, float] = {}
    for is_reachable in (False, True):
        selected = reachable == is_reachable
        reachability_rates[is_reachable] = _shrink(
            float(labels[selected].sum()),
            float(np.count_nonzero(selected)),
            base_rate,
            prior_strength,
        )

    rates: list[tuple[float, ...]] = []
    counts: list[tuple[int, ...]] = []
    for row_index in range(len(catalog)):
        row_rates = [0.0] * width
        row_counts = [0] * width
        for is_reachable in (False, True):
            # Level two: this skill at this reachability, shrunk toward the
            # reachability rate.
            selected = (rows == row_index) & (reachable == is_reachable)
            skill_rate = _shrink(
                float(labels[selected].sum()),
                float(np.count_nonzero(selected)),
                reachability_rates[is_reachable],
                prior_strength,
            )
            if not is_reachable:
                row_rates[UNREACHABLE_CELL] = skill_rate
                row_counts[UNREACHABLE_CELL] = int(np.count_nonzero(selected))
                continue
            # Level three: distance bands, shrunk toward the skill's own
            # reachable rate.
            for cell in range(1, width):
                in_cell = (rows == row_index) & (cells == cell)
                row_rates[cell] = _shrink(
                    float(labels[in_cell].sum()),
                    float(np.count_nonzero(in_cell)),
                    skill_rate,
                    prior_strength,
                )
                row_counts[cell] = int(np.count_nonzero(in_cell))
        rates.append(tuple(row_rates))
        counts.append(tuple(row_counts))

    return CompetencePrior(
        rates=tuple(rates),
        counts=tuple(counts),
        base_rate=base_rate,
        distance_edges=edges,
        floor=floor,
        config=config,
        skills=catalog,
    )
