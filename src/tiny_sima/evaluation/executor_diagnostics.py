"""Executor-only diagnostics for the gap between training and chain worlds.

The planner ladder showed that most chain failures are executor failures: the
scripted oracle plan succeeds in a quarter of episodes and a collect-first
reordering of the same skills in three fifths. Nothing about the planner
differs between those two, so the gap has to be explained by the states each
order hands the executor. This module measures that directly.

:func:`benchmark_executor` runs both fixed plans over the same worlds and
records the state at the start of every first attempt. :func:`chain_boundaries`
turns those states into labelled rows -- skill, how many times it has been
requested, interaction distance, what else is in view -- so success can be read
by skill and distance band. :func:`probe_counterfactuals` then edits the
boundary states one training-distribution difference at a time and re-runs the
option, which attributes failures to a cause instead of guessing at one, and
:func:`rollout_features` describes what a failed option actually did.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType

from tiny_sima.envs.skills import _scenario_inventory
from tiny_sima.evaluation.chains import (
    STONE_PICKAXE_AND_SWORD_CHAIN,
    TERSE_STONE_PICKAXE_CHAIN,
    WOOD_TO_STONE_CHAIN,
    WOOD_TO_STONE_PICKAXE_CHAIN,
    LearnedSkillExecutor,
    SkillChainSpec,
    SkillRolloutTrace,
    compare_paired_successes,
    evaluate_learned_skill_chain_batched,
    paired_success_summary,
    trace_skills_batched,
)
from tiny_sima.evaluation.goal_conditioned import evaluate_goal_conditioned_policy
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    DEFAULT_REACHABILITY,
    OBSERVATION_RADIUS,
    SkillGoal,
    SkillId,
    skill_interaction_distance,
    travel_distance_map,
    view_mask,
    walkable_mask,
)
from tiny_sima.planner.saycan import SkillOutcome

COLLECT_FIRST_PLAN: tuple[SkillId, ...] = (
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.PLACE_TABLE,
    SkillId.CRAFT_WOOD_PICKAXE,
    SkillId.COLLECT_STONE,
)

# The same reordering for the stone-pickaxe chain: all four wood before the
# table, so no collect_wood is requested from a world already thinned of trees.
STONE_PICKAXE_COLLECT_FIRST_PLAN: tuple[SkillId, ...] = (
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.PLACE_TABLE,
    SkillId.CRAFT_WOOD_PICKAXE,
    SkillId.COLLECT_STONE,
    SkillId.CRAFT_STONE_PICKAXE,
)

# And for the pickaxe-and-sword chain: all five wood, then both crafts at the
# one table.
STONE_PICKAXE_AND_SWORD_COLLECT_FIRST_PLAN: tuple[SkillId, ...] = (
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_WOOD,
    SkillId.PLACE_TABLE,
    SkillId.CRAFT_WOOD_PICKAXE,
    SkillId.CRAFT_WOOD_SWORD,
    SkillId.COLLECT_STONE,
    SkillId.CRAFT_STONE_PICKAXE,
)

COLLECT_FIRST_PLANS: dict[str, tuple[SkillId, ...]] = {
    WOOD_TO_STONE_CHAIN.name: COLLECT_FIRST_PLAN,
    WOOD_TO_STONE_PICKAXE_CHAIN.name: STONE_PICKAXE_COLLECT_FIRST_PLAN,
    TERSE_STONE_PICKAXE_CHAIN.name: STONE_PICKAXE_COLLECT_FIRST_PLAN,
    STONE_PICKAXE_AND_SWORD_CHAIN.name: STONE_PICKAXE_AND_SWORD_COLLECT_FIRST_PLAN,
}

# Held out from the planner calibration seeds (7, 23, 51), so that executor
# comparisons never select on worlds the planner ladder was tuned against.
EXECUTOR_BENCHMARK_SEED = 101

TARGET_BLOCKS: dict[SkillId, BlockType] = {
    SkillId.COLLECT_WOOD: BlockType.TREE,
    SkillId.COLLECT_STONE: BlockType.STONE,
    SkillId.DRINK_WATER: BlockType.WATER,
    SkillId.CRAFT_WOOD_PICKAXE: BlockType.CRAFTING_TABLE,
    SkillId.CRAFT_WOOD_SWORD: BlockType.CRAFTING_TABLE,
    SkillId.CRAFT_STONE_PICKAXE: BlockType.CRAFTING_TABLE,
}

_CRAFTING_SKILLS = frozenset(
    {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
        SkillId.CRAFT_STONE_PICKAXE,
    }
)
_MOVEMENT_ACTIONS = frozenset(
    {Action.LEFT.value, Action.RIGHT.value, Action.UP.value, Action.DOWN.value}
)

# Interaction distances are lower bounds in steps, so a target at Manhattan
# distance d in a cleared window has bound d. Training covered 2-4.
DISTANCE_BANDS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 7, "5-7"),
    (8, None, "8+"),
)


def distance_band(distance: int | None) -> str:
    """Name the band an interaction distance falls in."""

    if distance is None:
        return "unreachable"
    for lower, upper, name in DISTANCE_BANDS:
        if distance >= lower and (upper is None or distance <= upper):
            return name
    return "0"


@dataclass(frozen=True)
class ChainBoundary:
    """The world at the start of one first attempt inside a chain episode.

    ``occurrence`` counts requests for the same skill so far, so the third
    ``collect_wood`` of a plan has occurrence 3. ``interaction_distance`` is
    the in-view lower bound from :mod:`tiny_sima.goals.reachability`, or
    ``None`` when no target is reachable inside the observation window.
    """

    plan: str
    episode: int
    plan_step: int
    skill_id: SkillId
    occurrence: int
    outcome: SkillOutcome
    steps: int
    interaction_distance: int | None
    targets_in_view: int
    table_in_view: bool
    clutter_in_view: int
    uncleared_in_view: int
    player_offset: tuple[int, int]
    wood: int
    drink: int
    timestep: int
    light_level: float
    state: Any

    @property
    def succeeded(self) -> bool:
        return self.outcome is SkillOutcome.SUCCESS

    @property
    def band(self) -> str:
        return distance_band(self.interaction_distance)

    def as_row(self) -> dict[str, Any]:
        """Return every field except the state, for tables and JSON."""

        return {
            "plan": self.plan,
            "episode": self.episode,
            "plan_step": self.plan_step,
            "skill_id": self.skill_id.value,
            "occurrence": self.occurrence,
            "outcome": self.outcome.value,
            "succeeded": self.succeeded,
            "steps": self.steps,
            "interaction_distance": self.interaction_distance,
            "band": self.band,
            "targets_in_view": self.targets_in_view,
            "table_in_view": self.table_in_view,
            "clutter_in_view": self.clutter_in_view,
            "uncleared_in_view": self.uncleared_in_view,
            "player_offset": self.player_offset,
            "wood": self.wood,
            "drink": self.drink,
            "timestep": self.timestep,
            "light_level": self.light_level,
        }


def _cleared_window(state: Any) -> jax.Array:
    """Return the tiles the chain scenario cleared around the map centre."""

    world_map = jnp.asarray(state.map)
    rows = jnp.arange(world_map.shape[0])[:, None]
    columns = jnp.arange(world_map.shape[1])[None, :]
    return (jnp.abs(rows - world_map.shape[0] // 2) <= OBSERVATION_RADIUS[0]) & (
        jnp.abs(columns - world_map.shape[1] // 2) <= OBSERVATION_RADIUS[1]
    )


def _target_mask(state: Any, skill_id: SkillId) -> jax.Array:
    world_map = jnp.asarray(state.map)
    block = TARGET_BLOCKS.get(skill_id)
    if block is None:
        return jnp.zeros(world_map.shape, dtype=jnp.bool_)
    return world_map == block.value


def _single_boundary_features(state: Any, skill_index: jax.Array) -> dict[str, Any]:
    distances = travel_distance_map(state)
    view = view_mask(state, OBSERVATION_RADIUS)
    walkable = walkable_mask(state)
    bounds = jnp.stack(
        [
            skill_interaction_distance(SkillGoal(skill_id), state, distances=distances)
            for skill_id in ALL_SKILL_IDS
        ]
    )
    targets = jnp.stack(
        [
            jnp.count_nonzero(_target_mask(state, skill_id) & view)
            for skill_id in ALL_SKILL_IDS
        ]
    )
    obstacles = jnp.count_nonzero(view & ~walkable)
    target_count = targets[skill_index]
    centre = jnp.asarray(
        [state.map.shape[0] // 2, state.map.shape[1] // 2],
        dtype=jnp.int32,
    )
    return {
        "distance": bounds[skill_index],
        "targets": target_count,
        "table": jnp.any(
            view & (jnp.asarray(state.map) == BlockType.CRAFTING_TABLE.value)
        ),
        "clutter": obstacles - target_count,
        "uncleared": jnp.count_nonzero(view & ~_cleared_window(state)),
        "offset": jnp.asarray(state.player_position, dtype=jnp.int32) - centre,
    }


_boundary_features = jax.jit(jax.vmap(_single_boundary_features))


def chain_boundaries(
    result: dict[str, Any],
    *,
    plan_name: str | None = None,
) -> list[ChainBoundary]:
    """Label every first attempt recorded by a batched chain evaluation.

    ``result`` must come from :func:`evaluate_learned_skill_chain_batched`
    with ``record_boundaries=True``. Plan steps an episode never reached are
    skipped.
    """

    if "boundary_states" not in result:
        raise ValueError("Evaluate the chain with record_boundaries=True first.")
    plan = [SkillId(value) for value in result["plan"]]
    outcomes = result["first_attempt_outcomes"]
    steps = result["first_attempt_steps"]
    states = result["boundary_states"]
    num_episodes = len(outcomes)
    flat_states = jax.tree_util.tree_map(
        lambda leaf: np.reshape(leaf, (num_episodes * len(plan), *leaf.shape[2:])),
        states,
    )
    flat_skills = jnp.asarray(
        [
            ALL_SKILL_IDS.index(skill_id)
            for _ in range(num_episodes)
            for skill_id in plan
        ],
        dtype=jnp.int32,
    )
    features = jax.device_get(_boundary_features(flat_states, flat_skills))
    name = plan_name or "+".join(skill_id.value for skill_id in plan)

    occurrences = []
    counts: dict[SkillId, int] = defaultdict(int)
    for skill_id in plan:
        counts[skill_id] += 1
        occurrences.append(counts[skill_id])

    boundaries = []
    for episode in range(num_episodes):
        for plan_step, skill_id in enumerate(plan):
            outcome = outcomes[episode][plan_step]
            if outcome is None:
                continue
            flat = episode * len(plan) + plan_step
            state = jax.tree_util.tree_map(
                lambda leaf, flat=flat: leaf[flat],
                flat_states,
            )
            distance = int(features["distance"][flat])
            boundaries.append(
                ChainBoundary(
                    plan=name,
                    episode=episode,
                    plan_step=plan_step,
                    skill_id=skill_id,
                    occurrence=occurrences[plan_step],
                    outcome=SkillOutcome(outcome),
                    steps=int(steps[episode][plan_step]),
                    interaction_distance=(
                        None
                        if distance >= DEFAULT_REACHABILITY.unreachable
                        else distance
                    ),
                    targets_in_view=int(features["targets"][flat]),
                    table_in_view=bool(features["table"][flat]),
                    clutter_in_view=int(features["clutter"][flat]),
                    uncleared_in_view=int(features["uncleared"][flat]),
                    player_offset=(
                        int(features["offset"][flat][0]),
                        int(features["offset"][flat][1]),
                    ),
                    wood=int(np.asarray(state.inventory.wood)),
                    drink=int(np.asarray(state.player_drink)),
                    timestep=int(np.asarray(state.timestep)),
                    light_level=float(np.asarray(state.light_level)),
                    state=state,
                )
            )
    return boundaries


def success_table(
    records: Iterable[Any],
    key: Callable[[Any], Hashable],
    *,
    succeeded: Callable[[Any], bool] = lambda record: bool(record.succeeded),
) -> dict[Hashable, dict[str, Any]]:
    """Count records and successes per group, sorted by group key."""

    groups: dict[Hashable, list[bool]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(succeeded(record))
    return {
        group: {
            "count": len(values),
            "successes": sum(values),
            "success_rate": sum(values) / len(values),
        }
        for group, values in sorted(groups.items(), key=lambda item: str(item[0]))
    }


@dataclass(frozen=True)
class CounterfactualEdit:
    """One training-distribution difference undone on a boundary state.

    ``apply`` receives a single state and a static skill. Skills outside
    ``applies_to`` would lose their target or precondition under the edit, so
    they are left out of that comparison rather than reported as failures.
    """

    name: str
    description: str
    apply: Callable[[Any, SkillId], Any]
    applies_to: frozenset[SkillId] = frozenset(ALL_SKILL_IDS)


def _remove_tables(state: Any, skill_id: SkillId) -> Any:
    del skill_id
    world_map = jnp.asarray(state.map)
    return state.replace(
        map=jnp.where(
            world_map == BlockType.CRAFTING_TABLE.value,
            BlockType.GRASS.value,
            world_map,
        )
    )


def _clear_view(state: Any, skill_id: SkillId) -> Any:
    world_map = jnp.asarray(state.map)
    keep = _target_mask(state, skill_id) | ~view_mask(state, OBSERVATION_RADIUS)
    return state.replace(map=jnp.where(keep, world_map, BlockType.GRASS.value))


def _training_inventory(state: Any, skill_id: SkillId) -> Any:
    inventory = _scenario_inventory(skill_id)
    return state.replace(
        inventory=jax.tree_util.tree_map(
            lambda value, current: jnp.asarray(value, dtype=jnp.asarray(current).dtype),
            inventory,
            state.inventory,
        )
    )


def _training_vitals(state: Any, skill_id: SkillId) -> Any:
    return state.replace(
        player_health=jnp.asarray(9, dtype=jnp.asarray(state.player_health).dtype),
        player_food=jnp.asarray(9, dtype=jnp.asarray(state.player_food).dtype),
        player_drink=jnp.asarray(
            5 if skill_id is SkillId.DRINK_WATER else 9,
            dtype=jnp.asarray(state.player_drink).dtype,
        ),
        player_energy=jnp.asarray(9, dtype=jnp.asarray(state.player_energy).dtype),
    )


def _training_clock(state: Any, skill_id: SkillId) -> Any:
    del skill_id
    return state.replace(
        timestep=jnp.zeros_like(jnp.asarray(state.timestep)),
        light_level=jnp.ones_like(jnp.asarray(state.light_level)),
    )


_NON_TABLE_SKILLS = frozenset(set(ALL_SKILL_IDS) - _CRAFTING_SKILLS)

COUNTERFACTUAL_EDITS: tuple[CounterfactualEdit, ...] = (
    CounterfactualEdit(
        "remove_tables",
        "Turn every crafting table back into grass.",
        _remove_tables,
        _NON_TABLE_SKILLS,
    ),
    CounterfactualEdit(
        "clear_view",
        "Clear every visible tile except the skill's targets to grass.",
        _clear_view,
    ),
    CounterfactualEdit(
        "training_inventory",
        "Replace the inventory with the one the skill was trained with.",
        _training_inventory,
    ),
    CounterfactualEdit(
        "training_vitals",
        "Restore health, food, drink and energy to training values.",
        _training_vitals,
    ),
    CounterfactualEdit(
        "training_clock",
        "Reset the clock to dawn, so light_level matches training.",
        _training_clock,
    ),
)


def _compose(edits: Sequence[CounterfactualEdit]) -> CounterfactualEdit:
    def apply(state: Any, skill_id: SkillId) -> Any:
        for edit in edits:
            if skill_id in edit.applies_to:
                state = edit.apply(state, skill_id)
        return state

    return CounterfactualEdit(
        "all",
        "Apply every edit that applies to the skill.",
        apply,
    )


@partial(jax.jit, static_argnames=("edit", "skill_id"))
def _apply_edit(state: Any, *, edit: CounterfactualEdit, skill_id: SkillId) -> Any:
    return edit.apply(state, skill_id)


def probe_counterfactuals(
    executor: LearnedSkillExecutor,
    boundaries: Sequence[ChainBoundary],
    *,
    edits: Sequence[CounterfactualEdit] = COUNTERFACTUAL_EDITS,
    include_all: bool = True,
    skill_horizon: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-run each boundary's option unedited and under every edit.

    The unedited probe replays the chain's own first attempt, so its success
    rate should match the recorded outcomes; the edited probes then read how
    much each training-distribution difference costs. Every comparison is
    paired against the unedited probe of the same boundaries.
    """

    if not boundaries:
        raise ValueError("No boundaries to probe.")
    variants = list(edits)
    if include_all:
        variants.append(_compose(edits))
    rng = jax.random.PRNGKey(seed)

    def run(
        states: Sequence[Any], skill_ids: Sequence[SkillId]
    ) -> list[SkillRolloutTrace]:
        return list(
            trace_skills_batched(
                executor,
                states,
                skill_ids,
                rng,
                skill_horizon=skill_horizon,
            )
        )

    skill_ids = [boundary.skill_id for boundary in boundaries]
    original = run([boundary.state for boundary in boundaries], skill_ids)
    comparisons: dict[str, Any] = {}
    traces: dict[str, list[SkillRolloutTrace | None]] = {"none": list(original)}
    for edit in variants:
        selected = [
            index
            for index, boundary in enumerate(boundaries)
            if boundary.skill_id in edit.applies_to
        ]
        if not selected:
            traces[edit.name] = [None] * len(boundaries)
            continue
        # Boundaries the edit does not apply to run unedited as padding, so
        # every variant reuses the probe compiled for the unedited batch.
        edited = run(
            [
                _apply_edit(boundary.state, edit=edit, skill_id=boundary.skill_id)
                if boundary.skill_id in edit.applies_to
                else boundary.state
                for boundary in boundaries
            ],
            skill_ids,
        )
        traces[edit.name] = [
            trace if boundary.skill_id in edit.applies_to else None
            for trace, boundary in zip(edited, boundaries, strict=True)
        ]
        comparisons[edit.name] = {
            "description": edit.description,
            **paired_success_summary(
                [original[index].probe.succeeded for index in selected],
                [edited[index].probe.succeeded for index in selected],
            ),
        }

    return {
        "num_boundaries": len(boundaries),
        "skill_horizon": skill_horizon,
        "replay_agreement": sum(
            trace.probe.succeeded == boundary.succeeded
            for trace, boundary in zip(original, boundaries, strict=True)
        )
        / len(boundaries),
        "comparisons": comparisons,
        "traces": traces,
    }


def _near(position: np.ndarray, targets: np.ndarray, *, diagonal: bool) -> bool:
    if targets.size == 0:
        return False
    delta = np.abs(targets - position[None, :])
    if diagonal:
        return bool(np.any(np.max(delta, axis=1) == 1))
    return bool(np.any(np.sum(delta, axis=1) == 1))


def _facing(
    position: np.ndarray, direction: int, target_set: set[tuple[int, int]]
) -> bool:
    front = position + np.asarray(DIRECTIONS[direction])
    return (int(front[0]), int(front[1])) in target_set


def rollout_features(
    state: Any,
    skill_id: SkillId,
    trace: SkillRolloutTrace,
) -> dict[str, Any]:
    """Describe what one option did, relative to its targets at the start.

    Targets are read from the starting map. Trees and water persist and a
    stone only disappears on success, so the starting map is exact for every
    failed collect or drink option. ``place_table`` has no target and reports
    only the movement features.
    """

    block = TARGET_BLOCKS.get(skill_id)
    targets = (
        np.zeros((0, 2), dtype=np.int64)
        if block is None
        else np.argwhere(np.asarray(state.map) == block.value)
    )
    target_set = {(int(row), int(column)) for row, column in targets}
    diagonal = skill_id in _CRAFTING_SKILLS
    steps = trace.probe.steps
    start_position = np.asarray(state.player_position, dtype=np.int64)
    start_direction = int(np.asarray(state.player_direction))
    positions = [start_position, *np.asarray(trace.positions[:steps], dtype=np.int64)]
    directions = [start_direction, *[int(value) for value in trace.directions[:steps]]]
    actions = [int(value) for value in trace.actions[:steps]]

    near = [_near(position, targets, diagonal=diagonal) for position in positions]
    facing = [
        _facing(position, direction, target_set)
        for position, direction in zip(positions, directions, strict=True)
    ]
    distinct = {(int(position[0]), int(position[1])) for position in positions}
    moved = [
        not np.array_equal(positions[index], positions[index + 1])
        for index in range(steps)
    ]
    blocked_moves = sum(
        action in _MOVEMENT_ACTIONS
        and not moved[index]
        and directions[index] == directions[index + 1]
        for index, action in enumerate(actions)
    )
    nearest = (
        int(np.min(np.sum(np.abs(targets - positions[-1][None, :]), axis=1)))
        if targets.size
        else None
    )
    return {
        "skill_id": skill_id.value,
        "outcome": trace.probe.outcome.value,
        "steps": steps,
        "started_near": near[0],
        "ever_near": any(near),
        "ever_facing": any(facing),
        "steps_near": sum(near[1:]),
        "interact_while_facing": sum(
            action == Action.DO.value and facing[index]
            for index, action in enumerate(actions)
        ),
        "noop_steps": sum(action == Action.NOOP.value for action in actions),
        "blocked_moves": blocked_moves,
        "distinct_positions": len(distinct),
        "revisit_fraction": 1.0 - len(distinct) / (steps + 1),
        "final_target_distance": nearest,
    }


def summarize_rollout_features(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric and boolean rollout features over a set of options."""

    if not rows:
        return {"count": 0}
    summary: dict[str, Any] = {"count": len(rows)}
    for name, value in rows[0].items():
        if isinstance(value, bool | int | float) or value is None:
            values = [row[name] for row in rows if row[name] is not None]
            summary[name] = float(np.mean(values)) if values else None
    return summary


def failed_rollout_features(
    boundaries: Sequence[ChainBoundary],
    traces: Sequence[SkillRolloutTrace | None],
) -> list[dict[str, Any]]:
    """Rollout features for every failed probe, tagged with its boundary."""

    rows = []
    for boundary, trace in zip(boundaries, traces, strict=True):
        if trace is None or trace.probe.succeeded:
            continue
        if trace.probe.outcome is SkillOutcome.INFEASIBLE:
            continue
        rows.append(
            {
                "plan": boundary.plan,
                "occurrence": boundary.occurrence,
                "band": boundary.band,
                **rollout_features(boundary.state, boundary.skill_id, trace),
            }
        )
    return rows


def chain_plans(chain: SkillChainSpec) -> dict[str, tuple[SkillId, ...]]:
    """Return the chain's reference plan and its collect-first reordering."""

    return {
        "default": chain.reference_skills,
        "collect_first": COLLECT_FIRST_PLANS[chain.name],
    }


PLANS: dict[str, tuple[SkillId, ...]] = chain_plans(WOOD_TO_STONE_CHAIN)


def benchmark_executor(
    executor: LearnedSkillExecutor,
    *,
    num_episodes: int = 512,
    seed: int = EXECUTOR_BENCHMARK_SEED,
    skill_horizon: int = 64,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    plans: dict[str, tuple[SkillId, ...]] | None = None,
    local_episodes_per_skill: int = 256,
    local_horizon: int = 32,
) -> dict[str, Any]:
    """Run the executor-only benchmark that planner work is compared against.

    Every plan runs over the same ``num_episodes`` worlds so plans and
    checkpoints pair episode by episode. ``local_navigation`` at its training
    horizon is the in-distribution regression check: a new training
    distribution that raises chain success by giving up the original skills
    has to show up there.
    """

    selected_plans = chain_plans(chain) if plans is None else plans
    chains = {
        name: evaluate_learned_skill_chain_batched(
            executor,
            chain=chain,
            plan=plan,
            num_episodes=num_episodes,
            seed=seed,
            skill_horizon=skill_horizon,
            record_boundaries=True,
        )
        for name, plan in selected_plans.items()
    }
    boundaries = {
        name: chain_boundaries(result, plan_name=name)
        for name, result in chains.items()
    }
    names = list(chains)
    plan_comparisons = {
        f"{first}_vs_{second}": compare_paired_successes(
            chains[first],
            chains[second],
        )
        for index, first in enumerate(names)
        for second in names[index + 1 :]
    }
    local = (
        evaluate_goal_conditioned_policy(
            executor.params,
            executor.config,
            num_episodes_per_skill=local_episodes_per_skill,
            seed=seed,
            scenario="local_navigation",
            horizon=local_horizon,
        )
        if local_episodes_per_skill
        else []
    )
    return {
        "seed": seed,
        "num_episodes": num_episodes,
        "skill_horizon": skill_horizon,
        "chains": chains,
        "boundaries": boundaries,
        "plan_comparisons": plan_comparisons,
        "local_navigation": local,
    }


def compare_executor_benchmarks(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Pair two checkpoints' benchmarks plan by plan and boundary band.

    Plan comparisons pair whole episodes. Boundary bands cannot be paired,
    because a different executor reaches different states, so they are
    reported side by side instead.
    """

    plans = [name for name in reference["chains"] if name in candidate["chains"]]
    candidate_local = {
        result["skill_id"]: result for result in candidate["local_navigation"]
    }
    return {
        "plans": {
            name: compare_paired_successes(
                reference["chains"][name],
                candidate["chains"][name],
            )
            for name in plans
        },
        "bands": {
            name: {
                "reference": success_table(
                    reference["boundaries"][name],
                    lambda boundary: (boundary.skill_id.value, boundary.band),
                ),
                "candidate": success_table(
                    candidate["boundaries"][name],
                    lambda boundary: (boundary.skill_id.value, boundary.band),
                ),
            }
            for name in plans
        },
        # Paired by skill, over the skills both executors know, so a run that
        # adds a skill still compares against one that lacks it.
        "local_navigation": [
            {
                "skill_id": old["skill_id"],
                "reference": old["success_rate"],
                "candidate": candidate_local[old["skill_id"]]["success_rate"],
            }
            for old in reference["local_navigation"]
            if old["skill_id"] in candidate_local
        ],
    }


def json_ready_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Drop recorded states so a benchmark can be written as JSON."""

    return {
        **benchmark,
        "chains": {
            name: {
                key: value for key, value in result.items() if key != "boundary_states"
            }
            for name, result in benchmark["chains"].items()
        },
        "boundaries": {
            name: [boundary.as_row() for boundary in boundaries]
            for name, boundaries in benchmark["boundaries"].items()
        },
    }
