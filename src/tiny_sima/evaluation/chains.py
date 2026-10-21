"""Closed-loop evaluation for persistent SayCan-style skill chains."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import (
    DIRECTIONS,
    SOLID_BLOCKS,
    Action,
    BlockType,
)

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.chains import (
    SkillChainEnvironment,
    make_skill_chain_environment,
)
from tiny_sima.envs.craftax import step_batch
from tiny_sima.envs.skills import (
    SkillEnvironment,
    SkillEpisodeState,
    SkillScenarioConfig,
)
from tiny_sima.evaluation.goal_conditioned import (
    LoadedGoalConditionedCheckpoint,
    load_goal_conditioned_checkpoint,
)
from tiny_sima.executor.model import (
    GoalConditionedActorCriticRNN,
    apply_goal_conditioned_actor_critic,
    initial_hidden,
    make_goal_conditioned_actor_critic,
    select_action,
)
from tiny_sima.goals import (
    SKILL_IDS,
    SkillGoal,
    SkillId,
    get_skill_spec,
    skill_index,
    skill_metric,
    skill_precondition,
)
from tiny_sima.planner.saycan import (
    PlannerContext,
    PlannerDecision,
    PlannerSelection,
    PlannerSelectionStatus,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillAttempt,
    SkillOutcome,
    SkillScore,
)

ObjectivePredicate = Callable[[Any], bool]
JaxObjectivePredicate = Callable[[Any], jax.Array]
ValidNextSkills = Callable[[Any], tuple[SkillId, ...]]


class SkillChainSpec(NamedTuple):
    """A state-based objective with an optional scripted reference plan."""

    name: str
    objective: str
    reference_skills: tuple[SkillId, ...]
    is_complete: ObjectivePredicate
    is_complete_jax: JaxObjectivePredicate | None = None
    valid_next_skills: ValidNextSkills | None = None


def _wood_to_stone_complete_jax(state: Any) -> jax.Array:
    inventory = state.inventory
    return (jnp.asarray(inventory.wood_pickaxe) > 0) & (
        jnp.asarray(inventory.stone) > 0
    )


def _wood_to_stone_complete(state: Any) -> bool:
    return bool(np.asarray(_wood_to_stone_complete_jax(state)))


def _wood_to_stone_valid_next_skills(state: Any) -> tuple[SkillId, ...]:
    """Return objective-progressing skills, including valid plan reorderings."""

    inventory = state.inventory
    wood = int(np.asarray(inventory.wood))
    stone = int(np.asarray(inventory.stone))
    wood_pickaxe = int(np.asarray(inventory.wood_pickaxe))
    has_table = bool(
        np.any(np.asarray(state.map) == BlockType.CRAFTING_TABLE.value)
    )
    if wood_pickaxe > 0 and stone > 0:
        return ()
    if wood_pickaxe > 0:
        return (SkillId.COLLECT_STONE,)
    if has_table:
        if wood >= 1:
            return (SkillId.CRAFT_WOOD_PICKAXE,)
        return (SkillId.COLLECT_WOOD,)
    if wood >= 3:
        return (SkillId.PLACE_TABLE,)
    if wood == 2:
        return (SkillId.COLLECT_WOOD, SkillId.PLACE_TABLE)
    return (SkillId.COLLECT_WOOD,)


WOOD_TO_STONE_CHAIN = SkillChainSpec(
    name="wood_to_stone",
    objective="Craft a wooden pickaxe and use it to collect stone.",
    reference_skills=(
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.PLACE_TABLE,
        SkillId.COLLECT_WOOD,
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.COLLECT_STONE,
    ),
    is_complete=_wood_to_stone_complete,
    is_complete_jax=_wood_to_stone_complete_jax,
    valid_next_skills=_wood_to_stone_valid_next_skills,
)


class ChainTermination(str, Enum):
    """Why a hierarchical episode stopped."""

    SUCCESS = "success"
    NO_VIABLE_SKILL = "no_viable_skill"
    FAILURE_LIMIT = "failure_limit"
    DECISION_LIMIT = "decision_limit"
    INVALID_PLANNER_OUTPUT = "invalid_planner_output"


@dataclass(frozen=True)
class SkillExecutionResult:
    """State and terminal diagnostics returned by one executor call."""

    state: Any
    outcome: SkillOutcome
    steps: int
    skill_return: float
    native_return: float


@dataclass(frozen=True)
class ChainStepTrace:
    """One planner decision followed by one completed executor attempt."""

    decision_index: int
    requested_skill: SkillId
    outcome: SkillOutcome
    steps: int
    native_return: float
    utility: float
    feasibility: float
    combined_score: float
    candidate_scores: tuple[SkillScore, ...]
    planner_ranking: tuple[SkillId, ...]
    planner_raw_response: str | None


@dataclass(frozen=True)
class ChainEpisodeResult:
    """Final metrics and trace for one persistent chain episode."""

    succeeded: bool
    completed_skills: int
    decisions: int
    environment_steps: int
    native_return: float
    termination: ChainTermination
    planner_error: str | None
    planner_raw_response: str | None
    trace: tuple[ChainStepTrace, ...]


class BatchedChainCarry(NamedTuple):
    """JAX state for parallel fixed-plan chain evaluation."""

    rng: jax.Array
    state: Any
    observation: jax.Array
    hidden: jax.Array
    plan_index: jax.Array
    option_steps: jax.Array
    initial_metric: jax.Array
    failures_by_skill: jax.Array
    finished: jax.Array
    succeeded: jax.Array
    termination: jax.Array
    completed_skills: jax.Array
    decisions: jax.Array
    environment_steps: jax.Array
    native_return: jax.Array
    last_failed_skill: jax.Array
    first_outcome: jax.Array
    first_steps: jax.Array
    boundary_states: Any


class BatchedChainOutput(NamedTuple):
    """Per-episode metrics produced by the compiled chain evaluator.

    ``first_outcome`` and ``first_steps`` have one column per plan step and
    describe the first attempt at that step; ``-1`` marks a step never
    reached. ``boundary_states`` holds the world at the start of each first
    attempt when recording was requested, and is ``None`` otherwise.
    """

    succeeded: jax.Array
    termination: jax.Array
    completed_skills: jax.Array
    decisions: jax.Array
    environment_steps: jax.Array
    native_return: jax.Array
    last_failed_skill: jax.Array
    first_outcome: jax.Array
    first_steps: jax.Array
    boundary_states: Any


class SkillExecutor(Protocol):
    """Boundary implemented by scripted and learned low-level policies."""

    def execute(
        self,
        environment: SkillChainEnvironment,
        state: Any,
        skill_id: SkillId,
        rng: jax.Array,
        *,
        horizon: int,
    ) -> SkillExecutionResult:
        """Attempt one requested skill without resetting the world."""

        ...


class NavigationTarget(NamedTuple):
    """Shortest-path action and final direction toward an interaction target."""

    movement_action: int | None
    target_direction: int


_INTERACTION_BLOCKS = {
    SkillId.COLLECT_WOOD: BlockType.TREE,
    SkillId.COLLECT_STONE: BlockType.STONE,
    SkillId.DRINK_WATER: BlockType.WATER,
    SkillId.CRAFT_WOOD_PICKAXE: BlockType.CRAFTING_TABLE,
    SkillId.CRAFT_WOOD_SWORD: BlockType.CRAFTING_TABLE,
}

_CARDINAL_ACTIONS = (
    Action.LEFT,
    Action.RIGHT,
    Action.UP,
    Action.DOWN,
)


def _is_walkable(map_array: np.ndarray, position: tuple[int, int]) -> bool:
    row, column = position
    if not (0 <= row < map_array.shape[0] and 0 <= column < map_array.shape[1]):
        return False
    blocked = set(np.asarray(SOLID_BLOCKS).tolist()) | {BlockType.LAVA.value}
    return int(map_array[row, column]) not in blocked


def _nearest_navigation_target(
    state: Any,
    block: BlockType,
) -> NavigationTarget | None:
    map_array = np.asarray(state.map)
    player_position = np.asarray(state.player_position)
    start = (int(player_position[0]), int(player_position[1]))
    targets = {
        tuple(int(value) for value in position)
        for position in np.argwhere(map_array == block.value)
    }
    if not targets:
        return None

    queue: deque[tuple[tuple[int, int], int | None]] = deque([(start, None)])
    visited = {start}
    while queue:
        position, first_action = queue.popleft()
        for action in _CARDINAL_ACTIONS:
            direction = np.asarray(DIRECTIONS[action.value])
            adjacent = (
                position[0] + int(direction[0]),
                position[1] + int(direction[1]),
            )
            if adjacent in targets:
                return NavigationTarget(
                    movement_action=first_action,
                    target_direction=action.value,
                )

        for action in _CARDINAL_ACTIONS:
            direction = np.asarray(DIRECTIONS[action.value])
            neighbor = (
                position[0] + int(direction[0]),
                position[1] + int(direction[1]),
            )
            if neighbor in visited or not _is_walkable(map_array, neighbor):
                continue
            visited.add(neighbor)
            queue.append(
                (
                    neighbor,
                    action.value if first_action is None else first_action,
                )
            )
    return None


def _scripted_action(state: Any, skill_id: SkillId) -> int:
    if skill_id is SkillId.PLACE_TABLE:
        front = np.asarray(state.player_position) + np.asarray(
            DIRECTIONS[int(state.player_direction)]
        )
        if _is_walkable(np.asarray(state.map), (int(front[0]), int(front[1]))):
            return Action.PLACE_TABLE.value
        for action in _CARDINAL_ACTIONS:
            candidate = np.asarray(state.player_position) + np.asarray(
                DIRECTIONS[action.value]
            )
            if _is_walkable(
                np.asarray(state.map),
                (int(candidate[0]), int(candidate[1])),
            ):
                return action.value
        return Action.NOOP.value

    target = _nearest_navigation_target(state, _INTERACTION_BLOCKS[skill_id])
    if target is None:
        return Action.NOOP.value
    if target.movement_action is not None:
        return target.movement_action
    if skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
    }:
        return get_skill_spec(skill_id).interaction_action.value
    if int(state.player_direction) != target.target_direction:
        return target.target_direction
    return Action.DO.value


def _start_skill(
    environment: SkillChainEnvironment,
    state: Any,
    skill_id: SkillId,
    horizon: int,
) -> tuple[SkillEnvironment, SkillEpisodeState]:
    skill_environment = SkillEnvironment(
        bundle=environment.bundle,
        goal=SkillGoal(skill_id, horizon=horizon),
        scenario=SkillScenarioConfig(),
    )
    position = jnp.asarray(state.player_position, dtype=jnp.int32)
    episode_state = SkillEpisodeState(
        env_state=state,
        initial_metric=skill_metric(skill_id, state),
        steps=jnp.asarray(0, dtype=jnp.int32),
        target_direction=jnp.asarray(state.player_direction, dtype=jnp.int32),
        target_position=position,
        approach_position=position,
    )
    return skill_environment, episode_state


def _outcome_from_step(result: Any) -> SkillOutcome | None:
    if bool(np.asarray(result.info["skill_success"])):
        return SkillOutcome.SUCCESS
    if not bool(np.asarray(result.done)):
        return None
    if not bool(np.asarray(result.info["skill_precondition_after"])):
        return SkillOutcome.PRECONDITION_LOST
    if bool(np.asarray(result.info["skill_timeout"])):
        return SkillOutcome.TIMEOUT
    return SkillOutcome.ENVIRONMENT_TERMINATED


@dataclass(frozen=True)
class ScriptedSkillExecutor:
    """Shortest-path oracle used to validate chain construction."""

    def execute(
        self,
        environment: SkillChainEnvironment,
        state: Any,
        skill_id: SkillId,
        rng: jax.Array,
        *,
        horizon: int,
    ) -> SkillExecutionResult:
        skill_environment, episode_state = _start_skill(
            environment,
            state,
            skill_id,
            horizon,
        )
        if not bool(np.asarray(skill_precondition(skill_environment.goal, state))):
            return SkillExecutionResult(
                state=state,
                outcome=SkillOutcome.INFEASIBLE,
                steps=0,
                skill_return=0.0,
                native_return=0.0,
            )

        skill_return = 0.0
        native_return = 0.0
        current = episode_state
        step_keys = jax.random.split(rng, horizon)
        for step_index in range(horizon):
            action = _scripted_action(current.env_state, skill_id)
            result = skill_environment.step(
                step_keys[step_index],
                current,
                jnp.asarray(action, dtype=jnp.int32),
            )
            current = result.state
            skill_return += float(np.asarray(result.reward))
            native_return += float(np.asarray(result.info["native_reward"]))
            outcome = _outcome_from_step(result)
            if outcome is not None:
                return SkillExecutionResult(
                    state=current.env_state,
                    outcome=outcome,
                    steps=step_index + 1,
                    skill_return=skill_return,
                    native_return=native_return,
                )
        return SkillExecutionResult(
            state=current.env_state,
            outcome=SkillOutcome.TIMEOUT,
            steps=horizon,
            skill_return=skill_return,
            native_return=native_return,
        )


@runtime_checkable
class BatchedSkillExecutor(Protocol):
    """An executor that runs several independent skill attempts at once."""

    def execute_many(
        self,
        environment: SkillChainEnvironment,
        states: Sequence[Any],
        skill_ids: Sequence[SkillId],
        rngs: Sequence[jax.Array],
        *,
        horizon: int,
        batch_size: int | None = None,
    ) -> list[SkillExecutionResult]:
        """Attempt ``skill_ids[i]`` from ``states[i]`` with ``rngs[i]``."""

        ...


def _execute_many(
    executor: SkillExecutor,
    environment: SkillChainEnvironment,
    states: Sequence[Any],
    skill_ids: Sequence[SkillId],
    rngs: Sequence[jax.Array],
    *,
    horizon: int,
    batch_size: int | None = None,
) -> list[SkillExecutionResult]:
    """Run attempts together when the executor can, one by one otherwise."""

    if isinstance(executor, BatchedSkillExecutor):
        return executor.execute_many(
            environment,
            states,
            skill_ids,
            rngs,
            horizon=horizon,
            batch_size=batch_size,
        )
    return [
        executor.execute(environment, state, skill_id, rng, horizon=horizon)
        for state, skill_id, rng in zip(states, skill_ids, rngs, strict=True)
    ]


@dataclass(frozen=True)
class LearnedSkillExecutor:
    """Checkpoint-backed low-level policy used in persistent worlds.

    Every attempt runs inside one compiled loop, so nothing returns to the
    host until the option ends. Stepping the environment from Python instead
    dispatched each Craftax operation separately and read four flags back per
    step, which made a 64-step option cost seconds rather than milliseconds.
    """

    params: Any
    config: SkillPPOConfig
    model: GoalConditionedActorCriticRNN
    deterministic: bool = True

    def execute(
        self,
        environment: SkillChainEnvironment,
        state: Any,
        skill_id: SkillId,
        rng: jax.Array,
        *,
        horizon: int,
    ) -> SkillExecutionResult:
        return self.execute_many(
            environment,
            [state],
            [skill_id],
            [rng],
            horizon=horizon,
        )[0]

    def execute_many(
        self,
        environment: SkillChainEnvironment,
        states: Sequence[Any],
        skill_ids: Sequence[SkillId],
        rngs: Sequence[jax.Array],
        *,
        horizon: int,
        batch_size: int | None = None,
    ) -> list[SkillExecutionResult]:
        """Run independent attempts in parallel on device.

        Each attempt draws from its own key exactly as a lone attempt would,
        so batching changes how long the call takes, not what happens.
        ``batch_size`` pads the call to a fixed shape: a caller whose batch
        shrinks as episodes finish reuses one compiled loop instead of
        compiling one per size.
        """

        count = len(states)
        if len(skill_ids) != count or len(rngs) != count:
            raise ValueError("states, skill_ids and rngs must have equal length.")
        if count == 0:
            return []
        size = count if batch_size is None else batch_size
        if size < count:
            raise ValueError(f"batch_size {size} is smaller than {count} attempts.")
        if horizon < 1:
            raise ValueError("horizon must be positive.")

        padding = size - count
        run = _batched_skill_execution(
            environment.bundle.env,
            batch_size=size,
            skill_horizon=horizon,
            num_actions=environment.bundle.spec.num_actions,
            hidden_size=self.config.hidden_size,
            skill_embedding_size=self.config.skill_embedding_size,
            deterministic=self.deterministic,
        )
        output = run(
            self.params,
            environment.bundle.params,
            _stack_host_states([*states, *[states[0]] * padding]),
            jnp.asarray(
                [skill_index(skill_id) for skill_id in skill_ids] + [0] * padding,
                dtype=jnp.int32,
            ),
            jnp.asarray([True] * count + [False] * padding),
            jnp.stack([*rngs, *[rngs[0]] * padding]),
        )
        # One transfer for the whole batch. The planner reads these states
        # field by field, which costs nothing on host arrays.
        output = jax.device_get(output)
        return [
            SkillExecutionResult(
                state=jax.tree_util.tree_map(
                    lambda leaf, index=index: leaf[index],
                    output.state,
                ),
                outcome=_PROBE_OUTCOMES[int(output.outcome[index])],
                steps=int(output.steps[index]),
                skill_return=float(
                    _PROBE_OUTCOMES[int(output.outcome[index])]
                    is SkillOutcome.SUCCESS
                ),
                native_return=float(output.native_return[index]),
            )
            for index in range(count)
        ]


def make_learned_skill_executor(
    checkpoint: LoadedGoalConditionedCheckpoint,
    *,
    deterministic: bool = True,
) -> LearnedSkillExecutor:
    """Construct a persistent-world executor from a loaded checkpoint."""

    environment = make_skill_chain_environment()
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=checkpoint.config.hidden_size,
        skill_embedding_size=checkpoint.config.skill_embedding_size,
    )
    return LearnedSkillExecutor(
        params=checkpoint.params,
        config=checkpoint.config,
        model=model,
        deterministic=deterministic,
    )


def learned_skill_executor_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    deterministic: bool = True,
) -> LearnedSkillExecutor:
    """Load a checkpoint and adapt it to persistent skill execution."""

    return make_learned_skill_executor(
        load_goal_conditioned_checkpoint(checkpoint_dir),
        deterministic=deterministic,
    )


def _dynamic_skill_metric(skill_id: jax.Array, state: Any) -> jax.Array:
    """Evaluate a categorical skill metric inside compiled control flow."""

    branches = (
        lambda value: jnp.asarray(value.inventory.wood),
        lambda value: jnp.asarray(value.inventory.stone),
        lambda value: jnp.asarray(value.player_drink),
        lambda value: jnp.count_nonzero(
            value.map == BlockType.CRAFTING_TABLE.value
        ),
        lambda value: jnp.asarray(value.inventory.wood_pickaxe),
        lambda value: jnp.asarray(value.inventory.wood_sword),
    )
    return jax.lax.switch(skill_id, branches, state)


def _dynamic_skill_precondition(
    skill_id: jax.Array,
    state: Any,
) -> jax.Array:
    """Evaluate fixed skill preconditions for a traced categorical ID."""

    def collect_wood(value: Any) -> jax.Array:
        return (jnp.asarray(value.player_health) > 0) & jnp.any(
            value.map == BlockType.TREE.value
        )

    def collect_stone(value: Any) -> jax.Array:
        return (
            (jnp.asarray(value.player_health) > 0)
            & (jnp.asarray(value.inventory.wood_pickaxe) > 0)
            & jnp.any(value.map == BlockType.STONE.value)
        )

    def drink_water(value: Any) -> jax.Array:
        return (
            (jnp.asarray(value.player_health) > 0)
            & (jnp.asarray(value.player_drink) < 9)
            & jnp.any(value.map == BlockType.WATER.value)
        )

    def place_table(value: Any) -> jax.Array:
        return (jnp.asarray(value.player_health) > 0) & (
            jnp.asarray(value.inventory.wood) >= 2
        )

    def craft_wood_tool(value: Any) -> jax.Array:
        return (
            (jnp.asarray(value.player_health) > 0)
            & (jnp.asarray(value.inventory.wood) >= 1)
            & jnp.any(value.map == BlockType.CRAFTING_TABLE.value)
        )

    branches = (
        collect_wood,
        collect_stone,
        drink_water,
        place_table,
        craft_wood_tool,
        craft_wood_tool,
    )
    return jax.lax.switch(skill_id, branches, state)


def _batched_skill_metric(skill_ids: jax.Array, state: Any) -> jax.Array:
    return jax.vmap(_dynamic_skill_metric)(skill_ids, state)


def _batched_skill_precondition(
    skill_ids: jax.Array,
    state: Any,
) -> jax.Array:
    return jax.vmap(_dynamic_skill_precondition)(skill_ids, state)


def _select_batched_state(mask: jax.Array, selected: Any, fallback: Any) -> Any:
    def select_leaf(
        selected_leaf: jax.Array,
        fallback_leaf: jax.Array,
    ) -> jax.Array:
        mask_shape = (mask.shape[0],) + (1,) * (selected_leaf.ndim - 1)
        return jnp.where(mask.reshape(mask_shape), selected_leaf, fallback_leaf)

    return jax.tree_util.tree_map(select_leaf, selected, fallback)


def _record_skill_failures(
    failures_by_skill: jax.Array,
    skill_ids: jax.Array,
    failed: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Accumulate failures across every occurrence of each requested skill."""

    failure_delta = jax.nn.one_hot(
        skill_ids,
        len(SKILL_IDS),
        dtype=jnp.int32,
    ) * failed[:, None]
    updated = failures_by_skill + failure_delta
    selected_failures = jnp.take_along_axis(
        updated,
        skill_ids[:, None],
        axis=1,
    )[:, 0]
    return updated, selected_failures


BatchedRollout = Callable[
    [Any, jax.Array, jax.Array],
    BatchedChainOutput,
]


@lru_cache(maxsize=16)
def _make_batched_chain_rollout(
    *,
    num_episodes: int,
    skill_horizon: int,
    max_decisions: int,
    max_failures_per_skill: int,
    distractor_count: int,
    hidden_size: int,
    skill_embedding_size: int,
    deterministic: bool,
    is_complete: JaxObjectivePredicate,
    record_boundaries: bool = False,
) -> BatchedRollout:
    """Build and cache one compiled evaluator for a static execution shape."""

    success_outcome = _PROBE_OUTCOMES.index(SkillOutcome.SUCCESS)
    timeout_outcome = _PROBE_OUTCOMES.index(SkillOutcome.TIMEOUT)
    terminated_outcome = _PROBE_OUTCOMES.index(SkillOutcome.ENVIRONMENT_TERMINATED)
    lost_outcome = _PROBE_OUTCOMES.index(SkillOutcome.PRECONDITION_LOST)
    infeasible_outcome = _PROBE_OUTCOMES.index(SkillOutcome.INFEASIBLE)
    environment = make_skill_chain_environment(
        distractor_count=distractor_count,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=hidden_size,
        skill_embedding_size=skill_embedding_size,
    )
    max_primitive_steps = max_decisions * skill_horizon
    success_code = list(ChainTermination).index(ChainTermination.SUCCESS)
    no_viable_code = list(ChainTermination).index(
        ChainTermination.NO_VIABLE_SKILL
    )
    failure_limit_code = list(ChainTermination).index(
        ChainTermination.FAILURE_LIMIT
    )
    decision_limit_code = list(ChainTermination).index(
        ChainTermination.DECISION_LIMIT
    )

    def rollout(
        params: Any,
        plan: jax.Array,
        rng: jax.Array,
    ) -> BatchedChainOutput:
        plan_length = plan.shape[0]
        reset_rng, rollout_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, num_episodes)
        reset = jax.vmap(environment.reset)(reset_keys)
        initial_skill_ids = jnp.full(
            (num_episodes,),
            plan[0],
            dtype=jnp.int32,
        )
        initially_succeeded = jax.vmap(is_complete)(reset.state)
        episode_rows = jnp.arange(num_episodes)
        boundary_states = (
            jax.tree_util.tree_map(
                lambda leaf: jnp.repeat(leaf[:, None], plan_length, axis=1),
                reset.state,
            )
            if record_boundaries
            else None
        )
        initial_carry = BatchedChainCarry(
            rng=rollout_rng,
            state=reset.state,
            observation=reset.observation,
            hidden=initial_hidden(num_episodes, hidden_size),
            plan_index=jnp.zeros((num_episodes,), dtype=jnp.int32),
            option_steps=jnp.zeros((num_episodes,), dtype=jnp.int32),
            initial_metric=_batched_skill_metric(
                initial_skill_ids,
                reset.state,
            ),
            failures_by_skill=jnp.zeros(
                (num_episodes, len(SKILL_IDS)),
                dtype=jnp.int32,
            ),
            finished=initially_succeeded,
            succeeded=initially_succeeded,
            termination=jnp.where(
                initially_succeeded,
                success_code,
                decision_limit_code,
            ),
            completed_skills=jnp.zeros((num_episodes,), dtype=jnp.int32),
            decisions=jnp.zeros((num_episodes,), dtype=jnp.int32),
            environment_steps=jnp.zeros((num_episodes,), dtype=jnp.int32),
            native_return=jnp.zeros((num_episodes,), dtype=jnp.float32),
            last_failed_skill=jnp.full(
                (num_episodes,),
                -1,
                dtype=jnp.int32,
            ),
            first_outcome=jnp.full(
                (num_episodes, plan_length),
                -1,
                dtype=jnp.int32,
            ),
            first_steps=jnp.full(
                (num_episodes, plan_length),
                -1,
                dtype=jnp.int32,
            ),
            boundary_states=boundary_states,
        )

        def continue_evaluation(
            loop_state: tuple[jax.Array, BatchedChainCarry],
        ) -> jax.Array:
            iteration, carry = loop_state
            return (iteration < max_primitive_steps) & ~jnp.all(carry.finished)

        def step(
            loop_state: tuple[jax.Array, BatchedChainCarry],
        ) -> tuple[jax.Array, BatchedChainCarry]:
            iteration, carry = loop_state
            rng, action_rng, environment_rng = jax.random.split(carry.rng, 3)
            plan_exhausted = carry.plan_index >= plan_length
            safe_plan_index = jnp.minimum(carry.plan_index, plan_length - 1)
            skill_ids = plan[safe_plan_index]
            at_option_boundary = carry.option_steps == 0
            precondition = _batched_skill_precondition(skill_ids, carry.state)
            decision_limit_reached = (
                ~carry.finished
                & at_option_boundary
                & (carry.decisions >= max_decisions)
            )
            no_viable_skill = (
                ~carry.finished
                & ~decision_limit_reached
                & at_option_boundary
                & (plan_exhausted | ~precondition)
            )
            active = (
                ~carry.finished
                & ~decision_limit_reached
                & ~no_viable_skill
            )
            # Retries reuse a plan step, so a step whose outcome is already
            # set is not a first attempt and must not overwrite its record.
            first_attempt = carry.first_outcome[episode_rows, safe_plan_index] < 0
            boundary_states = carry.boundary_states
            if record_boundaries:
                starting = active & at_option_boundary & first_attempt

                def record_start(buffer: jax.Array, leaf: jax.Array) -> jax.Array:
                    mask_shape = (num_episodes,) + (1,) * (leaf.ndim - 1)
                    current = buffer[episode_rows, safe_plan_index]
                    return buffer.at[episode_rows, safe_plan_index].set(
                        jnp.where(starting.reshape(mask_shape), leaf, current)
                    )

                boundary_states = jax.tree_util.tree_map(
                    record_start,
                    carry.boundary_states,
                    carry.state,
                )

            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    skill_ids[None, ...],
                    (at_option_boundary | carry.finished)[None, ...],
                ),
            )
            action = select_action(
                model_output.logits[0],
                action_rng,
                deterministic=deterministic,
            )
            action = jnp.where(active, action, Action.NOOP.value)
            environment_step = step_batch(
                environment.bundle,
                environment_rng,
                carry.state,
                action,
            )
            state = _select_batched_state(
                active,
                environment_step.state,
                carry.state,
            )
            observation = jnp.where(
                active[:, None],
                environment_step.observation,
                carry.observation,
            )

            option_steps = carry.option_steps + active.astype(jnp.int32)
            metric = _batched_skill_metric(skill_ids, state)
            success = active & (metric - carry.initial_metric >= 1)
            timeout = active & (option_steps >= skill_horizon) & ~success
            environment_terminated = (
                active & environment_step.done & ~success & ~timeout
            )
            option_ended = success | timeout | environment_terminated
            failed = option_ended & ~success
            failures_by_skill, selected_failures = _record_skill_failures(
                carry.failures_by_skill,
                skill_ids,
                failed,
            )
            failure_limit = failed & (
                selected_failures >= max_failures_per_skill
            )

            precondition_after = _batched_skill_precondition(skill_ids, state)
            retry_is_infeasible = (
                failed
                & ~failure_limit
                & (~precondition_after | environment_terminated)
            )
            # Outcome codes follow the probe so the two can be compared.
            outcome_code = jnp.where(timeout, timeout_outcome, terminated_outcome)
            outcome_code = jnp.where(
                failed & ~precondition_after,
                lost_outcome,
                outcome_code,
            )
            outcome_code = jnp.where(success, success_outcome, outcome_code)
            infeasible_start = no_viable_skill & ~plan_exhausted
            outcome_code = jnp.where(
                infeasible_start,
                infeasible_outcome,
                outcome_code,
            )
            record_outcome = (option_ended | infeasible_start) & first_attempt
            first_outcome = carry.first_outcome.at[
                episode_rows, safe_plan_index
            ].set(
                jnp.where(
                    record_outcome,
                    outcome_code,
                    carry.first_outcome[episode_rows, safe_plan_index],
                )
            )
            first_steps = carry.first_steps.at[episode_rows, safe_plan_index].set(
                jnp.where(
                    record_outcome,
                    jnp.where(infeasible_start, 0, option_steps),
                    carry.first_steps[episode_rows, safe_plan_index],
                )
            )
            plan_index = carry.plan_index + success.astype(jnp.int32)
            objective_complete = jax.vmap(is_complete)(state)
            chain_succeeded = option_ended & objective_complete
            finished = (
                carry.finished
                | decision_limit_reached
                | no_viable_skill
                | failure_limit
                | retry_is_infeasible
                | chain_succeeded
            )

            termination = jnp.where(
                no_viable_skill | retry_is_infeasible,
                no_viable_code,
                carry.termination,
            )
            termination = jnp.where(
                failure_limit,
                failure_limit_code,
                termination,
            )
            termination = jnp.where(
                chain_succeeded,
                success_code,
                termination,
            )

            next_safe_plan_index = jnp.minimum(plan_index, plan_length - 1)
            next_skill_ids = plan[next_safe_plan_index]
            next_initial_metric = _batched_skill_metric(
                next_skill_ids,
                state,
            )
            initial_metric = jnp.where(
                option_ended,
                next_initial_metric,
                carry.initial_metric,
            )
            next_carry = BatchedChainCarry(
                rng=rng,
                state=state,
                observation=observation,
                hidden=model_output.hidden,
                plan_index=plan_index,
                option_steps=jnp.where(option_ended, 0, option_steps),
                initial_metric=initial_metric,
                failures_by_skill=failures_by_skill,
                finished=finished,
                succeeded=carry.succeeded | chain_succeeded,
                termination=termination,
                completed_skills=(
                    carry.completed_skills + success.astype(jnp.int32)
                ),
                decisions=carry.decisions + option_ended.astype(jnp.int32),
                environment_steps=(
                    carry.environment_steps + active.astype(jnp.int32)
                ),
                native_return=(
                    carry.native_return
                    + jnp.where(active, environment_step.reward, 0.0)
                ),
                last_failed_skill=jnp.where(
                    failed | no_viable_skill,
                    skill_ids,
                    carry.last_failed_skill,
                ),
                first_outcome=first_outcome,
                first_steps=first_steps,
                boundary_states=boundary_states,
            )
            return iteration + 1, next_carry

        _, final_carry = jax.lax.while_loop(
            continue_evaluation,
            step,
            (jnp.asarray(0, dtype=jnp.int32), initial_carry),
        )
        return BatchedChainOutput(
            succeeded=final_carry.succeeded,
            termination=final_carry.termination,
            completed_skills=final_carry.completed_skills,
            decisions=final_carry.decisions,
            environment_steps=final_carry.environment_steps,
            native_return=final_carry.native_return,
            last_failed_skill=final_carry.last_failed_skill,
            first_outcome=final_carry.first_outcome,
            first_steps=final_carry.first_steps,
            boundary_states=final_carry.boundary_states,
        )

    return jax.jit(rollout)


def evaluate_learned_skill_chain_batched(
    executor: LearnedSkillExecutor,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 64,
    seed: int = 0,
    skill_horizon: int = 64,
    max_decisions: int = 12,
    max_failures_per_skill: int = 2,
    distractor_count: int = 3,
    plan: Sequence[SkillId] | None = None,
    record_boundaries: bool = False,
) -> dict[str, Any]:
    """Evaluate a fixed plan across parallel JAX environments.

    This fast path intentionally executes a fixed plan, by default
    ``chain.reference_skills``, rather than calling an arbitrary Python
    planner inside ``jax.jit``. Use :func:`evaluate_skill_chain` when planner
    decisions or detailed traces are required.

    The same seed and episode count produce the same starting worlds for any
    plan, so two plans can be compared episode by episode with
    :func:`compare_paired_successes`. ``first_attempt_outcomes`` records how
    the first attempt at each plan step ended. With ``record_boundaries`` the
    result also carries ``boundary_states``, a host pytree whose leaves have
    leading ``(episode, plan step)`` axes holding the world at the start of
    each first attempt; entries for unreached steps hold the reset world and
    must be ignored where ``first_attempt_outcomes`` is ``None``.
    """

    plan_skills = tuple(chain.reference_skills if plan is None else plan)
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    if skill_horizon < 1:
        raise ValueError("skill_horizon must be positive.")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive.")
    if max_failures_per_skill < 1:
        raise ValueError("max_failures_per_skill must be positive.")
    if not plan_skills:
        raise ValueError("A batched chain requires at least one planned skill.")
    if chain.is_complete_jax is None:
        raise ValueError(
            "A batched chain requires a JAX-compatible completion predicate."
        )

    plan_array = jnp.asarray(
        [skill_index(skill_id) for skill_id in plan_skills],
        dtype=jnp.int32,
    )
    cache_hits_before = _make_batched_chain_rollout.cache_info().hits
    rollout = _make_batched_chain_rollout(
        num_episodes=num_episodes,
        skill_horizon=skill_horizon,
        max_decisions=max_decisions,
        max_failures_per_skill=max_failures_per_skill,
        distractor_count=distractor_count,
        hidden_size=executor.config.hidden_size,
        skill_embedding_size=executor.config.skill_embedding_size,
        deterministic=executor.deterministic,
        is_complete=chain.is_complete_jax,
        record_boundaries=record_boundaries,
    )
    compile_cache_hit = (
        _make_batched_chain_rollout.cache_info().hits > cache_hits_before
    )
    start = time.perf_counter()
    output = rollout(executor.params, plan_array, jax.random.PRNGKey(seed))
    output = jax.device_get(output)
    total_seconds = time.perf_counter() - start

    succeeded = np.asarray(output.succeeded)
    terminations = np.asarray(output.termination)
    last_failed_skills = np.asarray(output.last_failed_skill)
    termination_counts = {
        termination.value: int(
            np.count_nonzero(
                terminations == list(ChainTermination).index(termination)
            )
        )
        for termination in ChainTermination
    }
    final_failed_skill_counts = {
        skill_id.value: int(
            np.count_nonzero(
                ~succeeded & (last_failed_skills == skill_index(skill_id))
            )
        )
        for skill_id in SKILL_IDS
        if np.any(~succeeded & (last_failed_skills == skill_index(skill_id)))
    }
    first_outcomes = np.asarray(output.first_outcome)
    first_steps = np.asarray(output.first_steps)
    result: dict[str, Any] = {
        "chain": chain.name,
        "objective": chain.objective,
        "reference_skills": [
            skill.value for skill in chain.reference_skills
        ],
        "plan": [skill.value for skill in plan_skills],
        "num_episodes": num_episodes,
        "seed": seed,
        "batched": True,
        "compile_cache_hit": compile_cache_hit,
        "success_rate": float(succeeded.mean()),
        "mean_completed_skills": float(
            np.asarray(output.completed_skills).mean()
        ),
        "mean_decisions": float(np.asarray(output.decisions).mean()),
        "mean_environment_steps": float(
            np.asarray(output.environment_steps).mean()
        ),
        "mean_native_return": float(np.asarray(output.native_return).mean()),
        "termination_counts": termination_counts,
        "final_failed_skill_counts": final_failed_skill_counts,
        "total_seconds_including_compile": total_seconds,
        "episode_successes": [bool(value) for value in succeeded],
        "episode_completed_skills": [
            int(value) for value in np.asarray(output.completed_skills)
        ],
        "episode_terminations": [
            list(ChainTermination)[int(code)].value for code in terminations
        ],
        "first_attempt_outcomes": [
            [
                None if code < 0 else _PROBE_OUTCOMES[int(code)].value
                for code in row
            ]
            for row in first_outcomes
        ],
        "first_attempt_steps": [
            [None if steps < 0 else int(steps) for steps in row]
            for row in first_steps
        ],
    }
    if record_boundaries:
        result["boundary_states"] = output.boundary_states
    return result


class BatchedProbeCarry(NamedTuple):
    """JAX state for probing one option per batch entry in lockstep."""

    rng: jax.Array
    state: Any
    observation: jax.Array
    hidden: jax.Array
    steps: jax.Array
    initial_metric: jax.Array
    finished: jax.Array
    outcome: jax.Array
    positions: jax.Array
    directions: jax.Array
    actions: jax.Array


class BatchedProbeOutput(NamedTuple):
    """Per-entry probe results: how it ended and how long it took.

    ``positions``, ``directions`` and ``actions`` are time-major traces of
    shape ``(skill_horizon, batch)``; row ``t`` describes the state after
    step ``t``. Rows at or beyond an entry's ``steps`` are padding.
    """

    outcome: jax.Array
    steps: jax.Array
    positions: jax.Array
    directions: jax.Array
    actions: jax.Array


@dataclass(frozen=True)
class SkillProbe:
    """How one skill ended when attempted from a given state.

    The resulting world state is deliberately absent: a probe exists to label
    a boundary state, and every probe runs on a throwaway copy so that nothing
    it does can perturb the trajectory being labelled.
    """

    skill_id: SkillId
    outcome: SkillOutcome
    steps: int

    @property
    def succeeded(self) -> bool:
        return self.outcome is SkillOutcome.SUCCESS


@dataclass(frozen=True)
class SkillRolloutTrace:
    """Where the player went during one probed option.

    ``positions[t]`` and ``directions[t]`` hold the player after action
    ``actions[t]``; only the first ``probe.steps`` entries are meaningful.
    """

    probe: SkillProbe
    positions: np.ndarray
    directions: np.ndarray
    actions: np.ndarray


_PROBE_OUTCOMES = tuple(SkillOutcome)


def _stack_states(states: Sequence[Any]) -> Any:
    """Stack per-entry environment states into one batched pytree."""

    if not states:
        raise ValueError("Cannot stack zero states.")
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack([jnp.asarray(leaf) for leaf in leaves]),
        *states,
    )


BatchedProbe = Callable[[Any, Any, jax.Array, jax.Array], BatchedProbeOutput]


@lru_cache(maxsize=16)
def _make_batched_skill_probe(
    *,
    batch_size: int,
    skill_horizon: int,
    distractor_count: int,
    hidden_size: int,
    skill_embedding_size: int,
    deterministic: bool,
) -> BatchedProbe:
    """Build and cache one compiled probe for a static batch shape.

    The unbatched :meth:`LearnedSkillExecutor.execute` reads ``reward``,
    ``done`` and four info flags back to the host on every primitive step, so
    a probe sweep spends most of its time on device synchronisation rather
    than on rollout. Running the whole batch inside one ``jax.lax.while_loop``
    removes every per-step round trip; only the final outcome codes come back.
    """

    environment = make_skill_chain_environment(distractor_count=distractor_count)
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=hidden_size,
        skill_embedding_size=skill_embedding_size,
    )
    success_code = _PROBE_OUTCOMES.index(SkillOutcome.SUCCESS)
    timeout_code = _PROBE_OUTCOMES.index(SkillOutcome.TIMEOUT)
    precondition_lost_code = _PROBE_OUTCOMES.index(SkillOutcome.PRECONDITION_LOST)
    terminated_code = _PROBE_OUTCOMES.index(SkillOutcome.ENVIRONMENT_TERMINATED)
    infeasible_code = _PROBE_OUTCOMES.index(SkillOutcome.INFEASIBLE)

    def probe(
        params: Any,
        states: Any,
        skill_ids: jax.Array,
        rng: jax.Array,
    ) -> BatchedProbeOutput:
        precondition = _batched_skill_precondition(skill_ids, states)
        initial_carry = BatchedProbeCarry(
            rng=rng,
            state=states,
            observation=jax.vmap(environment.bundle.env.get_obs)(states),
            hidden=initial_hidden(batch_size, hidden_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            initial_metric=_batched_skill_metric(skill_ids, states),
            finished=~precondition,
            outcome=jnp.where(precondition, timeout_code, infeasible_code),
            positions=jnp.zeros((skill_horizon, batch_size, 2), dtype=jnp.int32),
            directions=jnp.zeros((skill_horizon, batch_size), dtype=jnp.int32),
            actions=jnp.zeros((skill_horizon, batch_size), dtype=jnp.int32),
        )

        def continue_probing(
            loop_state: tuple[jax.Array, BatchedProbeCarry],
        ) -> jax.Array:
            iteration, carry = loop_state
            return (iteration < skill_horizon) & ~jnp.all(carry.finished)

        def step(
            loop_state: tuple[jax.Array, BatchedProbeCarry],
        ) -> tuple[jax.Array, BatchedProbeCarry]:
            iteration, carry = loop_state
            rng, action_rng, environment_rng = jax.random.split(carry.rng, 3)
            active = ~carry.finished

            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    skill_ids[None, ...],
                    (carry.steps == 0)[None, ...],
                ),
            )
            action = select_action(
                model_output.logits[0],
                action_rng,
                deterministic=deterministic,
            )
            action = jnp.where(active, action, Action.NOOP.value)
            environment_step = step_batch(
                environment.bundle,
                environment_rng,
                carry.state,
                action,
            )
            state = _select_batched_state(
                active,
                environment_step.state,
                carry.state,
            )
            observation = jnp.where(
                active[:, None],
                environment_step.observation,
                carry.observation,
            )

            steps = carry.steps + active.astype(jnp.int32)
            metric = _batched_skill_metric(skill_ids, state)
            success = active & (metric - carry.initial_metric >= 1)
            timeout = active & ~success & (steps >= skill_horizon)
            terminated = active & ~success & environment_step.done
            ended = success | timeout | terminated
            # The skill environment reports a lost precondition only once the
            # option has ended, so the same ordering is reproduced here.
            precondition_after = _batched_skill_precondition(skill_ids, state)
            precondition_lost = ended & ~success & ~precondition_after
            outcome = jnp.where(
                timeout,
                timeout_code,
                terminated_code,
            )
            outcome = jnp.where(precondition_lost, precondition_lost_code, outcome)
            outcome = jnp.where(success, success_code, outcome)

            next_carry = BatchedProbeCarry(
                rng=rng,
                state=state,
                observation=observation,
                hidden=model_output.hidden,
                steps=steps,
                initial_metric=carry.initial_metric,
                finished=carry.finished | ended,
                outcome=jnp.where(ended, outcome, carry.outcome),
                positions=carry.positions.at[iteration].set(
                    jnp.asarray(state.player_position, dtype=jnp.int32)
                ),
                directions=carry.directions.at[iteration].set(
                    jnp.asarray(state.player_direction, dtype=jnp.int32)
                ),
                actions=carry.actions.at[iteration].set(action),
            )
            return iteration + 1, next_carry

        _, final_carry = jax.lax.while_loop(
            continue_probing,
            step,
            (jnp.asarray(0, dtype=jnp.int32), initial_carry),
        )
        return BatchedProbeOutput(
            outcome=final_carry.outcome,
            steps=final_carry.steps,
            positions=final_carry.positions,
            directions=final_carry.directions,
            actions=final_carry.actions,
        )

    return jax.jit(probe)


def probe_skills_batched(
    executor: LearnedSkillExecutor,
    states: Sequence[Any],
    skill_ids: Sequence[SkillId],
    rng: jax.Array,
    *,
    skill_horizon: int = 64,
    distractor_count: int = 3,
) -> tuple[SkillProbe, ...]:
    """Run one option per ``(state, skill)`` pair, all in parallel on device.

    ``states[i]`` is paired with ``skill_ids[i]``, so a caller that wants the
    whole catalog probed at several boundaries flattens those pairs itself.
    Keeping the batch length fixed across calls reuses the compiled probe;
    padding with throwaway entries is cheaper than recompiling.
    """

    return tuple(
        trace.probe
        for trace in trace_skills_batched(
            executor,
            states,
            skill_ids,
            rng,
            skill_horizon=skill_horizon,
            distractor_count=distractor_count,
        )
    )


def trace_skills_batched(
    executor: LearnedSkillExecutor,
    states: Sequence[Any],
    skill_ids: Sequence[SkillId],
    rng: jax.Array,
    *,
    skill_horizon: int = 64,
    distractor_count: int = 3,
) -> tuple[SkillRolloutTrace, ...]:
    """Probe like :func:`probe_skills_batched` and keep each player path.

    The paths are what failure diagnosis needs: whether a failed option ever
    stood beside its target, faced it, or stalled in place.
    """

    if len(states) != len(skill_ids):
        raise ValueError("States and skill IDs must describe the same pairs.")
    if not states:
        return ()
    if skill_horizon < 1:
        raise ValueError("skill_horizon must be positive.")

    probe = _make_batched_skill_probe(
        batch_size=len(states),
        skill_horizon=skill_horizon,
        distractor_count=distractor_count,
        hidden_size=executor.config.hidden_size,
        skill_embedding_size=executor.config.skill_embedding_size,
        deterministic=executor.deterministic,
    )
    output = jax.device_get(
        probe(
            executor.params,
            _stack_states(states),
            jnp.asarray(
                [skill_index(skill_id) for skill_id in skill_ids],
                dtype=jnp.int32,
            ),
            rng,
        )
    )
    outcomes = np.asarray(output.outcome)
    steps = np.asarray(output.steps)
    positions = np.asarray(output.positions)
    directions = np.asarray(output.directions)
    actions = np.asarray(output.actions)
    return tuple(
        SkillRolloutTrace(
            probe=SkillProbe(
                skill_id=skill_id,
                outcome=_PROBE_OUTCOMES[int(outcomes[index])],
                steps=int(steps[index]),
            ),
            positions=positions[:, index],
            directions=directions[:, index],
            actions=actions[:, index],
        )
        for index, skill_id in enumerate(skill_ids)
    )


class BatchedExecutionCarry(NamedTuple):
    """JAX state for executing one option per batch entry in lockstep."""

    state: Any
    observation: jax.Array
    hidden: jax.Array
    steps: jax.Array
    finished: jax.Array
    outcome: jax.Array
    native_return: jax.Array


class BatchedExecutionOutput(NamedTuple):
    """Per-entry world state and diagnostics after one option."""

    state: Any
    outcome: jax.Array
    steps: jax.Array
    native_return: jax.Array


BatchedExecution = Callable[
    [Any, Any, Any, jax.Array, jax.Array, jax.Array],
    BatchedExecutionOutput,
]

_BATCHED_EXECUTIONS: dict[tuple[Any, ...], BatchedExecution] = {}


def _stack_host_states(states: Sequence[Any]) -> Any:
    """Stack states on the host so every call sees the same leaf dtypes.

    A freshly reset state carries weakly typed leaves and a state returned by
    a previous call does not; passing both straight to ``jit`` would compile
    the loop twice for one shape.
    """

    return jax.tree_util.tree_map(
        lambda *leaves: np.stack([np.asarray(leaf) for leaf in leaves]),
        *states,
    )


def _batched_skill_execution(
    env: Any,
    *,
    batch_size: int,
    skill_horizon: int,
    num_actions: int,
    hidden_size: int,
    skill_embedding_size: int,
    deterministic: bool,
) -> BatchedExecution:
    """Build and cache one compiled executor for a static batch shape.

    The cache is keyed on the environment's class and static parameters
    rather than on the instance, because evaluations build a fresh but
    identical environment per call and each would otherwise recompile. Its
    dynamic parameters are an argument, so a caller's settings are honoured.

    Unlike :func:`_make_batched_skill_probe`, which shares one key across the
    batch, entry ``i`` splits ``rngs[i]`` into a policy key and an
    environment key per step, the schedule a lone attempt has always used.
    """

    key = (
        type(env),
        getattr(env, "static_env_params", None),
        batch_size,
        skill_horizon,
        num_actions,
        hidden_size,
        skill_embedding_size,
        deterministic,
    )
    cached = _BATCHED_EXECUTIONS.get(key)
    if cached is not None:
        return cached

    model = make_goal_conditioned_actor_critic(
        num_actions=num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=hidden_size,
        skill_embedding_size=skill_embedding_size,
    )
    success_code = _PROBE_OUTCOMES.index(SkillOutcome.SUCCESS)
    timeout_code = _PROBE_OUTCOMES.index(SkillOutcome.TIMEOUT)
    precondition_lost_code = _PROBE_OUTCOMES.index(SkillOutcome.PRECONDITION_LOST)
    terminated_code = _PROBE_OUTCOMES.index(SkillOutcome.ENVIRONMENT_TERMINATED)
    infeasible_code = _PROBE_OUTCOMES.index(SkillOutcome.INFEASIBLE)

    def sample_action(logits: jax.Array, rng: jax.Array) -> jax.Array:
        return select_action(logits[None], rng, deterministic=deterministic)[0]

    def execute(
        params: Any,
        env_params: Any,
        states: Any,
        skill_ids: jax.Array,
        active: jax.Array,
        rngs: jax.Array,
    ) -> BatchedExecutionOutput:
        step_keys = jax.vmap(
            lambda rng: jax.random.split(rng, 2 * skill_horizon)
        )(rngs)
        runnable = active & _batched_skill_precondition(skill_ids, states)
        initial_metric = _batched_skill_metric(skill_ids, states)
        initial_carry = BatchedExecutionCarry(
            state=states,
            observation=jax.vmap(env.get_obs)(states),
            hidden=initial_hidden(batch_size, hidden_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            finished=~runnable,
            outcome=jnp.where(runnable, timeout_code, infeasible_code),
            native_return=jnp.zeros((batch_size,), dtype=jnp.float32),
        )

        def continue_executing(
            loop_state: tuple[jax.Array, BatchedExecutionCarry],
        ) -> jax.Array:
            iteration, carry = loop_state
            return (iteration < skill_horizon) & ~jnp.all(carry.finished)

        def step(
            loop_state: tuple[jax.Array, BatchedExecutionCarry],
        ) -> tuple[jax.Array, BatchedExecutionCarry]:
            iteration, carry = loop_state
            running = ~carry.finished
            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    skill_ids[None, ...],
                    (carry.steps == 0)[None, ...],
                ),
            )
            action = jax.vmap(sample_action)(
                model_output.logits[0],
                step_keys[:, 2 * iteration],
            )
            action = jnp.where(running, action, Action.NOOP.value)
            observation, stepped_state, native_reward, done, _ = jax.vmap(
                env.step,
                in_axes=(0, 0, 0, None),
            )(step_keys[:, 2 * iteration + 1], carry.state, action, env_params)
            state = _select_batched_state(running, stepped_state, carry.state)
            observation = jnp.where(
                running[:, None],
                observation,
                carry.observation,
            )

            steps = carry.steps + running.astype(jnp.int32)
            metric = _batched_skill_metric(skill_ids, state)
            success = running & (metric - initial_metric >= 1)
            timeout = running & ~success & (steps >= skill_horizon)
            terminated = running & ~success & done
            ended = success | timeout | terminated
            # Same ordering as the skill environment: a lost precondition is
            # only reported once the option has ended for another reason.
            precondition_lost = ended & ~success & ~_batched_skill_precondition(
                skill_ids,
                state,
            )
            outcome = jnp.where(timeout, timeout_code, terminated_code)
            outcome = jnp.where(precondition_lost, precondition_lost_code, outcome)
            outcome = jnp.where(success, success_code, outcome)

            next_carry = BatchedExecutionCarry(
                state=state,
                observation=observation,
                hidden=model_output.hidden,
                steps=steps,
                finished=carry.finished | ended,
                outcome=jnp.where(ended, outcome, carry.outcome),
                native_return=carry.native_return
                + jnp.where(running, native_reward, 0.0),
            )
            return iteration + 1, next_carry

        _, final_carry = jax.lax.while_loop(
            continue_executing,
            step,
            (jnp.asarray(0, dtype=jnp.int32), initial_carry),
        )
        return BatchedExecutionOutput(
            state=final_carry.state,
            outcome=final_carry.outcome,
            steps=final_carry.steps,
            native_return=final_carry.native_return,
        )

    compiled = jax.jit(execute)
    _BATCHED_EXECUTIONS[key] = compiled
    return compiled


@dataclass
class _ChainCursor:
    """One episode's progress while every episode advances in lockstep."""

    state: Any
    context: PlannerContext
    rng: jax.Array
    trace: list[ChainStepTrace]
    environment_steps: int = 0
    native_return: float = 0.0
    termination: ChainTermination = ChainTermination.DECISION_LIMIT
    planner_error: str | None = None
    planner_raw_response: str | None = None
    finished: bool = False


def evaluate_chain_episodes(
    planner: SayCanPlanner,
    executor: SkillExecutor,
    environment: SkillChainEnvironment,
    rngs: Sequence[jax.Array],
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    skill_horizon: int = 64,
    max_decisions: int = 12,
    max_failures_per_skill: int = 2,
) -> list[ChainEpisodeResult]:
    """Run independent planner/executor episodes in lockstep by decision.

    A decision depends only on its own episode's state and history, so at
    each decision index every unfinished episode is planned in one
    :meth:`SayCanPlanner.select_many` call, which a batched utility model
    decodes together, and its skills run in one
    :meth:`LearnedSkillExecutor.execute_many` call. Each episode keeps its
    own key, so the results are those of running the episodes one by one.
    """

    cursors: list[_ChainCursor] = []
    for rng in rngs:
        reset_rng, episode_rng = jax.random.split(rng)
        cursors.append(
            _ChainCursor(
                state=environment.reset(reset_rng).state,
                context=PlannerContext(objective=chain.objective),
                rng=episode_rng,
                trace=[],
            )
        )

    for decision_index in range(max_decisions):
        for cursor in cursors:
            if not cursor.finished and chain.is_complete(cursor.state):
                cursor.termination = ChainTermination.SUCCESS
                cursor.finished = True
        deciding = [cursor for cursor in cursors if not cursor.finished]
        if not deciding:
            break
        selections = planner.select_many(
            [cursor.state for cursor in deciding],
            [cursor.context for cursor in deciding],
        )

        executing: list[
            tuple[_ChainCursor, PlannerSelection, PlannerDecision, jax.Array]
        ] = []
        for cursor, selection in zip(deciding, selections, strict=True):
            decision = selection.decision
            if decision is None:
                cursor.planner_error = selection.utility_error
                cursor.planner_raw_response = selection.utility_raw_response
                cursor.termination = (
                    ChainTermination.INVALID_PLANNER_OUTPUT
                    if selection.status is PlannerSelectionStatus.INVALID_UTILITY
                    else ChainTermination.NO_VIABLE_SKILL
                )
                cursor.finished = True
                continue
            cursor.rng, execution_rng = jax.random.split(cursor.rng)
            executing.append((cursor, selection, decision, execution_rng))
        if not executing:
            continue

        results = _execute_many(
            executor,
            environment,
            [cursor.state for cursor, _, _, _ in executing],
            [decision.skill_id for _, _, decision, _ in executing],
            [execution_rng for _, _, _, execution_rng in executing],
            horizon=skill_horizon,
            # Padded to the episode count so one compiled loop serves every
            # decision, however many episodes are still running.
            batch_size=len(cursors),
        )
        for (cursor, selection, decision, _), result in zip(
            executing,
            results,
            strict=True,
        ):
            selected_score = next(
                score
                for score in decision.scores
                if score.skill_id is decision.skill_id
            )
            cursor.trace.append(
                ChainStepTrace(
                    decision_index=decision_index,
                    requested_skill=decision.skill_id,
                    outcome=result.outcome,
                    steps=result.steps,
                    native_return=result.native_return,
                    utility=selected_score.utility,
                    feasibility=selected_score.feasibility,
                    combined_score=selected_score.combined,
                    candidate_scores=decision.scores,
                    planner_ranking=selection.utility_ranking,
                    planner_raw_response=selection.utility_raw_response,
                )
            )
            cursor.state = result.state
            cursor.environment_steps += result.steps
            cursor.native_return += result.native_return
            cursor.context = PlannerContext(
                objective=cursor.context.objective,
                history=(
                    *cursor.context.history,
                    SkillAttempt(
                        skill_id=decision.skill_id,
                        outcome=result.outcome,
                        steps=result.steps,
                        native_return=result.native_return,
                    ),
                ),
            )
            failures = sum(
                item.skill_id is decision.skill_id and not item.succeeded
                for item in cursor.context.history
            )
            if failures >= max_failures_per_skill:
                cursor.termination = ChainTermination.FAILURE_LIMIT
                cursor.finished = True

    episodes: list[ChainEpisodeResult] = []
    for cursor in cursors:
        succeeded = chain.is_complete(cursor.state)
        episodes.append(
            ChainEpisodeResult(
                succeeded=succeeded,
                completed_skills=sum(
                    attempt.succeeded for attempt in cursor.context.history
                ),
                decisions=len(cursor.trace),
                environment_steps=cursor.environment_steps,
                native_return=cursor.native_return,
                termination=(
                    ChainTermination.SUCCESS if succeeded else cursor.termination
                ),
                planner_error=cursor.planner_error,
                planner_raw_response=cursor.planner_raw_response,
                trace=tuple(cursor.trace),
            )
        )
    return episodes


def evaluate_chain_episode(
    planner: SayCanPlanner,
    executor: SkillExecutor,
    environment: SkillChainEnvironment,
    rng: jax.Array,
    *,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    skill_horizon: int = 64,
    max_decisions: int = 12,
    max_failures_per_skill: int = 2,
) -> ChainEpisodeResult:
    """Run one event-driven planner/executor episode without state resets."""

    return evaluate_chain_episodes(
        planner,
        executor,
        environment,
        [rng],
        chain=chain,
        skill_horizon=skill_horizon,
        max_decisions=max_decisions,
        max_failures_per_skill=max_failures_per_skill,
    )[0]


def evaluate_skill_chain(
    executor: SkillExecutor,
    *,
    planner: SayCanPlanner | None = None,
    chain: SkillChainSpec = WOOD_TO_STONE_CHAIN,
    num_episodes: int = 16,
    seed: int = 0,
    skill_horizon: int = 64,
    max_decisions: int = 12,
    max_failures_per_skill: int = 2,
    distractor_count: int = 3,
) -> dict[str, Any]:
    """Evaluate a planner/executor pair over randomized persistent worlds."""

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    active_planner = planner or SayCanPlanner(
        utility_scorer=SequentialUtilityScorer(chain.reference_skills)
    )
    environment = make_skill_chain_environment(
        distractor_count=distractor_count,
    )
    episode_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    episodes = evaluate_chain_episodes(
        active_planner,
        executor,
        environment,
        list(episode_keys),
        chain=chain,
        skill_horizon=skill_horizon,
        max_decisions=max_decisions,
        max_failures_per_skill=max_failures_per_skill,
    )
    successes = np.asarray([episode.succeeded for episode in episodes])
    termination_counts = {
        termination.value: sum(
            episode.termination is termination for episode in episodes
        )
        for termination in ChainTermination
    }
    planner_error_counts = dict(
        Counter(
            episode.planner_error
            for episode in episodes
            if episode.planner_error is not None
        )
    )
    final_failure_traces = [
        failed_trace
        for episode in episodes
        if not episode.succeeded
        for failed_trace in [
            next(
                (
                    trace
                    for trace in reversed(episode.trace)
                    if trace.outcome is not SkillOutcome.SUCCESS
                ),
                None,
            )
        ]
        if failed_trace is not None
    ]
    final_failed_skill_counts = dict(
        Counter(
            trace.requested_skill.value
            for trace in final_failure_traces
        )
    )
    final_failure_outcome_counts = dict(
        Counter(trace.outcome.value for trace in final_failure_traces)
    )
    return {
        "chain": chain.name,
        "objective": chain.objective,
        "reference_skills": [skill.value for skill in chain.reference_skills],
        "num_episodes": num_episodes,
        "seed": seed,
        "success_rate": float(successes.mean()),
        "mean_completed_skills": float(
            np.mean([episode.completed_skills for episode in episodes])
        ),
        "mean_decisions": float(np.mean([episode.decisions for episode in episodes])),
        "mean_environment_steps": float(
            np.mean([episode.environment_steps for episode in episodes])
        ),
        "mean_native_return": float(
            np.mean([episode.native_return for episode in episodes])
        ),
        "termination_counts": termination_counts,
        "planner_error_counts": planner_error_counts,
        "final_failed_skill_counts": final_failed_skill_counts,
        "final_failure_outcome_counts": final_failure_outcome_counts,
        "episodes": episodes,
    }


def compare_paired_successes(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two chain results episode by episode.

    Either result may come from :func:`evaluate_skill_chain` or from
    :func:`evaluate_learned_skill_chain_batched`; both reset episode ``i``
    into the same world for the same seed and episode count, so their
    outcomes pair up. The exact McNemar test reads only the discordant pairs,
    which removes the variance between worlds that an unpaired test on two
    success rates has to carry: at 64 episodes and success near 0.25,
    Fisher's test cannot separate 0.30 from 0.22.
    """

    for key in ("seed", "num_episodes"):
        if reference[key] != candidate[key]:
            raise ValueError(
                f"Paired results need the same {key}: "
                f"{reference[key]} and {candidate[key]}."
            )
    return paired_success_summary(
        _episode_successes(reference),
        _episode_successes(candidate),
    )


def _episode_successes(result: dict[str, Any]) -> list[bool]:
    if "episode_successes" in result:
        return [bool(value) for value in result["episode_successes"]]
    return [episode.succeeded for episode in result["episodes"]]


def paired_success_summary(
    reference: Sequence[bool],
    candidate: Sequence[bool],
) -> dict[str, Any]:
    """Run the exact McNemar test on two aligned success sequences."""

    pairs = list(zip(reference, candidate, strict=True))
    if not pairs:
        raise ValueError("A paired comparison needs at least one pair.")
    reference_only = sum(first and not second for first, second in pairs)
    candidate_only = sum(second and not first for first, second in pairs)
    discordant = reference_only + candidate_only
    # Two-sided exact test: under the null each discordant pair is a fair coin.
    tail = sum(
        math.comb(discordant, count)
        for count in range(min(reference_only, candidate_only) + 1)
    )
    p_value = min(1.0, 2.0 * tail / 2**discordant) if discordant else 1.0
    return {
        "num_pairs": len(pairs),
        "both": sum(first and second for first, second in pairs),
        "neither": sum(not first and not second for first, second in pairs),
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "difference": (candidate_only - reference_only) / len(pairs),
        "p_value": p_value,
    }


def _json_ready_result(result: dict[str, Any]) -> dict[str, Any]:
    episodes = []
    for episode in result["episodes"]:
        episodes.append(
            {
                "succeeded": episode.succeeded,
                "completed_skills": episode.completed_skills,
                "decisions": episode.decisions,
                "environment_steps": episode.environment_steps,
                "native_return": episode.native_return,
                "termination": episode.termination.value,
                "planner_error": episode.planner_error,
                "planner_raw_response": episode.planner_raw_response,
                "trace": [
                    {
                        "decision_index": trace.decision_index,
                        "requested_skill": trace.requested_skill.value,
                        "outcome": trace.outcome.value,
                        "steps": trace.steps,
                        "native_return": trace.native_return,
                        "utility": trace.utility,
                        "feasibility": trace.feasibility,
                        "combined_score": trace.combined_score,
                        "planner_ranking": [
                            skill_id.value
                            for skill_id in trace.planner_ranking
                        ],
                        "planner_raw_response": trace.planner_raw_response,
                        "candidate_scores": [
                            {
                                "skill_id": score.skill_id.value,
                                "utility": score.utility,
                                "feasibility": score.feasibility,
                                "failure_penalty": score.failure_penalty,
                                "combined": score.combined,
                            }
                            for score in trace.candidate_scores
                        ],
                    }
                    for trace in episode.trace
                ],
            }
        )
    return {**result, "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executor",
        choices=("scripted", "learned"),
        default="scripted",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required when --executor=learned.",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skill-horizon", type=int, default=64)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--max-failures-per-skill", type=int, default=2)
    parser.add_argument("--distractor-count", type=int, default=3)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample learned executor actions instead of using greedy actions.",
    )
    args = parser.parse_args()

    if args.executor == "learned":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --executor=learned.")
        executor: SkillExecutor = learned_skill_executor_from_checkpoint(
            args.checkpoint,
            deterministic=not args.stochastic,
        )
    else:
        executor = ScriptedSkillExecutor()

    result = evaluate_skill_chain(
        executor,
        num_episodes=args.episodes,
        seed=args.seed,
        skill_horizon=args.skill_horizon,
        max_decisions=args.max_decisions,
        max_failures_per_skill=args.max_failures_per_skill,
        distractor_count=args.distractor_count,
    )
    print(json.dumps(_json_ready_result(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
