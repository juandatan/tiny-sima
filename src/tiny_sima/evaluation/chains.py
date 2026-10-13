"""Closed-loop evaluation for persistent SayCan-style skill chains."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Protocol

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
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillAttempt,
    SkillOutcome,
    SkillScore,
)

ObjectivePredicate = Callable[[Any], bool]
JaxObjectivePredicate = Callable[[Any], jax.Array]


class SkillChainSpec(NamedTuple):
    """A state-based objective with an optional scripted reference plan."""

    name: str
    objective: str
    reference_skills: tuple[SkillId, ...]
    is_complete: ObjectivePredicate
    is_complete_jax: JaxObjectivePredicate | None = None


def _wood_to_stone_complete_jax(state: Any) -> jax.Array:
    inventory = state.inventory
    return (jnp.asarray(inventory.wood_pickaxe) > 0) & (
        jnp.asarray(inventory.stone) > 0
    )


def _wood_to_stone_complete(state: Any) -> bool:
    return bool(np.asarray(_wood_to_stone_complete_jax(state)))


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
)


class ChainTermination(str, Enum):
    """Why a hierarchical episode stopped."""

    SUCCESS = "success"
    NO_VIABLE_SKILL = "no_viable_skill"
    FAILURE_LIMIT = "failure_limit"
    DECISION_LIMIT = "decision_limit"


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


@dataclass(frozen=True)
class ChainEpisodeResult:
    """Final metrics and trace for one persistent chain episode."""

    succeeded: bool
    completed_skills: int
    decisions: int
    environment_steps: int
    native_return: float
    termination: ChainTermination
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


class BatchedChainOutput(NamedTuple):
    """Per-episode metrics produced by the compiled chain evaluator."""

    succeeded: jax.Array
    termination: jax.Array
    completed_skills: jax.Array
    decisions: jax.Array
    environment_steps: jax.Array
    native_return: jax.Array
    last_failed_skill: jax.Array


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


PolicyStep = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]


@dataclass(frozen=True)
class LearnedSkillExecutor:
    """Checkpoint-backed low-level policy used in persistent worlds."""

    params: Any
    config: SkillPPOConfig
    model: GoalConditionedActorCriticRNN
    deterministic: bool = True
    _policy_step: PolicyStep | None = None

    def __post_init__(self) -> None:
        def policy_step(
            hidden: jax.Array,
            observation: jax.Array,
            categorical_skill_id: jax.Array,
            reset: jax.Array,
            rng: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            output = apply_goal_conditioned_actor_critic(
                self.model,
                self.params,
                hidden,
                (
                    observation[None, None, ...],
                    categorical_skill_id[None, None],
                    reset[None, None],
                ),
            )
            action = select_action(
                output.logits[0],
                rng,
                deterministic=self.deterministic,
            )
            return output.hidden, action[0]

        object.__setattr__(self, "_policy_step", jax.jit(policy_step))

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

        if self._policy_step is None:
            raise RuntimeError("Learned executor policy step was not initialized.")
        hidden = initial_hidden(1, self.config.hidden_size)
        categorical_skill_id = jnp.asarray(skill_index(skill_id), dtype=jnp.int32)
        skill_return = 0.0
        native_return = 0.0
        current = episode_state
        step_keys = jax.random.split(rng, horizon * 2)
        for step_index in range(horizon):
            observation = environment.bundle.env.get_obs(current.env_state)
            hidden, action = self._policy_step(
                hidden,
                observation,
                categorical_skill_id,
                jnp.asarray(step_index == 0),
                step_keys[2 * step_index],
            )
            result = skill_environment.step(
                step_keys[2 * step_index + 1],
                current,
                action,
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
) -> BatchedRollout:
    """Build and cache one compiled evaluator for a static execution shape."""

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
) -> dict[str, Any]:
    """Evaluate a fixed reference plan across parallel JAX environments.

    This fast path intentionally executes ``chain.reference_skills`` rather
    than calling an arbitrary Python planner inside ``jax.jit``. Use
    :func:`evaluate_skill_chain` when planner decisions or detailed traces are
    required.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    if skill_horizon < 1:
        raise ValueError("skill_horizon must be positive.")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive.")
    if max_failures_per_skill < 1:
        raise ValueError("max_failures_per_skill must be positive.")
    if not chain.reference_skills:
        raise ValueError("A batched chain requires at least one reference skill.")
    if chain.is_complete_jax is None:
        raise ValueError(
            "A batched chain requires a JAX-compatible completion predicate."
        )

    plan = jnp.asarray(
        [skill_index(skill_id) for skill_id in chain.reference_skills],
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
    )
    compile_cache_hit = (
        _make_batched_chain_rollout.cache_info().hits > cache_hits_before
    )
    start = time.perf_counter()
    output = rollout(executor.params, plan, jax.random.PRNGKey(seed))
    jax.block_until_ready(output)
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
    return {
        "chain": chain.name,
        "objective": chain.objective,
        "reference_skills": [
            skill.value for skill in chain.reference_skills
        ],
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
    }


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

    reset_rng, episode_rng = jax.random.split(rng)
    reset = environment.reset(reset_rng)
    state = reset.state
    context = PlannerContext(objective=chain.objective)
    trace: list[ChainStepTrace] = []
    environment_steps = 0
    native_return = 0.0
    termination = ChainTermination.DECISION_LIMIT

    for decision_index in range(max_decisions):
        if chain.is_complete(state):
            termination = ChainTermination.SUCCESS
            break
        decision = planner.select(state, context)
        if decision is None:
            termination = ChainTermination.NO_VIABLE_SKILL
            break
        episode_rng, execution_rng = jax.random.split(episode_rng)
        result = executor.execute(
            environment,
            state,
            decision.skill_id,
            execution_rng,
            horizon=skill_horizon,
        )
        selected_score = next(
            score
            for score in decision.scores
            if score.skill_id is decision.skill_id
        )
        trace.append(
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
            )
        )
        state = result.state
        environment_steps += result.steps
        native_return += result.native_return
        attempt = SkillAttempt(
            skill_id=decision.skill_id,
            outcome=result.outcome,
            steps=result.steps,
            native_return=result.native_return,
        )
        context = PlannerContext(
            objective=context.objective,
            history=(*context.history, attempt),
        )
        failures = sum(
            item.skill_id is decision.skill_id and not item.succeeded
            for item in context.history
        )
        if failures >= max_failures_per_skill:
            termination = ChainTermination.FAILURE_LIMIT
            break

    succeeded = chain.is_complete(state)
    if succeeded:
        termination = ChainTermination.SUCCESS
    return ChainEpisodeResult(
        succeeded=succeeded,
        completed_skills=sum(attempt.succeeded for attempt in context.history),
        decisions=len(trace),
        environment_steps=environment_steps,
        native_return=native_return,
        termination=termination,
        trace=tuple(trace),
    )


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
    episodes = [
        evaluate_chain_episode(
            active_planner,
            executor,
            environment,
            episode_key,
            chain=chain,
            skill_horizon=skill_horizon,
            max_decisions=max_decisions,
            max_failures_per_skill=max_failures_per_skill,
        )
        for episode_key in episode_keys
    ]
    successes = np.asarray([episode.succeeded for episode in episodes])
    termination_counts = {
        termination.value: sum(
            episode.termination is termination for episode in episodes
        )
        for termination in ChainTermination
    }
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
        "episodes": episodes,
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
