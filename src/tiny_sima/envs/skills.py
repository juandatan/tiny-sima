"""Short, reproducible Craftax skill episodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import DIRECTIONS, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import Inventory

from tiny_sima.envs.craftax import (
    CraftaxBundle,
    EnvironmentSpec,
    make_environment,
)
from tiny_sima.goals.skills import (
    SKILL_IDS,
    SkillGoal,
    SkillId,
    skill_index,
    skill_metric,
    skill_precondition,
    skill_progress_from_metric,
    skill_success_from_metric,
)


class SkillEpisodeState(NamedTuple):
    """Dynamic state needed by a short skill episode."""

    env_state: Any
    initial_metric: jax.Array
    steps: jax.Array
    target_direction: jax.Array


class SkillReset(NamedTuple):
    """Output of a single or batched skill reset."""

    observation: jax.Array
    state: SkillEpisodeState


class SkillStep(NamedTuple):
    """Output of a single or batched skill step."""

    observation: jax.Array
    state: SkillEpisodeState
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]


class PreparedSkillScenario(NamedTuple):
    """Craftax state and target metadata produced by scenario preparation."""

    state: Any
    target_direction: jax.Array


class GoalConditionedSkillReset(NamedTuple):
    """Batched observations, state, and categorical goals after reset."""

    observation: jax.Array
    state: SkillEpisodeState
    skill_id: jax.Array


class GoalConditionedSkillStep(NamedTuple):
    """Batched transition returned during goal-conditioned training."""

    observation: jax.Array
    state: SkillEpisodeState
    skill_id: jax.Array
    reward: jax.Array
    done: jax.Array
    info: dict[str, jax.Array]


@dataclass(frozen=True)
class SkillEnvironment:
    """A Craftax environment configured for one short objective."""

    bundle: CraftaxBundle
    goal: SkillGoal

    def reset(self, rng: jax.Array) -> SkillReset:
        world_rng, scenario_rng = jax.random.split(rng)
        _, state = self.bundle.env.reset(world_rng, self.bundle.params)
        scenario = prepare_skill_scenario(
            state,
            self.goal,
            scenario_rng,
        )
        observation = self.bundle.env.get_obs(scenario.state)
        episode_state = SkillEpisodeState(
            env_state=scenario.state,
            initial_metric=skill_metric(self.goal.skill_id, scenario.state),
            steps=jnp.array(0, dtype=jnp.int32),
            target_direction=scenario.target_direction,
        )
        return SkillReset(observation=observation, state=episode_state)

    def step(
        self,
        rng: jax.Array,
        state: SkillEpisodeState,
        action: jax.Array,
    ) -> SkillStep:
        previous_metric = skill_metric(self.goal.skill_id, state.env_state)
        precondition = skill_precondition(self.goal, state.env_state)
        observation, env_state, native_reward, env_done, info = self.bundle.env.step(
            rng,
            state.env_state,
            action,
            self.bundle.params,
        )
        steps = state.steps + 1
        success = skill_success_from_metric(
            self.goal,
            state.initial_metric,
            env_state,
        )
        previous_success = previous_metric - state.initial_metric >= self.goal.quantity
        reward = (success & ~previous_success).astype(jnp.float32)
        timeout = (steps >= self.goal.effective_horizon) & ~success
        done = env_done | success | timeout
        next_state = state._replace(env_state=env_state, steps=steps)
        skill_info = {
            **info,
            "native_reward": native_reward,
            "skill_success": success,
            "skill_timeout": timeout,
            "skill_progress": skill_progress_from_metric(
                self.goal,
                state.initial_metric,
                env_state,
            ),
            "skill_precondition": precondition,
        }
        return SkillStep(
            observation=observation,
            state=next_state,
            reward=reward,
            done=done,
            info=skill_info,
        )


def _clear_mobs(state: Any) -> Any:
    def clear(mobs: Any) -> Any:
        return mobs.replace(mask=jnp.zeros_like(mobs.mask))

    return state.replace(
        mob_map=jnp.zeros_like(state.mob_map),
        zombies=clear(state.zombies),
        cows=clear(state.cows),
        skeletons=clear(state.skeletons),
        arrows=clear(state.arrows),
    )


def prepare_skill_scenario(
    state: Any,
    goal: SkillGoal,
    rng: jax.Array,
) -> PreparedSkillScenario:
    """Inject a valid nearby target while retaining real Craftax dynamics."""

    if goal.quantity != 1:
        raise ValueError("The nearby-target curriculum currently supports quantity=1.")

    target_direction = jax.random.randint(
        rng,
        shape=(),
        minval=Action.LEFT.value,
        maxval=Action.DOWN.value + 1,
        dtype=jnp.int32,
    )
    player_position = jnp.array(
        [state.map.shape[0] // 2, state.map.shape[1] // 2],
        dtype=jnp.int32,
    )
    clean_map = jnp.where(
        state.map == BlockType.CRAFTING_TABLE.value,
        BlockType.GRASS.value,
        state.map,
    )
    clean_map = clean_map.at[player_position[0], player_position[1]].set(
        BlockType.GRASS.value
    )
    for direction in range(Action.LEFT.value, Action.DOWN.value + 1):
        position = player_position + DIRECTIONS[direction]
        clean_map = clean_map.at[position[0], position[1]].set(
            BlockType.GRASS.value
        )

    target_position = player_position + DIRECTIONS[target_direction]
    skill_id = goal.skill_id
    inventory = Inventory()
    player_drink = 9
    player_direction = target_direction

    if skill_id is SkillId.COLLECT_WOOD:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.TREE.value
        )
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.COLLECT_STONE:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.STONE.value
        )
        inventory = replace(inventory, wood_pickaxe=1)
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.DRINK_WATER:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.WATER.value
        )
        player_drink = 5
        player_direction = target_direction % 4 + 1
    elif skill_id is SkillId.PLACE_TABLE:
        inventory = replace(inventory, wood=2)
    elif skill_id in {
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
    }:
        clean_map = clean_map.at[target_position[0], target_position[1]].set(
            BlockType.CRAFTING_TABLE.value
        )
        inventory = replace(inventory, wood=1)
    else:
        raise AssertionError(f"Unhandled skill: {skill_id}")

    state = _clear_mobs(state)
    state = state.replace(
        map=clean_map,
        player_position=player_position,
        player_direction=player_direction,
        player_health=9,
        player_food=9,
        player_drink=player_drink,
        player_energy=9,
        is_sleeping=False,
        player_recover=0.0,
        player_hunger=0.0,
        player_thirst=0.0,
        player_fatigue=0.0,
        inventory=inventory,
        achievements=jnp.zeros_like(state.achievements),
        timestep=0,
        light_level=1.0,
    )
    return PreparedSkillScenario(state=state, target_direction=target_direction)


def make_skill_environment(
    skill_id: SkillId | str,
    *,
    horizon: int = 8,
) -> SkillEnvironment:
    """Create the first-stage nearby-target benchmark for one skill."""

    goal = SkillGoal(SkillId(skill_id), horizon=horizon)
    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        auto_reset=False,
        max_episode_steps=max(128, horizon + 1),
    )
    params = replace(
        bundle.params,
        spawn_cow_chance=0.0,
        spawn_zombie_base_chance=0.0,
        spawn_zombie_night_chance=0.0,
        spawn_skeleton_chance=0.0,
    )
    return SkillEnvironment(bundle=replace(bundle, params=params), goal=goal)


def reset_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    num_envs: int,
) -> SkillReset:
    """Reset a vector of independent skill scenarios."""

    reset_keys = jax.random.split(rng, num_envs)
    return jax.vmap(environment.reset)(reset_keys)


def step_skill_batch(
    environment: SkillEnvironment,
    rng: jax.Array,
    state: SkillEpisodeState,
    action: jax.Array,
) -> SkillStep:
    """Step a vector of independent skill scenarios."""

    step_keys = jax.random.split(rng, action.shape[0])
    return jax.vmap(environment.step)(step_keys, state, action)


def _concatenate_pytrees(*values: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.concatenate(leaves, axis=0),
        *values,
    )


def _select_completed(
    done: jax.Array,
    reset_value: Any,
    stepped_value: Any,
) -> Any:
    def select_leaf(reset_leaf: jax.Array, stepped_leaf: jax.Array) -> jax.Array:
        mask_shape = (done.shape[0],) + (1,) * (reset_leaf.ndim - 1)
        return jnp.where(done.reshape(mask_shape), reset_leaf, stepped_leaf)

    return jax.tree_util.tree_map(select_leaf, reset_value, stepped_value)


@dataclass(frozen=True)
class VectorizedSkillEnvironment:
    """Balanced multi-skill environment with automatic episode resets."""

    environments: tuple[SkillEnvironment, ...]
    envs_per_skill: int

    @property
    def num_envs(self) -> int:
        return len(self.environments) * self.envs_per_skill

    @property
    def num_skills(self) -> int:
        return len(self.environments)

    @property
    def spec(self) -> EnvironmentSpec:
        return self.environments[0].bundle.spec

    @property
    def skill_ids(self) -> jax.Array:
        return jnp.repeat(
            jnp.arange(self.num_skills, dtype=jnp.int32),
            self.envs_per_skill,
        )

    def reset(self, rng: jax.Array) -> GoalConditionedSkillReset:
        reset_keys = jax.random.split(rng, self.num_skills)
        resets = tuple(
            reset_skill_batch(environment, reset_keys[index], self.envs_per_skill)
            for index, environment in enumerate(self.environments)
        )
        return GoalConditionedSkillReset(
            observation=jnp.concatenate(
                tuple(reset.observation for reset in resets),
                axis=0,
            ),
            state=_concatenate_pytrees(*(reset.state for reset in resets)),
            skill_id=self.skill_ids,
        )

    def step(
        self,
        rng: jax.Array,
        state: SkillEpisodeState,
        action: jax.Array,
    ) -> GoalConditionedSkillStep:
        keys = jax.random.split(rng, self.num_skills * 2)
        steps: list[SkillStep] = []
        next_states: list[SkillEpisodeState] = []
        next_observations: list[jax.Array] = []

        for index, environment in enumerate(self.environments):
            start = index * self.envs_per_skill
            stop = start + self.envs_per_skill

            def slice_group(
                value: jax.Array,
                group_start: int = start,
                group_stop: int = stop,
            ) -> jax.Array:
                return value[group_start:group_stop]

            group_state = jax.tree_util.tree_map(
                slice_group,
                state,
            )
            result = step_skill_batch(
                environment,
                keys[2 * index],
                group_state,
                action[start:stop],
            )
            reset = reset_skill_batch(
                environment,
                keys[2 * index + 1],
                self.envs_per_skill,
            )
            steps.append(result)
            next_states.append(
                _select_completed(result.done, reset.state, result.state)
            )
            next_observations.append(
                _select_completed(
                    result.done,
                    reset.observation,
                    result.observation,
                )
            )

        skill_ids = self.skill_ids
        info = _concatenate_pytrees(*(step.info for step in steps))
        info = {**info, "skill_index": skill_ids}
        return GoalConditionedSkillStep(
            observation=jnp.concatenate(tuple(next_observations), axis=0),
            state=_concatenate_pytrees(*next_states),
            skill_id=skill_ids,
            reward=jnp.concatenate(tuple(step.reward for step in steps), axis=0),
            done=jnp.concatenate(tuple(step.done for step in steps), axis=0),
            info=info,
        )


def make_vectorized_skill_environment(
    *,
    num_envs: int,
    horizon: int = 8,
) -> VectorizedSkillEnvironment:
    """Create a balanced batch covering every registered skill."""

    num_skills = len(SKILL_IDS)
    if num_envs < num_skills or num_envs % num_skills:
        raise ValueError(
            f"num_envs must be divisible by the {num_skills} registered skills."
        )
    environments = tuple(
        make_skill_environment(skill_id, horizon=horizon)
        for skill_id in SKILL_IDS
    )
    expected_indices = tuple(
        skill_index(environment.goal.skill_id) for environment in environments
    )
    if expected_indices != tuple(range(num_skills)):
        raise AssertionError("Skill environment order does not match skill indices.")
    return VectorizedSkillEnvironment(
        environments=environments,
        envs_per_skill=num_envs // num_skills,
    )
