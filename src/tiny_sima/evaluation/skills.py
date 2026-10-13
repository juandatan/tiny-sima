"""Scripted and random baselines for the Phase 2 skill benchmark."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action

from tiny_sima.envs.skills import (
    SkillEnvironment,
    SkillEpisodeState,
    SkillScenario,
    make_skill_environment,
    reset_skill_batch,
    skills_for_scenario,
    step_skill_batch,
)
from tiny_sima.evaluation.skill_rollout import (
    SkillEpisodeTracker,
    SkillRolloutOutput,
    finish_skill_tracker,
    initialize_skill_tracker,
    update_skill_tracker,
)
from tiny_sima.goals.skills import SkillId, get_skill_spec

PolicyName = Literal["scripted", "random"]


class SkillRolloutCarry(NamedTuple):
    """State carried between steps of a batched skill rollout."""

    rng: jax.Array
    state: SkillEpisodeState
    tracker: SkillEpisodeTracker


_INTERACTION_SKILLS = {
    SkillId.COLLECT_WOOD,
    SkillId.COLLECT_STONE,
    SkillId.DRINK_WATER,
}


def scripted_skill_action(
    environment: SkillEnvironment,
    state: SkillEpisodeState,
) -> jax.Array:
    """Return a shortest-path action for a controlled skill scenario."""

    skill_id = environment.goal.skill_id
    if (
        environment.scenario.kind is SkillScenario.LOCAL_NAVIGATION
    ):
        player_position = state.env_state.player_position
        row_delta = state.approach_position[..., 0] - player_position[..., 0]
        column_delta = state.approach_position[..., 1] - player_position[..., 1]
        movement = jnp.where(
            row_delta < 0,
            Action.UP.value,
            jnp.where(
                row_delta > 0,
                Action.DOWN.value,
                jnp.where(
                    column_delta < 0,
                    Action.LEFT.value,
                    Action.RIGHT.value,
                ),
            ),
        )
        at_approach = (row_delta == 0) & (column_delta == 0)
        if skill_id in _INTERACTION_SKILLS:
            interaction = jnp.where(
                state.env_state.player_direction == state.target_direction,
                Action.DO.value,
                state.target_direction,
            )
        else:
            interaction = jnp.full_like(
                state.steps,
                get_skill_spec(skill_id).interaction_action.value,
                dtype=jnp.int32,
            )
        return jnp.where(at_approach, interaction, movement).astype(jnp.int32)

    if skill_id in _INTERACTION_SKILLS:
        return jnp.where(
            state.env_state.player_direction == state.target_direction,
            Action.DO.value,
            state.target_direction,
        ).astype(jnp.int32)

    action = get_skill_spec(skill_id).interaction_action.value
    return jnp.full_like(state.steps, action, dtype=jnp.int32)


def evaluate_skill_policy(
    skill_id: SkillId | str,
    *,
    policy: PolicyName,
    num_episodes: int = 256,
    horizon: int = 8,
    seed: int = 0,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
) -> dict[str, Any]:
    """Evaluate one baseline on one controlled skill distribution."""

    if policy not in {"scripted", "random"}:
        raise ValueError(f"Unknown skill policy: {policy}")
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    environment = make_skill_environment(
        skill_id,
        horizon=horizon,
        scenario=scenario,
        min_distance=min_distance,
        max_distance=max_distance,
        distractor_count=distractor_count,
    )

    def rollout(rng: jax.Array) -> SkillRolloutOutput:
        rng, reset_rng = jax.random.split(rng)
        _, state = reset_skill_batch(environment, reset_rng, num_episodes)
        initial_carry = SkillRolloutCarry(
            rng=rng,
            state=state,
            tracker=initialize_skill_tracker(num_episodes),
        )

        def step(
            carry: SkillRolloutCarry,
            _unused: None,
        ) -> tuple[SkillRolloutCarry, None]:
            rng, action_rng, step_rng = jax.random.split(carry.rng, 3)
            if policy == "scripted":
                action = scripted_skill_action(environment, carry.state)
            else:
                action = jax.random.randint(
                    action_rng,
                    shape=(num_episodes,),
                    minval=0,
                    maxval=environment.bundle.spec.num_actions,
                )

            result = step_skill_batch(
                environment,
                step_rng,
                carry.state,
                action,
            )
            next_carry = SkillRolloutCarry(
                rng=rng,
                state=result.state,
                tracker=update_skill_tracker(carry.tracker, result),
            )
            return next_carry, None

        final_carry, _ = jax.lax.scan(
            step,
            initial_carry,
            None,
            length=horizon,
        )
        return finish_skill_tracker(final_carry.tracker)

    output = jax.jit(rollout)(jax.random.PRNGKey(seed))
    succeeded = np.asarray(output.succeeded)
    episode_return = np.asarray(output.episode_return)
    episode_length = np.asarray(output.episode_length)
    failed = ~succeeded
    precondition_lost = np.asarray(output.precondition_lost) & failed
    feasible_timeout = np.asarray(output.timed_out) & failed & ~precondition_lost
    environment_terminated = (
        np.asarray(output.environment_terminated) & failed & ~precondition_lost
    )
    return {
        "skill_id": SkillId(skill_id).value,
        "policy": policy,
        "scenario": environment.scenario.kind.value,
        "num_episodes": num_episodes,
        "horizon": horizon,
        "min_distance": environment.scenario.min_distance,
        "max_distance": environment.scenario.max_distance,
        "distractor_count": environment.scenario.distractor_count,
        "seed": seed,
        "success_rate": float(succeeded.mean()),
        "mean_return": float(episode_return.mean()),
        "mean_episode_length": float(episode_length.mean()),
        "precondition_lost_rate": float(precondition_lost.mean()),
        "feasible_timeout_rate": float(feasible_timeout.mean()),
        "environment_termination_rate": float(environment_terminated.mean()),
    }


def evaluate_skill_suite(
    *,
    num_episodes: int = 256,
    horizon: int = 8,
    seed: int = 0,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
) -> list[dict[str, Any]]:
    """Evaluate scripted and random policies on every registered skill."""

    scenario_skills = skills_for_scenario(scenario)
    return [
        evaluate_skill_policy(
            skill_id,
            policy=policy,
            num_episodes=num_episodes,
            horizon=horizon,
            seed=seed,
            scenario=scenario,
            min_distance=min_distance,
            max_distance=max_distance,
            distractor_count=distractor_count,
        )
        for skill_id in scenario_skills
        for policy in ("scripted", "random")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=tuple(scenario.value for scenario in SkillScenario),
        default=SkillScenario.NEARBY.value,
    )
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--distractor-count", type=int, default=6)
    args = parser.parse_args()

    results = evaluate_skill_suite(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        seed=args.seed,
        scenario=args.scenario,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        distractor_count=args.distractor_count,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
