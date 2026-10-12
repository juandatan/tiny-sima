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
    make_skill_environment,
    reset_skill_batch,
    step_skill_batch,
)
from tiny_sima.evaluation.skill_rollout import (
    SkillEpisodeTracker,
    SkillRolloutOutput,
    finish_skill_tracker,
    initialize_skill_tracker,
    update_skill_tracker,
)
from tiny_sima.goals.skills import SKILL_REGISTRY, SkillId, get_skill_spec

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
    """Return the shortest action for the injected nearby-target scenario."""

    skill_id = environment.goal.skill_id
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
) -> dict[str, Any]:
    """Evaluate one baseline on one nearby-target skill."""

    if policy not in {"scripted", "random"}:
        raise ValueError(f"Unknown skill policy: {policy}")
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    environment = make_skill_environment(skill_id, horizon=horizon)

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
    return {
        "skill_id": SkillId(skill_id).value,
        "policy": policy,
        "num_episodes": num_episodes,
        "horizon": horizon,
        "seed": seed,
        "success_rate": float(succeeded.mean()),
        "mean_return": float(episode_return.mean()),
        "mean_episode_length": float(episode_length.mean()),
    }


def evaluate_skill_suite(
    *,
    num_episodes: int = 256,
    horizon: int = 8,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate scripted and random policies on every registered skill."""

    return [
        evaluate_skill_policy(
            skill_id,
            policy=policy,
            num_episodes=num_episodes,
            horizon=horizon,
            seed=seed,
        )
        for skill_id in SKILL_REGISTRY
        for policy in ("scripted", "random")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = evaluate_skill_suite(
        num_episodes=args.num_episodes,
        horizon=args.horizon,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
