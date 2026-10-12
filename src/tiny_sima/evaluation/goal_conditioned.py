"""Evaluate categorical goal-conditioned executor checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tiny_sima.config import SkillPPOConfig, load_skill_config
from tiny_sima.envs.skills import (
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
from tiny_sima.executor.checkpoints import load_checkpoint
from tiny_sima.executor.model import (
    apply_goal_conditioned_actor_critic,
    initial_hidden,
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
    select_action,
)
from tiny_sima.goals import SKILL_IDS, SkillId, skill_index


class GoalEvaluationCarry(NamedTuple):
    """State carried through a batched learned-skill evaluation."""

    rng: jax.Array
    state: SkillEpisodeState
    observation: jax.Array
    hidden: jax.Array
    tracker: SkillEpisodeTracker


def evaluate_goal_conditioned_skill(
    params: Any,
    config: SkillPPOConfig,
    skill_id: SkillId | str,
    *,
    num_episodes: int = 256,
    seed: int = 1,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate one requested skill over independently generated scenarios."""

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    normalized_skill_id = SkillId(skill_id)
    environment = make_skill_environment(
        normalized_skill_id,
        horizon=config.skill_horizon,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    categorical_skill_id = skill_index(normalized_skill_id)

    def rollout(rng: jax.Array) -> SkillRolloutOutput:
        rng, reset_rng = jax.random.split(rng)
        reset = reset_skill_batch(environment, reset_rng, num_episodes)
        initial_carry = GoalEvaluationCarry(
            rng=rng,
            state=reset.state,
            observation=reset.observation,
            hidden=initial_hidden(num_episodes, config.hidden_size),
            tracker=initialize_skill_tracker(num_episodes),
        )
        skill_ids = jnp.full(
            (num_episodes,),
            categorical_skill_id,
            dtype=jnp.int32,
        )

        def step(
            carry: GoalEvaluationCarry,
            _unused: None,
        ) -> tuple[GoalEvaluationCarry, None]:
            rng, action_rng, env_rng = jax.random.split(carry.rng, 3)
            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    skill_ids[None, ...],
                    carry.tracker.finished[None, ...],
                ),
            )
            logits = model_output.logits[0]
            action = select_action(
                logits,
                action_rng,
                deterministic=deterministic,
            )
            result = step_skill_batch(
                environment,
                env_rng,
                carry.state,
                action,
            )
            return (
                GoalEvaluationCarry(
                    rng=rng,
                    state=result.state,
                    observation=result.observation,
                    hidden=model_output.hidden,
                    tracker=update_skill_tracker(carry.tracker, result),
                ),
                None,
            )

        final_carry, _ = jax.lax.scan(
            step,
            initial_carry,
            None,
            length=config.skill_horizon,
        )
        return finish_skill_tracker(final_carry.tracker)

    output = jax.jit(rollout)(jax.random.PRNGKey(seed))
    succeeded = np.asarray(output.succeeded)
    episode_length = np.asarray(output.episode_length)
    return {
        "skill_id": normalized_skill_id.value,
        "num_episodes": num_episodes,
        "seed": seed,
        "deterministic": deterministic,
        "success_rate": float(succeeded.mean()),
        "mean_episode_length": float(episode_length.mean()),
    }


def evaluate_goal_conditioned_policy(
    params: Any,
    config: SkillPPOConfig,
    *,
    num_episodes_per_skill: int = 256,
    seed: int = 1,
    deterministic: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate every requested skill independently."""

    return [
        evaluate_goal_conditioned_skill(
            params,
            config,
            skill_id,
            num_episodes=num_episodes_per_skill,
            seed=seed + index,
            deterministic=deterministic,
        )
        for index, skill_id in enumerate(SKILL_IDS)
    ]


def evaluate_goal_conditioned_checkpoint(
    checkpoint_dir: str | Path,
    *,
    num_episodes_per_skill: int = 256,
    seed: int = 1,
    deterministic: bool = True,
) -> list[dict[str, Any]]:
    """Load and evaluate a saved goal-conditioned executor."""

    checkpoint_path = Path(checkpoint_dir)
    config = load_skill_config(checkpoint_path / "config.yaml")
    environment = make_skill_environment(
        SKILL_IDS[0],
        horizon=config.skill_horizon,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=len(SKILL_IDS),
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    template_params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        environment.bundle.spec.observation_shape,
    )
    params, _ = load_checkpoint(checkpoint_path, template_params)
    return evaluate_goal_conditioned_policy(
        params,
        config,
        num_episodes_per_skill=num_episodes_per_skill,
        seed=seed,
        deterministic=deterministic,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Goal-conditioned checkpoint directory.")
    parser.add_argument("--episodes-per-skill", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of selecting the most likely action.",
    )
    args = parser.parse_args()
    results = evaluate_goal_conditioned_checkpoint(
        args.checkpoint,
        num_episodes_per_skill=args.episodes_per_skill,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
