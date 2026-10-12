"""Evaluate saved Tiny-SIMA PPO-RNN checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tiny_sima.config import PPOConfig, load_config
from tiny_sima.envs.craftax import make_environment, reset_batch, step_batch
from tiny_sima.executor.checkpoints import load_checkpoint
from tiny_sima.executor.model import (
    ActorCriticRNN,
    apply_actor_critic,
    initial_hidden,
    initialize_parameters,
)


class EvaluationRolloutCarry(NamedTuple):
    """Dynamic state carried through checkpoint evaluation."""

    rng: jax.Array
    env_state: Any
    observation: jax.Array
    done: jax.Array
    hidden: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array


class CompletedEpisodes(NamedTuple):
    """Episode metrics emitted by each vectorized environment step."""

    episode_return: jax.Array
    episode_length: jax.Array


def evaluate_policy(
    params: Any,
    config: PPOConfig,
    *,
    num_episodes: int = 16,
    num_envs: int = 8,
    seed: int = 1,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate parameters using vectorized, auto-reset Craftax episodes."""

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")

    bundle = make_environment(
        config.env_name,
        auto_reset=True,
        max_episode_steps=config.max_episode_steps,
    )
    model = ActorCriticRNN(
        num_actions=bundle.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    batches_needed = math.ceil(num_episodes / num_envs)
    rollout_steps = config.max_episode_steps * batches_needed

    def rollout(rng: jax.Array) -> CompletedEpisodes:
        rng, reset_rng = jax.random.split(rng)
        observation, env_state = reset_batch(bundle, reset_rng, num_envs)
        initial_carry = EvaluationRolloutCarry(
            rng=rng,
            env_state=env_state,
            observation=observation,
            done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            hidden=initial_hidden(num_envs, config.hidden_size),
            episode_return=jnp.zeros((num_envs,), dtype=jnp.float32),
            episode_length=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

        def step(
            carry: EvaluationRolloutCarry,
            _unused: None,
        ) -> tuple[EvaluationRolloutCarry, CompletedEpisodes]:
            rng, action_rng, env_rng = jax.random.split(carry.rng, 3)
            model_output = apply_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    carry.done[None, ...],
                ),
            )
            logits = model_output.logits[0]
            if deterministic:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = jax.random.categorical(action_rng, logits, axis=-1)

            env_step = step_batch(
                bundle,
                env_rng,
                carry.env_state,
                action,
            )
            episode_return = carry.episode_return + env_step.reward
            episode_length = carry.episode_length + 1
            completed_return = jnp.where(
                env_step.done,
                episode_return,
                jnp.nan,
            )
            completed_length = jnp.where(
                env_step.done,
                episode_length.astype(jnp.float32),
                jnp.nan,
            )
            next_carry = EvaluationRolloutCarry(
                rng=rng,
                env_state=env_step.state,
                observation=env_step.observation,
                done=env_step.done,
                hidden=model_output.hidden,
                episode_return=jnp.where(
                    env_step.done,
                    0.0,
                    episode_return,
                ),
                episode_length=jnp.where(
                    env_step.done,
                    0,
                    episode_length,
                ),
            )
            completed = CompletedEpisodes(
                episode_return=completed_return,
                episode_length=completed_length,
            )
            return next_carry, completed

        _, completed = jax.lax.scan(
            step,
            initial_carry,
            None,
            length=rollout_steps,
        )
        return completed

    completed = jax.jit(rollout)(jax.random.PRNGKey(seed))
    completed_return = np.asarray(completed.episode_return).reshape(-1)
    completed_length = np.asarray(completed.episode_length).reshape(-1)
    valid = ~np.isnan(completed_return)
    returns = completed_return[valid][:num_episodes]
    lengths = completed_length[valid][:num_episodes]
    if len(returns) < num_episodes:
        raise RuntimeError(
            f"Only {len(returns)} episodes completed; expected {num_episodes}."
        )

    return {
        "environment": config.env_name,
        "device": str(jax.devices()[0]),
        "num_episodes": num_episodes,
        "num_envs": num_envs,
        "deterministic": deterministic,
        "seed": seed,
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_episode_length": float(lengths.mean()),
    }


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    *,
    num_episodes: int = 16,
    num_envs: int = 8,
    seed: int = 1,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it with the saved configuration."""

    checkpoint_path = Path(checkpoint_dir)
    config = load_config(checkpoint_path / "config.yaml")
    bundle = make_environment(
        config.env_name,
        auto_reset=True,
        max_episode_steps=config.max_episode_steps,
    )
    model = ActorCriticRNN(
        num_actions=bundle.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    template_params = initialize_parameters(
        model,
        jax.random.PRNGKey(0),
        bundle.spec.observation_shape,
        batch_size=1,
    )
    params, checkpoint_metadata = load_checkpoint(
        checkpoint_path,
        template_params,
    )
    metrics = evaluate_policy(
        params,
        config,
        num_episodes=num_episodes,
        num_envs=num_envs,
        seed=seed,
        deterministic=deterministic,
    )
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_timesteps"] = checkpoint_metadata.get("timesteps")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint directory to evaluate.")
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of taking the highest-probability action.",
    )
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        args.checkpoint,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        seed=args.seed,
        deterministic=not args.stochastic,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
