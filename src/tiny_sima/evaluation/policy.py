"""Evaluate saved Tiny-SIMA PPO-RNN checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tiny_sima.config import PPOConfig, load_config
from tiny_sima.envs.craftax import make_environment, reset_batch, step_batch
from tiny_sima.executor.checkpoints import load_checkpoint
from tiny_sima.executor.model import (
    ActorCriticRNN,
    initial_hidden,
    initialize_parameters,
)


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

    def rollout(rng: jax.Array):
        rng, reset_rng = jax.random.split(rng)
        observation, env_state = reset_batch(bundle, reset_rng, num_envs)
        hidden = initial_hidden(num_envs, config.hidden_size)
        done = jnp.zeros((num_envs,), dtype=jnp.bool_)
        episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
        episode_length = jnp.zeros((num_envs,), dtype=jnp.int32)

        def step(carry, _):
            (
                rng,
                env_state,
                observation,
                done,
                hidden,
                episode_return,
                episode_length,
            ) = carry
            rng, action_rng, env_rng = jax.random.split(rng, 3)
            hidden, logits, _ = model.apply(
                params,
                hidden,
                (observation[None, ...], done[None, ...]),
            )
            logits = logits[0]
            if deterministic:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = jax.random.categorical(action_rng, logits, axis=-1)

            observation, env_state, reward, done, _ = step_batch(
                bundle,
                env_rng,
                env_state,
                action,
            )
            episode_return = episode_return + reward
            episode_length = episode_length + 1
            completed_return = jnp.where(done, episode_return, jnp.nan)
            completed_length = jnp.where(
                done,
                episode_length.astype(jnp.float32),
                jnp.nan,
            )
            next_carry = (
                rng,
                env_state,
                observation,
                done,
                hidden,
                jnp.where(done, 0.0, episode_return),
                jnp.where(done, 0, episode_length),
            )
            return next_carry, (completed_return, completed_length)

        _, completed = jax.lax.scan(
            step,
            (
                rng,
                env_state,
                observation,
                done,
                hidden,
                episode_return,
                episode_length,
            ),
            None,
            length=rollout_steps,
        )
        return completed

    completed_return, completed_length = jax.jit(rollout)(jax.random.PRNGKey(seed))
    completed_return = np.asarray(completed_return).reshape(-1)
    completed_length = np.asarray(completed_length).reshape(-1)
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
