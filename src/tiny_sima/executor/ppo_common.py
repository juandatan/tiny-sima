"""Shared optimizer, advantage, and loss utilities for PPO executors."""

from __future__ import annotations

from typing import Any, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from tiny_sima.config import PPOConfig
from tiny_sima.executor.model import (
    categorical_entropy,
    categorical_log_probability,
)

UpdateMetrics = dict[str, jax.Array]
TransitionType = TypeVar("TransitionType")


class GAECarry(NamedTuple):
    """Values propagated backward while calculating GAE."""

    advantage: jax.Array
    next_value: jax.Array
    next_done: jax.Array


class PPOLossBatch(NamedTuple):
    """Model-independent values consumed by the clipped PPO objective."""

    action: jax.Array
    old_value: jax.Array
    old_log_probability: jax.Array
    advantage: jax.Array
    target: jax.Array


class OptimizationCarry(NamedTuple):
    """Optimizer state and RNG carried between PPO epochs."""

    train_state: TrainState
    rng: jax.Array


class RecurrentMinibatch(NamedTuple, Generic[TransitionType]):
    """One environment-major recurrent PPO minibatch."""

    hidden: jax.Array
    transitions: TransitionType
    advantages: jax.Array
    targets: jax.Array


def create_train_state(
    model: nn.Module,
    params: Any,
    config: PPOConfig,
) -> TrainState:
    """Create the optimizer shared by baseline and skill-conditioned PPO."""

    optimizer_steps = config.num_updates * config.update_epochs * config.num_minibatches
    if config.anneal_learning_rate:
        learning_rate: float | optax.Schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=max(1, optimizer_steps),
        )
    else:
        learning_rate = config.learning_rate

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def calculate_gae(
    rewards: jax.Array,
    values: jax.Array,
    resets: jax.Array,
    last_value: jax.Array,
    last_done: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array]:
    """Calculate generalized advantage estimates for a rollout."""

    def step(
        carry: GAECarry,
        transition: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[GAECarry, jax.Array]:
        reward, value, reset = transition
        not_done = 1.0 - carry.next_done.astype(jnp.float32)
        delta = reward + config.gamma * carry.next_value * not_done - value
        advantage = (
            delta + config.gamma * config.gae_lambda * not_done * carry.advantage
        )
        return (
            GAECarry(
                advantage=advantage,
                next_value=value,
                next_done=reset,
            ),
            advantage,
        )

    _, advantages = jax.lax.scan(
        step,
        GAECarry(
            advantage=jnp.zeros_like(last_value),
            next_value=last_value,
            next_done=last_done,
        ),
        (rewards, values, resets),
        reverse=True,
    )
    return advantages, advantages + values


def clipped_ppo_loss(
    logits: jax.Array,
    predicted_value: jax.Array,
    batch: PPOLossBatch,
    config: PPOConfig,
) -> tuple[jax.Array, UpdateMetrics]:
    """Calculate the shared clipped actor-critic PPO objective."""

    log_probability = categorical_log_probability(logits, batch.action)
    value_clipped = batch.old_value + jnp.clip(
        predicted_value - batch.old_value,
        -config.clip_epsilon,
        config.clip_epsilon,
    )
    value_loss = (
        0.5
        * jnp.maximum(
            jnp.square(predicted_value - batch.target),
            jnp.square(value_clipped - batch.target),
        ).mean()
    )

    normalized_advantage = (batch.advantage - batch.advantage.mean()) / (
        batch.advantage.std() + 1e-8
    )
    log_ratio = log_probability - batch.old_log_probability
    ratio = jnp.exp(log_ratio)
    actor_loss = jnp.maximum(
        -ratio * normalized_advantage,
        -jnp.clip(
            ratio,
            1.0 - config.clip_epsilon,
            1.0 + config.clip_epsilon,
        )
        * normalized_advantage,
    ).mean()
    entropy = categorical_entropy(logits).mean()
    total_loss = (
        actor_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    metrics = {
        "loss": total_loss,
        "actor_loss": actor_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approximate_kl": ((ratio - 1.0) - log_ratio).mean(),
        "clip_fraction": (jnp.abs(ratio - 1.0) > config.clip_epsilon).mean(),
    }
    return total_loss, metrics


def host_metrics(metrics: UpdateMetrics) -> dict[str, float]:
    """Copy scalar device metrics to JSON-friendly host floats."""

    return {name: float(np.asarray(value)) for name, value in metrics.items()}
