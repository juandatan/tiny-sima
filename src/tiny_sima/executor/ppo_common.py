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


def normalize_advantages_by_group(
    advantages: jax.Array,
    group_ids: jax.Array,
    num_groups: int,
) -> jax.Array:
    """Normalize advantages independently within each requested task."""

    membership = jax.nn.one_hot(
        group_ids,
        num_groups,
        dtype=advantages.dtype,
    )
    reduction_axes = tuple(range(advantages.ndim))
    counts = membership.sum(axis=reduction_axes)
    safe_counts = jnp.maximum(counts, 1.0)
    means = (advantages[..., None] * membership).sum(axis=reduction_axes) / safe_counts
    centered = advantages - means[group_ids]
    variances = (jnp.square(centered)[..., None] * membership).sum(
        axis=reduction_axes
    ) / safe_counts
    return centered / jnp.sqrt(variances[group_ids] + 1e-8)


def stratified_minibatch_indices(
    rng: jax.Array,
    *,
    num_groups: int,
    items_per_group: int,
    num_minibatches: int,
) -> jax.Array:
    """Shuffle within groups while placing every group in every minibatch."""

    if items_per_group % num_minibatches:
        raise ValueError(
            "items_per_group must be divisible by num_minibatches for "
            "stratified minibatching."
        )

    group_indices = jnp.arange(
        num_groups * items_per_group,
        dtype=jnp.int32,
    ).reshape(num_groups, items_per_group)
    group_keys = jax.random.split(rng, num_groups)
    shuffled = jax.vmap(jax.random.permutation)(group_keys, group_indices)
    items_per_group_per_minibatch = items_per_group // num_minibatches
    return (
        shuffled.reshape(
            num_groups,
            num_minibatches,
            items_per_group_per_minibatch,
        )
        .transpose(1, 0, 2)
        .reshape(num_minibatches, -1)
    )


def _group_balanced_mean(
    values: jax.Array,
    group_ids: jax.Array | None,
    num_groups: int | None,
) -> jax.Array:
    if group_ids is None:
        return values.mean()
    if num_groups is None:
        raise ValueError("num_groups is required when group_ids are supplied.")

    membership = jax.nn.one_hot(
        group_ids,
        num_groups,
        dtype=values.dtype,
    )
    reduction_axes = tuple(range(values.ndim))
    counts = membership.sum(axis=reduction_axes)
    group_means = (values[..., None] * membership).sum(
        axis=reduction_axes
    ) / jnp.maximum(counts, 1.0)
    present = counts > 0
    return jnp.where(present, group_means, 0.0).sum() / jnp.maximum(
        present.sum(),
        1,
    )


def clipped_ppo_loss(
    logits: jax.Array,
    predicted_value: jax.Array,
    batch: PPOLossBatch,
    config: PPOConfig,
    *,
    group_ids: jax.Array | None = None,
    num_groups: int | None = None,
    normalize_advantage: bool = True,
) -> tuple[jax.Array, UpdateMetrics]:
    """Calculate the shared clipped actor-critic PPO objective."""

    log_probability = categorical_log_probability(logits, batch.action)
    value_clipped = batch.old_value + jnp.clip(
        predicted_value - batch.old_value,
        -config.clip_epsilon,
        config.clip_epsilon,
    )
    value_loss = 0.5 * _group_balanced_mean(
        jnp.maximum(
            jnp.square(predicted_value - batch.target),
            jnp.square(value_clipped - batch.target),
        ),
        group_ids,
        num_groups,
    )

    normalized_advantage = batch.advantage
    if normalize_advantage:
        normalized_advantage = (batch.advantage - batch.advantage.mean()) / (
            batch.advantage.std() + 1e-8
        )
    log_ratio = log_probability - batch.old_log_probability
    ratio = jnp.exp(log_ratio)
    actor_loss = _group_balanced_mean(
        jnp.maximum(
            -ratio * normalized_advantage,
            -jnp.clip(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            )
            * normalized_advantage,
        ),
        group_ids,
        num_groups,
    )
    entropy = _group_balanced_mean(
        categorical_entropy(logits),
        group_ids,
        num_groups,
    )
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
        "approximate_kl": _group_balanced_mean(
            (ratio - 1.0) - log_ratio,
            group_ids,
            num_groups,
        ),
        "clip_fraction": _group_balanced_mean(
            (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32),
            group_ids,
            num_groups,
        ),
    }
    return total_loss, metrics


def host_metrics(metrics: UpdateMetrics) -> dict[str, float]:
    """Copy scalar device metrics to JSON-friendly host floats."""

    return {name: float(np.asarray(value)) for name, value in metrics.items()}
