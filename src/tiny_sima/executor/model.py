"""Recurrent actor-critic network used by the Phase 1 baseline."""

from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ScannedGRU(nn.Module):
    """A GRU cell scanned over time, with episode-boundary resets."""

    hidden_size: int

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self,
        carry: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        features, resets = inputs
        carry = jnp.where(resets[:, None], jnp.zeros_like(carry), carry)
        return nn.GRUCell(features=self.hidden_size)(carry, features)


class ActorCriticRNN(nn.Module):
    """A compact recurrent policy for flat symbolic Craftax observations."""

    num_actions: int
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        hidden: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        observations, resets = inputs

        embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="observation_encoder",
        )(observations)
        embedding = nn.relu(embedding)

        hidden, embedding = ScannedGRU(
            hidden_size=self.hidden_size,
            name="memory",
        )(hidden, (embedding, resets))

        actor = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="actor_hidden",
        )(embedding)
        actor = nn.relu(actor)
        logits = nn.Dense(
            self.num_actions,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_logits",
        )(actor)

        critic = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="critic_hidden",
        )(embedding)
        critic = nn.relu(critic)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_value",
        )(critic)

        return hidden, logits, jnp.squeeze(value, axis=-1)


def initial_hidden(batch_size: int, hidden_size: int) -> jax.Array:
    """Create an all-zero recurrent state."""

    return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


def initialize_parameters(
    model: ActorCriticRNN,
    rng: jax.Array,
    observation_shape: tuple[int, ...],
    *,
    batch_size: int = 1,
):
    """Initialize model parameters for a given environment observation shape."""

    observations = jnp.zeros(
        (1, batch_size, *observation_shape),
        dtype=jnp.float32,
    )
    resets = jnp.zeros((1, batch_size), dtype=jnp.bool_)
    hidden = initial_hidden(batch_size, model.hidden_size)
    return model.init(rng, hidden, (observations, resets))


def categorical_log_probability(
    logits: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Return log probabilities for selected categorical actions."""

    all_log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        all_log_probabilities,
        actions[..., None],
        axis=-1,
    )[..., 0]


def categorical_entropy(logits: jax.Array) -> jax.Array:
    """Return the entropy of categorical action distributions."""

    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    probabilities = jnp.exp(log_probabilities)
    return -jnp.sum(probabilities * log_probabilities, axis=-1)
