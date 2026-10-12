"""Recurrent actor-critic network used by the Phase 1 baseline."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class ActorCriticOutput(NamedTuple):
    """Typed outputs produced by the recurrent actor-critic."""

    hidden: jax.Array
    logits: jax.Array
    value: jax.Array


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


def _encode_observation(
    observations: jax.Array,
    hidden_size: int,
) -> jax.Array:
    embedding = nn.Dense(
        hidden_size,
        kernel_init=orthogonal(jnp.sqrt(2.0)),
        bias_init=constant(0.0),
        name="observation_encoder",
    )(observations)
    return nn.relu(embedding)


def _apply_recurrent_actor_critic(
    hidden: jax.Array,
    embedding: jax.Array,
    resets: jax.Array,
    *,
    num_actions: int,
    hidden_size: int,
) -> ActorCriticOutput:
    hidden, embedding = ScannedGRU(
        hidden_size=hidden_size,
        name="memory",
    )(hidden, (embedding, resets))

    actor = nn.Dense(
        hidden_size,
        kernel_init=orthogonal(jnp.sqrt(2.0)),
        bias_init=constant(0.0),
        name="actor_hidden",
    )(embedding)
    actor = nn.relu(actor)
    logits = nn.Dense(
        num_actions,
        kernel_init=orthogonal(0.01),
        bias_init=constant(0.0),
        name="actor_logits",
    )(actor)

    critic = nn.Dense(
        hidden_size,
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

    return ActorCriticOutput(
        hidden=hidden,
        logits=logits,
        value=jnp.squeeze(value, axis=-1),
    )


class ActorCriticRNN(nn.Module):
    """A compact recurrent policy for flat symbolic Craftax observations."""

    num_actions: int
    hidden_size: int = 128

    @nn.compact
    def __call__(
        self,
        hidden: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> ActorCriticOutput:
        observations, resets = inputs
        embedding = _encode_observation(
            observations,
            self.hidden_size,
        )
        return _apply_recurrent_actor_critic(
            hidden,
            embedding,
            resets,
            num_actions=self.num_actions,
            hidden_size=self.hidden_size,
        )


class GoalConditionedActorCriticRNN(nn.Module):
    """Recurrent actor-critic conditioned on a categorical skill ID."""

    num_actions: int
    num_skills: int
    hidden_size: int = 128
    skill_embedding_size: int = 32

    @nn.compact
    def __call__(
        self,
        hidden: jax.Array,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> ActorCriticOutput:
        observations, skill_ids, resets = inputs

        observation_embedding = _encode_observation(
            observations,
            self.hidden_size,
        )

        skill_embedding = nn.Embed(
            num_embeddings=self.num_skills,
            features=self.skill_embedding_size,
            embedding_init=orthogonal(1.0),
            name="skill_embedding",
        )(skill_ids)
        embedding = jnp.concatenate(
            (observation_embedding, skill_embedding),
            axis=-1,
        )
        embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
            name="goal_fusion",
        )(embedding)
        embedding = nn.relu(embedding)

        return _apply_recurrent_actor_critic(
            hidden,
            embedding,
            resets,
            num_actions=self.num_actions,
            hidden_size=self.hidden_size,
        )


def make_actor_critic(
    *,
    num_actions: int,
    hidden_size: int,
) -> ActorCriticRNN:
    """Construct the baseline executor architecture."""

    return ActorCriticRNN(
        num_actions=num_actions,
        hidden_size=hidden_size,
    )


def make_goal_conditioned_actor_critic(
    *,
    num_actions: int,
    num_skills: int,
    hidden_size: int,
    skill_embedding_size: int,
) -> GoalConditionedActorCriticRNN:
    """Construct the categorical goal-conditioned executor architecture."""

    return GoalConditionedActorCriticRNN(
        num_actions=num_actions,
        num_skills=num_skills,
        hidden_size=hidden_size,
        skill_embedding_size=skill_embedding_size,
    )


def _apply_model(
    model: nn.Module,
    params: Any,
    hidden: jax.Array,
    inputs: Any,
) -> ActorCriticOutput:
    return cast(ActorCriticOutput, model.apply(params, hidden, inputs))


def apply_actor_critic(
    model: ActorCriticRNN,
    params: Any,
    hidden: jax.Array,
    inputs: tuple[jax.Array, jax.Array],
) -> ActorCriticOutput:
    """Apply a Flax model behind a precisely typed project boundary."""

    return _apply_model(model, params, hidden, inputs)


def apply_goal_conditioned_actor_critic(
    model: GoalConditionedActorCriticRNN,
    params: Any,
    hidden: jax.Array,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
) -> ActorCriticOutput:
    """Apply a goal-conditioned Flax model through a typed boundary."""

    return _apply_model(model, params, hidden, inputs)


def initial_hidden(batch_size: int, hidden_size: int) -> jax.Array:
    """Create an all-zero recurrent state."""

    return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


def _dummy_observations(
    observation_shape: tuple[int, ...],
    batch_size: int,
) -> jax.Array:
    return jnp.zeros(
        (1, batch_size, *observation_shape),
        dtype=jnp.float32,
    )


def initialize_parameters(
    model: ActorCriticRNN,
    rng: jax.Array,
    observation_shape: tuple[int, ...],
    *,
    batch_size: int = 1,
) -> Any:
    """Initialize model parameters for a given environment observation shape."""

    observations = _dummy_observations(observation_shape, batch_size)
    resets = jnp.zeros((1, batch_size), dtype=jnp.bool_)
    hidden = initial_hidden(batch_size, model.hidden_size)
    return model.init(rng, hidden, (observations, resets))


def initialize_goal_conditioned_parameters(
    model: GoalConditionedActorCriticRNN,
    rng: jax.Array,
    observation_shape: tuple[int, ...],
    *,
    batch_size: int = 1,
) -> Any:
    """Initialize goal-conditioned parameters for an observation shape."""

    observations = _dummy_observations(observation_shape, batch_size)
    skill_ids = jnp.zeros((1, batch_size), dtype=jnp.int32)
    resets = jnp.zeros((1, batch_size), dtype=jnp.bool_)
    hidden = initial_hidden(batch_size, model.hidden_size)
    return model.init(rng, hidden, (observations, skill_ids, resets))


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


def select_action(
    logits: jax.Array,
    rng: jax.Array,
    *,
    deterministic: bool,
) -> jax.Array:
    """Choose greedy or sampled categorical actions from policy logits."""

    if deterministic:
        return jnp.argmax(logits, axis=-1)
    return jax.random.categorical(rng, logits, axis=-1)
