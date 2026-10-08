import jax
import jax.numpy as jnp

from tiny_sima.executor.model import (
    ActorCriticRNN,
    categorical_entropy,
    categorical_log_probability,
    initial_hidden,
    initialize_parameters,
)


def test_actor_critic_shapes_and_finite_outputs() -> None:
    model = ActorCriticRNN(num_actions=7, hidden_size=32)
    params = initialize_parameters(
        model,
        jax.random.PRNGKey(0),
        (19,),
        batch_size=3,
    )
    hidden = initial_hidden(3, 32)
    observations = jnp.zeros((5, 3, 19), dtype=jnp.float32)
    resets = jnp.zeros((5, 3), dtype=jnp.bool_)

    next_hidden, logits, values = model.apply(
        params,
        hidden,
        (observations, resets),
    )

    assert next_hidden.shape == (3, 32)
    assert logits.shape == (5, 3, 7)
    assert values.shape == (5, 3)
    assert jnp.isfinite(logits).all()
    assert jnp.isfinite(values).all()


def test_categorical_helpers() -> None:
    logits = jnp.array([[1.0, 2.0, 3.0]])
    actions = jnp.array([2])

    log_probability = categorical_log_probability(logits, actions)
    entropy = categorical_entropy(logits)

    assert log_probability.shape == (1,)
    assert entropy.shape == (1,)
    assert log_probability[0] <= 0.0
    assert entropy[0] > 0.0
