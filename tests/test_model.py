import jax
import jax.numpy as jnp

from tiny_sima.executor.model import (
    ActorCriticRNN,
    GoalConditionedActorCriticRNN,
    apply_actor_critic,
    apply_goal_conditioned_actor_critic,
    categorical_entropy,
    categorical_log_probability,
    initial_hidden,
    initialize_goal_conditioned_parameters,
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

    output = apply_actor_critic(
        model,
        params,
        hidden,
        (observations, resets),
    )

    assert output.hidden.shape == (3, 32)
    assert output.logits.shape == (5, 3, 7)
    assert output.value.shape == (5, 3)
    assert jnp.isfinite(output.logits).all()
    assert jnp.isfinite(output.value).all()


def test_categorical_helpers() -> None:
    logits = jnp.array([[1.0, 2.0, 3.0]])
    actions = jnp.array([2])

    log_probability = categorical_log_probability(logits, actions)
    entropy = categorical_entropy(logits)

    assert log_probability.shape == (1,)
    assert entropy.shape == (1,)
    assert log_probability[0] <= 0.0
    assert entropy[0] > 0.0


def test_goal_conditioned_actor_critic_shapes() -> None:
    model = GoalConditionedActorCriticRNN(
        num_actions=17,
        num_skills=6,
        hidden_size=32,
        skill_embedding_size=8,
    )
    params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        (19,),
        batch_size=3,
    )
    output = apply_goal_conditioned_actor_critic(
        model,
        params,
        initial_hidden(3, 32),
        (
            jnp.zeros((5, 3, 19), dtype=jnp.float32),
            jnp.array([[0, 1, 2]] * 5, dtype=jnp.int32),
            jnp.zeros((5, 3), dtype=jnp.bool_),
        ),
    )

    assert output.hidden.shape == (3, 32)
    assert output.logits.shape == (5, 3, 17)
    assert output.value.shape == (5, 3)
