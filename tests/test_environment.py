import jax
import numpy as np

from tiny_sima.envs.craftax import make_environment


def test_craftax_classic_reset_and_step() -> None:
    bundle = make_environment(
        "Craftax-Classic-Symbolic-v1",
        max_episode_steps=32,
    )
    reset_rng, action_rng, step_rng = jax.random.split(
        jax.random.PRNGKey(0),
        3,
    )

    observation, state = bundle.env.reset(reset_rng, bundle.params)
    action = bundle.env.action_space(bundle.params).sample(action_rng)
    next_observation, _, reward, done, info = bundle.env.step(
        step_rng,
        state,
        action,
        bundle.params,
    )

    assert observation.shape == bundle.spec.observation_shape
    assert next_observation.shape == bundle.spec.observation_shape
    assert np.isfinite(np.asarray(reward))
    assert np.asarray(done).shape == ()
    assert "discount" in info
    assert bundle.spec.num_actions == 17
    assert bundle.spec.max_episode_steps == 32
