import jax
import numpy as np

from tiny_sima.envs.craftax import make_environment, render_state


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


def test_compiled_render_matches_the_uncompiled_craftax_renderer() -> None:
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

    bundle = make_environment("Craftax-Classic-Symbolic-v1")
    _, state = bundle.env.reset(jax.random.PRNGKey(0), bundle.params)

    pixels = render_state(state, bundle.spec.name, block_pixel_size=16)
    with jax.disable_jit():
        reference = np.asarray(make_craftax_pixel_renderer(16)(state))

    assert pixels.dtype == np.uint8
    assert pixels.ndim == 3 and pixels.shape[-1] == 3
    np.testing.assert_array_equal(
        pixels,
        np.clip(reference, 0, 255).astype(np.uint8),
    )
