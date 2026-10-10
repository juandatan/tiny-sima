"""Small, tested adapters around the Craftax Gymnax-style API."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name


@dataclass(frozen=True)
class EnvironmentSpec:
    """Static information needed to construct policies."""

    name: str
    observation_shape: tuple[int, ...]
    num_actions: int
    max_episode_steps: int


@dataclass(frozen=True)
class CraftaxBundle:
    """An environment together with the parameters used by its functions."""

    env: Any
    params: Any
    spec: EnvironmentSpec


@dataclass(frozen=True)
class VectorizedCraftax:
    """A batched environment with optional efficient optimistic resets."""

    bundle: CraftaxBundle
    num_envs: int
    optimistic_reset_ratio: int | None = None

    @property
    def spec(self) -> EnvironmentSpec:
        return self.bundle.spec

    def reset(self, rng: jax.Array) -> tuple[jax.Array, Any]:
        return reset_batch(self.bundle, rng, self.num_envs)

    def step(
        self,
        rng: jax.Array,
        state: Any,
        action: jax.Array,
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, jax.Array]]:
        if self.optimistic_reset_ratio is None:
            return step_batch(self.bundle, rng, state, action)
        return optimistic_step_batch(
            self.bundle,
            rng,
            state,
            action,
            reset_ratio=self.optimistic_reset_ratio,
        )


class BenchmarkResult(NamedTuple):
    environment: str
    device: str
    num_envs: int
    num_steps: int
    reset_mode: str
    optimistic_reset_ratio: int | None
    compile_seconds: float
    run_seconds: float
    steps_per_second: float


def make_environment(
    env_name: str = "Craftax-Classic-Symbolic-v1",
    *,
    auto_reset: bool = True,
    max_episode_steps: int | None = None,
) -> CraftaxBundle:
    """Create a Craftax environment and describe its observation/action spaces."""

    env = make_craftax_env_from_name(env_name, auto_reset=auto_reset)
    params = env.default_params
    if max_episode_steps is not None:
        if not hasattr(params, "replace"):
            raise TypeError("Craftax environment parameters do not support replace().")
        params = params.replace(max_timesteps=max_episode_steps)

    observation_shape = tuple(env.observation_space(params).shape)
    num_actions = int(env.action_space(params).n)
    effective_max_steps = int(getattr(params, "max_timesteps", 10_000))
    spec = EnvironmentSpec(
        name=env_name,
        observation_shape=observation_shape,
        num_actions=num_actions,
        max_episode_steps=effective_max_steps,
    )
    return CraftaxBundle(env=env, params=params, spec=spec)


def reset_batch(
    bundle: CraftaxBundle,
    rng: jax.Array,
    num_envs: int,
) -> tuple[jax.Array, Any]:
    """Reset a batch of independent environments."""

    reset_keys = jax.random.split(rng, num_envs)
    return jax.vmap(bundle.env.reset, in_axes=(0, None))(reset_keys, bundle.params)


def step_batch(
    bundle: CraftaxBundle,
    rng: jax.Array,
    state: Any,
    action: jax.Array,
) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Step a batch of independent environments."""

    step_keys = jax.random.split(rng, action.shape[0])
    return jax.vmap(bundle.env.step, in_axes=(0, 0, 0, None))(
        step_keys,
        state,
        action,
        bundle.params,
    )


def optimistic_step_batch(
    bundle: CraftaxBundle,
    rng: jax.Array,
    state: Any,
    action: jax.Array,
    *,
    reset_ratio: int,
) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, jax.Array]]:
    """Step non-auto-reset environments using a small shared reset pool.

    This follows the optimistic reset strategy used by the official Craftax
    baselines. It may reuse a reset state when more environments terminate in
    one step than the reset pool contains.
    """

    num_envs = action.shape[0]
    if reset_ratio < 1 or num_envs % reset_ratio:
        raise ValueError("reset_ratio must be positive and divide num_envs.")
    num_resets = num_envs // reset_ratio

    step_rng, reset_rng, selection_rng = jax.random.split(rng, 3)
    stepped_observation, stepped_state, reward, done, info = step_batch(
        bundle,
        step_rng,
        state,
        action,
    )
    reset_observation, reset_state = reset_batch(
        bundle,
        reset_rng,
        num_resets,
    )

    random_scores = jax.random.uniform(selection_rng, shape=(num_envs,))
    done_scores = jnp.where(done, random_scores, -1.0)
    random_reset_indices = jnp.argsort(done_scores)[-num_resets:]
    deterministic_reset_indices = jnp.argsort(done)[-num_resets:]
    selected_reset_indices = jnp.where(
        done.sum() < num_resets,
        deterministic_reset_indices,
        random_reset_indices,
    )

    reset_assignment = jnp.arange(num_resets).repeat(reset_ratio)
    reset_assignment = reset_assignment.at[selected_reset_indices].set(
        jnp.arange(num_resets)
    )
    expanded_reset_observation = reset_observation[reset_assignment]
    expanded_reset_state = jax.tree_util.tree_map(
        lambda value: value[reset_assignment],
        reset_state,
    )

    observation = jax.vmap(
        lambda terminal, reset_value, stepped_value: jax.lax.select(
            terminal,
            reset_value,
            stepped_value,
        )
    )(done, expanded_reset_observation, stepped_observation)

    def select_state(terminal: jax.Array, reset_value: Any, stepped_value: Any):
        return jax.tree_util.tree_map(
            lambda reset_leaf, stepped_leaf: jax.lax.select(
                terminal,
                reset_leaf,
                stepped_leaf,
            ),
            reset_value,
            stepped_value,
        )

    next_state = jax.vmap(select_state)(
        done,
        expanded_reset_state,
        stepped_state,
    )
    return observation, next_state, reward, done, info


def make_vectorized_environment(
    env_name: str,
    *,
    num_envs: int,
    max_episode_steps: int,
    use_optimistic_resets: bool,
    optimistic_reset_ratio: int,
) -> VectorizedCraftax:
    """Create the vectorized environment used by PPO training."""

    if use_optimistic_resets:
        if optimistic_reset_ratio < 1 or num_envs % optimistic_reset_ratio:
            raise ValueError(
                "optimistic_reset_ratio must be positive and divide num_envs."
            )
        bundle = make_environment(
            env_name,
            auto_reset=False,
            max_episode_steps=max_episode_steps,
        )
        return VectorizedCraftax(
            bundle=bundle,
            num_envs=num_envs,
            optimistic_reset_ratio=optimistic_reset_ratio,
        )

    bundle = make_environment(
        env_name,
        auto_reset=True,
        max_episode_steps=max_episode_steps,
    )
    return VectorizedCraftax(bundle=bundle, num_envs=num_envs)


def render_state(
    state: Any,
    env_name: str,
    *,
    block_pixel_size: int = 16,
) -> np.ndarray:
    """Render a symbolic-environment state as an RGB uint8 image."""

    supported_sizes = (7, 16, 64)
    if block_pixel_size not in supported_sizes:
        raise ValueError(
            f"block_pixel_size must be one of {supported_sizes}; "
            f"received {block_pixel_size}."
        )

    if "Classic" in env_name:
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
    else:
        from craftax.craftax.renderer import make_craftax_pixel_renderer

    renderer = make_craftax_pixel_renderer(block_pixel_size)
    pixels = np.asarray(renderer(state))
    if np.issubdtype(pixels.dtype, np.floating) and pixels.max(initial=0) <= 1.0:
        pixels = pixels * 255.0
    return np.clip(pixels, 0, 255).astype(np.uint8)


def benchmark_environment(
    *,
    env_name: str = "Craftax-Classic-Symbolic-v1",
    num_envs: int = 64,
    num_steps: int = 256,
    seed: int = 0,
    max_episode_steps: int = 1000,
    use_optimistic_resets: bool = True,
    optimistic_reset_ratio: int = 16,
) -> BenchmarkResult:
    """Compile and time a vectorized random-policy rollout."""

    vector_env = make_vectorized_environment(
        env_name,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        use_optimistic_resets=use_optimistic_resets,
        optimistic_reset_ratio=optimistic_reset_ratio,
    )

    def rollout(rng: jax.Array) -> jax.Array:
        rng, reset_rng = jax.random.split(rng)
        observation, state = vector_env.reset(reset_rng)

        def step(carry: tuple[jax.Array, Any], _: None):
            step_rng, env_state = carry
            step_rng, action_rng, env_rng = jax.random.split(step_rng, 3)
            action = jax.random.randint(
                action_rng,
                shape=(num_envs,),
                minval=0,
                maxval=vector_env.spec.num_actions,
            )
            next_observation, next_state, _, _, _ = vector_env.step(
                env_rng,
                env_state,
                action,
            )
            return (step_rng, next_state), next_observation

        (_, _), observations = jax.lax.scan(
            step,
            (rng, state),
            None,
            length=num_steps,
        )
        return observations[-1] + observation * 0.0

    compiled_rollout = jax.jit(rollout)
    rng = jax.random.PRNGKey(seed)

    compile_start = time.perf_counter()
    compiled_rollout(rng).block_until_ready()
    compile_seconds = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    compiled_rollout(rng).block_until_ready()
    run_seconds = time.perf_counter() - run_start
    total_steps = num_envs * num_steps

    return BenchmarkResult(
        environment=env_name,
        device=str(jax.devices()[0]),
        num_envs=num_envs,
        num_steps=num_steps,
        reset_mode="optimistic" if use_optimistic_resets else "automatic",
        optimistic_reset_ratio=(
            optimistic_reset_ratio if use_optimistic_resets else None
        ),
        compile_seconds=compile_seconds,
        run_seconds=run_seconds,
        steps_per_second=total_steps / run_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-name",
        default="Craftax-Classic-Symbolic-v1",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument(
        "--no-optimistic-resets",
        action="store_true",
        help="Use regular per-environment automatic resets.",
    )
    parser.add_argument("--optimistic-reset-ratio", type=int, default=16)
    args = parser.parse_args()

    result = benchmark_environment(
        env_name=args.env_name,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        use_optimistic_resets=not args.no_optimistic_resets,
        optimistic_reset_ratio=args.optimistic_reset_ratio,
    )
    print(json.dumps(result._asdict(), indent=2))


if __name__ == "__main__":
    main()
