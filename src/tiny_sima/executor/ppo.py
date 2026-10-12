"""Notebook-friendly recurrent PPO training for Craftax."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from tiny_sima.config import PPOConfig, load_config, save_config
from tiny_sima.envs.craftax import (
    VectorizedCraftax,
    make_vectorized_environment,
)
from tiny_sima.evaluation.gpu_monitor import (
    NvidiaSmiMonitor,
    jax_allocator_summary,
)
from tiny_sima.executor.model import (
    ActorCriticRNN,
    apply_actor_critic,
    categorical_log_probability,
    initial_hidden,
    initialize_parameters,
    make_actor_critic,
)
from tiny_sima.executor.ppo_common import (
    OptimizationCarry,
    PPOLossBatch,
    RecurrentMinibatch,
    UpdateMetrics,
    calculate_gae,
    clipped_ppo_loss,
    create_train_state,
)
from tiny_sima.executor.training import run_training_loop, save_final_run


class Transition(NamedTuple):
    reset: jax.Array
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_probability: jax.Array
    observation: jax.Array
    completed_return: jax.Array
    completed_length: jax.Array


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    observation: jax.Array
    done: jax.Array
    hidden: jax.Array
    rng: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    completed_return_sum: jax.Array
    completed_length_sum: jax.Array
    completed_episodes: jax.Array


UpdateFunction = Callable[[RunnerState], tuple[RunnerState, UpdateMetrics]]


def _runner_params(runner: RunnerState) -> Any:
    return runner.train_state.params


def _format_progress(
    update_index: int,
    timesteps: int,
    metrics: dict[str, float],
    record: dict[str, Any],
) -> str:
    return (
        f"update={update_index}/{record['total_updates']} "
        f"steps={timesteps} "
        f"reward={metrics['reward_mean']:.4f} "
        f"sps={record['steps_per_second_including_compile']:.0f} "
        f"steady_sps={record['steps_per_second_after_compile'] or 0:.0f}"
    )


def initialize_runner(
    config: PPOConfig,
    vector_env: VectorizedCraftax,
    model: ActorCriticRNN,
) -> RunnerState:
    """Initialize policy parameters and a vector of Craftax environments."""

    rng = jax.random.PRNGKey(config.seed)
    rng, parameter_rng, reset_rng, runner_rng = jax.random.split(rng, 4)
    params = initialize_parameters(
        model,
        parameter_rng,
        vector_env.spec.observation_shape,
        batch_size=config.num_envs,
    )
    train_state = create_train_state(model, params, config)
    observation, env_state = vector_env.reset(reset_rng)

    return RunnerState(
        train_state=train_state,
        env_state=env_state,
        observation=observation,
        done=jnp.zeros((config.num_envs,), dtype=jnp.bool_),
        hidden=initial_hidden(config.num_envs, config.hidden_size),
        rng=runner_rng,
        episode_return=jnp.zeros((config.num_envs,), dtype=jnp.float32),
        episode_length=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        completed_return_sum=jnp.array(0.0, dtype=jnp.float32),
        completed_length_sum=jnp.array(0, dtype=jnp.int32),
        completed_episodes=jnp.array(0, dtype=jnp.int32),
    )


def make_update(
    config: PPOConfig,
    vector_env: VectorizedCraftax,
    model: ActorCriticRNN,
) -> UpdateFunction:
    """Build one compiled rollout-and-update function."""

    def update(runner: RunnerState) -> tuple[RunnerState, UpdateMetrics]:
        initial_recurrent_state = runner.hidden

        def collect_step(
            current: RunnerState,
            _unused: None,
        ) -> tuple[RunnerState, Transition]:
            rng, action_rng, env_rng = jax.random.split(current.rng, 3)

            network_input = (
                current.observation[None, ...],
                current.done[None, ...],
            )
            model_output = apply_actor_critic(
                model,
                current.train_state.params,
                current.hidden,
                network_input,
            )
            logits = model_output.logits[0]
            value = model_output.value[0]
            action = jax.random.categorical(action_rng, logits, axis=-1)
            log_probability = categorical_log_probability(logits, action)

            env_step = vector_env.step(
                env_rng,
                current.env_state,
                action,
            )

            episode_return = current.episode_return + env_step.reward
            episode_length = current.episode_length + 1
            completed_return = jnp.where(env_step.done, episode_return, 0.0)
            completed_length = jnp.where(env_step.done, episode_length, 0)

            transition = Transition(
                reset=current.done,
                done=env_step.done,
                action=action,
                value=value,
                reward=env_step.reward,
                log_probability=log_probability,
                observation=current.observation,
                completed_return=completed_return,
                completed_length=completed_length,
            )
            next_runner = RunnerState(
                train_state=current.train_state,
                env_state=env_step.state,
                observation=env_step.observation,
                done=env_step.done,
                hidden=model_output.hidden,
                rng=rng,
                episode_return=jnp.where(
                    env_step.done,
                    0.0,
                    episode_return,
                ),
                episode_length=jnp.where(
                    env_step.done,
                    0,
                    episode_length,
                ),
                completed_return_sum=(
                    current.completed_return_sum + completed_return.sum()
                ),
                completed_length_sum=(
                    current.completed_length_sum + completed_length.sum()
                ),
                completed_episodes=(current.completed_episodes + env_step.done.sum()),
            )
            return next_runner, transition

        runner, transitions = jax.lax.scan(
            collect_step,
            runner,
            None,
            length=config.num_steps,
        )

        network_input = (
            runner.observation[None, ...],
            runner.done[None, ...],
        )
        last_model_output = apply_actor_critic(
            model,
            runner.train_state.params,
            runner.hidden,
            network_input,
        )
        advantages, targets = calculate_gae(
            transitions.reward,
            transitions.value,
            transitions.reset,
            last_model_output.value[0],
            runner.done,
            config,
        )

        def loss(
            params: Any,
            minibatch_hidden: jax.Array,
            minibatch_transitions: Transition,
            minibatch_advantages: jax.Array,
            minibatch_targets: jax.Array,
        ) -> tuple[jax.Array, UpdateMetrics]:
            model_output = apply_actor_critic(
                model,
                params,
                minibatch_hidden,
                (
                    minibatch_transitions.observation,
                    minibatch_transitions.reset,
                ),
            )
            return clipped_ppo_loss(
                model_output.logits,
                model_output.value,
                PPOLossBatch(
                    action=minibatch_transitions.action,
                    old_value=minibatch_transitions.value,
                    old_log_probability=(minibatch_transitions.log_probability),
                    advantage=minibatch_advantages,
                    target=minibatch_targets,
                ),
                config,
            )

        def update_minibatch(
            train_state: TrainState,
            minibatch: RecurrentMinibatch[Transition],
        ) -> tuple[TrainState, UpdateMetrics]:
            gradient_function = jax.value_and_grad(loss, has_aux=True)
            (_, metrics), gradients = gradient_function(
                train_state.params,
                minibatch.hidden,
                minibatch.transitions,
                minibatch.advantages,
                minibatch.targets,
            )
            train_state = train_state.apply_gradients(grads=gradients)
            return train_state, metrics

        def update_epoch(
            carry: OptimizationCarry,
            _unused: None,
        ) -> tuple[OptimizationCarry, UpdateMetrics]:
            rng, permutation_rng = jax.random.split(carry.rng)
            permutation = jax.random.permutation(
                permutation_rng,
                config.num_envs,
            )
            minibatch_indices = permutation.reshape(
                config.num_minibatches,
                config.envs_per_minibatch,
            )

            minibatch_hidden = initial_recurrent_state[minibatch_indices]
            minibatch_transitions = jax.tree_util.tree_map(
                lambda value: jnp.swapaxes(value[:, minibatch_indices], 0, 1),
                transitions,
            )
            minibatch_advantages = jnp.swapaxes(
                advantages[:, minibatch_indices],
                0,
                1,
            )
            minibatch_targets = jnp.swapaxes(
                targets[:, minibatch_indices],
                0,
                1,
            )

            train_state, metrics = jax.lax.scan(
                update_minibatch,
                carry.train_state,
                RecurrentMinibatch(
                    hidden=minibatch_hidden,
                    transitions=minibatch_transitions,
                    advantages=minibatch_advantages,
                    targets=minibatch_targets,
                ),
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            next_carry = OptimizationCarry(
                train_state=train_state,
                rng=rng,
            )
            return next_carry, metrics

        optimization, optimization_metrics = jax.lax.scan(
            update_epoch,
            OptimizationCarry(
                train_state=runner.train_state,
                rng=runner.rng,
            ),
            None,
            length=config.update_epochs,
        )
        optimization_metrics = jax.tree_util.tree_map(
            jnp.mean,
            optimization_metrics,
        )

        completed_episodes = transitions.done.sum()
        rollout_metrics = {
            "reward_mean": transitions.reward.mean(),
            "completed_episodes": completed_episodes,
            "completed_return_sum": transitions.completed_return.sum(),
            "completed_length_sum": transitions.completed_length.sum(),
        }
        metrics = {**optimization_metrics, **rollout_metrics}
        runner = runner._replace(
            train_state=optimization.train_state,
            rng=optimization.rng,
        )
        return runner, metrics

    return cast(UpdateFunction, jax.jit(update))


def train(
    config: PPOConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train the Phase 1 PPO-RNN baseline and save a final checkpoint."""

    config = config.validate()
    run_dir = Path(output_dir or config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    gpu_monitor = NvidiaSmiMonitor(
        enabled=config.monitor_gpu,
        interval_seconds=config.gpu_monitor_interval_seconds,
    )
    gpu_monitor.start()

    vector_env = make_vectorized_environment(
        config.env_name,
        num_envs=config.num_envs,
        max_episode_steps=config.max_episode_steps,
        use_optimistic_resets=config.use_optimistic_resets,
        optimistic_reset_ratio=config.optimistic_reset_ratio,
    )
    model = make_actor_critic(
        num_actions=vector_env.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    runner = initialize_runner(config, vector_env, model)
    update = make_update(config, vector_env, model)

    loop = run_training_loop(
        config=config,
        run_dir=run_dir,
        initial_runner=runner,
        update=update,
        monitor=gpu_monitor,
        get_params=_runner_params,
        format_progress=_format_progress,
    )
    runner = loop.runner
    jax_memory = jax_allocator_summary(jax.devices()[0])
    gpu_summary = gpu_monitor.stop(
        run_dir,
        steady_state_start_seconds=loop.steady_state_start_seconds,
        tail_start_seconds=loop.tail_gpu_start_seconds,
    )
    completed_return_sum = float(np.asarray(runner.completed_return_sum))
    completed_length_sum = float(np.asarray(runner.completed_length_sum))
    completed_episodes = int(np.asarray(runner.completed_episodes))
    mean_episode_return = (
        completed_return_sum / completed_episodes if completed_episodes else None
    )
    mean_episode_length = (
        completed_length_sum / completed_episodes if completed_episodes else None
    )
    summary = {
        "environment": config.env_name,
        "device": str(jax.devices()[0]),
        "timesteps": config.total_timesteps,
        "updates": config.num_updates,
        "total_seconds": loop.total_seconds,
        "compile_and_first_update_seconds": (loop.compile_and_first_update_seconds),
        "post_compile_seconds": loop.post_compile_seconds,
        "steps_per_second_including_compile": (
            config.total_timesteps / loop.total_seconds
        ),
        "steps_per_second_after_compile": (loop.steps_per_second_after_compile),
        "tail_steps_per_second": loop.tail_steps_per_second,
        "reset_mode": ("optimistic" if config.use_optimistic_resets else "automatic"),
        "optimistic_reset_ratio": (
            config.optimistic_reset_ratio if config.use_optimistic_resets else None
        ),
        "mean_completed_episode_return": mean_episode_return,
        "mean_completed_episode_length": mean_episode_length,
        "completed_episodes": completed_episodes,
        "latest_metrics": loop.latest_metrics,
        "jax_memory": jax_memory,
        "gpu_monitor": gpu_summary,
    }

    return save_final_run(
        run_dir=run_dir,
        params=runner.train_state.params,
        config=config,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ppo_debug.yaml",
        help="Path to a PPO YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory in the configuration.",
    )
    args = parser.parse_args()

    summary = train(load_config(args.config), output_dir=args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
