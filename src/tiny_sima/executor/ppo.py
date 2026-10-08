"""Notebook-friendly recurrent PPO training for Craftax."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from tiny_sima.config import PPOConfig, load_config, save_config
from tiny_sima.envs.craftax import (
    CraftaxBundle,
    make_environment,
    reset_batch,
    step_batch,
)
from tiny_sima.executor.checkpoints import save_checkpoint
from tiny_sima.executor.model import (
    ActorCriticRNN,
    categorical_entropy,
    categorical_log_probability,
    initial_hidden,
    initialize_parameters,
)


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


def create_train_state(
    model: ActorCriticRNN,
    params: Any,
    config: PPOConfig,
) -> TrainState:
    """Create the optimizer and Flax training state."""

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


def initialize_runner(
    config: PPOConfig,
    bundle: CraftaxBundle,
    model: ActorCriticRNN,
) -> RunnerState:
    """Initialize policy parameters and a vector of Craftax environments."""

    rng = jax.random.PRNGKey(config.seed)
    rng, parameter_rng, reset_rng, runner_rng = jax.random.split(rng, 4)
    params = initialize_parameters(
        model,
        parameter_rng,
        bundle.spec.observation_shape,
        batch_size=config.num_envs,
    )
    train_state = create_train_state(model, params, config)
    observation, env_state = reset_batch(bundle, reset_rng, config.num_envs)

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


def _calculate_gae(
    transitions: Transition,
    last_value: jax.Array,
    last_done: jax.Array,
    config: PPOConfig,
) -> tuple[jax.Array, jax.Array]:
    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        transition: Transition,
    ):
        gae, next_value, next_done = carry
        not_done = 1.0 - next_done.astype(jnp.float32)
        delta = (
            transition.reward + config.gamma * next_value * not_done - transition.value
        )
        gae = delta + config.gamma * config.gae_lambda * not_done * gae
        return (gae, transition.value, transition.reset), gae

    (_, _, _), advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value, last_done),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def make_update(
    config: PPOConfig,
    bundle: CraftaxBundle,
    model: ActorCriticRNN,
):
    """Build one compiled rollout-and-update function."""

    def update(runner: RunnerState) -> tuple[RunnerState, dict[str, jax.Array]]:
        initial_recurrent_state = runner.hidden

        def collect_step(
            current: RunnerState,
            _: None,
        ) -> tuple[RunnerState, Transition]:
            rng, action_rng, env_rng = jax.random.split(current.rng, 3)

            network_input = (
                current.observation[None, ...],
                current.done[None, ...],
            )
            hidden, logits, value = model.apply(
                current.train_state.params,
                current.hidden,
                network_input,
            )
            logits = logits[0]
            value = value[0]
            action = jax.random.categorical(action_rng, logits, axis=-1)
            log_probability = categorical_log_probability(logits, action)

            observation, env_state, reward, done, _ = step_batch(
                bundle,
                env_rng,
                current.env_state,
                action,
            )

            episode_return = current.episode_return + reward
            episode_length = current.episode_length + 1
            completed_return = jnp.where(done, episode_return, 0.0)
            completed_length = jnp.where(done, episode_length, 0)

            transition = Transition(
                reset=current.done,
                done=done,
                action=action,
                value=value,
                reward=reward,
                log_probability=log_probability,
                observation=current.observation,
                completed_return=completed_return,
                completed_length=completed_length,
            )
            next_runner = RunnerState(
                train_state=current.train_state,
                env_state=env_state,
                observation=observation,
                done=done,
                hidden=hidden,
                rng=rng,
                episode_return=jnp.where(done, 0.0, episode_return),
                episode_length=jnp.where(done, 0, episode_length),
                completed_return_sum=(
                    current.completed_return_sum + completed_return.sum()
                ),
                completed_length_sum=(
                    current.completed_length_sum + completed_length.sum()
                ),
                completed_episodes=current.completed_episodes + done.sum(),
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
        _, _, last_value = model.apply(
            runner.train_state.params,
            runner.hidden,
            network_input,
        )
        advantages, targets = _calculate_gae(
            transitions,
            last_value[0],
            runner.done,
            config,
        )

        def loss(
            params: Any,
            minibatch_hidden: jax.Array,
            minibatch_transitions: Transition,
            minibatch_advantages: jax.Array,
            minibatch_targets: jax.Array,
        ):
            _, logits, values = model.apply(
                params,
                minibatch_hidden,
                (
                    minibatch_transitions.observation,
                    minibatch_transitions.reset,
                ),
            )
            log_probability = categorical_log_probability(
                logits,
                minibatch_transitions.action,
            )

            value_clipped = minibatch_transitions.value + jnp.clip(
                values - minibatch_transitions.value,
                -config.clip_epsilon,
                config.clip_epsilon,
            )
            value_loss_unclipped = jnp.square(values - minibatch_targets)
            value_loss_clipped = jnp.square(value_clipped - minibatch_targets)
            value_loss = (
                0.5
                * jnp.maximum(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
            )

            normalized_advantages = (
                minibatch_advantages - minibatch_advantages.mean()
            ) / (minibatch_advantages.std() + 1e-8)
            log_ratio = log_probability - minibatch_transitions.log_probability
            ratio = jnp.exp(log_ratio)
            actor_loss_unclipped = -ratio * normalized_advantages
            actor_loss_clipped = (
                -jnp.clip(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                )
                * normalized_advantages
            )
            actor_loss = jnp.maximum(
                actor_loss_unclipped,
                actor_loss_clipped,
            ).mean()

            entropy = categorical_entropy(logits).mean()
            total_loss = (
                actor_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            approximate_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (jnp.abs(ratio - 1.0) > config.clip_epsilon).mean()
            metrics = {
                "loss": total_loss,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approximate_kl": approximate_kl,
                "clip_fraction": clip_fraction,
            }
            return total_loss, metrics

        def update_minibatch(
            train_state: TrainState,
            minibatch: tuple[
                jax.Array,
                Transition,
                jax.Array,
                jax.Array,
            ],
        ):
            (
                minibatch_hidden,
                minibatch_transitions,
                minibatch_advantages,
                minibatch_targets,
            ) = minibatch
            gradient_function = jax.value_and_grad(loss, has_aux=True)
            (_, metrics), gradients = gradient_function(
                train_state.params,
                minibatch_hidden,
                minibatch_transitions,
                minibatch_advantages,
                minibatch_targets,
            )
            train_state = train_state.apply_gradients(grads=gradients)
            return train_state, metrics

        def update_epoch(
            carry: tuple[TrainState, jax.Array],
            _: None,
        ):
            train_state, rng = carry
            rng, permutation_rng = jax.random.split(rng)
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
                train_state,
                (
                    minibatch_hidden,
                    minibatch_transitions,
                    minibatch_advantages,
                    minibatch_targets,
                ),
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return (train_state, rng), metrics

        (train_state, rng), optimization_metrics = jax.lax.scan(
            update_epoch,
            (runner.train_state, runner.rng),
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
        runner = runner._replace(train_state=train_state, rng=rng)
        return runner, metrics

    return jax.jit(update)


def _host_metrics(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(np.asarray(value)) for name, value in metrics.items()}


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

    bundle = make_environment(
        config.env_name,
        auto_reset=True,
        max_episode_steps=config.max_episode_steps,
    )
    model = ActorCriticRNN(
        num_actions=bundle.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    runner = initialize_runner(config, bundle, model)
    update = make_update(config, bundle, model)

    metrics_path = run_dir / "metrics.jsonl"
    start = time.perf_counter()

    compile_start = time.perf_counter()
    runner, device_metrics = update(runner)
    jax.block_until_ready(device_metrics)
    compile_and_first_update_seconds = time.perf_counter() - compile_start

    latest_metrics = _host_metrics(device_metrics)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for update_index in range(config.num_updates):
            if update_index > 0:
                runner, device_metrics = update(runner)

            should_log = (
                update_index == 0
                or (update_index + 1) % config.log_every == 0
                or update_index + 1 == config.num_updates
            )
            if should_log:
                jax.block_until_ready(device_metrics)
                latest_metrics = _host_metrics(device_metrics)

                timesteps = (update_index + 1) * config.batch_size
                elapsed = time.perf_counter() - start
                record = {
                    "update": update_index + 1,
                    "timesteps": timesteps,
                    "elapsed_seconds": elapsed,
                    "steps_per_second_including_compile": timesteps / elapsed,
                    **latest_metrics,
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(
                    f"update={update_index + 1}/{config.num_updates} "
                    f"steps={timesteps} "
                    f"reward={latest_metrics['reward_mean']:.4f} "
                    f"sps={record['steps_per_second_including_compile']:.0f}"
                )

            if (
                config.checkpoint_every > 0
                and (update_index + 1) % config.checkpoint_every == 0
            ):
                checkpoint_name = f"step-{(update_index + 1) * config.batch_size}"
                save_checkpoint(
                    run_dir / "checkpoints" / checkpoint_name,
                    runner.train_state.params,
                    config,
                    timesteps=(update_index + 1) * config.batch_size,
                    metrics=latest_metrics,
                )

    jax.block_until_ready(runner.train_state.params)
    completed_return_sum = float(np.asarray(runner.completed_return_sum))
    completed_length_sum = float(np.asarray(runner.completed_length_sum))
    completed_episodes = int(np.asarray(runner.completed_episodes))
    total_seconds = time.perf_counter() - start
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
        "total_seconds": total_seconds,
        "compile_and_first_update_seconds": compile_and_first_update_seconds,
        "steps_per_second_including_compile": config.total_timesteps / total_seconds,
        "mean_completed_episode_return": mean_episode_return,
        "mean_completed_episode_length": mean_episode_length,
        "completed_episodes": completed_episodes,
        "latest_metrics": latest_metrics,
    }

    checkpoint_dir = save_checkpoint(
        run_dir / "checkpoints" / "final",
        runner.train_state.params,
        config,
        timesteps=config.total_timesteps,
        metrics=summary,
    )
    summary["checkpoint"] = str(checkpoint_dir)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


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
