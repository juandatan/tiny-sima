"""Goal-conditioned recurrent PPO training for the Tiny-SIMA skill suite."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from tiny_sima.config import SkillPPOConfig, load_skill_config, save_config
from tiny_sima.envs.skills import (
    SkillEpisodeState,
    VectorizedSkillEnvironment,
    make_vectorized_skill_environment,
)
from tiny_sima.evaluation.gpu_monitor import (
    NvidiaSmiMonitor,
    jax_allocator_summary,
)
from tiny_sima.executor.checkpoints import save_checkpoint
from tiny_sima.executor.model import (
    GoalConditionedActorCriticRNN,
    apply_goal_conditioned_actor_critic,
    categorical_entropy,
    categorical_log_probability,
    initial_hidden,
    initialize_goal_conditioned_parameters,
)
from tiny_sima.goals import SKILL_IDS


class SkillTransition(NamedTuple):
    """One timestep of batched goal-conditioned experience."""

    reset: jax.Array
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_probability: jax.Array
    observation: jax.Array
    skill_id: jax.Array
    completed_success: jax.Array
    completed_episode: jax.Array
    completed_length: jax.Array


class SkillRunnerState(NamedTuple):
    """Training state carried between compiled PPO updates."""

    train_state: TrainState
    env_state: SkillEpisodeState
    observation: jax.Array
    skill_id: jax.Array
    done: jax.Array
    hidden: jax.Array
    rng: jax.Array
    episode_length: jax.Array
    completed_successes: jax.Array
    completed_episodes: jax.Array
    completed_length_sum: jax.Array


class GAECarry(NamedTuple):
    """Values propagated backward while calculating GAE."""

    advantage: jax.Array
    next_value: jax.Array
    next_done: jax.Array


class SkillMinibatch(NamedTuple):
    """One recurrent goal-conditioned PPO minibatch."""

    hidden: jax.Array
    transitions: SkillTransition
    advantages: jax.Array
    targets: jax.Array


class OptimizationCarry(NamedTuple):
    """Optimizer state and RNG carried between PPO epochs."""

    train_state: TrainState
    rng: jax.Array


UpdateMetrics = dict[str, jax.Array]
UpdateFunction = Callable[
    [SkillRunnerState],
    tuple[SkillRunnerState, UpdateMetrics],
]


def create_skill_train_state(
    model: GoalConditionedActorCriticRNN,
    params: Any,
    config: SkillPPOConfig,
) -> TrainState:
    """Create the optimizer and Flax training state."""

    optimizer_steps = (
        config.num_updates * config.update_epochs * config.num_minibatches
    )
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


def initialize_skill_runner(
    config: SkillPPOConfig,
    environment: VectorizedSkillEnvironment,
    model: GoalConditionedActorCriticRNN,
) -> SkillRunnerState:
    """Initialize parameters and a balanced batch of skill episodes."""

    rng = jax.random.PRNGKey(config.seed)
    rng, parameter_rng, reset_rng, runner_rng = jax.random.split(rng, 4)
    params = initialize_goal_conditioned_parameters(
        model,
        parameter_rng,
        environment.spec.observation_shape,
        batch_size=config.num_envs,
    )
    train_state = create_skill_train_state(model, params, config)
    reset = environment.reset(reset_rng)
    num_skills = environment.num_skills

    return SkillRunnerState(
        train_state=train_state,
        env_state=reset.state,
        observation=reset.observation,
        skill_id=reset.skill_id,
        done=jnp.zeros((config.num_envs,), dtype=jnp.bool_),
        hidden=initial_hidden(config.num_envs, config.hidden_size),
        rng=runner_rng,
        episode_length=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        completed_successes=jnp.zeros((num_skills,), dtype=jnp.int32),
        completed_episodes=jnp.zeros((num_skills,), dtype=jnp.int32),
        completed_length_sum=jnp.zeros((num_skills,), dtype=jnp.int32),
    )


def _calculate_gae(
    transitions: SkillTransition,
    last_value: jax.Array,
    last_done: jax.Array,
    config: SkillPPOConfig,
) -> tuple[jax.Array, jax.Array]:
    def step(
        carry: GAECarry,
        transition: SkillTransition,
    ) -> tuple[GAECarry, jax.Array]:
        not_done = 1.0 - carry.next_done.astype(jnp.float32)
        delta = (
            transition.reward
            + config.gamma * carry.next_value * not_done
            - transition.value
        )
        advantage = (
            delta
            + config.gamma
            * config.gae_lambda
            * not_done
            * carry.advantage
        )
        next_carry = GAECarry(
            advantage=advantage,
            next_value=transition.value,
            next_done=transition.reset,
        )
        return next_carry, advantage

    _, advantages = jax.lax.scan(
        step,
        GAECarry(
            advantage=jnp.zeros_like(last_value),
            next_value=last_value,
            next_done=last_done,
        ),
        transitions,
        reverse=True,
    )
    return advantages, advantages + transitions.value


def make_skill_update(
    config: SkillPPOConfig,
    environment: VectorizedSkillEnvironment,
    model: GoalConditionedActorCriticRNN,
) -> UpdateFunction:
    """Build one compiled goal-conditioned rollout-and-update function."""

    num_skills = environment.num_skills

    def update(
        runner: SkillRunnerState,
    ) -> tuple[SkillRunnerState, UpdateMetrics]:
        initial_recurrent_state = runner.hidden

        def collect_step(
            current: SkillRunnerState,
            _unused: None,
        ) -> tuple[SkillRunnerState, SkillTransition]:
            rng, action_rng, env_rng = jax.random.split(current.rng, 3)
            model_output = apply_goal_conditioned_actor_critic(
                model,
                current.train_state.params,
                current.hidden,
                (
                    current.observation[None, ...],
                    current.skill_id[None, ...],
                    current.done[None, ...],
                ),
            )
            logits = model_output.logits[0]
            value = model_output.value[0]
            action = jax.random.categorical(action_rng, logits, axis=-1)
            log_probability = categorical_log_probability(logits, action)
            env_step = environment.step(
                env_rng,
                current.env_state,
                action,
            )

            episode_length = current.episode_length + 1
            completed_episode = env_step.done.astype(jnp.int32)
            completed_success = (
                env_step.done & env_step.info["skill_success"]
            ).astype(jnp.int32)
            completed_length = jnp.where(env_step.done, episode_length, 0)
            skill_one_hot = jax.nn.one_hot(
                current.skill_id,
                num_skills,
                dtype=jnp.int32,
            )

            transition = SkillTransition(
                reset=current.done,
                done=env_step.done,
                action=action,
                value=value,
                reward=env_step.reward,
                log_probability=log_probability,
                observation=current.observation,
                skill_id=current.skill_id,
                completed_success=completed_success,
                completed_episode=completed_episode,
                completed_length=completed_length,
            )
            next_runner = SkillRunnerState(
                train_state=current.train_state,
                env_state=env_step.state,
                observation=env_step.observation,
                skill_id=env_step.skill_id,
                done=env_step.done,
                hidden=model_output.hidden,
                rng=rng,
                episode_length=jnp.where(
                    env_step.done,
                    0,
                    episode_length,
                ),
                completed_successes=(
                    current.completed_successes
                    + (skill_one_hot * completed_success[:, None]).sum(axis=0)
                ),
                completed_episodes=(
                    current.completed_episodes
                    + (skill_one_hot * completed_episode[:, None]).sum(axis=0)
                ),
                completed_length_sum=(
                    current.completed_length_sum
                    + (skill_one_hot * completed_length[:, None]).sum(axis=0)
                ),
            )
            return next_runner, transition

        runner, transitions = jax.lax.scan(
            collect_step,
            runner,
            None,
            length=config.num_steps,
        )

        last_model_output = apply_goal_conditioned_actor_critic(
            model,
            runner.train_state.params,
            runner.hidden,
            (
                runner.observation[None, ...],
                runner.skill_id[None, ...],
                runner.done[None, ...],
            ),
        )
        advantages, targets = _calculate_gae(
            transitions,
            last_model_output.value[0],
            runner.done,
            config,
        )

        def loss(
            params: Any,
            minibatch: SkillMinibatch,
        ) -> tuple[jax.Array, UpdateMetrics]:
            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                minibatch.hidden,
                (
                    minibatch.transitions.observation,
                    minibatch.transitions.skill_id,
                    minibatch.transitions.reset,
                ),
            )
            log_probability = categorical_log_probability(
                model_output.logits,
                minibatch.transitions.action,
            )
            value_clipped = minibatch.transitions.value + jnp.clip(
                model_output.value - minibatch.transitions.value,
                -config.clip_epsilon,
                config.clip_epsilon,
            )
            value_loss_unclipped = jnp.square(
                model_output.value - minibatch.targets
            )
            value_loss_clipped = jnp.square(
                value_clipped - minibatch.targets
            )
            value_loss = 0.5 * jnp.maximum(
                value_loss_unclipped,
                value_loss_clipped,
            ).mean()

            normalized_advantages = (
                minibatch.advantages - minibatch.advantages.mean()
            ) / (minibatch.advantages.std() + 1e-8)
            log_ratio = (
                log_probability - minibatch.transitions.log_probability
            )
            ratio = jnp.exp(log_ratio)
            actor_loss = jnp.maximum(
                -ratio * normalized_advantages,
                -jnp.clip(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                )
                * normalized_advantages,
            ).mean()
            entropy = categorical_entropy(model_output.logits).mean()
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
                "approximate_kl": ((ratio - 1.0) - log_ratio).mean(),
                "clip_fraction": (
                    jnp.abs(ratio - 1.0) > config.clip_epsilon
                ).mean(),
            }
            return total_loss, metrics

        def update_minibatch(
            train_state: TrainState,
            minibatch: SkillMinibatch,
        ) -> tuple[TrainState, UpdateMetrics]:
            gradient_function = jax.value_and_grad(loss, has_aux=True)
            (_, metrics), gradients = gradient_function(
                train_state.params,
                minibatch,
            )
            return (
                train_state.apply_gradients(grads=gradients),
                metrics,
            )

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
            minibatches = SkillMinibatch(
                hidden=initial_recurrent_state[minibatch_indices],
                transitions=jax.tree_util.tree_map(
                    lambda value: jnp.swapaxes(
                        value[:, minibatch_indices],
                        0,
                        1,
                    ),
                    transitions,
                ),
                advantages=jnp.swapaxes(
                    advantages[:, minibatch_indices],
                    0,
                    1,
                ),
                targets=jnp.swapaxes(
                    targets[:, minibatch_indices],
                    0,
                    1,
                ),
            )
            train_state, metrics = jax.lax.scan(
                update_minibatch,
                carry.train_state,
                minibatches,
            )
            return (
                OptimizationCarry(train_state=train_state, rng=rng),
                jax.tree_util.tree_map(jnp.mean, metrics),
            )

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
        completed_episodes = transitions.completed_episode.sum()
        completed_successes = transitions.completed_success.sum()
        rollout_metrics = {
            "reward_mean": transitions.reward.mean(),
            "completed_episodes": completed_episodes,
            "completed_successes": completed_successes,
            "success_rate": jnp.where(
                completed_episodes > 0,
                completed_successes / completed_episodes,
                0.0,
            ),
        }
        runner = runner._replace(
            train_state=optimization.train_state,
            rng=optimization.rng,
        )
        return runner, {**optimization_metrics, **rollout_metrics}

    return cast(UpdateFunction, jax.jit(update))


def _host_metrics(metrics: UpdateMetrics) -> dict[str, float]:
    return {name: float(np.asarray(value)) for name, value in metrics.items()}


def train_goal_conditioned(
    config: SkillPPOConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and checkpoint the categorical goal-conditioned executor."""

    config = config.validate()
    if config.num_envs % len(SKILL_IDS):
        raise ValueError(
            f"num_envs must be divisible by {len(SKILL_IDS)} skills."
        )

    run_dir = Path(output_dir or config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    monitor = NvidiaSmiMonitor(
        enabled=config.monitor_gpu,
        interval_seconds=config.gpu_monitor_interval_seconds,
    )
    monitor.start()
    environment = make_vectorized_skill_environment(
        num_envs=config.num_envs,
        horizon=config.skill_horizon,
    )
    model = GoalConditionedActorCriticRNN(
        num_actions=environment.spec.num_actions,
        num_skills=environment.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    runner = initialize_skill_runner(config, environment, model)
    update = make_skill_update(config, environment, model)

    metrics_path = run_dir / "metrics.jsonl"
    start = time.perf_counter()
    compile_start = time.perf_counter()
    runner, device_metrics = update(runner)
    jax.block_until_ready(device_metrics)
    compile_seconds = time.perf_counter() - compile_start
    steady_state_start = monitor.elapsed_seconds()
    post_compile_start = time.perf_counter()
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
                    f"success={latest_metrics['success_rate']:.3f} "
                    f"sps={record['steps_per_second_including_compile']:.0f}"
                )

            if (
                config.checkpoint_every > 0
                and (update_index + 1) % config.checkpoint_every == 0
            ):
                timesteps = (update_index + 1) * config.batch_size
                save_checkpoint(
                    run_dir / "checkpoints" / f"step-{timesteps}",
                    runner.train_state.params,
                    config,
                    timesteps=timesteps,
                    metrics=latest_metrics,
                )

    jax.block_until_ready(runner.train_state.params)
    total_seconds = time.perf_counter() - start
    post_compile_seconds = time.perf_counter() - post_compile_start
    successes = np.asarray(runner.completed_successes)
    episodes = np.asarray(runner.completed_episodes)
    lengths = np.asarray(runner.completed_length_sum)
    per_skill = {
        skill_id.value: {
            "completed_episodes": int(episodes[index]),
            "success_rate": (
                float(successes[index] / episodes[index])
                if episodes[index]
                else None
            ),
            "mean_episode_length": (
                float(lengths[index] / episodes[index])
                if episodes[index]
                else None
            ),
        }
        for index, skill_id in enumerate(SKILL_IDS)
    }
    summary = {
        "environment": "Tiny-SIMA-Nearby-Skills-v0",
        "device": str(jax.devices()[0]),
        "timesteps": config.total_timesteps,
        "updates": config.num_updates,
        "num_envs": config.num_envs,
        "num_skills": len(SKILL_IDS),
        "skill_horizon": config.skill_horizon,
        "total_seconds": total_seconds,
        "compile_and_first_update_seconds": compile_seconds,
        "post_compile_seconds": post_compile_seconds,
        "steps_per_second_including_compile": (
            config.total_timesteps / total_seconds
        ),
        "latest_metrics": latest_metrics,
        "per_skill": per_skill,
        "jax_memory": jax_allocator_summary(jax.devices()[0]),
        "gpu_monitor": monitor.stop(
            run_dir,
            steady_state_start_seconds=steady_state_start,
        ),
    }
    checkpoint = save_checkpoint(
        run_dir / "checkpoints" / "final",
        runner.train_state.params,
        config,
        timesteps=config.total_timesteps,
        metrics=summary,
    )
    summary["checkpoint"] = str(checkpoint)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/skill_ppo_debug.yaml",
        help="Path to a goal-conditioned PPO YAML configuration.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory in the configuration.",
    )
    args = parser.parse_args()
    summary = train_goal_conditioned(
        load_skill_config(args.config),
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
