"""Goal-conditioned recurrent PPO training for the Tiny-SIMA skill suite."""

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
from tiny_sima.executor.model import (
    GoalConditionedActorCriticRNN,
    apply_goal_conditioned_actor_critic,
    categorical_log_probability,
    initial_hidden,
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
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


UpdateFunction = Callable[
    [SkillRunnerState],
    tuple[SkillRunnerState, UpdateMetrics],
]


def _runner_params(runner: SkillRunnerState) -> Any:
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
        f"success={metrics['success_rate']:.3f} "
        f"sps={record['steps_per_second_including_compile']:.0f}"
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
    train_state = create_train_state(model, params, config)
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
            completed_success = (env_step.done & env_step.info["skill_success"]).astype(
                jnp.int32
            )
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
            minibatch: RecurrentMinibatch[SkillTransition],
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
            return clipped_ppo_loss(
                model_output.logits,
                model_output.value,
                PPOLossBatch(
                    action=minibatch.transitions.action,
                    old_value=minibatch.transitions.value,
                    old_log_probability=(minibatch.transitions.log_probability),
                    advantage=minibatch.advantages,
                    target=minibatch.targets,
                ),
                config,
            )

        def update_minibatch(
            train_state: TrainState,
            minibatch: RecurrentMinibatch[SkillTransition],
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
            minibatches = RecurrentMinibatch(
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


def train_goal_conditioned(
    config: SkillPPOConfig,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and checkpoint the categorical goal-conditioned executor."""

    config = config.validate()
    if config.num_envs % len(SKILL_IDS):
        raise ValueError(f"num_envs must be divisible by {len(SKILL_IDS)} skills.")

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
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.spec.num_actions,
        num_skills=environment.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    runner = initialize_skill_runner(config, environment, model)
    update = make_skill_update(config, environment, model)

    loop = run_training_loop(
        config=config,
        run_dir=run_dir,
        initial_runner=runner,
        update=update,
        monitor=monitor,
        get_params=_runner_params,
        format_progress=_format_progress,
    )
    runner = loop.runner
    successes = np.asarray(runner.completed_successes)
    episodes = np.asarray(runner.completed_episodes)
    lengths = np.asarray(runner.completed_length_sum)
    per_skill = {
        skill_id.value: {
            "completed_episodes": int(episodes[index]),
            "success_rate": (
                float(successes[index] / episodes[index]) if episodes[index] else None
            ),
            "mean_episode_length": (
                float(lengths[index] / episodes[index]) if episodes[index] else None
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
        "total_seconds": loop.total_seconds,
        "compile_and_first_update_seconds": (loop.compile_and_first_update_seconds),
        "post_compile_seconds": loop.post_compile_seconds,
        "steps_per_second_including_compile": (
            config.total_timesteps / loop.total_seconds
        ),
        "steps_per_second_after_compile": (loop.steps_per_second_after_compile),
        "tail_steps_per_second": loop.tail_steps_per_second,
        "latest_metrics": loop.latest_metrics,
        "per_skill": per_skill,
        "jax_memory": jax_allocator_summary(jax.devices()[0]),
        "gpu_monitor": monitor.stop(
            run_dir,
            steady_state_start_seconds=loop.steady_state_start_seconds,
            tail_start_seconds=loop.tail_gpu_start_seconds,
        ),
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
