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
    scenario_randomization_from_config,
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
    normalize_advantages_by_group,
    stratified_minibatch_indices,
)
from tiny_sima.executor.training import run_training_loop, save_final_run
from tiny_sima.goals import SKILL_IDS

_ENVIRONMENT_NAMES = {
    "nearby": "Tiny-SIMA-Nearby-Skills-v0",
    "local_navigation": "Tiny-SIMA-Local-Navigation-Skills-v0",
    "randomized": "Tiny-SIMA-Randomized-Skills-v0",
}

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
    completed_feasible_timeout: jax.Array
    completed_precondition_loss: jax.Array
    completed_environment_termination: jax.Array


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
    episode_precondition_lost: jax.Array
    completed_successes: jax.Array
    completed_episodes: jax.Array
    completed_length_sum: jax.Array
    completed_feasible_timeouts: jax.Array
    completed_precondition_losses: jax.Array
    completed_environment_terminations: jax.Array


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
        f"macro_success={metrics['macro_success_rate']:.3f} "
        f"weighted_success={metrics['success_rate']:.3f} "
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
        episode_precondition_lost=jnp.zeros(
            (config.num_envs,),
            dtype=jnp.bool_,
        ),
        completed_successes=jnp.zeros((num_skills,), dtype=jnp.int32),
        completed_episodes=jnp.zeros((num_skills,), dtype=jnp.int32),
        completed_length_sum=jnp.zeros((num_skills,), dtype=jnp.int32),
        completed_feasible_timeouts=jnp.zeros(
            (num_skills,),
            dtype=jnp.int32,
        ),
        completed_precondition_losses=jnp.zeros(
            (num_skills,),
            dtype=jnp.int32,
        ),
        completed_environment_terminations=jnp.zeros(
            (num_skills,),
            dtype=jnp.int32,
        ),
    )


def make_skill_update(
    config: SkillPPOConfig,
    environment: VectorizedSkillEnvironment,
    model: GoalConditionedActorCriticRNN,
) -> UpdateFunction:
    """Build one compiled goal-conditioned rollout-and-update function."""

    num_skills = environment.num_skills
    if environment.envs_per_skill % config.num_minibatches:
        raise ValueError(
            "Goal-conditioned PPO requires envs_per_skill to be divisible "
            "by num_minibatches so every minibatch contains every skill."
        )

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
            episode_precondition_lost = (
                current.episode_precondition_lost
                | ~env_step.info["skill_precondition_after"]
            )
            failed = env_step.done & ~env_step.info["skill_success"]
            completed_precondition_loss = (failed & episode_precondition_lost).astype(
                jnp.int32
            )
            completed_feasible_timeout = (
                failed & env_step.info["skill_timeout"] & ~episode_precondition_lost
            ).astype(jnp.int32)
            completed_environment_termination = (
                failed & ~env_step.info["skill_timeout"] & ~episode_precondition_lost
            ).astype(jnp.int32)
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
                completed_feasible_timeout=completed_feasible_timeout,
                completed_precondition_loss=completed_precondition_loss,
                completed_environment_termination=(completed_environment_termination),
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
                episode_precondition_lost=jnp.where(
                    env_step.done,
                    False,
                    episode_precondition_lost,
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
                completed_feasible_timeouts=(
                    current.completed_feasible_timeouts
                    + (skill_one_hot * completed_feasible_timeout[:, None]).sum(axis=0)
                ),
                completed_precondition_losses=(
                    current.completed_precondition_losses
                    + (skill_one_hot * completed_precondition_loss[:, None]).sum(axis=0)
                ),
                completed_environment_terminations=(
                    current.completed_environment_terminations
                    + (skill_one_hot * completed_environment_termination[:, None]).sum(
                        axis=0
                    )
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
        if config.normalize_advantages_per_skill:
            advantages = normalize_advantages_by_group(
                advantages,
                transitions.skill_id,
                num_skills,
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
                group_ids=minibatch.transitions.skill_id,
                num_groups=num_skills,
                normalize_advantage=(not config.normalize_advantages_per_skill),
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
            minibatch_indices = stratified_minibatch_indices(
                permutation_rng,
                num_groups=num_skills,
                items_per_group=environment.envs_per_skill,
                num_minibatches=config.num_minibatches,
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
        membership = jax.nn.one_hot(
            transitions.skill_id,
            num_skills,
            dtype=jnp.int32,
        )

        def sum_by_skill(values: jax.Array) -> jax.Array:
            return (membership * values[..., None]).sum(axis=(0, 1))

        completed_by_skill = sum_by_skill(transitions.completed_episode)
        successes_by_skill = sum_by_skill(transitions.completed_success)
        feasible_timeouts_by_skill = sum_by_skill(
            transitions.completed_feasible_timeout
        )
        precondition_losses_by_skill = sum_by_skill(
            transitions.completed_precondition_loss
        )
        environment_terminations_by_skill = sum_by_skill(
            transitions.completed_environment_termination
        )
        success_rates_by_skill = jnp.where(
            completed_by_skill > 0,
            successes_by_skill / completed_by_skill,
            0.0,
        )
        represented_skills = completed_by_skill > 0
        macro_success_rate = jnp.where(
            represented_skills.any(),
            jnp.where(
                represented_skills,
                success_rates_by_skill,
                0.0,
            ).sum()
            / represented_skills.sum(),
            0.0,
        )
        completed_episodes = completed_by_skill.sum()
        completed_successes = successes_by_skill.sum()
        rollout_metrics = {
            "reward_mean": transitions.reward.mean(),
            "completed_episodes": completed_episodes,
            "completed_successes": completed_successes,
            "success_rate": jnp.where(
                completed_episodes > 0,
                completed_successes / completed_episodes,
                0.0,
            ),
            "macro_success_rate": macro_success_rate,
        }
        for index, skill_id in enumerate(SKILL_IDS):
            prefix = f"skill/{skill_id.value}"
            episodes = completed_by_skill[index]
            rollout_metrics[f"{prefix}/completed_episodes"] = episodes
            rollout_metrics[f"{prefix}/success_rate"] = success_rates_by_skill[index]
            rollout_metrics[f"{prefix}/feasible_timeout_rate"] = jnp.where(
                episodes > 0,
                feasible_timeouts_by_skill[index] / episodes,
                0.0,
            )
            rollout_metrics[f"{prefix}/precondition_lost_rate"] = jnp.where(
                episodes > 0,
                precondition_losses_by_skill[index] / episodes,
                0.0,
            )
            rollout_metrics[f"{prefix}/environment_termination_rate"] = jnp.where(
                episodes > 0,
                environment_terminations_by_skill[index] / episodes,
                0.0,
            )
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
    randomization = scenario_randomization_from_config(config)
    environment = make_vectorized_skill_environment(
        num_envs=config.num_envs,
        horizon=config.skill_horizon,
        scenario=config.skill_scenario,
        min_distance=config.skill_min_distance,
        max_distance=config.skill_max_distance,
        distractor_count=config.skill_distractor_count,
        randomization=randomization,
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
    feasible_timeouts = np.asarray(runner.completed_feasible_timeouts)
    precondition_losses = np.asarray(runner.completed_precondition_losses)
    environment_terminations = np.asarray(runner.completed_environment_terminations)
    per_skill = {
        skill_id.value: {
            "completed_episodes": int(episodes[index]),
            "success_rate": (
                float(successes[index] / episodes[index]) if episodes[index] else None
            ),
            "mean_episode_length": (
                float(lengths[index] / episodes[index]) if episodes[index] else None
            ),
            "feasible_timeout_rate": (
                float(feasible_timeouts[index] / episodes[index])
                if episodes[index]
                else None
            ),
            "precondition_lost_rate": (
                float(precondition_losses[index] / episodes[index])
                if episodes[index]
                else None
            ),
            "environment_termination_rate": (
                float(environment_terminations[index] / episodes[index])
                if episodes[index]
                else None
            ),
        }
        for index, skill_id in enumerate(SKILL_IDS)
    }
    per_skill_success_rates = [
        metrics["success_rate"]
        for metrics in per_skill.values()
        if metrics["success_rate"] is not None
    ]
    total_completed_episodes = int(episodes.sum())
    total_completed_successes = int(successes.sum())
    summary = {
        "environment": _ENVIRONMENT_NAMES[config.skill_scenario],
        "device": str(jax.devices()[0]),
        "timesteps": config.total_timesteps,
        "updates": config.num_updates,
        "num_envs": config.num_envs,
        "num_skills": len(SKILL_IDS),
        "skill_horizon": config.skill_horizon,
        "skill_scenario": config.skill_scenario,
        "skill_min_distance": config.skill_min_distance,
        "skill_max_distance": config.skill_max_distance,
        "skill_distractor_count": config.skill_distractor_count,
        "skill_randomization": (
            None if randomization is None else randomization.to_dict()
        ),
        "stratified_minibatches": True,
        "normalize_advantages_per_skill": (config.normalize_advantages_per_skill),
        "total_seconds": loop.total_seconds,
        "compile_and_first_update_seconds": (loop.compile_and_first_update_seconds),
        "post_compile_seconds": loop.post_compile_seconds,
        "steps_per_second_including_compile": (
            config.total_timesteps / loop.total_seconds
        ),
        "steps_per_second_after_compile": (loop.steps_per_second_after_compile),
        "tail_steps_per_second": loop.tail_steps_per_second,
        "macro_success_rate": (
            float(np.mean(per_skill_success_rates)) if per_skill_success_rates else None
        ),
        "episode_weighted_success_rate": (
            total_completed_successes / total_completed_episodes
            if total_completed_episodes
            else None
        ),
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
