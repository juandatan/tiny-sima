"""Evaluate categorical goal-conditioned executor checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action

from tiny_sima.config import SkillPPOConfig, load_skill_config
from tiny_sima.envs.skills import (
    RESOURCE_SKILLS,
    ScenarioRandomization,
    SkillEpisodeState,
    SkillScenario,
    make_skill_environment,
    reset_skill_batch,
    skills_for_scenario,
    step_skill_batch,
)
from tiny_sima.evaluation.skill_rollout import (
    SkillEpisodeTracker,
    SkillRolloutOutput,
    finish_skill_tracker,
    initialize_skill_tracker,
    update_skill_tracker,
)
from tiny_sima.executor.checkpoints import load_checkpoint
from tiny_sima.executor.model import (
    apply_goal_conditioned_actor_critic,
    categorical_entropy,
    initial_hidden,
    initialize_goal_conditioned_parameters,
    make_goal_conditioned_actor_critic,
    select_action,
)
from tiny_sima.goals import (
    SKILL_IDS,
    SkillId,
    executor_skill_ids,
    get_skill_spec,
    skill_index,
)


class GoalEvaluationCarry(NamedTuple):
    """State carried through a batched learned-skill evaluation."""

    rng: jax.Array
    state: SkillEpisodeState
    observation: jax.Array
    hidden: jax.Array
    tracker: SkillEpisodeTracker


class ActionProbe(NamedTuple):
    """Per-scenario action statistics at one controlled skill stage."""

    top_action: jax.Array
    top_action_correct: jax.Array
    correct_action_probability: jax.Array
    entropy: jax.Array


class InteractionSkillProbe(NamedTuple):
    """Action probes before and after an oracle supplies the correct turn."""

    initial_turn: ActionProbe
    after_oracle_turn: ActionProbe


class LoadedGoalConditionedCheckpoint(NamedTuple):
    """Parameters and configuration restored from one executor checkpoint."""

    params: Any
    config: SkillPPOConfig


def _require_executor_skill(config: SkillPPOConfig, skill_id: SkillId) -> None:
    # An out-of-range embedding index clamps rather than failing in JAX.
    if skill_id not in executor_skill_ids(config.num_skills):
        raise ValueError(
            f"An executor with {config.num_skills} skills cannot run "
            f"{skill_id.value}."
        )


def _probe_actions(
    logits: jax.Array,
    correct_action: jax.Array,
) -> ActionProbe:
    probabilities = jax.nn.softmax(logits, axis=-1)
    top_action = jnp.argmax(logits, axis=-1)
    return ActionProbe(
        top_action=top_action,
        top_action_correct=top_action == correct_action,
        correct_action_probability=jnp.take_along_axis(
            probabilities,
            correct_action[..., None],
            axis=-1,
        )[..., 0],
        entropy=categorical_entropy(logits),
    )


def _action_name(action: int) -> str:
    return Action(action).name.lower()


def _summarize_action_probe(
    probe: ActionProbe,
    *,
    target: str,
    num_actions: int,
) -> dict[str, Any]:
    top_actions = np.asarray(probe.top_action)
    action_values, action_counts = np.unique(top_actions, return_counts=True)
    action_histogram = {
        _action_name(int(action)): float(count / len(top_actions))
        for action, count in zip(action_values, action_counts, strict=True)
    }
    mean_entropy = float(np.asarray(probe.entropy).mean())
    return {
        "target": target,
        "top1_accuracy": float(np.asarray(probe.top_action_correct).mean()),
        "mean_target_probability": float(
            np.asarray(probe.correct_action_probability).mean()
        ),
        "mean_action_entropy": mean_entropy,
        "normalized_action_entropy": (mean_entropy / float(np.log(num_actions))),
        "top1_action_histogram": action_histogram,
    }


def diagnose_goal_conditioned_skill(
    params: Any,
    config: SkillPPOConfig,
    skill_id: SkillId | str,
    *,
    num_scenarios: int = 256,
    seed: int = 1,
) -> dict[str, Any]:
    """Probe action ranking at controlled stages of one requested skill."""

    if num_scenarios < 1:
        raise ValueError("num_scenarios must be positive.")

    normalized_skill_id = SkillId(skill_id)
    _require_executor_skill(config, normalized_skill_id)
    environment = make_skill_environment(
        normalized_skill_id,
        horizon=config.skill_horizon,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=config.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    categorical_skill_id = skill_index(normalized_skill_id)
    skill_ids = jnp.full(
        (num_scenarios,),
        categorical_skill_id,
        dtype=jnp.int32,
    )

    if normalized_skill_id in RESOURCE_SKILLS:

        def probe_interaction(rng: jax.Array) -> InteractionSkillProbe:
            reset_rng, step_rng = jax.random.split(rng)
            reset = reset_skill_batch(
                environment,
                reset_rng,
                num_scenarios,
            )
            hidden = initial_hidden(num_scenarios, config.hidden_size)
            initial_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                hidden,
                (
                    reset.observation[None, ...],
                    skill_ids[None, ...],
                    jnp.zeros(
                        (1, num_scenarios),
                        dtype=jnp.bool_,
                    ),
                ),
            )
            correct_turn = reset.state.target_direction
            initial_probe = _probe_actions(
                initial_output.logits[0],
                correct_turn,
            )
            oracle_step = step_skill_batch(
                environment,
                step_rng,
                reset.state,
                correct_turn,
            )
            followup_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                initial_output.hidden,
                (
                    oracle_step.observation[None, ...],
                    skill_ids[None, ...],
                    oracle_step.done[None, ...],
                ),
            )
            correct_interaction = jnp.full(
                (num_scenarios,),
                Action.DO.value,
                dtype=jnp.int32,
            )
            return InteractionSkillProbe(
                initial_turn=initial_probe,
                after_oracle_turn=_probe_actions(
                    followup_output.logits[0],
                    correct_interaction,
                ),
            )

        probe = jax.jit(probe_interaction)(jax.random.PRNGKey(seed))
        initial = _summarize_action_probe(
            probe.initial_turn,
            target="face_target",
            num_actions=environment.bundle.spec.num_actions,
        )
        followup = _summarize_action_probe(
            probe.after_oracle_turn,
            target=_action_name(Action.DO.value),
            num_actions=environment.bundle.spec.num_actions,
        )
        initial_correct = np.asarray(probe.initial_turn.top_action_correct)
        followup_correct = np.asarray(probe.after_oracle_turn.top_action_correct)
        estimated_sequence_probability = np.asarray(
            probe.initial_turn.correct_action_probability
            * probe.after_oracle_turn.correct_action_probability
        )
        return {
            "skill_id": normalized_skill_id.value,
            "skill_structure": "turn_then_interact",
            "num_scenarios": num_scenarios,
            "seed": seed,
            "initial": initial,
            "after_oracle_turn": followup,
            "deterministic_sequence_accuracy": float(
                (initial_correct & followup_correct).mean()
            ),
            "estimated_stochastic_sequence_probability": float(
                estimated_sequence_probability.mean()
            ),
        }

    correct_action_value = get_skill_spec(normalized_skill_id).interaction_action.value

    def probe_direct(rng: jax.Array) -> ActionProbe:
        reset = reset_skill_batch(
            environment,
            rng,
            num_scenarios,
        )
        output = apply_goal_conditioned_actor_critic(
            model,
            params,
            initial_hidden(num_scenarios, config.hidden_size),
            (
                reset.observation[None, ...],
                skill_ids[None, ...],
                jnp.zeros((1, num_scenarios), dtype=jnp.bool_),
            ),
        )
        correct_action = jnp.full(
            (num_scenarios,),
            correct_action_value,
            dtype=jnp.int32,
        )
        return _probe_actions(output.logits[0], correct_action)

    direct_probe = jax.jit(probe_direct)(jax.random.PRNGKey(seed))
    initial = _summarize_action_probe(
        direct_probe,
        target=_action_name(correct_action_value),
        num_actions=environment.bundle.spec.num_actions,
    )
    return {
        "skill_id": normalized_skill_id.value,
        "skill_structure": "direct",
        "num_scenarios": num_scenarios,
        "seed": seed,
        "initial": initial,
        "after_oracle_turn": None,
        "deterministic_sequence_accuracy": initial["top1_accuracy"],
        "estimated_stochastic_sequence_probability": initial["mean_target_probability"],
    }


def diagnose_goal_conditioned_policy(
    params: Any,
    config: SkillPPOConfig,
    *,
    num_scenarios_per_skill: int = 256,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Run controlled action probes for every requested skill."""

    return [
        diagnose_goal_conditioned_skill(
            params,
            config,
            skill_id,
            num_scenarios=num_scenarios_per_skill,
            seed=seed + index,
        )
        for index, skill_id in enumerate(executor_skill_ids(config.num_skills))
    ]


def evaluate_goal_conditioned_skill(
    params: Any,
    config: SkillPPOConfig,
    skill_id: SkillId | str,
    *,
    num_episodes: int = 256,
    seed: int = 1,
    deterministic: bool = True,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    horizon: int | None = None,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
    randomization: ScenarioRandomization | None = None,
) -> dict[str, Any]:
    """Evaluate one requested skill over independently generated scenarios."""

    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    normalized_skill_id = SkillId(skill_id)
    _require_executor_skill(config, normalized_skill_id)
    evaluation_horizon = config.skill_horizon if horizon is None else horizon
    environment = make_skill_environment(
        normalized_skill_id,
        horizon=evaluation_horizon,
        scenario=scenario,
        min_distance=min_distance,
        max_distance=max_distance,
        distractor_count=distractor_count,
        randomization=randomization,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=config.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    categorical_skill_id = skill_index(normalized_skill_id)

    def rollout(rng: jax.Array) -> SkillRolloutOutput:
        rng, reset_rng = jax.random.split(rng)
        reset = reset_skill_batch(environment, reset_rng, num_episodes)
        initial_carry = GoalEvaluationCarry(
            rng=rng,
            state=reset.state,
            observation=reset.observation,
            hidden=initial_hidden(num_episodes, config.hidden_size),
            tracker=initialize_skill_tracker(num_episodes),
        )
        skill_ids = jnp.full(
            (num_episodes,),
            categorical_skill_id,
            dtype=jnp.int32,
        )

        def step(
            carry: GoalEvaluationCarry,
            _unused: None,
        ) -> tuple[GoalEvaluationCarry, None]:
            rng, action_rng, env_rng = jax.random.split(carry.rng, 3)
            model_output = apply_goal_conditioned_actor_critic(
                model,
                params,
                carry.hidden,
                (
                    carry.observation[None, ...],
                    skill_ids[None, ...],
                    carry.tracker.finished[None, ...],
                ),
            )
            logits = model_output.logits[0]
            action = select_action(
                logits,
                action_rng,
                deterministic=deterministic,
            )
            result = step_skill_batch(
                environment,
                env_rng,
                carry.state,
                action,
            )
            return (
                GoalEvaluationCarry(
                    rng=rng,
                    state=result.state,
                    observation=result.observation,
                    hidden=model_output.hidden,
                    tracker=update_skill_tracker(carry.tracker, result),
                ),
                None,
            )

        final_carry, _ = jax.lax.scan(
            step,
            initial_carry,
            None,
            length=evaluation_horizon,
        )
        return finish_skill_tracker(final_carry.tracker)

    output = jax.jit(rollout)(jax.random.PRNGKey(seed))
    succeeded = np.asarray(output.succeeded)
    episode_return = np.asarray(output.episode_return)
    episode_length = np.asarray(output.episode_length)
    failed = ~succeeded
    precondition_lost = np.asarray(output.precondition_lost) & failed
    feasible_timeout = np.asarray(output.timed_out) & failed & ~precondition_lost
    environment_terminated = (
        np.asarray(output.environment_terminated) & failed & ~precondition_lost
    )
    return {
        "skill_id": normalized_skill_id.value,
        "scenario": environment.scenario.kind.value,
        "num_episodes": num_episodes,
        "horizon": evaluation_horizon,
        "min_distance": environment.scenario.min_distance,
        "max_distance": environment.scenario.max_distance,
        "distractor_count": environment.scenario.distractor_count,
        "randomization": (
            None
            if environment.scenario.randomization is None
            else environment.scenario.randomization.to_dict()
        ),
        "seed": seed,
        "deterministic": deterministic,
        "success_rate": float(succeeded.mean()),
        "mean_return": float(episode_return.mean()),
        "mean_episode_length": float(episode_length.mean()),
        "precondition_lost_rate": float(precondition_lost.mean()),
        "feasible_timeout_rate": float(feasible_timeout.mean()),
        "environment_termination_rate": float(environment_terminated.mean()),
    }


def evaluate_goal_conditioned_policy(
    params: Any,
    config: SkillPPOConfig,
    *,
    num_episodes_per_skill: int = 256,
    seed: int = 1,
    deterministic: bool = True,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    horizon: int | None = None,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
    randomization: ScenarioRandomization | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every requested skill independently."""

    scenario_skills = skills_for_scenario(
        scenario,
        executor_skill_ids(config.num_skills),
    )
    return [
        evaluate_goal_conditioned_skill(
            params,
            config,
            skill_id,
            num_episodes=num_episodes_per_skill,
            seed=seed + index,
            deterministic=deterministic,
            scenario=scenario,
            horizon=horizon,
            min_distance=min_distance,
            max_distance=max_distance,
            distractor_count=distractor_count,
            randomization=randomization,
        )
        for index, skill_id in enumerate(scenario_skills)
    ]


def evaluate_goal_conditioned_checkpoint(
    checkpoint_dir: str | Path,
    *,
    num_episodes_per_skill: int = 256,
    seed: int = 1,
    deterministic: bool = True,
    scenario: SkillScenario | str = SkillScenario.NEARBY,
    horizon: int | None = None,
    min_distance: int = 2,
    max_distance: int = 4,
    distractor_count: int = 6,
    randomization: ScenarioRandomization | None = None,
) -> list[dict[str, Any]]:
    """Load and evaluate a saved goal-conditioned executor."""

    checkpoint = load_goal_conditioned_checkpoint(checkpoint_dir)
    return evaluate_goal_conditioned_policy(
        checkpoint.params,
        checkpoint.config,
        num_episodes_per_skill=num_episodes_per_skill,
        seed=seed,
        deterministic=deterministic,
        scenario=scenario,
        horizon=horizon,
        min_distance=min_distance,
        max_distance=max_distance,
        distractor_count=distractor_count,
        randomization=randomization,
    )


def diagnose_goal_conditioned_checkpoint(
    checkpoint_dir: str | Path,
    *,
    num_scenarios_per_skill: int = 256,
    seed: int = 1,
) -> list[dict[str, Any]]:
    """Load a checkpoint and probe its action ranking for every skill."""

    checkpoint = load_goal_conditioned_checkpoint(checkpoint_dir)
    return diagnose_goal_conditioned_policy(
        checkpoint.params,
        checkpoint.config,
        num_scenarios_per_skill=num_scenarios_per_skill,
        seed=seed,
    )


def load_goal_conditioned_checkpoint(
    checkpoint_dir: str | Path,
) -> LoadedGoalConditionedCheckpoint:
    """Load executor parameters and configuration for custom evaluations."""

    checkpoint_path = Path(checkpoint_dir)
    config = load_skill_config(checkpoint_path / "config.yaml")
    environment = make_skill_environment(
        SKILL_IDS[0],
        horizon=config.skill_horizon,
    )
    model = make_goal_conditioned_actor_critic(
        num_actions=environment.bundle.spec.num_actions,
        num_skills=config.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    template_params = initialize_goal_conditioned_parameters(
        model,
        jax.random.PRNGKey(0),
        environment.bundle.spec.observation_shape,
    )
    params, _ = load_checkpoint(checkpoint_path, template_params)
    return LoadedGoalConditionedCheckpoint(
        params=params,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Goal-conditioned checkpoint directory.")
    parser.add_argument("--episodes-per-skill", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--scenario",
        choices=tuple(scenario.value for scenario in SkillScenario),
        default=SkillScenario.NEARBY.value,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override the checkpoint's skill horizon for evaluation.",
    )
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--max-distance", type=int, default=4)
    parser.add_argument("--distractor-count", type=int, default=6)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of selecting the most likely action.",
    )
    args = parser.parse_args()
    results = evaluate_goal_conditioned_checkpoint(
        args.checkpoint,
        num_episodes_per_skill=args.episodes_per_skill,
        seed=args.seed,
        deterministic=not args.stochastic,
        scenario=args.scenario,
        horizon=args.horizon,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        distractor_count=args.distractor_count,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
